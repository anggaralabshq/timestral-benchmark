"""Backtest rolling-origin. Origin = hari pertama yang diramal; histori = 1024 hari sebelumnya."""

from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pangan.calendar import covariates
from pangan.metrics import EXTRA_Q, QUANTILES, coverage, mase, sort_quantiles, wql

HISTORY = 1024
HORIZON = 28
FIRST_ORIGIN = date(2025, 1, 6)
LEBARAN_WINDOWS = [(date(2025, 3, 31) - timedelta(days=45), date(2025, 3, 31)),
                   (date(2026, 3, 20) - timedelta(days=45), date(2026, 3, 20))]
QCOLS = [f"q{q}" for q in sorted(QUANTILES + EXTRA_Q)]


def origins(last_date: date) -> list[date]:
    return [d.date() for d in pd.date_range(FIRST_ORIGIN, last_date - timedelta(days=HORIZON), freq="W-MON")]


def in_lebaran_window(origin: date) -> bool:
    return any(lo <= origin <= hi for lo, hi in LEBARAN_WINDOWS)


def load_series(raw_dir: Path) -> dict[str, pd.Series]:
    from pangan.pihps import COMMODITIES, load
    from pangan.series import to_daily
    df = load(raw_dir)
    return {c: to_daily(df, c) for c in COMMODITIES}


def _history(s: pd.Series, origin: date) -> pd.Series:
    o = pd.Timestamp(origin)
    return s[o - pd.Timedelta(days=HISTORY):o - pd.Timedelta(days=1)]


def _one(args) -> pd.DataFrame:
    fn, use_cov, name, s, origin = args
    hist = _history(s, origin)
    fut = pd.date_range(pd.Timestamp(origin), periods=HORIZON, freq="D")
    cov = covariates(hist.index.append(fut)) if use_cov else None
    fc = fn(hist, HORIZON, cov)
    out = pd.DataFrame({"commodity": name, "origin": pd.Timestamp(origin), "date": fut,
                        "h": np.arange(1, HORIZON + 1), "y": s.reindex(fut).to_numpy()})
    for q, col in zip(sorted(QUANTILES + EXTRA_Q), QCOLS):
        out[col] = fc[q].to_numpy() if q in fc.columns else np.nan
    return out


def run(model: str, use_cov: bool, series: dict[str, pd.Series], out_dir: Path, models=None, jobs: int = 1) -> Path:
    if models is None:
        from pangan.models import MODELS as models
    fn, supports = models[model]
    assert supports or not use_cov, f"{model} tidak mendukung covariates"
    last = min(s.dropna().index[-1] for s in series.values()).date()
    tasks = [(fn, use_cov, n, s, o) for n, s in series.items() for o in origins(last)]
    if jobs > 1:  # ponytail: proses paralel hanya untuk model statistik; foundation model sudah multi-thread
        with ProcessPoolExecutor(jobs) as ex:
            parts = list(ex.map(_one, tasks, chunksize=4))
    else:
        parts = [_one(t) for t in tasks]
    path = Path(out_dir) / "forecasts" / f"{model}__{'cov' if use_cov else 'nocov'}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(path)
    return path


def evaluate(forecast_dir: Path, series: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(forecast_dir).glob("*.parquet")):
        model, cov = f.stem.split("__")
        df = pd.read_parquet(f)
        for (name, origin), g in df.groupby(["commodity", "origin"]):
            g = g.sort_values("h")
            hist = _history(series[name], origin.date()).to_numpy()
            q = sort_quantiles(g[[f"q{x}" for x in QUANTILES]].to_numpy())
            for H in (14, 28):
                y = g.y.to_numpy()[:H]
                lo95, hi95 = g["q0.05"].to_numpy()[:H], g["q0.95"].to_numpy()[:H]
                rows.append({
                    "model": model, "cov": cov == "cov", "commodity": name, "origin": origin.date(), "horizon": H,
                    "mase": mase(y, q[:H, 4], hist), "wql": wql(y, q[:H]),
                    "cov80": coverage(y, q[:H, 0], q[:H, -1]),
                    "cov90": coverage(y, lo95, hi95) if not np.isnan(lo95).any() else np.nan,
                    "lebaran": in_lebaran_window(origin.date()),
                })
    return pd.DataFrame(rows)


def _geo(x: pd.Series) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    key = ["commodity", "origin", "horizon"]
    base = metrics[(metrics.model == "snaive") & ~metrics["cov"]][key + ["mase", "wql"]]
    m = metrics.merge(base, on=key, suffixes=("", "_base"))
    m["r_mase"], m["r_wql"] = m.mase / m.mase_base, m.wql / m.wql_base
    out = []
    for sl, part in [("all", m), ("lebaran", m[m.lebaran])]:
        for (model, cov, H), g in part.groupby(["model", "cov", "horizon"]):
            out.append({"model": model, "cov": cov, "slice": sl, "horizon": H, "rel_mase": _geo(g.r_mase),
                        "rel_wql": _geo(g.r_wql), "cov80": g.cov80.mean(), "cov90": g.cov90.mean(), "n": len(g)})
    return pd.DataFrame(out)
