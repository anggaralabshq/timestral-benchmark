"""Adapter model. Kontrak semua adapter:

    fn(history: pd.Series, horizon: int, cov: pd.DataFrame | None) -> pd.DataFrame

- history: seri harian (boleh NaN), panjang <= 1024
- cov: covariates untuk history.index ∪ `horizon` hari ke depan; None = tanpa covariates
- return: index = `horizon` tanggal setelah history.index[-1]; kolom float QUANTILES (+ EXTRA_Q jika ada)
"""

import functools
import logging
from typing import Callable

import numpy as np
import pandas as pd

from pangan.metrics import EXTRA_Q, QUANTILES

FOUNDATION = {"timestral_5m", "timestral_209m", "chronos2", "timesfm25"}
LEVELS = (20, 40, 60, 80, 90)  # interval simetris -> kuantil 0.05..0.95


def fill_for_model(history: pd.Series) -> pd.Series:
    return history.ffill().bfill()


def _future_index(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def _split_cov(cov: pd.DataFrame, history: pd.Series, horizon: int):
    """Covariates untuk histori dan masa depan; kolom konstan di histori dibuang (singular untuk regresi)."""
    fut = _future_index(history, horizon)
    h, f = cov.loc[history.index], cov.loc[fut]
    keep = [c for c in h.columns if h[c].nunique() > 1]
    return h[keep].astype(float), f[keep].astype(float)


# ---------- statsforecast ----------

def _statsforecast(model, history: pd.Series, horizon: int, cov: pd.DataFrame | None) -> pd.DataFrame:
    from statsforecast import StatsForecast

    y = fill_for_model(history)
    df = pd.DataFrame({"unique_id": "s", "ds": y.index, "y": y.values})
    X_df = None
    if cov is not None:
        h, f = _split_cov(cov, history, horizon)
        df = pd.concat([df, h.reset_index(drop=True)], axis=1)
        X_df = pd.concat([pd.DataFrame({"unique_id": "s", "ds": f.index}), f.reset_index(drop=True)], axis=1)
    sf = StatsForecast(models=[model], freq="D")
    fc = sf.forecast(df=df, h=horizon, level=list(LEVELS), X_df=X_df)
    name = repr(model)
    out = pd.DataFrame(index=_future_index(history, horizon))
    out[0.5] = fc[name].values
    for lv in LEVELS:
        out[round(0.5 - lv / 200, 2)] = fc[f"{name}-lo-{lv}"].values
        out[round(0.5 + lv / 200, 2)] = fc[f"{name}-hi-{lv}"].values
    return out[sorted(out.columns)]


def snaive(history, horizon, cov=None):
    from statsforecast.models import SeasonalNaive
    return _statsforecast(SeasonalNaive(season_length=7), history, horizon, None)


def ets(history, horizon, cov=None):
    from statsforecast.models import AutoETS
    return _statsforecast(AutoETS(), history, horizon, None)


def arima(history, horizon, cov=None):
    from statsforecast.models import AutoARIMA
    return _statsforecast(AutoARIMA(), history, horizon, cov)


# ---------- Prophet ----------

def prophet(history, horizon, cov=None):
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    df = pd.DataFrame({"ds": history.index, "y": history.values})  # Prophet menerima y NaN
    fut = pd.DataFrame({"ds": _future_index(history, horizon)})
    m = Prophet(uncertainty_samples=1000)
    if cov is not None:
        h, f = _split_cov(cov, history, horizon)
        m.add_country_holidays(country_name="ID")
        for c in h.columns:
            m.add_regressor(c)
        df = pd.concat([df, h.reset_index(drop=True)], axis=1)
        fut = pd.concat([fut, f.reset_index(drop=True)], axis=1)
    m.fit(df)
    samples = m.predictive_samples(fut)["yhat"]  # (horizon, n_samples)
    levels = sorted(QUANTILES + EXTRA_Q)
    q = np.quantile(samples, levels, axis=1).T
    return pd.DataFrame(q, index=_future_index(history, horizon), columns=levels)


# ---------- foundation model (CPU, dimuat sekali per proses) ----------

def _frame(q: np.ndarray, levels, history, horizon) -> pd.DataFrame:
    out = pd.DataFrame(np.asarray(q, dtype=float), index=_future_index(history, horizon), columns=[round(l, 2) for l in levels])
    return out[[c for c in out.columns if c in QUANTILES + EXTRA_Q]]


@functools.cache
def _timestral(repo: str):
    from timestral import TimestralForecaster
    return TimestralForecaster.from_pretrained(repo, device="cpu")


def _timestral_fn(repo: str):
    def fn(history, horizon, cov=None):
        m = _timestral(repo)
        known = None
        if cov is not None:
            idx = history.index.append(_future_index(history, horizon))
            known = cov.loc[idx].to_numpy(dtype=np.float32).T[None]  # (1, C, T+H)
        fc = m.forecast(history.to_numpy(dtype=np.float32), prediction_length=horizon, known_covariates=known)
        return _frame(fc.quantiles[0, 0].numpy(), fc.quantile_levels, history, horizon)  # NaN didukung native
    return fn


@functools.cache
def _chronos2():
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")


def chronos2(history, horizon, cov=None):
    inp = {"target": history.to_numpy(dtype=np.float32)}  # NaN didukung native
    if cov is not None:
        fut = _future_index(history, horizon)
        inp["past_covariates"] = {c: cov.loc[history.index, c].to_numpy(dtype=np.float32) for c in cov.columns}
        inp["future_covariates"] = {c: cov.loc[fut, c].to_numpy(dtype=np.float32) for c in cov.columns}
    levels = sorted(QUANTILES + EXTRA_Q)
    q, _ = _chronos2().predict_quantiles([inp], prediction_length=horizon, quantile_levels=levels)
    return _frame(q[0][0].numpy(), levels, history, horizon)


@functools.cache
def _timesfm():
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    m.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=128, normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True,
        return_backcast=True,  # wajib untuk XReg; dipakai di kedua mode agar setara
    ))
    return m


def timesfm25(history, horizon, cov=None):
    m = _timesfm()
    y = fill_for_model(history).to_numpy(dtype=float)  # TimesFM tidak menerima NaN
    if cov is None:
        _, q = m.forecast(horizon=horizon, inputs=[y])
        q = q[0][:horizon]
    else:
        idx = history.index.append(_future_index(history, horizon))
        dyn = {c: [cov.loc[idx, c].to_numpy(dtype=float)] for c in cov.columns}
        _, qs = m.forecast_with_covariates(inputs=[y], dynamic_numerical_covariates=dyn)
        q = np.asarray(qs[0])[:horizon]
    return _frame(q[:, 1:], QUANTILES, history, horizon)  # kolom 0 = mean, 1..9 = q0.1..0.9


MODELS: dict[str, tuple[Callable, bool]] = {
    "snaive": (snaive, False),
    "ets": (ets, False),
    "arima": (arima, True),
    "prophet": (prophet, True),
    "timestral_5m": (_timestral_fn("mistralai/Timestral-5M"), True),
    "timestral_209m": (_timestral_fn("mistralai/Timestral-209M"), True),
    "chronos2": (chronos2, True),
    "timesfm25": (timesfm25, True),
}
