"""Headline chart: prices before Eid al-Fitr 2026, Timestral vs the best rival (English for README, Indonesian for the post)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tsbench.backtest import LEBARAN_WINDOWS

NAME = {"snaive": "Seasonal naive", "ets": "ETS", "arima": "ARIMA", "prophet": "Prophet",
        "chronos2": "Chronos-2", "timesfm25": "TimesFM 2.5"}
TEXT = {
    "en": {"cabai_merah": "Curly red chili", "cabai_rawit": "Red bird's eye chili", "bawang_merah": "Shallot",
           "beras": "Medium rice", "actual": "Actual price", "band": "Timestral: 80% interval",
           "main": "Timestral 209M (+calendar)", "cal": " (+calendar)", "best": "best rival", "eid": " Eid al-Fitr",
           "start": " forecast start", "title": "{c} before Eid al-Fitr 2026\n28-day forecast from {d}",
           "y": "IDR thousand / kg", "src": "Data: PIHPS Nasional, Bank Indonesia. Models run zero-shot on a CPU laptop."},
    "id": {"cabai_merah": "Cabai merah keriting", "cabai_rawit": "Cabai rawit merah", "bawang_merah": "Bawang merah",
           "beras": "Beras medium", "actual": "Harga aktual", "band": "Timestral: interval 80%",
           "main": "Timestral 209M (+kalender)", "cal": " (+kalender)", "best": "pesaing terbaik", "eid": " Lebaran",
           "start": " mulai ramal", "title": "{c} menjelang Lebaran 2026\nramalan 28 hari dari {d}",
           "y": "Rp ribu / kg", "src": "Sumber data: PIHPS Nasional, Bank Indonesia. Model dijalankan zero-shot di CPU laptop."},
}


def pick_commodity(series: dict[str, pd.Series]) -> str:
    """Cabai dengan lonjakan terbesar (maks 45 hari pra-Lebaran / median Januari 2026). Tidak melihat hasil model."""
    lo, hi = LEBARAN_WINDOWS[1]
    jump = {c: series[c][str(lo):str(hi)].max() / series[c]["2026-01"].median() for c in ("cabai_merah", "cabai_rawit")}
    return max(jump, key=jump.get)


def best_rival(summary: pd.DataFrame) -> tuple[str, bool]:
    r = summary[~summary.model.str.startswith("timestral") & (summary.slice == "all") & (summary.horizon == 28)]
    best = r.loc[r.rel_mase.idxmin()]
    return best.model, bool(best["cov"])


def plot(series: dict[str, pd.Series], summary: pd.DataFrame, forecast_dir: Path, out: Path,
         commodity: str | None = None, lang: str = "en") -> Path:
    tx = TEXT[lang]
    commodity = commodity or pick_commodity(series)
    lo, lebaran = LEBARAN_WINDOWS[1]
    origin = max(o for o in pd.date_range(lo, lebaran, freq="W-MON") if o <= pd.Timestamp(lebaran) - pd.Timedelta(days=14))
    rival, rival_cov = best_rival(summary)

    def fc(model, cov):
        df = pd.read_parquet(Path(forecast_dir) / f"{model}__{'cov' if cov else 'nocov'}.parquet")
        return df[(df.commodity == commodity) & (df.origin == origin)].set_index("date")

    t, r = fc("timestral_209m", True), fc(rival, rival_cov)
    actual = series[commodity]["2026-01-01":"2026-04-30"].dropna()

    plt.rcParams.update({"font.size": 20})
    fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
    ax.plot(actual.index, actual / 1000, color="#222", lw=2.5, label=tx["actual"])
    ax.fill_between(t.index, t["q0.1"] / 1000, t["q0.9"] / 1000, color="#e8590c", alpha=0.22, label=tx["band"])
    ax.plot(t.index, t["q0.5"] / 1000, color="#e8590c", lw=3, label=tx["main"])
    ax.plot(r.index, r["q0.5"] / 1000, color="#1c7ed6", lw=3, ls="--",
            label=f"{NAME[rival]}{tx['cal'] if rival_cov else ''}, {tx['best']}")
    ax.axvline(pd.Timestamp(lebaran), color="#888", ls=":", lw=2)
    ax.text(pd.Timestamp(lebaran), ax.get_ylim()[1], tx["eid"], va="top", color="#555")
    ax.axvline(origin, color="#bbb", lw=1.5)
    ax.text(origin, ax.get_ylim()[0], tx["start"], va="bottom", color="#777", fontsize=16)
    ax.set_title(tx["title"].format(c=tx[commodity], d=f"{origin:%d %b %Y}"), loc="left")
    ax.set_ylabel(tx["y"])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=16, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.01, 0.01, tx["src"], fontsize=13, color="#777")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = Path(out)
    fig.savefig(out)
    plt.close(fig)
    return out
