"""Benchmark CPU untuk K3. Jalankan saat tidak ada proses berat lain."""

import platform
import statistics
import time
from pathlib import Path

import pandas as pd

from pangan.backtest import HISTORY, HORIZON
from pangan.calendar import covariates

REPOS = {
    "timestral_5m": "mistralai/Timestral-5M",
    "timestral_209m": "mistralai/Timestral-209M",
    "chronos2": "amazon/chronos-2",
    "timesfm25": "google/timesfm-2.5-200m-pytorch",
}


def _loader(model: str):
    from pangan import models as m
    return {"timestral_5m": lambda: m._timestral(REPOS["timestral_5m"]),
            "timestral_209m": lambda: m._timestral(REPOS["timestral_209m"]),
            "chronos2": m._chronos2, "timesfm25": m._timesfm}[model]


def _file_mb(repo: str) -> float:
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(repo, local_files_only=True))
    return sum(f.stat().st_size for f in d.rglob("*") if f.suffix in {".safetensors", ".bin", ".ckpt", ".pt"}) / 1e6


def speed(model: str, series: dict[str, pd.Series], runs: int = 5) -> dict:
    from pangan.models import MODELS
    fn = MODELS[model][0]
    loader = _loader(model)
    t0 = time.perf_counter()
    loader()
    load_s = time.perf_counter() - t0

    inputs = []
    for s in series.values():
        h = s.iloc[-HISTORY:]
        inputs.append((h, covariates(h.index.append(pd.date_range(h.index[-1] + pd.Timedelta(days=1), periods=HORIZON)))))
    fn(*inputs[0][:1], HORIZON, inputs[0][1])  # warm-up

    def timed(batch):
        t = time.perf_counter()
        for h, c in batch:
            fn(h, HORIZON, c)
        return time.perf_counter() - t

    return {"model": model, "file_mb": round(_file_mb(REPOS[model]), 1), "load_s": round(load_s, 2),
            "per_series_s": statistics.median(timed(inputs[:1]) for _ in range(runs)),
            "batch_all_s": statistics.median(timed(inputs) for _ in range(runs)),
            "machine": f"{platform.processor() or platform.machine()} / {platform.platform()}"}
