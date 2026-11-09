"""Metrik forecast. Semua fungsi mengabaikan target NaN (hari tanpa observasi)."""

import numpy as np

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
EXTRA_Q = (0.05, 0.95)


def sort_quantiles(q: np.ndarray) -> np.ndarray:
    return np.sort(q, axis=1)


def mase(y: np.ndarray, median: np.ndarray, history: np.ndarray) -> float:
    scale = np.nanmean(np.abs(np.diff(history)))
    ok = ~np.isnan(y)
    if not ok.any() or not scale > 0:  # skala 0/NaN (harga datar) -> NaN, bukan inf
        return float("nan")
    return float(np.mean(np.abs(y[ok] - median[ok])) / scale)


def wql(y: np.ndarray, q: np.ndarray, levels=QUANTILES) -> float:
    ok = ~np.isnan(y)
    y, q = y[ok, None], q[ok]
    tau = np.asarray(levels)[None, :]
    diff = y - q
    pinball = np.maximum(tau * diff, (tau - 1) * diff)
    denom = len(levels) * np.abs(y).sum()
    return float(pinball.sum() / denom) if denom > 0 else float("nan")


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    ok = ~np.isnan(y)
    if not ok.any():
        return float("nan")
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])))
