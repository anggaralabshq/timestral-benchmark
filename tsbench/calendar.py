"""Kalender hari besar Indonesia sebagai known-future covariates.

Tanggal Islam: versi pemerintah (Kemenag) bila berbeda dengan ormas.
Sumber: id.wikipedia.org/wiki/Ramadan, /Idulfitri, /Iduladha (diakses 2026-11-09);
situs Kemenag/SKB tidak bisa diakses otomatis. Lebaran 2026 bisa bergeser ±1 hari
dari keputusan sidang isbat; 2027 hanya estimasi (tidak berpengaruh: semua horizon
uji berakhir sebelum 2027 dan days_to_lebaran dipotong 60).
"""

from datetime import date

import numpy as np
import pandas as pd

EVENTS: dict[str, list[date]] = {
    "ramadan_start": [date(2018, 5, 17), date(2019, 5, 6), date(2020, 4, 24), date(2021, 4, 13), date(2022, 4, 3),
                      date(2023, 3, 23), date(2024, 3, 12), date(2025, 3, 1), date(2026, 2, 19), date(2027, 2, 8)],
    "lebaran": [date(2018, 6, 15), date(2019, 6, 5), date(2020, 5, 24), date(2021, 5, 13), date(2022, 5, 2),
                date(2023, 4, 22), date(2024, 4, 10), date(2025, 3, 31), date(2026, 3, 20), date(2027, 3, 10)],
    "idul_adha": [date(2018, 8, 22), date(2019, 8, 11), date(2020, 7, 31), date(2021, 7, 20), date(2022, 7, 10),
                  date(2023, 6, 29), date(2024, 6, 17), date(2025, 6, 6), date(2026, 5, 27), date(2027, 5, 16)],
    "natal": [date(y, 12, 25) for y in range(2017, 2028)],
    "tahun_baru": [date(y, 1, 1) for y in range(2018, 2029)],
}

WINDOW = 7  # H-7..H+7
MAX_DAYS = 60


def covariates(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index).normalize()
    leb = pd.DatetimeIndex(EVENTS["lebaran"])
    nxt = leb[np.searchsorted(leb, idx)]  # lebaran berikutnya (atau hari ini)
    out = pd.DataFrame(index=idx)
    out["days_to_lebaran"] = np.minimum((nxt - idx).days, MAX_DAYS)
    ram = np.zeros(len(idx), dtype=int)
    for s, e in zip(EVENTS["ramadan_start"], EVENTS["lebaran"]):
        ram |= (idx >= pd.Timestamp(s)) & (idx < pd.Timestamp(e))
    out["ramadan"] = ram
    for ev in ["lebaran", "idul_adha", "natal", "tahun_baru"]:
        d = pd.DatetimeIndex(EVENTS[ev])
        dist = np.abs((idx.values[:, None] - d.values[None, :]).astype("timedelta64[D]").astype(int)).min(axis=1)
        out[f"win_{ev}"] = (dist <= WINDOW).astype(int)
    return out
