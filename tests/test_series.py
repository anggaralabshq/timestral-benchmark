import pandas as pd

from tsbench.series import to_daily


def test_to_daily_fills_calendar_with_nan():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-06", "2026-11-09", "2026-11-06"]),  # Jum, Sen; baris Aceh
        "region": ["Semua Provinsi", "Semua Provinsi", "Aceh"],
        "level": [0, 0, 1],
        "commodity": ["beras"] * 3,
        "price": [16700.0, 16650.0, 99.0],
    })
    s = to_daily(df, "beras")
    assert s.index.freqstr == "D" and len(s) == 4  # Jum, Sab, Min, Sen
    assert s.isna().sum() == 2
    assert s.iloc[0] == 16700.0 and s.name == "beras"
