import pandas as pd

from pangan.calendar import EVENTS, covariates


def test_lebaran_2025_features():
    cov = covariates(pd.date_range("2025-03-01", "2025-04-10"))
    assert cov.loc["2025-03-31", "days_to_lebaran"] == 0
    assert cov.loc["2025-03-21", "days_to_lebaran"] == 10
    assert cov.loc["2025-03-24", "win_lebaran"] == 1
    assert cov.loc["2025-04-08", "win_lebaran"] == 0
    assert cov.loc["2025-03-15", "ramadan"] == 1
    assert cov.loc["2025-04-05", "ramadan"] == 0


def test_days_to_lebaran_clipped_and_columns():
    cov = covariates(pd.date_range("2025-06-01", "2025-06-03"))
    assert (cov.days_to_lebaran == 60).all()
    assert list(cov.columns) == ["days_to_lebaran", "ramadan", "win_lebaran", "win_idul_adha", "win_natal", "win_tahun_baru"]


def test_natal_window_crosses_year():
    cov = covariates(pd.date_range("2025-12-17", "2026-01-09"))
    assert cov.loc["2025-12-18", "win_natal"] == 1 and cov.loc["2025-12-17", "win_natal"] == 0
    assert cov.loc["2026-01-08", "win_tahun_baru"] == 1 and cov.loc["2026-01-09", "win_tahun_baru"] == 0


def test_events_cover_range():
    for k, ds in EVENTS.items():
        assert {d.year for d in ds} >= set(range(2018, 2028)), k
