from datetime import date

import numpy as np
import pandas as pd
import pytest

from tsbench.backtest import evaluate, in_lebaran_window, origins, run, summarize
from tsbench.metrics import EXTRA_Q, QUANTILES

idx = pd.date_range("2022-01-01", "2026-11-09", freq="D")
rng = np.random.default_rng(1)
SERIES = {"beras": pd.Series(15000 + rng.normal(0, 100, len(idx)).cumsum(), index=idx, name="beras")}


def test_origins_are_mondays_and_leave_room():
    os_ = origins(date(2026, 11, 9))
    assert os_[0] == date(2025, 1, 6) and all(o.weekday() == 0 for o in os_)
    assert os_[-1] == date(2026, 10, 12)


def test_lebaran_window():
    assert in_lebaran_window(date(2025, 3, 3)) and in_lebaran_window(date(2026, 3, 16))
    assert not in_lebaran_window(date(2025, 4, 7)) and not in_lebaran_window(date(2025, 2, 10))


def _fake(seen):
    def fn(history, horizon, cov):
        seen.append((history.index[0], history.index[-1], len(history), cov))
        fut = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        last = history.dropna().iloc[-1]
        return pd.DataFrame({q: last + (q - 0.5) * 1000 for q in QUANTILES}, index=fut)
    return fn


def test_history_window_has_no_future_leak(tmp_path):
    seen = []
    path = run("fake", True, SERIES, tmp_path, models={"fake": (_fake(seen), True)})
    df = pd.read_parquet(path)
    first_origin = pd.Timestamp("2025-01-06")
    start, end, n, cov = seen[0]
    assert end == first_origin - pd.Timedelta(days=1) and n == 1024
    assert cov.index[-1] == first_origin + pd.Timedelta(days=27)  # cov mencakup horizon
    assert set(df.h) == set(range(1, 29))
    assert df["q0.05"].isna().all() and df["q0.5"].notna().all()  # kuantil absen -> NaN


def test_evaluate_and_summarize_end_to_end(tmp_path):
    run("fake", False, SERIES, tmp_path, models={"fake": (_fake([]), False)})
    run("snaive", False, SERIES, tmp_path, models={"snaive": (_fake([]), False)})
    m = evaluate(tmp_path / "forecasts", SERIES)
    assert set(m.horizon) == {14, 28} and m.lebaran.any()
    s = summarize(m)
    row = s[(s.model == "fake") & (s.slice == "all") & (s.horizon == 28)].iloc[0]
    assert row.rel_mase == pytest.approx(1.0) and np.isnan(row.cov90)


def test_summarize_relative_to_snaive():
    base = pd.DataFrame({"model": "snaive", "cov": False, "commodity": "beras", "origin": [1, 2],
                         "horizon": 28, "mase": [2.0, 4.0], "wql": [0.2, 0.4], "cov80": 0.8, "cov90": 0.9,
                         "lebaran": [True, False]})
    other = base.assign(model="x", mase=[1.0, 2.0], wql=[0.1, 0.2])
    s = summarize(pd.concat([base, other]))
    r = s[(s.model == "x") & (s.slice == "all")].iloc[0]
    assert r.rel_mase == pytest.approx(0.5) and r.rel_wql == pytest.approx(0.5) and r.n == 2
    assert s[(s.model == "x") & (s.slice == "lebaran")].iloc[0].n == 1


def test_run_pins_origins_to_as_of_date(tmp_path):
    # Reruns after 2026-11-09 must reproduce the published 93 origins, not add new ones
    longer = {"beras": pd.Series(15000.0, index=pd.date_range("2022-01-01", "2027-03-31", freq="D"), name="beras")}
    df = pd.read_parquet(run("fake", False, longer, tmp_path, models={"fake": (_fake([]), False)}))
    assert df.origin.max() == pd.Timestamp("2026-10-12") and df.origin.nunique() == 93


def test_summarize_adds_commodity_slices_and_mean_rank():
    base = pd.DataFrame({"model": "snaive", "cov": False, "commodity": ["beras", "cabai_merah"], "origin": [1, 1],
                         "horizon": 28, "mase": [2.0, 4.0], "wql": [0.2, 0.4], "cov80": 0.8, "cov90": 0.9,
                         "lebaran": False})
    other = base.assign(model="x", mase=[1.0, 8.0], wql=[0.1, 0.8])
    s = summarize(pd.concat([base, other])).set_index(["model", "slice"])
    assert s.loc[("x", "beras")].rel_mase == pytest.approx(0.5) and s.loc[("x", "cabai_merah")].rel_mase == pytest.approx(2.0)
    assert s.loc[("x", "all")].mean_rank == pytest.approx(1.5) and s.loc[("snaive", "beras")].mean_rank == 2.0
