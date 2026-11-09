import numpy as np
import pytest

from tsbench.metrics import QUANTILES, coverage, mase, sort_quantiles, wql


def test_mase_ignores_nan_targets():
    # skala naive-1 history = mean|diff([8,9,10,11])| = 1; error = mean(|10-11|, |12-11|) = 1
    assert mase(np.array([10, np.nan, 12.]), np.array([11, 999, 11.]), np.array([8, 9, 10, 11.])) == 1.0


def test_mase_scale_ignores_nan_history():
    assert mase(np.array([12.]), np.array([10.]), np.array([8, np.nan, 10, 12.])) == pytest.approx(1.0)


def test_mase_zero_scale_is_nan():
    assert np.isnan(mase(np.array([5.]), np.array([5.]), np.array([5, 5, 5.])))


def test_wql_perfect_is_zero():
    y = np.array([1., 2.])
    assert wql(y, np.tile(y[:, None], (1, len(QUANTILES)))) == 0


def test_wql_known_value():
    # y=10, semua kuantil=8: pinball_tau = tau*2; Σ_tau = 2*4.5 = 9; / (n_tau=9 * |y|=10) = 0.1
    assert wql(np.array([10.]), np.full((1, 9), 8.)) == pytest.approx(0.1)


def test_wql_ignores_nan_targets():
    q = np.full((2, 9), 8.)
    assert wql(np.array([10., np.nan]), q) == pytest.approx(0.1)


def test_sort_quantiles_fixes_crossing():
    assert (np.diff(sort_quantiles(np.array([[3, 1, 2.]])), axis=1) >= 0).all()


def test_coverage_skips_nan():
    assert coverage(np.array([1, np.nan, 5.]), np.array([0, 0, 0.]), np.array([2, 2, 2.])) == 0.5
