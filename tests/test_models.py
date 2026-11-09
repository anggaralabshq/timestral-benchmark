import numpy as np
import pandas as pd
import pytest

from pangan.calendar import covariates
from pangan.metrics import QUANTILES
from pangan.models import FOUNDATION, MODELS, fill_for_model

rng = np.random.default_rng(0)
idx = pd.date_range("2024-01-01", periods=400, freq="D")
vals = 30000 + 50 * np.arange(400) + 3000 * np.sin(np.arange(400) / 20) + rng.normal(0, 300, 400)
vals[rng.random(400) < 0.1] = np.nan
hist = pd.Series(vals, index=idx, name="cabai_merah")
cov_full = covariates(pd.date_range(idx[0], periods=400 + 28, freq="D"))


def _params():
    for name in MODELS:
        for use_cov in (False, True):
            marks = [pytest.mark.slow] if name in FOUNDATION else []
            yield pytest.param(name, use_cov, marks=marks, id=f"{name}-{'cov' if use_cov else 'nocov'}")


@pytest.mark.parametrize("name,use_cov", list(_params()))
def test_adapter_contract(name, use_cov):
    fn, supports = MODELS[name]
    if use_cov and not supports:
        pytest.skip("model tanpa dukungan covariates")
    out = fn(hist, 28, cov_full if use_cov else None)
    assert len(out) == 28 and out.index[0] == hist.index[-1] + pd.Timedelta(days=1)
    assert all(q in out.columns for q in QUANTILES)
    assert out[list(QUANTILES)].notna().all().all()
    med = out[0.5]
    assert med.between(10000, 80000).all()  # skala masuk akal, bukan ter-normalisasi


def test_fill_for_model_has_no_nan():
    s = hist.copy()
    s.iloc[0] = np.nan
    assert fill_for_model(s).notna().all()
