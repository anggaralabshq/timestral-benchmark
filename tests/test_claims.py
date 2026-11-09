import pandas as pd

from pangan.claims import judge

RIVALS = ["snaive", "ets", "arima", "prophet", "chronos2", "timesfm25"]


def make_summary(t209=(0.8, 0.8), t209_leb=(0.8, 0.8), t209_nocov=(0.8, 0.8), t209_nocov_leb=(0.8, 0.8),
                 rival=(0.9, 0.9), rival_leb=(0.9, 0.9), t5=(0.85, 0.85), cov80=0.8, cov90=0.9):
    rows = []

    def add(model, cov, sl, mase, wql, c80=0.8, c90=0.9):
        rows.append(dict(model=model, cov=cov, slice=sl, horizon=28, rel_mase=mase, rel_wql=wql, cov80=c80, cov90=c90, n=10))

    add("timestral_209m", True, "all", *t209, cov80, cov90)
    add("timestral_209m", True, "lebaran", *t209_leb)
    add("timestral_209m", False, "all", *t209_nocov)
    add("timestral_209m", False, "lebaran", *t209_nocov_leb)
    add("timestral_5m", True, "all", *t5)
    add("timestral_5m", True, "lebaran", *t5)
    for r in RIVALS:
        for cov in (False, True):
            add(r, cov, "all", *rival)
            add(r, cov, "lebaran", *rival_leb)
    return pd.DataFrame(rows)


SPEED = pd.DataFrame({"model": ["timestral_5m", "timestral_209m", "chronos2", "timesfm25"],
                      "per_series_s": [0.01, 0.2, 0.1, 0.3]})


def status(summary, claim, speed=SPEED):
    j = judge(summary, speed)
    return j.set_index("claim").loc[claim, "status"]


def test_k1_terbukti_when_rank1_both_scores():
    assert status(make_summary(), "K1") == "proven"


def test_k1_sebagian_when_rank1_one_score():
    assert status(make_summary(t209=(0.8, 0.95)), "K1") == "partly proven"


def test_k1_sebagian_when_rank1_only_lebaran():
    assert status(make_summary(t209=(0.95, 0.95), t209_leb=(0.5, 0.5)), "K1") == "partly proven"


def test_k1_tidak_when_beaten():
    assert status(make_summary(t209=(0.95, 0.95), t209_leb=(0.95, 0.95)), "K1") == "not proven"


def test_k1_uses_rival_best_config():
    s = make_summary()
    s.loc[(s.model == "chronos2") & ~s["cov"], ["rel_mase", "rel_wql"]] = 0.7
    assert status(s, "K1") == "not proven"


def test_k1_ignores_other_horizon():
    s = make_summary(t209=(0.95, 0.95), t209_leb=(0.95, 0.95))
    s14 = s.assign(horizon=14)
    s14.loc[(s14.model == "timestral_209m"), ["rel_mase", "rel_wql"]] = 0.1
    assert status(pd.concat([s, s14]), "K1") == "not proven"


def test_k2_terbukti_at_5pct_and_no_harm():
    assert status(make_summary(t209_leb=(0.75, 0.8), t209_nocov_leb=(0.8, 0.8), t209=(0.81, 0.8)), "K2") == "proven"


def test_k2_sebagian_when_gain_below_5pct():
    assert status(make_summary(t209_leb=(0.78, 0.8), t209_nocov_leb=(0.8, 0.8)), "K2") == "partly proven"


def test_k2_sebagian_when_harms_all_slice():
    assert status(make_summary(t209_leb=(0.7, 0.8), t209=(0.83, 0.8), t209_nocov=(0.8, 0.8)), "K2") == "partly proven"


def test_k2_tidak_when_no_gain_in_lebaran():
    assert status(make_summary(t209_leb=(0.8, 0.8), t209_nocov_leb=(0.8, 0.8)), "K2") == "not proven"


def test_k3_requires_within_10pct_and_fastest():
    assert status(make_summary(t5=(0.87, 0.9)), "K3") == "proven"
    assert status(make_summary(t5=(0.9, 0.9)), "K3") == "partly proven"  # 12.5% lebih buruk
    slow = SPEED.assign(per_series_s=[0.5, 0.2, 0.1, 0.3])
    assert status(make_summary(t5=(0.9, 0.9)), "K3", slow) == "not proven"


def test_k4_ranges_80_and_90():
    assert status(make_summary(cov80=0.78, cov90=0.88), "K4") == "proven"
    assert status(make_summary(cov80=0.70, cov90=0.88), "K4") == "partly proven"
    assert status(make_summary(cov80=0.70, cov90=0.97), "K4") == "not proven"


def test_judge_has_evidence_for_all_claims():
    j = judge(make_summary(), SPEED)
    assert list(j.claim) == ["K1", "K2", "K3", "K4"] and j.evidence.str.len().gt(10).all()
