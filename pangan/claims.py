"""Claim verdicts K1–K4 using the rules in spec §8 (locked before the experiment)."""

import pandas as pd

MAIN = ("timestral_209m", True)
SMALL = ("timestral_5m", True)
FOUNDATION = ["timestral_5m", "timestral_209m", "chronos2", "timesfm25"]
H = 28


def _get(s: pd.DataFrame, model: str, cov: bool, sl: str) -> pd.Series:
    r = s[(s.model == model) & (s["cov"] == cov) & (s.slice == sl) & (s.horizon == H)]
    assert len(r) == 1, f"no row for {model} cov={cov} {sl}"
    return r.iloc[0]


def _best_rival(s: pd.DataFrame, sl: str, score: str) -> tuple[float, str]:
    """Best configuration of each rival, per score (deliberately favours the rivals)."""
    r = s[~s.model.str.startswith("timestral") & (s.slice == sl) & (s.horizon == H)].dropna(subset=[score])
    best = r.loc[r[score].idxmin()]
    return float(best[score]), f"{best.model}{'+cov' if best['cov'] else ''}"


def _k1(s):
    wins = {}
    for sl in ("all", "lebaran"):
        m = _get(s, *MAIN, sl)
        wins[sl] = {sc: m[sc] < _best_rival(s, sl, sc)[0] for sc in ("rel_mase", "rel_wql")}
    n_all = sum(wins["all"].values())
    status = "proven" if n_all == 2 else "partly proven" if n_all == 1 or all(wins["lebaran"].values()) else "not proven"
    m = _get(s, *MAIN, "all")
    (bm, bm_name), (bw, bw_name) = _best_rival(s, "all", "rel_mase"), _best_rival(s, "all", "rel_wql")
    ev = (f"relative MASE {m.rel_mase:.3f} vs best rival {bm:.3f} ({bm_name}); "
          f"relative CRPS {m.rel_wql:.3f} vs {bw:.3f} ({bw_name})")
    return status, ev


def _k2(s):
    leb_c, leb_n = _get(s, "timestral_209m", True, "lebaran").rel_mase, _get(s, "timestral_209m", False, "lebaran").rel_mase
    all_c, all_n = _get(s, "timestral_209m", True, "all").rel_mase, _get(s, "timestral_209m", False, "all").rel_mase
    gain, harm = 1 - leb_c / leb_n, all_c / all_n - 1
    status = "not proven" if gain <= 0 else "proven" if gain >= 0.05 and harm <= 0.02 else "partly proven"
    return status, f"calendar covariates: MASE {gain:+.1%} better in the Eid window; {harm:+.1%} error change across all origins"


def _k3(s, speed):
    small, main = _get(s, *SMALL, "all").rel_mase, _get(s, *MAIN, "all").rel_mase
    gap = small / main - 1
    sp = speed[speed.model.isin(FOUNDATION)].set_index("model").per_series_s
    fastest = sp.idxmin() == "timestral_5m"
    ok = int(gap <= 0.10) + int(fastest)  # int(): numpy bool + bool is a logical OR, not 2
    status = "proven" if ok == 2 else "partly proven" if ok == 1 else "not proven"
    return status, f"5M vs 209M MASE {gap:+.1%}; 5M {sp['timestral_5m'] * 1000:.0f} ms per series, fastest: {sp.idxmin()}"


def _k4(s):
    m = _get(s, *MAIN, "all")
    ok = int(0.75 <= m.cov80 <= 0.85) + int(0.85 <= m.cov90 <= 0.95)
    status = "proven" if ok == 2 else "partly proven" if ok == 1 else "not proven"
    return status, f"80% interval covers {m.cov80:.1%}; 90% interval covers {m.cov90:.1%}"


def judge(summary: pd.DataFrame, speed: pd.DataFrame) -> pd.DataFrame:
    rows = [("K1", *_k1(summary)), ("K2", *_k2(summary)), ("K3", *_k3(summary, speed)), ("K4", *_k4(summary))]
    return pd.DataFrame(rows, columns=["claim", "status", "evidence"])
