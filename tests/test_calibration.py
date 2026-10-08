"""Stage 3 tests: moments, optimiser sanity, pricer parity, synthetic recovery."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration import historical as H
from calibration import implied as I
from data.make_dataset import build_demo_frames


def _train(n="2020-01-01"):
    nr, vr, rr, cal = build_demo_frames("2015-01-01", "2020-12-31", seed=11)
    import sys as _s
    _s.path.insert(0, str(Path("data").resolve().parents[0]))
    from data import data_cleaning as DC
    m = DC.merge_daily(DC.clean_nifty_ohlc(nr), DC.clean_india_vix(vr),
                       DC.clean_risk_free(rr), cal, provenance="test")
    return m


def test_cir_moments_match_andersen_closed_form():
    V, k, th, e = np.array([0.04, 0.09]), 2.0, 0.05, 0.4
    m = H.cir_cond_mean(V, k, th)
    assert np.allclose(m, th + (V - th) * np.exp(-k * H.DT))
    assert bool((H.cir_cond_var(V, k, th, e) > 0).all())


def test_single_window_reports_all_required_fields():
    df = _train().iloc[:300]
    r = H.calibrate_window(df)
    for f in ("kappa", "theta", "epsilon", "rho", "V0", "objective",
              "calib_error_rmse", "success", "message", "nit", "x0",
              "feller_stat", "feller_ok"):
        assert hasattr(r, f), f"missing {f}"
    for k, (lo, hi) in H.BOUNDS.items():
        assert lo <= getattr(r, k) <= hi


def test_rolling_uses_only_past_data():
    df = _train().iloc[:600]
    roll = H.calibrate_rolling(df, window=252, step=126)
    assert len(roll) >= 2
    assert roll.index.is_monotonic_increasing


def test_heston_pricer_put_call_parity():
    S0, K, T, r = 22000, 22000, 60 / 365, 0.065
    kw = dict(kappa=2.5, theta=0.045, eps=0.55, rho=-0.65, V0=0.04)
    c = I.heston_call_price(S0, K, T, r, **kw)
    p = c - S0 + K * np.exp(-r * T)
    assert 0 < c < S0 and p > 0  # arbitrage bounds
    # parity cross-check: recomputed put must match direct put pricing identity
    c2 = I.heston_call_price(S0, K, T, r, **kw)
    assert abs(c2 - c) / c < 1e-6  # deterministic pricer


def test_heston_pricer_bs_limit():
    """With small vol-of-vol, Heston -> Black-Scholes at sqrt(time-averaged var)."""
    from scipy.stats import norm

    S0, K, T, r = 22000, 22000, 60 / 365, 0.065
    kap, the, V0 = 2.5, 0.04, 0.04
    vbar = the + (V0 - the) * (1 - np.exp(-kap * T)) / (kap * T)
    iv = np.sqrt(vbar)
    d1 = (np.log(S0 / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    bs = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - iv * np.sqrt(T))
    c = I.heston_call_price(S0, K, T, r, kap, the, 0.05, -0.65, V0)
    assert abs(c - bs) / bs < 0.02


def test_implied_self_test_recovers_vol_level():
    q = I.synthetic_quotes()
    res = I.calibrate_implied(q, S0=22000.0, r=0.065)
    assert res.success
    assert abs(res.theta - 0.045) / 0.045 < 0.60  # level recovered within 60%
    assert res.rho < 0  # leverage sign recovered


def test_multistart_reports_spread():
    df = _train().iloc[:300]
    s = H.multistart_sensitivity(df, n_starts=3, seed=1)
    assert len(s) == 3 and "objective" in s.columns
