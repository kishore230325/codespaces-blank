"""Stage-5 tests: PIT discipline, determinism, sanity, metrics identities."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signals import forecast as F
from signals import metrics as M
from signals import signals as S
from simulation.heston_simulator import HestonParams


def _toy_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    r = rng.normal(0.0003, 0.01, n)
    close = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame({"close": close, "log_ret": np.r_[np.nan, np.diff(np.log(close))],
                          "india_vix_close": 15.0}, index=idx)


def test_trailing_state_uses_slice_only():
    df = _toy_df()
    a = F.trailing_state(df.iloc[:100])
    b = F.trailing_state(df.iloc[:100])  # same slice, same answer
    assert a == b
    c = F.trailing_state(df.iloc[:101])  # one more day -> may differ
    assert isinstance(c["V0"], float) and c["V0"] > 0


def test_forecast_determinism_and_shapes():
    df = _toy_df()
    t = df.index[500]
    v = F.Vintage(t, HestonParams(2.0, 0.04, 0.3, -0.5, 0.04), 0.05)
    f1 = F.forecast_day(df.loc[:t], t, 1, v, 5000, 123)
    f2 = F.forecast_day(df.loc[:t], t, 1, v, 5000, 123)
    assert np.array_equal(f1.S_T, f2.S_T) and np.array_equal(f1.Ibar, f2.Ibar)
    assert (f1.S_T > 0).all() and (f1.Ibar >= 0).all()


def test_dist_stats_sanity():
    rng = np.random.default_rng(0)
    S_T = 100 * np.exp(rng.normal(0.001, 0.02, 20000))
    st = S.dist_stats(S_T, 100.0, np.full(20000, 0.04 / 252), 1 / 252, vix_t=16.0)
    assert 0 <= st["p_pos"] <= 1 and 0 <= st["p_below_m1"] <= 1
    assert st["q05"] <= st["q25"] <= st["median_ret"] <= st["q75"] <= st["q95"]
    assert st["exp_vol"] > 0 and np.isfinite(st["up_down_ratio"])
    assert st["downside_prob"] == st["p_below_m1"]
    d, conflict = S.candidate_direction(0.6, 0.3)
    assert d == "LONG" and not conflict
    d, conflict = S.candidate_direction(0.6, 0.6)
    assert d == "FLAT" and conflict


def test_metrics_identities():
    y = pd.Series([0.01, -0.02, 0.03, -0.01])
    perfect = pd.Series([0.9, 0.1, 0.8, 0.2])
    ds = M.directional_stats(perfect, y, 0.5)
    assert ds["accuracy"] == 1.0 and ds["balanced_accuracy"] == 1.0
    assert M.brier_score(perfect, y) < 0.05
    assert M.brier_score(pd.Series([1.0, 0.0, 1.0, 0.0]), y) == 0.0
    rng = np.random.default_rng(0)
    n = 200
    sig = pd.Series(rng.normal(0, 1, n))
    real = pd.Series(0.5 * sig + rng.normal(0, 0.5, n))
    cr = M.forecast_corr(sig, real)
    assert cr["pearson"] > 0.5 and cr["n"] == n
