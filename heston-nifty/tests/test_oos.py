"""OOS evaluation tests: PIT discipline, equity accounting, bootstrap sanity."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.oos_evaluation import (
    benchmark_frames,
    daily_equity,
    paired_sharpe_diff_p,
    volfilter_frame,
)


def _ohlc(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = np.r_[close[0] * 0.999, close[:-1]]
    return pd.DataFrame({"open": open_, "close": close,
                         "high": close * 1.005, "low": close * 0.995,
                         "india_vix_close": 15.0, "rf_daily": 0.065 / 252}, index=idx)


def test_benchmark_frames_point_in_time():
    ohlc = _ohlc()
    frames = benchmark_frames(ohlc, ohlc.index[300], ohlc.index[320])
    mom = frames["MOM20"]
    t = mom["t"].iloc[5]
    # manual trailing-20d check using data <= t only
    p = ohlc.index.get_loc(t)
    expect = "LONG" if float(np.log(ohlc["close"].iloc[p - 19:p + 1] /
                                    ohlc["close"].iloc[p - 20:p]).sum()) > 0 else "FLAT"
    assert mom[mom["t"] == t]["direction"].iloc[0] == expect


def test_volfilter_abstains_without_history_and_above_median():
    rows = []
    idx = pd.bdate_range("2023-01-01", periods=100)
    for i, t in enumerate(idx):
        rows.append({"t": t, "h": 5, "direction": "LONG", "conflict": False,
                     "exp_vol": 0.20 if i < 90 else 0.01, "vintage_date": t})
    oos = pd.DataFrame(rows)
    val = pd.DataFrame([{"t": t, "h": 5, "exp_vol": 0.15}
                        for t in pd.bdate_range("2022-10-01", periods=10)])
    out = volfilter_frame(oos, val)
    # early rows: insufficient 63d history -> FLAT
    assert (out.iloc[:50]["direction"] == "FLAT").all()
    # late rows: exp_vol 0.01 << median -> keep LONG
    assert (out.iloc[95:]["direction"] == "LONG").all()


def test_daily_equity_accounts_costs_exactly():
    ohlc = _ohlc(n=30)
    tr = pd.DataFrame([{"t_entry": ohlc.index[5], "t_exit": ohlc.index[10],
                        "h": 5, "direction": "LONG",
                        "cost_in": 0.0007, "cost_out": 0.0007}])
    eq = daily_equity(tr, ohlc)
    active = ohlc.loc[tr.iloc[0]["t_entry"]:tr.iloc[0]["t_exit"]].index
    assert (eq.loc[~eq.index.isin(active)] == 0).all()
    # cost drag exact: (in+out)/h deducted across entry/exit days
    assert abs(-(eq.loc[active].sum()
                 - (ohlc["close"].pct_change().loc[active].sum() / 5))
               - 0.0014 / 5) < 1e-12


def test_paired_sharpe_bootstrap_sanity():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 500)
    p_same, _ = paired_sharpe_diff_p(x, x.copy())
    assert p_same == 1.0  # identical series: no evidence against H0, by construction
    p_diff, _ = paired_sharpe_diff_p(x + 0.005, x)
    assert p_diff < 0.05
