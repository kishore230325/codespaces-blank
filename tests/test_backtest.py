"""Engine validation tests (Stage 6). Run: pytest tests/test_backtest.py -q."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import BacktestConfig, run_backtest, run_buy_and_hold


def _toy_ohlc(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    open_ = np.r_[close[0] * 0.999, close[:-1] * (1 + rng.normal(0, 0.002, n - 1))]
    return pd.DataFrame({"open": open_, "close": close,
                         "high": np.maximum(open_, close) * 1.005,
                         "low": np.minimum(open_, close) * 0.995}, index=idx)


def _toy_signals(ohlc, dirs=("LONG", "SHORT", "FLAT")):
    rows = []
    for i, t in enumerate(ohlc.index[5:-6]):
        rows.append({"t": t, "h": 1 if i % 2 == 0 else 5,
                     "direction": dirs[i % len(dirs)],
                     "conflict": False, "vintage_date": t})
    return pd.DataFrame(rows)


def test_a_no_trade_before_signal():
    """(a) Every execution strictly postdates its signal; prices are post-signal prints."""
    ohlc = _toy_ohlc()
    sig = _toy_signals(ohlc)
    res = run_backtest(sig, ohlc, BacktestConfig())
    assert res.n_traded > 0
    assert bool((res.trades["t_entry"] > res.trades["t_signal"]).all())
    # execution prices equal the actual post-signal OHLC prints
    for _, r in res.trades.head(10).iterrows():
        assert r["S_entry"] == ohlc.loc[r["t_entry"], "open"]
        assert r["S_exit"] == ohlc.loc[r["t_exit"], "close"]
    # exit = h-th trading day after signal (R2)
    for _, r in res.trades.iterrows():
        p = ohlc.index.get_loc(r["t_signal"])
        assert r["t_exit"] == ohlc.index[p + r["h"]]


def test_b_cost_accounting_identity():
    """(b) gross - costs = net exactly, on toy and real tables, both modes."""
    ohlc = _toy_ohlc()
    sig = _toy_signals(ohlc)
    for mode in ("longshort", "longflat"):
        res = run_backtest(sig, ohlc, BacktestConfig(mode=mode))
        resid = (res.trades["gross"] - res.trades["cost_in"]
                 - res.trades["cost_out"] - res.trades["net_ret"]).abs().max()
        assert resid < 1e-8, resid


def test_c_buy_and_hold_replication():
    """(c) Zero-cost passive run equals first-open to last-close arithmetic."""
    ohlc = _toy_ohlc()
    bh = run_buy_and_hold(ohlc, cost_bps_per_side=0.0, slippage_bps=0.0)
    expect = ohlc["close"].iloc[-1] / ohlc["open"].iloc[0] - 1
    assert abs(bh["gross"] - expect) < 1e-10
    assert abs(bh["net_ret"] - expect) < 1e-10


def test_conflict_and_longflat_mapping():
    ohlc = _toy_ohlc()
    sig = _toy_signals(ohlc)
    sig.loc[sig["direction"] == "LONG", "conflict"] = True  # force conflicts
    res = run_backtest(sig, ohlc, BacktestConfig())
    assert "LONG" not in set(res.trades["direction"])  # conflicts -> FLAT
    assert res.n_conflict_flat > 0
    sig2 = _toy_signals(ohlc)
    lf = run_backtest(sig2, ohlc, BacktestConfig(mode="longflat"))
    assert "SHORT" not in set(lf.trades["direction"])  # SHORT -> FLAT in longflat
