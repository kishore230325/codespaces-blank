"""Event-driven backtest engine (Stage 6). Mechanics only — no strategy selection.

Frozen execution rules (V1):
  R1. signal(t close, data <= t) -> entry at the NEXT trading day's open.
      t_entry > t_signal strictly, enforced by construction and asserted in tests.
      (Equals "t+1 open" when sessions are consecutive; skips holidays.)
  R2. Exit at the close of the h-th trading day after t (t+h in trading-day
      steps), i.e. the same date Stage 5 used for close-to-close scoring, so
      engine trades stay comparable to Stage-5 diagnostics.
  R3. Max holding = h. No pyramiding, no early exit, no re-entry in V1.
  R4. Conflict rows, missing opens/closes, or signals with no executable
      entry/exit inside the OHLC index -> FLAT (logged in summary, never traded).
  R5. Modes: 'longshort' trades LONG and SHORT; 'longflat' maps SHORT -> FLAT.
  R6. Costs via backtest.costs (frictional bps per side, deducted from notional).

The engine NEVER reads realised future columns (real_R etc.) — only t, h,
direction, conflict. Scoring columns pass through untouched for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.costs import side_cost_frac

_DIR_SIGN = {"LONG": 1.0, "SHORT": -1.0, "FLAT": 0.0}


@dataclass
class BacktestConfig:
    cost_bps_per_side: float = 5.0
    slippage_bps: float = 2.0
    mode: str = "longshort"  # or 'longflat'

    def __post_init__(self):
        if self.mode not in ("longshort", "longflat"):
            raise ValueError("mode must be longshort|longflat")


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    n_signals: int
    n_traded: int
    n_flat_signal: int
    n_conflict_flat: int
    n_dropped_no_execution: int
    config: BacktestConfig


def _positions(ohlc: pd.DataFrame) -> dict:
    return {pd.Timestamp(d): i for i, d in enumerate(ohlc.index)}


def run_backtest(signals: pd.DataFrame, ohlc: pd.DataFrame,
                 cfg: BacktestConfig) -> BacktestResult:
    sig = signals.copy()
    sig["t"] = pd.to_datetime(sig["t"])
    ohlc = ohlc.sort_index()
    pos_of = _positions(ohlc)
    cost = side_cost_frac(cfg.cost_bps_per_side, cfg.slippage_bps)
    opens = ohlc["open"]
    closes = ohlc["close"]

    trades, n_flat, n_conf, n_drop = [], 0, 0, 0
    for _, r in sig.iterrows():
        t, h = pd.Timestamp(r["t"]), int(r["h"])
        direction = str(r["direction"])
        if direction == "FLAT" or bool(r.get("conflict", False)):
            n_flat += 1
            n_conf += int(bool(r.get("conflict", False)))
            continue
        if cfg.mode == "longflat" and direction == "SHORT":
            n_flat += 1
            continue
        if t not in pos_of:
            n_drop += 1
            continue
        p = pos_of[t]
        if p + 1 >= len(ohlc) or p + h >= len(ohlc):
            n_drop += 1  # no executable entry/exit inside history
            continue
        t_entry, t_exit = ohlc.index[p + 1], ohlc.index[p + h]
        assert t_entry > t, "R1 violated: entry must be strictly after signal date"
        S_in, S_out = float(opens.iloc[p + 1]), float(closes.iloc[p + h])
        if not (np.isfinite(S_in) and np.isfinite(S_out) and S_in > 0):
            n_drop += 1
            continue
        d = _DIR_SIGN[direction]
        gross = d * (S_out / S_in - 1)
        net = gross - cost - cost  # identity: net = signed_gross - cost_in - cost_out
        trades.append({"t_signal": t, "h": h, "t_entry": t_entry, "t_exit": t_exit,
                       "direction": direction, "S_entry": S_in, "S_exit": S_out,
                       "gross": gross, "cost_in": cost, "cost_out": cost,
                       "net_ret": net, "holding_days": h,
                       "vintage_date": r.get("vintage_date", pd.NaT)})
    trades_df = pd.DataFrame(trades)
    return BacktestResult(trades=trades_df, n_signals=len(sig),
                          n_traded=len(trades_df), n_flat_signal=n_flat,
                          n_conflict_flat=n_conf, n_dropped_no_execution=n_drop,
                          config=cfg)


def run_buy_and_hold(ohlc: pd.DataFrame, start=None, end=None,
                     cost_bps_per_side: float = 0.0,
                     slippage_bps: float = 0.0) -> dict:
    """Passive LONG from first open to last close in range (zero costs default).

    Replication benchmark for test (c): validates price handling independently
    of any signal logic.
    """
    sub = ohlc.loc[start:end] if (start or end) else ohlc
    first, last = float(sub["open"].iloc[0]), float(sub["close"].iloc[-1])
    cost = side_cost_frac(cost_bps_per_side, slippage_bps)
    gross = last / first - 1
    return {"t_entry": sub.index[0], "t_exit": sub.index[-1], "S_entry": first,
            "S_exit": last, "gross": gross, "net_ret": gross - cost - cost}
