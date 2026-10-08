"""Smoke validation of the engine on real Stage-5 tables (Stage 6).

Runs checks (a)-(c) on data/processed artifacts and writes
reports/backtest_engine_validation.md. Reports MECHANICS ONLY
(counts, identity residuals, replication error) — no performance conclusions.

Usage (from heston-nifty/):  python -m backtest.validate_engine
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.engine import BacktestConfig, run_backtest, run_buy_and_hold


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "backtest_v1.yaml").read_text())
    sig = pd.read_parquet(root / cfg["inputs"]["signals"])
    ohlc = pd.read_parquet(root / cfg["inputs"]["ohlc"])

    lines = ["# Stage 6 — backtest engine validation (mechanics only)", ""]
    ok_all = True
    for mode in cfg["modes"]:
        bc = BacktestConfig(cost_bps_per_side=cfg["costs"]["cost_bps_per_side"],
                            slippage_bps=cfg["costs"]["slippage_bps"], mode=mode)
        res = run_backtest(sig, ohlc, bc)
        # (a) timing: entry strictly after signal; prices match post-signal prints
        a_tim = bool((res.trades["t_entry"] > res.trades["t_signal"]).all())
        sample = res.trades.head(50)
        a_px = bool(((sample["S_entry"] == ohlc.loc[sample["t_entry"], "open"].to_numpy()).all()
                     and (sample["S_exit"] == ohlc.loc[sample["t_exit"], "close"].to_numpy()).all()))
        # (b) cost identity
        resid = float((res.trades["gross"] - res.trades["cost_in"]
                       - res.trades["cost_out"] - res.trades["net_ret"]).abs().max())
        # (c) B&H replication on the validation span
        span = ohlc.loc[sig["t"].min():sig["real_t"].max()]
        bh = run_buy_and_hold(span)
        expect = float(span["close"].iloc[-1] / span["open"].iloc[0] - 1)
        bh_err = abs(bh["gross"] - expect)
        passed = a_tim and a_px and resid < 1e-8 and bh_err < 1e-10
        ok_all &= passed
        lines += [f"## mode={mode}: {'PASS' if passed else 'FAIL'}",
                  f"- signals={res.n_signals} traded={res.n_traded} "
                  f"flat_signal={res.n_flat_signal} conflict_flat={res.n_conflict_flat} "
                  f"dropped_no_execution={res.n_dropped_no_execution}",
                  f"- (a) entry-after-signal={a_tim}, prices-match-prints={a_px}",
                  f"- (b) max|gross-costs-net|={resid:.2e} (< 1e-8)",
                  f"- (c) B&H replication error={bh_err:.2e} (< 1e-10)", ""]
    lines += ["Verdict: " + ("PASS — engine cleared for Stage 7 strategy evaluation."
                             if ok_all else "FAIL — resolve before Stage 7.")]
    (root / cfg["outputs"]["report"]).write_text("\n".join(lines))
    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
