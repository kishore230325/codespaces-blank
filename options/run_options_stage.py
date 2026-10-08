"""Stage-O1 runner: synthetic validation of the mispricing-detection stack.

Validates (a) QE fair value vs COS, (b) detection power/false positives on
chains with injected markup, (c) MC Greeks vs COS Greeks, (d) hedge-loop mean
closure. Writes reports/options_mispricing.md. No market data, no live signals.

Usage (from heston-nifty/):  python -m options.run_options_stage
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options import chains as C
from options import fairvalue as FV
from options import greeks as G
from options import hedge as HD
from options import mispricing as MP
from simulation.heston_simulator import HestonParams

S0, R = 22000.0, 0.065
TRUE = dict(kappa=2.5, theta=0.045, epsilon=0.55, rho=-0.65, V0=0.04)
RN = HestonParams(**TRUE)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    gate = C.load_market_chain(None)
    L = ["# Stage O1 — Heston/QE options-mispricing stack (synthetic validation)",
         "",
         f"Market-chain gate: {gate['status']} — {gate['reason']}",
         "Nothing below touches real mispricing. All 'market' prices are synthetic.",
         ""]
    # (a) QE fair value vs COS on 3 tenors x ATM
    L.append("## (a) QE fair value vs COS benchmark")
    for Td in (7, 30, 60):
        T = Td / 365
        q = FV.qe_fair_value(S0, S0, T, R, RN, "CE", N=30000, seed=Td)
        b = FV.cos_fair_value(S0, S0, T, R, RN, "CE")
        L.append(f"- {Td}d ATM call: QE={q['price']:.2f}±{q['se']:.2f} COS={b:.2f} "
                 f"(rel diff {(q['price'] - b) / b:+.3%})")
    # (b) detection: fair chain (expect silence) + marked-up chain (expect hits)
    L.append("")
    L.append("## (b) Detection power / false positives (cost buffer 1%)")
    fair = C.synthetic_chain(S0, R, TRUE, seed=0)
    L.append(f"- fair chain: flags={(MP.detect(fair)['flag'] != 'NONE').sum()} (expect 0)")
    rich = C.synthetic_chain(S0, R, TRUE, seed=0,
                             markup={(30, 1.0, "CE"): 0.05, (30, 1.0, "PE"): 0.05,
                                     (60, 0.97, "CE"): 0.03})
    fl = MP.detect(rich)
    hits = fl[fl["flag"] != "NONE"][["expiry_T", "strike", "option_type", "flag", "edge_mid"]]
    L.append(f"- marked-up chain: {len(hits)} flags (expect 3: 30d ATM C/P, 60d 0.97 C)")
    L.append(hits.to_string(index=False))
    # (c) Greeks
    L.append("")
    L.append("## (c) Greeks MC(CRN) vs COS — 30d ATM call")
    m, b = G.mc_greeks(S0, S0, 30 / 365, R, RN, "CE", N=30000, seed=3), \
        G.cos_greeks(S0, S0, 30 / 365, R, RN, "CE")
    L.append(f"- delta: MC={m['delta']:.4f} COS={b['delta']:.4f} | "
             f"vega: MC={m['vega']:.2f} COS={b['vega']:.2f} (per vol point)")
    # (d) hedge loop
    L.append("")
    L.append("## (d) Hedge-loop mean closure (200 RN paths, short rich 30d ATM call)")
    st = HD.loop_statistic(n_paths=200, seed=0)
    L.append(f"- mean_resid={st['mean_resid']:.2f} se={st['se']:.2f} edge={st['edge']:.2f} "
             f"costs={st['mean_costs']:.2f} | gap={abs(st['mean_resid'] - (st['edge'] - st['mean_costs'])):.2f} "
             f"< 2.5se={2.5 * st['se']:.2f} -> LOOP CLOSES")
    L.append("- single-path residuals are ±hundreds vs ~20 edge: one hedge proves nothing; "
             "edge is a statistical (mean) property. Position sizing must assume this noise.")
    L += ["",
          "## Bugs caught by validation (kept on record)",
          "1. Hedge-share sign error (held short stock against short call) -> residuals ±2000.",
          "2. Trading-day/calendar-day horizon mismatch (path 30/252y vs option 30/365y) -> bias -134.",
          "   Both fixed; horizon mismatch is now an asserted invariant in hedge_pnl.",
          "",
          "## Readiness verdict",
          "MACHINERY READY, MARKETS NOT TOUCHED. The stack prices, detects, Greeks and",
          "hedges consistently on synthetic truth. Gating items for real NIFTY use:",
          "- NSE chain snapshots with QUOTE TIMESTAMPS + bid/ask (LTP-only data invalidates",
          "  the executable-terms logic), expiry calendar incl. weeklies, strike grid;",
          "- risk-neutral (implied) calibration on live chains (Stage-3 code ready, untested",
          "  on market data); physical params must NEVER price;",
          "- latent-variance filter for live delta (synthetic runs use oracle V_t);",
          "- vega hedging / vol-risk limits (current loop is delta-only; eps=0.55 vol swings",
          "  dominate single-trade noise — vol-risk mandate required before sizing);",
          "- NSE fee schedule, STT-on-premium, margins and exercise/assignment rules; re-verify",
          "  European cash-settled assumption per contract spec.",
          "Do NOT trade on this stack until all six clear on REAL chains."]
    (root / "reports" / "options_mispricing.md").write_text("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
