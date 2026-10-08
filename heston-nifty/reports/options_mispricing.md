# Stage O1 — Heston/QE options-mispricing stack (synthetic validation)

Market-chain gate: UNAVAILABLE — No NSE chain snapshot. Synthetic validation only.
Nothing below touches real mispricing. All 'market' prices are synthetic.

## (a) QE fair value vs COS benchmark
- 7d ATM call: QE=256.19±1.97 COS=256.43 (rel diff -0.092%)
- 30d ATM call: QE=557.41±4.02 COS=559.30 (rel diff -0.338%)
- 60d ATM call: QE=827.69±5.69 COS=823.54 (rel diff +0.503%)

## (b) Detection power / false positives (cost buffer 1%)
- fair chain: flags=0 (expect 0)
- marked-up chain: 3 flags (expect 3: 30d ATM C/P, 60d 0.97 C)
 expiry_T  strike option_type flag  edge_mid
 0.082192 22000.0          CE SELL -0.046746
 0.082192 22000.0          PE SELL -0.045692
 0.164384 21340.0          CE SELL -0.029639

## (c) Greeks MC(CRN) vs COS — 30d ATM call
- delta: MC=0.5973 COS=0.5971 | vega: MC=22.51 COS=22.55 (per vol point)

## (d) Hedge-loop mean closure (200 RN paths, short rich 30d ATM call)
- mean_resid=9.88 se=10.13 edge=20.68 costs=4.95 | gap=5.84 < 2.5se=25.33 -> LOOP CLOSES
- single-path residuals are ±hundreds vs ~20 edge: one hedge proves nothing; edge is a statistical (mean) property. Position sizing must assume this noise.

## Bugs caught by validation (kept on record)
1. Hedge-share sign error (held short stock against short call) -> residuals ±2000.
2. Trading-day/calendar-day horizon mismatch (path 30/252y vs option 30/365y) -> bias -134.
   Both fixed; horizon mismatch is now an asserted invariant in hedge_pnl.

## Readiness verdict
MACHINERY READY, MARKETS NOT TOUCHED. The stack prices, detects, Greeks and
hedges consistently on synthetic truth. Gating items for real NIFTY use:
- NSE chain snapshots with QUOTE TIMESTAMPS + bid/ask (LTP-only data invalidates
  the executable-terms logic), expiry calendar incl. weeklies, strike grid;
- risk-neutral (implied) calibration on live chains (Stage-3 code ready, untested
  on market data); physical params must NEVER price;
- latent-variance filter for live delta (synthetic runs use oracle V_t);
- vega hedging / vol-risk limits (current loop is delta-only; eps=0.55 vol swings
  dominate single-trade noise — vol-risk mandate required before sizing);
- NSE fee schedule, STT-on-premium, margins and exercise/assignment rules; re-verify
  European cash-settled assumption per contract spec.
Do NOT trade on this stack until all six clear on REAL chains.