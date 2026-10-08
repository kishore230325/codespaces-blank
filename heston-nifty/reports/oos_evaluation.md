# Directional strategy — walk-forward backtest + statistical significance

## Test block (untouched until this run): 2023-01-05..2025-12-31
Rules frozen a priori; 7 candidates reported (6 active + B&H ref); Bonferroni alpha = 0.0071. Costs 5bps/side + 2bps slip, t+1-open execution.

| rule | n | cover | hit¹ | binom p | mean/trade | t(non-overlap) | bootCI | Sharpe | alpha | beta | dSharpe-p [CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
¹hit = fraction of trades with gross return > 0 (pre-cost direction check).
| TRICH-h1 | 0 | — | — | — | — | — | — | — | — | — | — |
| TRICH-h5 | 428 | 0.574 | 0.514 | 0.595 | -0.00161 | -0.88 | [-0.00417,+0.00093] | -0.81 | -0.1686 | 0.52 | 0.0002 [-2.09,-0.59] |
| VOLFILT-h5 | 251 | 0.337 | 0.498 | 1 | -0.00230 | -1.03 | [-0.00532,+0.00075] | -1.15 | -0.1293 | 0.23 | 0.0033 [-2.80,-0.56] |
| MOM20 | 431 | 0.575 | 0.513 | 0.6301 | +0.00056 | +0.25 | [-0.00213,+0.00329] | -0.32 | -0.1166 | 0.62 | 0.011 [-1.52,-0.19] |
| SMA50 | 441 | 0.589 | 0.542 | 0.08636 | +0.00236 | +0.22 | [-0.00020,+0.00496] | +0.03 | -0.0650 | 0.63 | 0.1404 [-1.17,+0.17] |
| VIXFLT | 481 | 0.642 | 0.497 | 0.9274 | -0.00115 | -0.62 | [-0.00340,+0.00099] | -0.72 | -0.1568 | 0.50 | 0.003 [-2.08,-0.44] |

Buy-and-hold reference (same span, zero-cost): gross=+0.5709.

## Verdict (directional H0: no excess risk-adjusted return)
Applied decision rule (fixed before the run): reject directional H0 for a rule
only on Bonferroni-significant (p < 0.0071) superiority with consistent
validation-block behaviour. Outcome:
- No rule is significantly SUPERIOR to buy-and-hold on any pre-specified
  statistic. Directional H0 is NOT rejected — H1 (excess risk-adjusted
  returns from Heston/QE signals) finds no support.
- TRICH-h5 (p=0.0002) and VOLFILT-h5 (p=0.0033) are significantly INFERIOR to
  buy-and-hold on risk-adjusted return — the expected signature of a
  long/flat timing rule with ~50% hit rate in a +57% rising market after costs,
  not evidence of perverse signal (hit rates 0.51/0.50 ~= coin flip).
- The pre-registered vol-filter hypothesis (Stage-5 Signal E helps direction)
  is REJECTED as a directional aid: VOLFILT mean/trade (-0.00230) < TRICH (-0.00161).
  Signal E remains a volatility-forecast carrier (Stage-5/OOS vol-corr ~0.47),
  which was never a directional claim.
- Validation-block consistency: Stage-5 directional accuracy at base rate with
  Brier skill <= 0 on 2021-2022 replicates OOS on 2023-2025. Two independent
  blocks, same null. This is a replicated null, not an inconclusive one.
- Demo-data scope (load-bearing): synthetic DGP has constant drift and no true
  directional predictability, so the null is DGP-expected. The transferable
  products are the validated pipeline (Stages 2-6), the frozen decision
  framework, and the vol-forecast result — NOT a tradeable directional edge.