# Stage 2 — Data validation report

Source config: `configs/data_v1.yaml`
Generated: 2026-10-08T14:25:44 | rows: 2840 | range: 2015-01-01..2025-12-31

## Tradability rule (frozen V1)
signal(t close, using data <= t close) -> execution at t+1 open or later.
Overnight gaps measured by `overnight_gap = open_t/close_{t-1} - 1`.
Costs modelled separately in backtest (default 5 bps/side stress).

## Checks
| check | status | detail |
|---|---|---|
| missing_values | PASS | critical NaNs={'open': 0, 'high': 0, 'low': 0, 'close': 0}; aux NaNs={'india_vix_close': 0, 'rf_annual': 0} |
| duplicate_timestamps | PASS | duplicated index entries=0 |
| monotonic_unique_index | PASS | monotonic=True, unique=True |
| market_holidays | PASS | weekend rows=0; holiday rows=0 |
| ohlc_logic | PASS | range/logic violations=0 |
| index_continuity | PASS | rows with |log_ret|>20% = 0 (none) |
| stale_observations | PASS | stale-flagged rows=0 (review; feed freeze if clustered) |
| timestamp_alignment | PASS | rows=2840, range=2015-01-01..2025-12-31, aux NaNs={'india_vix_close': 0, 'rf_annual': 0} |
| survivorship | PASS | N/A for index level (published NIFTY history embeds historical constituents). Constituent-level strategies would need PIT lists — out of scope V1. |
| lookahead_audit | PASS | suspicious future columns=none; return recompute match=True; all rows on calendar=True. RULE: features@row t use cols<=t close; execution>=t+1 open. |

## Splits (embargo=5d)
| split | n | start | end | close range |
|---|---|---|---|---|
| train | 1566 | 2015-01-01 | 2020-12-31 | 8062.3→22127.8 |
| validation | 519 | 2021-01-05 | 2022-12-30 | 22068.8→22564.3 |
| test | 750 | 2023-01-05 | 2025-12-31 | 22147.1→34957.8 |

## Index methodology / survivorship
NIFTY 50 index-adjustment policy (V1, frozen):
- NIFTY 50 is an index, not a stock: constituent splits/bonuses/dividends do
  NOT create price discontinuities in the published index level; the divisor
  absorbs them. Therefore NO split/dividend adjustment is applied to OHLC.
- What IS logged: (a) NSE index methodology / base-year changes, (b) constituent
  reconstitutions (semi-annual), (c) special sessions (Muhurat) and exchange
  outages. These appear in reports/data_quality.md, not as price edits.
- Continuity assertion enforced in validation: abs(log_ret) > 20% on a normal
  trading day is flagged as suspected bad print (1987-style crash moves do not
  exist in NIFTY history at that scale; 2020-03 crash max single-day fall ~13%).
- Survivorship: the published index inherently reflects historical constituents
  (no survivorship bias CAN exist for an index level). Constituent-level
  backtests would need point-in-time constituent lists — out of scope for V1,
  which trades the index instrument only.


## Limitations / V2 items
- Options chains: UNAVAILABLE in V1 (loader raises OptionalDataUnavailable).
  Options-implied calibration + Signal F deferred to Stage 3+/V2.
- Risk-free: DEMO synthetic constant — not market data
- Calendar fallback: built-in 2024–25 holidays only if data/raw/nse_holidays.csv absent.
  Replace with full NSE holiday history before publication.
