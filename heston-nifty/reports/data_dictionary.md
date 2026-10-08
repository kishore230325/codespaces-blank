# Data dictionary — Stage 2 V1 (frozen)

All timestamps: NSE trading day, Asia/Kolkata, tz-naive midnight. A row dated `t`
contains information knowable **on or before t's close** only.

## 1. `data/processed/nifty_daily.parquet` (primary output)
| column | dtype | unit / example | knowable at | notes |
|---|---|---|---|---|
| date (index) | date | 2021-03-15 | t close | NSE trading day only; no weekends/holidays |
| open/high/low/close | float | index points, e.g. 17245.3 | t close | cleaned; high>=max(o,c), low<=min(o,c) asserted |
| volume | float | contracts/shares; NaN ok | t close | index feeds often omit; never imputed |
| log_ret | float | ln(C_t/C_{t-1}), e.g. -0.004 | after t close | NaN first row; recompute-audited |
| simple_ret | float | C_t/C_{t-1}-1 | after t close | — |
| overnight_gap | float | O_t/C_{t-1}-1 | t open | execution-realism; gap risk input for Stage 7 |
| india_vix_close | float | annualised %, e.g. 14.2 | t close | ffill(<=5d) only; leading NaN kept |
| rf_annual | float | decimal, e.g. 0.065 | t close | last-known 91D yield; ffill(<=5d) |
| rf_daily | float | rf_annual/252 | t close | simple daily conversion |
| is_trading_day | bool | True | — | always True in this file |
| stale_flag | bool | True/False | after t close | True if close flat for >=5 sessions |
| data_source | str | provenance chain | — | e.g. yfinance:^NSEI \| ... |

## 2. Raw inputs (`data/raw/`, git-ignored except samples)
- `nifty_spot_raw.csv`: [date, open, high, low, close, volume?] (NSE official preferred; else yfinance ^NSEI)
- `indiavix_raw.csv`: [date, india_vix_close|close] (% units; else yfinance ^INDIAVIX)
- `riskfree_91d.csv`: [date, rf_annual] (RBI 91D auction yield; % or decimal — auto-detected)
- `nse_holidays.csv`: [date, description, (is_special_session?)] — full history required before publication
- `options_snapshot_YYYYMMDD.csv` (V2 only): [date, quote_time_ist, expiry, strike, option_type(CE/PE), bid, ask, ltp, volume, open_interest, implied_vol, underlying_spot, source]

## 3. Calendar (`load_trading_calendar` output, in-memory)
[date, is_trading_day, is_weekend, is_holiday, holiday_description, calendar_source]

## 4. Splits (from `make_splits`, embargo=5d)
- train: 2015-01-01..2020-12-31 | validation: 2021-01-05..2022-12-31 | test: 2023-01-05..2025-12-31
- Rule: train.max < val.min, val.max < test.min (asserted); embargo >= max horizon h.
- Stage 3+ calibration windows draw ONLY from train (expanding/rolling); thresholds tuned on validation; test touched once in Stage 8.

## 5. Explicitly NOT in V1
Options chains, intraday bars, constituent lists, futures basis/roll series, financing rates beyond 91D. Each has a reserved schema/loader stub so V2 cannot silently leak future data.
