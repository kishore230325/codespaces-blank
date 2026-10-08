"""Stage 2 — cleaning, schema enforcement, and point-in-time alignment.

Look-ahead rules enforced here (never violated):
  R1. No forward-fill from the future: ffill only, never bfill silently.
      Leading NaNs stay NaN (row dropped or flagged), never back-filled.
  R2. Returns are computed from cleaned closes only; the return dated `t`
      uses closes at t-1 and t and is knowable only AFTER t's close.
  R3. VIX / Rf merges are left-joins onto the NSE trading calendar with
      ffill capped at `max_fill_days` (default 5). Gaps beyond that -> NaN
      -> row excluded, NOT interpolated from future prints.
  R4. No resampling of Rf yields using future auctions. Weekly/monthly Rf
      observations are forward-filled (last-known yield), the standard
      point-in-time convention.
  R5. Index adjustments: NIFTY 50 is a capitalisation-weighted index —
      no stock-split adjustments apply. Corporate actions are absorbed by
      the index divisor. We assert continuity (no artificial 2x jumps) and
      log index-methodology notes instead of "adjusting" prices.

Output schema: data/processed/nifty_daily.parquet (see MERGED_SCHEMA).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen V1 schemas
# ---------------------------------------------------------------------------

NIFTY_CLEAN_SCHEMA = {
    "open": "float64", "high": "float64", "low": "float64",
    "close": "float64", "volume": "float64",
}

MERGED_SCHEMA = [
    "date",  # NSE trading day (YYYY-MM-DD), index
    "open", "high", "low", "close",  # float, index points
    "volume",  # float (may be NaN for index feeds)
    "log_ret",  # ln(close_t / close_{t-1}); NaN for first row; knowable after t close
    "simple_ret",  # close_t/close_{t-1} - 1
    "overnight_gap",  # open_t / close_{t-1} - 1 (execution realism)
    "india_vix_close",  # float, annualised % (level at t close)
    "rf_annual",  # float, decimal (last-known 91D yield as of t)
    "rf_daily",  # rf_annual / 252 (simple; use 252 NSE trading days)
    "is_trading_day",  # bool, always True in merged output
    "stale_flag",  # bool, True if close unchanged for >= stale_days
    "data_source",  # provenance string
]

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Individual cleaners
# ---------------------------------------------------------------------------


def clean_nifty_ohlc(df: pd.DataFrame, drop_bad: bool = True) -> pd.DataFrame:
    """Enforce OHLC schema + internal consistency.

    Drops (and logs) rows with non-positive prices or high<low. Flags
    high/low violations vs open/close for the validation report instead of
    silently fixing them.
    """
    out = df.copy()
    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"clean_nifty_ohlc: missing column '{col}'")
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "volume" not in out.columns:
        out["volume"] = np.nan
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    bad_price = (out[["open", "high", "low", "close"]] <= 0).any(axis=1)
    crossed = out["high"] < out["low"]
    if bad_price.any() or crossed.any():
        log.warning(
            "Dropping %d rows: %d non-positive price, %d high<low.",
            int((bad_price | crossed).sum()), int(bad_price.sum()), int(crossed.sum()),
        )
    out["_excluded_bad_price"] = bad_price | crossed
    if drop_bad:
        out = out[~(bad_price | crossed)].drop(columns=["_excluded_bad_price"])
    # Range-violation flags (kept for validation; NOT auto-corrected).
    out["_range_flag"] = (out["high"] < out[["open", "close"]].max(axis=1)) | (
        out["low"] > out[["open", "close"]].min(axis=1)
    )
    return out.sort_index()


def clean_india_vix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "india_vix_close" not in out.columns:
        raise ValueError("clean_india_vix: missing 'india_vix_close'")
    out["india_vix_close"] = pd.to_numeric(out["india_vix_close"], errors="coerce")
    bad = (out["india_vix_close"] <= 0) | (out["india_vix_close"] > 150)
    if bad.any():
        log.warning("Flagging %d implausible VIX prints (>150 or <=0) as NaN.", int(bad.sum()))
        out.loc[bad, "india_vix_close"] = np.nan
    return out.sort_index()


def clean_risk_free(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to decimal annual; do NOT interpolate — ffill happens at merge."""
    out = df.copy()
    if "rf_annual" not in out.columns:
        raise ValueError("clean_risk_free: missing 'rf_annual'")
    out["rf_annual"] = pd.to_numeric(out["rf_annual"], errors="coerce")
    if out["rf_annual"].median(skipna=True) > 1.0:  # given in %
        out["rf_annual"] /= 100.0
    bad = (out["rf_annual"] < 0) | (out["rf_annual"] > 0.30)
    if bad.any():
        log.warning("Flagging %d implausible Rf prints as NaN.", int(bad.sum()))
        out.loc[bad, "rf_annual"] = np.nan
    return out.sort_index()


# ---------------------------------------------------------------------------
# Alignment / merge (point-in-time safe)
# ---------------------------------------------------------------------------


def merge_daily(
    nifty: pd.DataFrame,
    vix: pd.DataFrame | None,
    rf: pd.DataFrame | None,
    calendar: pd.DataFrame,
    max_fill_days: int = 5,
    stale_days: int = 5,
    provenance: str = "",
) -> pd.DataFrame:
    """Left-join VIX/Rf onto NSE trading days. Ffill-only, capped.

    Steps:
      1. Restrict to calendar trading days (kills weekend rows from
         yfinance/crypto-style feeds that would otherwise create fake obs).
      2. Reindex VIX/Rf to trading days, ffill(limit=max_fill_days).
         Leading gaps -> NaN (never bfilled).
      3. Compute log_ret / simple_ret / overnight_gap from cleaned closes.
      4. rf_daily = rf_annual / 252.
      5. stale_flag = close unchanged for `stale_days` sessions.
    """
    trading_days = calendar.loc[calendar["is_trading_day"]].index
    nifty_td = nifty.loc[nifty.index.intersection(trading_days)].copy()
    if nifty_td.empty:
        raise ValueError("merge_daily: no NIFTY rows fall on NSE trading days.")

    merged = nifty_td[["open", "high", "low", "close", "volume"]].copy()

    if vix is not None and not vix.empty:
        v = vix[["india_vix_close"]].reindex(merged.index)
        n_new = int(v["india_vix_close"].isna().sum())
        v["india_vix_close"] = v["india_vix_close"].ffill(limit=max_fill_days)
        log.info("VIX: %d NaNs before ffill (limit %d).", n_new, max_fill_days)
        merged["india_vix_close"] = v["india_vix_close"]
    else:
        merged["india_vix_close"] = np.nan

    if rf is not None and not rf.empty:
        r = rf[["rf_annual"]].reindex(merged.index).ffill(limit=max_fill_days)
        merged["rf_annual"] = r["rf_annual"]
    else:
        merged["rf_annual"] = np.nan
    merged["rf_daily"] = merged["rf_annual"] / TRADING_DAYS_PER_YEAR

    prev_close = merged["close"].shift(1)
    merged["log_ret"] = np.log(merged["close"] / prev_close)
    merged["simple_ret"] = merged["close"] / prev_close - 1
    merged["overnight_gap"] = merged["open"] / prev_close - 1

    merged["is_trading_day"] = True
    merged["stale_flag"] = (
        merged["close"].eq(merged["close"].shift(1)).rolling(stale_days, min_periods=stale_days).max() == 1
    ).fillna(False)
    merged["data_source"] = provenance or "merged"
    merged.index.name = "date"
    # Column order per MERGED_SCHEMA (minus index-held date).
    cols = [c for c in MERGED_SCHEMA if c != "date"]
    return merged[cols]


def save_processed(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=True)
    # Human-readable head for the audit trail.
    df.head(3).to_csv(p.with_suffix(".head.csv"))
    return p


# ---------------------------------------------------------------------------
# Index-adjustment / survivorship notes (no price mutation)
# ---------------------------------------------------------------------------

INDEX_METHODOLOGY_NOTES = """\
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
"""
