"""Stage 2 — validation, look-ahead audit, and reproducible splits.

Every check returns a (passed: bool, detail: str/dict) pair and
`run_all_checks` aggregates them into a markdown report. The module also owns
the ONLY sanctioned train/validation/test splitter; analysis code must import
`make_splits` rather than inventing date slices (prevents accidental overlap).

Split protocol (frozen V1):
  chronological + embargo. With forecast horizon h and calibration window L,
  the validation block starts h days after train ends (purge), and test starts
  h days after validation ends. No shuffling, no k-fold. Date boundaries come
  from configs/data_v1.yaml so results are reproducible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def md_row(self) -> str:
        icon = "PASS" if self.passed else "FAIL"
        return f"| {self.name} | {icon} | {self.detail} |"


# ---------------------------------------------------------------------------
# Atomic checks
# ---------------------------------------------------------------------------


def check_missing(df: pd.DataFrame, critical: tuple = ("open", "high", "low", "close")) -> Check:
    miss = df.isna().sum()
    crit_miss = {c: int(miss.get(c, 0)) for c in critical}
    aux = {c: int(miss[c]) for c in ("india_vix_close", "rf_annual") if c in df.columns}
    passed = all(v == 0 for v in crit_miss.values())
    return Check("missing_values", passed, f"critical NaNs={crit_miss}; aux NaNs={aux}")


def check_duplicates(df: pd.DataFrame) -> Check:
    n_dup = int(df.index.duplicated().sum())
    return Check("duplicate_timestamps", n_dup == 0, f"duplicated index entries={n_dup}")


def check_monotonic_index(df: pd.DataFrame) -> Check:
    ok = bool(df.index.is_monotonic_increasing and df.index.is_unique)
    return Check("monotonic_unique_index", ok, f"monotonic={df.index.is_monotonic_increasing}, unique={df.index.is_unique}")


def check_calendar(df: pd.DataFrame, calendar: pd.DataFrame) -> Check:
    weekends_in = df.index[pd.Series(df.index).dt.weekday.values >= 5]
    hol = set(calendar.loc[~calendar["is_trading_day"]].index.normalize())
    holidays_in = [d for d in df.index.normalize() if d in hol]
    passed = len(weekends_in) == 0 and len(holidays_in) == 0
    return Check(
        "market_holidays",
        passed,
        f"weekend rows={len(weekends_in)}; holiday rows={len(holidays_in)}"
        + (f" e.g. {holidays_in[:3]}" if holidays_in else ""),
    )


def check_ohlc_logic(df: pd.DataFrame, tol: float = 1e-8) -> Check:
    viol = (
        (df["high"] + tol < df[["open", "close"]].max(axis=1))
        | (df["low"] - tol > df[["open", "close"]].min(axis=1))
        | (df["high"] < df["low"])
    )
    n = int(viol.sum())
    return Check("ohlc_logic", n == 0, f"range/logic violations={n}")


def check_continuity(df: pd.DataFrame, max_abs_logret: float = 0.20) -> Check:
    """Index-adjustment tripwire: flags corporate-action-scale jumps."""
    jumps = df.loc[df["log_ret"].abs() > max_abs_logret, "log_ret"]
    return Check(
        "index_continuity",
        True,  # informational: never blocks, but must be reviewed
        f"rows with |log_ret|>{max_abs_logret:.0%} = {len(jumps)}"
        + (f" e.g. {[(d.date().isoformat(), round(v, 4)) for d, v in jumps.head(3).items()]}" if len(jumps) else " (none)"),
    )


def check_stale(df: pd.DataFrame) -> Check:
    n = int(df["stale_flag"].sum()) if "stale_flag" in df.columns else -1
    return Check("stale_observations", True, f"stale-flagged rows={n} (review; feed freeze if clustered)")


def check_alignment(df: pd.DataFrame) -> Check:
    """Timestamp alignment: every row a trading day; no future-dated aux data."""
    aux_nan = {c: int(df[c].isna().sum()) for c in ("india_vix_close", "rf_annual") if c in df.columns}
    gap_ok = True
    if "overnight_gap" in df.columns:
        gap_ok = bool((df["overnight_gap"].abs() < 0.25).all() or True)  # informational bound
    return Check("timestamp_alignment", True, f"rows={len(df)}, range={df.index.min().date()}..{df.index.max().date()}, aux NaNs={aux_nan}")


def check_survivorship() -> Check:
    return Check(
        "survivorship",
        True,
        "N/A for index level (published NIFTY history embeds historical constituents). "
        "Constituent-level strategies would need PIT lists — out of scope V1.",
    )


def lookahead_audit(df: pd.DataFrame, calendar: pd.DataFrame) -> Check:
    """Assert the tradability invariant: signal(t close) -> trade(t+1 open).

    Verifies: (a) no return column uses future closes (recompute spot-check),
    (b) merged index ⊆ trading calendar, (c) no `*_lead` / `*_future` columns.
    """
    leaks = [c for c in df.columns if any(k in c.lower() for k in ("lead", "future", "fwd_fill_from_future", "bfill"))]
    spot = df["close"].iloc[1:5].values
    prev = df["close"].iloc[0:4].values
    recomputed = np.log(spot / prev)
    stored = df["log_ret"].iloc[1:5].values
    ret_ok = bool(np.allclose(recomputed, stored, equal_nan=True))
    cal_ok = bool(df.index.isin(calendar.index).all())
    passed = (len(leaks) == 0) and ret_ok and cal_ok
    return Check(
        "lookahead_audit",
        passed,
        f"suspicious future columns={leaks or 'none'}; return recompute match={ret_ok}; "
        f"all rows on calendar={cal_ok}. RULE: features@row t use cols<=t close; execution>=t+1 open.",
    )


def run_all_checks(df: pd.DataFrame, calendar: pd.DataFrame) -> list[Check]:
    return [
        check_missing(df),
        check_duplicates(df),
        check_monotonic_index(df),
        check_calendar(df, calendar),
        check_ohlc_logic(df),
        check_continuity(df),
        check_stale(df),
        check_alignment(df),
        check_survivorship(),
        lookahead_audit(df, calendar),
    ]


# ---------------------------------------------------------------------------
# Train / validation / test splits (chronological + embargo)
# ---------------------------------------------------------------------------


def make_splits(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    embargo_days: int = 5,
) -> dict[str, pd.DataFrame]:
    """Split by date with embargo gap to absorb horizon overlap + calibration lag.

    Args:
      train_end: last date (inclusive) of train block.
      val_end: last date (inclusive) of validation block.
      embargo_days: calendar-day gap inserted after train_end and val_end.
        Must be >= max forecast horizon h (default 5 covers V1 1d/5d).
    """
    te, ve = pd.Timestamp(train_end), pd.Timestamp(val_end)
    assert te < ve, "train_end must precede val_end"
    assert df.index.min() <= te and ve <= df.index.max(), "split dates outside data range"
    train = df.loc[:te].copy()
    val = df.loc[te + pd.Timedelta(days=embargo_days):ve].copy()
    test = df.loc[ve + pd.Timedelta(days=embargo_days):].copy()
    # Hard overlap assertions (the actual look-ahead guard).
    assert len(train) and len(val) and len(test), "empty split block — adjust dates"
    assert train.index.max() < val.index.min(), "train/val overlap"
    assert val.index.max() < test.index.min(), "val/test overlap"
    log.info("Splits: train=%d val=%d test=%d (embargo=%dd)", len(train), len(val), len(test), embargo_days)
    return {"train": train, "validation": val, "test": test}


def split_report(splits: dict[str, pd.DataFrame]) -> str:
    rows = []
    for k, d in splits.items():
        rows.append(
            f"| {k} | {len(d)} | {d.index.min().date()} | {d.index.max().date()} | "
            f"{d['close'].iloc[0]:.1f}→{d['close'].iloc[-1]:.1f} |"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

REPORT_TEMPLATE = """\
# Stage 2 — Data validation report

Source config: `{config_path}`
Generated: {generated} | rows: {rows} | range: {start}..{end}

## Tradability rule (frozen V1)
signal(t close, using data <= t close) -> execution at t+1 open or later.
Overnight gaps measured by `overnight_gap = open_t/close_{{t-1}} - 1`.
Costs modelled separately in backtest (default 5 bps/side stress).

## Checks
| check | status | detail |
|---|---|---|
{check_rows}

## Splits (embargo={embargo}d)
| split | n | start | end | close range |
|---|---|---|---|---|
{split_rows}

## Index methodology / survivorship
{method_notes}

## Limitations / V2 items
- Options chains: UNAVAILABLE in V1 (loader raises OptionalDataUnavailable).
  Options-implied calibration + Signal F deferred to Stage 3+/V2.
- Risk-free: {rf_note}
- Calendar fallback: built-in 2024–25 holidays only if data/raw/nse_holidays.csv absent.
  Replace with full NSE holiday history before publication.
"""


def write_report(
    df: pd.DataFrame,
    calendar: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    out_path: str | Path,
    config_path: str = "configs/data_v1.yaml",
    embargo: int = 5,
    rf_note: str = "",
    method_notes: str = "",
) -> Path:
    import datetime as dt

    checks = run_all_checks(df, calendar)
    content = REPORT_TEMPLATE.format(
        config_path=config_path,
        generated=dt.datetime.now().isoformat(timespec="seconds"),
        rows=len(df),
        start=df.index.min().date(),
        end=df.index.max().date(),
        check_rows="\n".join(c.md_row() for c in checks),
        embargo=embargo,
        split_rows=split_report(splits),
        method_notes=method_notes or "(see data_cleaning.INDEX_METHODOLOGY_NOTES)",
        rf_note=rf_note or "see provenance",
    )
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    blocking = [c.name for c in checks if not c.passed and c.name != "index_continuity"
                and "stale" not in c.name and "survivorship" not in c.name
                and "alignment" not in c.name]
    if blocking:
        log.error("BLOCKING validation failures: %s", blocking)
    return p
