"""Stage 2 — raw data loading for Heston-NIFTY research.

Responsibility: fetch / load raw files ONLY. No cleaning, no forward-fill,
no return computation, no calibration. Every function stamps provenance
(source, download_time, timezone) so the look-ahead audit can verify
point-in-time correctness.

Source priority (frozen for V1):
  1. NSE official CSVs in data/raw/ (manual download, highest trust)
  2. yfinance (^NSEI for NIFTY 50, ^INDIAVIX for India VIX)
  3. Cached parquet in data/raw/ (reproducibility fallback)

Timezone convention: all dates are NSE trading days, Asia/Kolkata.
A `date` means "information released on or before that day's close".
Nothing returned by this module may contain data from date > requested end.

Options data: V1 does NOT require it. `load_options_snapshot` documents the
V2 schema and raises OptionalDataUnavailable when files are absent, so no
downstream code can silently invent implied-vol data.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent / "raw"

# ---------------------------------------------------------------------------
# Exceptions / metadata
# ---------------------------------------------------------------------------


class OptionalDataUnavailable(FileNotFoundError):
    """Raised when V2-only data (options chains) is requested in V1."""


@dataclass
class Provenance:
    source: str
    download_time_ist: str
    start: str
    end: str
    rows: int

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "download_time_ist": self.download_time_ist,
            "start": self.start,
            "end": self.end,
            "rows": self.rows,
        }


def _now_ist() -> str:
    try:
        from zoneinfo import ZoneInfo

        return _dt.datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    except Exception:
        return _dt.datetime.now().isoformat() + " (local, IST unknown)"


def _normalise_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce index/column to tz-naive DatetimeIndex normalised to midnight."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(
            "Asia/Kolkata"
        ).dt.tz_localize(None).dt.normalize()
        df = df.set_index("date").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        try:
            idx = idx.tz_convert("Asia/Kolkata").tz_localize(None).normalize()
        except Exception:
            idx = pd.to_datetime(df.index).normalize()
        df.index = idx
        df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Generic fetch helpers
# ---------------------------------------------------------------------------


def fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily OHLC from yfinance. Raises ImportError-friendly error."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is not installed. Run `pip install yfinance` "
            "or place an NSE official CSV in data/raw/."
        ) from exc
    # NOTE: yfinance `end` is exclusive; we keep that semantic and document it.
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    return _normalise_date_index(df.reset_index().rename(columns={"index": "date", "datetime": "date"}))


def load_local_csv(path: str | os.PathLike, **kwargs) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Local CSV not found: {p}")
    df = pd.read_csv(p, **kwargs)
    # Normalise common NSE header variants.
    rename = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename)
    if "date" not in df.columns:
        for cand in ("timestamp", "trade_date", "datetime"):
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    return _normalise_date_index(df)


# ---------------------------------------------------------------------------
# 1. NIFTY 50 spot / index
# ---------------------------------------------------------------------------

NIFTY_RAW_SCHEMA = ["date", "open", "high", "low", "close", "volume"]


def load_nifty_spot(
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    local_csv: Optional[str | os.PathLike] = None,
    symbol: str = "^NSEI",
) -> tuple[pd.DataFrame, Provenance]:
    """Load NIFTY 50 daily OHLC.

    Returns (df, provenance). df index = trading-day date, columns at minimum
    open/high/low/close (float, index points). Volume may be NaN for indices.
    """
    if local_csv is not None:
        df = load_local_csv(local_csv)
        prov = Provenance(f"local_csv:{local_csv}", _now_ist(), start, end, len(df))
    else:
        # Prefer a cached NSE file if present.
        cached = RAW_DIR / "nifty_spot_raw.csv"
        if cached.exists():
            df = load_local_csv(cached)
            prov = Provenance(f"local_csv:{cached}", _now_ist(), start, end, len(df))
        else:
            df = fetch_yfinance(symbol, start, end)
            prov = Provenance(f"yfinance:{symbol}", _now_ist(), start, end, len(df))
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    keep = [c for c in ["open", "high", "low", "close", "volume", "adj_close"] if c in df.columns]
    df = df[keep].astype(float, errors="ignore")
    df["source"] = prov.source
    return df, prov


# ---------------------------------------------------------------------------
# 2. India VIX
# ---------------------------------------------------------------------------


def load_india_vix(
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    local_csv: Optional[str | os.PathLike] = None,
    symbol: str = "^INDIAVIX",
) -> tuple[pd.DataFrame, Provenance]:
    """Load India VIX daily close (annualised %, e.g. 13.5 = 13.5%)."""
    if local_csv is not None:
        df = load_local_csv(local_csv)
        prov = Provenance(f"local_csv:{local_csv}", _now_ist(), start, end, len(df))
    else:
        cached = RAW_DIR / "indiavix_raw.csv"
        if cached.exists():
            df = load_local_csv(cached)
            prov = Provenance(f"local_csv:{cached}", _now_ist(), start, end, len(df))
        else:
            raw = fetch_yfinance(symbol, start, end)
            # yfinance VIX: use close column only.
            close_col = "close" if "close" in raw.columns else raw.columns[-1]
            df = raw[[close_col]].rename(columns={close_col: "india_vix_close"})
            prov = Provenance(f"yfinance:{symbol}", _now_ist(), start, end, len(df))
    if "india_vix_close" not in df.columns:
        # NSE header variants: 'close', 'vix'
        for cand in ("close", "vix", "adj_close"):
            if cand in df.columns:
                df = df.rename(columns={cand: "india_vix_close"})
                break
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    df = df[["india_vix_close"]].astype(float)
    df["source"] = prov.source
    return df, prov


# ---------------------------------------------------------------------------
# 3. Risk-free rate (91-day T-bill, RBI)
# ---------------------------------------------------------------------------


def load_risk_free_rate(
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    local_csv: Optional[str | os.PathLike] = None,
    fallback_annual: Optional[float] = None,
) -> tuple[pd.DataFrame, Provenance]:
    """Load risk-free rate as annualised decimal (0.065 = 6.5%).

    Preferred: local CSV with columns [date, rf_annual] from RBI 91D T-bill
    auction yields, resampled to daily (resampling happens in cleaning, NOT
    here — this function returns observations as-is to avoid look-ahead).

    fallback_annual: if given AND no CSV exists, broadcast a constant with
    source='CONSTANT_FALLBACK:DO_NOT_PUBLISH'. If None and no CSV, raise.
    """
    candidates = []
    if local_csv is not None:
        candidates.append(Path(local_csv))
    candidates += [RAW_DIR / "riskfree_91d.csv", RAW_DIR / "riskfree_raw.csv"]
    for c in candidates:
        if c.exists():
            df = load_local_csv(c)
            col = "rf_annual" if "rf_annual" in df.columns else (
                "yield" if "yield" in df.columns else df.columns[0]
            )
            out = df[[col]].rename(columns={col: "rf_annual"}).astype(float)
            # Convert % (e.g. 6.5) to decimal if needed.
            if out["rf_annual"].median() > 1.0:
                out["rf_annual"] = out["rf_annual"] / 100.0
            out = out.loc[(out.index >= pd.Timestamp(start)) & (out.index <= pd.Timestamp(end))]
            return out.assign(source=f"local_csv:{c}"), Provenance(
                f"local_csv:{c}", _now_ist(), start, end, len(out)
            )
    if fallback_annual is not None:
        idx = pd.bdate_range(start, end)  # business days; filtered to NSE later
        out = pd.DataFrame({"rf_annual": float(fallback_annual)}, index=idx)
        out.index.name = "date"
        out["source"] = "CONSTANT_FALLBACK:DO_NOT_PUBLISH"
        log.warning("Using constant Rf fallback %.4f — flag results accordingly.", fallback_annual)
        return out, Provenance("constant_fallback", _now_ist(), start, end, len(out))
    raise FileNotFoundError(
        "No risk-free CSV found in data/raw/ (expected riskfree_91d.csv with "
        "[date, rf_annual]). Place the RBI 91D T-bill series there or pass "
        "fallback_annual=0.065 explicitly for exploratory runs."
    )


# ---------------------------------------------------------------------------
# 4. Trading calendar (NSE)
# ---------------------------------------------------------------------------

DEFAULT_NSE_HOLIDAYS_2024_2025 = [
    # Minimal built-in fallback. MUST be replaced by data/raw/nse_holidays.csv
    # for production runs. Format: [date, description].
    ("2024-01-22", "Ram Mandir consecration (special)"),
    ("2024-01-26", "Republic Day"),
    ("2024-03-08", "Mahashivratri"),
    ("2024-03-25", "Holi"),
    ("2024-03-29", "Good Friday"),
    ("2024-04-11", "Id-Ul-Fitr"),
    ("2024-04-17", "Ram Navmi"),
    ("2024-05-01", "Maharashtra Day"),
    ("2024-05-20", "General election (Mumbai)"),
    ("2024-06-17", "Bakrid"),
    ("2024-07-17", "Moharram"),
    ("2024-08-15", "Independence Day"),
    ("2024-09-18", "Ganesh Chaturthi (special holiday)"),
    ("2024-10-02", "Gandhi Jayanti"),
    ("2024-11-01", "Diwali Laxmi Pujan"),
    ("2024-11-15", "Guru Nanak Jayanti"),
    ("2024-12-25", "Christmas"),
    ("2025-01-26", "Republic Day (Sun; no session)"),
    ("2025-02-26", "Mahashivratri"),
    ("2025-03-14", "Holi"),
    ("2025-03-31", "Id-Ul-Fitr"),
    ("2025-04-10", "Mahavir Jayanti"),
    ("2025-04-14", "Ambedkar Jayanti"),
    ("2025-04-18", "Good Friday"),
    ("2025-05-01", "Maharashtra Day"),
    ("2025-08-15", "Independence Day"),
    ("2025-08-27", "Ganesh Chaturthi"),
    ("2025-10-02", "Gandhi Jayanti"),
    ("2025-10-21", "Diwali Laxmi Pujan"),
    ("2025-11-05", "Guru Nanak Jayanti"),
    ("2025-12-25", "Christmas"),
]


def load_trading_calendar(
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    holidays_csv: Optional[str | os.PathLike] = None,
) -> pd.DataFrame:
    """Build NSE trading calendar: every calendar day -> trading flag.

    Trading day = Mon–Fri AND NOT in holiday list. Special sessions
    (e.g. Muhurat) are flagged via holidays_csv column `is_special_session`.
    """
    all_days = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame(index=all_days)
    cal.index.name = "date"
    cal["weekday"] = cal.index.weekday  # Mon=0
    holidays: dict[str, str] = {}
    src = "builtin_fallback_2024_2025:REPLACE_WITH_NSE_CSV"
    for cand in ([Path(holidays_csv)] if holidays_csv else []) + [RAW_DIR / "nse_holidays.csv"]:
        if cand.exists():
            h = pd.read_csv(cand)
            h.columns = [c.strip().lower() for c in h.columns]
            dcol = "date" if "date" in h.columns else h.columns[0]
            for _, r in h.iterrows():
                holidays[str(pd.Timestamp(r[dcol]).date())] = str(
                    r.get("description", r.get("holiday", "holiday"))
                )
            src = f"local_csv:{cand}"
            break
    else:
        for d, desc in DEFAULT_NSE_HOLIDAYS_2024_2025:
            holidays[d] = desc
    cal["holiday_description"] = cal.index.map(lambda d: holidays.get(str(d.date()), ""))
    cal["is_holiday"] = cal["holiday_description"] != ""
    cal["is_weekend"] = cal["weekday"] >= 5
    cal["is_trading_day"] = ~(cal["is_weekend"] | cal["is_holiday"])
    cal["calendar_source"] = src
    return cal[["is_trading_day", "is_weekend", "is_holiday", "holiday_description", "calendar_source"]]


# ---------------------------------------------------------------------------
# 5. NIFTY options (V2 — schema only in V1)
# ---------------------------------------------------------------------------

OPTIONS_SCHEMA = [
    "date",  # quote date (trading day)
    "quote_time_ist",  # snapshot timestamp — critical for look-ahead audit
    "expiry",  # YYYY-MM-DD
    "strike",  # float
    "option_type",  # 'CE' | 'PE'
    "bid", "ask", "ltp",  # floats
    "volume", "open_interest",  # ints
    "implied_vol",  # decimal, e.g. 0.14
    "underlying_spot",  # NIFTY level at quote_time
    "source",
]


def options_schema() -> list[str]:
    return list(OPTIONS_SCHEMA)


def load_options_snapshot(
    snapshot_csv: str | os.PathLike,
) -> tuple[pd.DataFrame, Provenance]:
    """Load ONE options-chain snapshot (V2). Strict schema enforcement.

    Raises OptionalDataUnavailable if the file is absent — callers must treat
    options-implied calibration / Signal F as unavailable in V1.
    """
    p = Path(snapshot_csv)
    if not p.exists():
        raise OptionalDataUnavailable(
            f"Options snapshot not found: {p}. Options-implied calibration and "
            f"Signal F are V2-only and must be reported as unavailable."
        )
    df = pd.read_csv(p)
    missing = [c for c in OPTIONS_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Options CSV missing columns {missing}. Required: {OPTIONS_SCHEMA}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df, Provenance(f"local_csv:{p}", _now_ist(), str(df['date'].min()), str(df['date'].max()), len(df))
