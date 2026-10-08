"""Pipeline smoke tests. Run: pytest tests/ -q (from heston-nifty/)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import data_cleaning as DC
from data import data_loader as DL
from data import data_validation as DV
from data.make_dataset import build_demo_frames


def _demo():
    nifty_raw, vix_raw, rf_raw, cal = build_demo_frames("2020-01-01", "2021-06-30", seed=7)
    merged = DC.merge_daily(DC.clean_nifty_ohlc(nifty_raw), DC.clean_india_vix(vix_raw),
                            DC.clean_risk_free(rf_raw), cal, provenance="test")
    return merged, cal


def test_no_duplicate_or_weekend_rows():
    merged, cal = _demo()
    assert check_pass(DV.check_duplicates(merged))
    assert check_pass(DV.check_calendar(merged, cal))


def test_ohlc_logic_and_missing():
    merged, cal = _demo()
    assert check_pass(DV.check_ohlc_logic(merged))
    assert check_pass(DV.check_missing(merged))


def test_lookahead_audit():
    merged, cal = _demo()
    assert check_pass(DV.lookahead_audit(merged, cal))


def test_splits_have_embargo_and_no_overlap():
    merged, _ = _demo()
    splits = DV.make_splits(merged, "2020-09-30", "2020-12-31", embargo_days=5)
    assert splits["train"].index.max() < splits["validation"].index.min()
    assert splits["validation"].index.max() < splits["test"].index.min()
    assert all(len(v) > 0 for v in splits.values())


def test_options_unavailable_in_v1():
    try:
        DL.load_options_snapshot("data/raw/nonexistent_options.csv")
    except DL.OptionalDataUnavailable:
        return
    raise AssertionError("options stub must raise OptionalDataUnavailable")


def test_ffill_never_backfills_leading_nan():
    idx = pd.bdate_range("2020-01-01", periods=10)
    cal = pd.DataFrame({"is_trading_day": True}, index=idx)
    nifty = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                           "close": 100.0, "volume": np.nan}, index=idx)
    vix = pd.DataFrame({"india_vix_close": [np.nan, np.nan, 15.0] + [16.0] * 7}, index=idx)
    rf = pd.DataFrame({"rf_annual": 0.06}, index=idx)
    merged = DC.merge_daily(DC.clean_nifty_ohlc(nifty), DC.clean_india_vix(vix),
                            DC.clean_risk_free(rf), cal)
    assert np.isnan(merged["india_vix_close"].iloc[0]), "leading VIX NaN must NOT be backfilled"


def check_pass(c) -> bool:
    return bool(c.passed)
