"""Orchestrator: raw -> clean -> validate -> splits. No calibration, no signals.

Usage:
  python -m data.make_dataset --config configs/data_v1.yaml
  python -m data.make_dataset --config configs/data_v1.yaml --demo   # offline synthetic data
  python -m data.make_dataset --config configs/data_v1.yaml --no-fetch # use data/raw/*.csv only

--demo builds a reproducible Heston-flavoured synthetic NIFTY path (seeded)
so the pipeline, checks, and splits can be verified without network access.
Demo output is clearly stamped DEMO_SYNTHETIC and must never be published as
market data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data import data_cleaning as DC
from data import data_loader as DL
from data import data_validation as DV

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("make_dataset")


def build_demo_frames(start: str, end: str, seed: int = 42):
    """Synthetic NIFTY-like Heston path + VIX + Rf on NSE business days."""
    rng = np.random.default_rng(seed)
    cal = DL.load_trading_calendar(start, end)
    tdays = cal.loc[cal["is_trading_day"]].index
    n = len(tdays)
    dt = 1 / 252
    kappa, theta, eps, rho, mu = 2.0, 0.04, 0.30, -0.65, 0.08
    V = np.empty(n)
    V[0] = theta
    for t in range(1, n):
        V[t] = np.maximum(
            V[t - 1] + kappa * (theta - V[t - 1]) * dt
            + eps * np.sqrt(max(V[t - 1], 0) * dt) * rng.standard_normal(),
            0.0,
        )
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    logret = (mu - 0.5 * V) * dt + np.sqrt(V * dt) * z2
    close = 8000 * np.exp(np.cumsum(logret))
    open_ = np.empty(n)
    open_[0] = close[0] * (1 - 0.001)
    open_[1:] = close[:-1] * (1 + 0.001 * rng.standard_normal(n - 1))
    spread = 0.004 * close * np.sqrt(V / theta)
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * spread
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * spread
    nifty = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.nan, "source": "DEMO_SYNTHETIC:Heston-flavoured"},
        index=tdays,
    )
    nifty.index.name = "date"
    vix = pd.DataFrame(
        {"india_vix_close": np.clip(100 * np.sqrt(V * 0.9 + 0.002) + rng.standard_normal(n) * 0.5, 8, 60),
         "source": "DEMO_SYNTHETIC"},
        index=tdays,
    )
    vix.index.name = "date"
    rf = pd.DataFrame({"rf_annual": 0.065, "source": "DEMO_SYNTHETIC:constant"},
                      index=tdays)
    rf.index.name = "date"
    return nifty, vix, rf, cal


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data_v1.yaml")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = (root / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    start, end = cfg["period"]["start"], cfg["period"]["end"]

    if args.demo:
        nifty_raw, vix_raw, rf_raw, cal = build_demo_frames(start, end, cfg.get("seed", 42))
        prov = "DEMO_SYNTHETIC(seed=%s)" % cfg.get("seed", 42)
    elif args.no_fetch:
        nifty_raw, p1 = DL.load_nifty_spot(start, end, local_csv=cfg["sources"].get("nifty_csv") or DL.RAW_DIR / "nifty_spot_raw.csv")
        vix_raw, p2 = DL.load_india_vix(start, end, local_csv=cfg["sources"].get("vix_csv") or DL.RAW_DIR / "indiavix_raw.csv")
        rf_raw, p3 = DL.load_risk_free_rate(start, end, local_csv=cfg["sources"].get("rf_csv"))
        cal = DL.load_trading_calendar(start, end, cfg["sources"].get("holidays_csv"))
        prov = f"{p1.source} | {p2.source} | {p3.source}"
    else:
        s = cfg["sources"]
        nifty_raw, p1 = DL.load_nifty_spot(start, end, local_csv=s.get("nifty_csv"))
        try:
            vix_raw, p2 = DL.load_india_vix(start, end, local_csv=s.get("vix_csv"))
        except Exception as e:
            log.warning("VIX load failed (%s); proceeding with NaN VIX.", e)
            vix_raw, p2 = pd.DataFrame(), type("P", (), {"source": f"FAILED:{e}"})()
        try:
            rf_raw, p3 = DL.load_risk_free_rate(start, end, local_csv=s.get("rf_csv"),
                                                fallback_annual=s.get("rf_fallback_annual"))
        except FileNotFoundError as e:
            log.error(str(e))
            return 2
        cal = DL.load_trading_calendar(start, end, s.get("holidays_csv"))
        prov = f"{p1.source} | {p2.source} | {p3.source}"

    # Clean + merge (point-in-time safe; see data_cleaning rules R1–R5).
    nifty = DC.clean_nifty_ohlc(nifty_raw)
    vix = DC.clean_india_vix(vix_raw) if len(vix_raw) else None
    rf = DC.clean_risk_free(rf_raw) if len(rf_raw) else None
    merged = DC.merge_daily(nifty, vix, rf, cal,
                            max_fill_days=cfg["cleaning"]["max_fill_days"],
                            stale_days=cfg["cleaning"]["stale_days"],
                            provenance=prov)

    out_parquet = root / cfg["outputs"]["processed"]
    DC.save_processed(merged, out_parquet)

    splits = DV.make_splits(merged, cfg["splits"]["train_end"], cfg["splits"]["val_end"],
                             cfg["splits"]["embargo_days"])
    rf_note = ("CONSTANT FALLBACK %.3f — replace with RBI 91D series before publication"
               % cfg["sources"].get("rf_fallback_annual", 0.065)) if "CONSTANT" in prov or "fallback" in prov.lower() \
        else "RBI 91D T-bill series from data/raw"
    if args.demo:
        rf_note = "DEMO synthetic constant — not market data"
    DV.write_report(merged, cal, splits, root / cfg["outputs"]["report"],
                    config_path=args.config, embargo=cfg["splits"]["embargo_days"],
                    rf_note=rf_note, method_notes=DC.INDEX_METHODOLOGY_NOTES)
    log.info("Wrote %s (%d rows) + validation report.", out_parquet, len(merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
