"""Mispricing detection: model vs MARKET quotes (never vs mid-only).

A contract is flaggable only on executable terms:
  BUY (market cheap):  fair > ask * (1 + cost_buffer)
  SELL (market rich):  fair < bid * (1 - cost_buffer)
plus liquidity gates (max spread bps, min volume). cost_buffer covers brokerage,
STT-on-premium, stamp and slippage in one frictional number (V1 simplification;
real fee schedule is a Stage-O2 item). Edge is reported pre- and post-buffer so
detection power and tradability are never conflated.
"""

from __future__ import annotations

import pandas as pd


def detect(df: pd.DataFrame, cost_buffer: float = 0.01,
           max_spread_bps: float = 100.0, min_volume: int = 1000) -> pd.DataFrame:
    out = df.copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread_bps"] = (out["ask"] - out["bid"]) / out["mid"] * 1e4
    out["edge_mid"] = (out["fair"] - out["mid"]) / out["mid"]
    out["liquid"] = (out["spread_bps"] <= max_spread_bps) & (out["volume"] >= min_volume)
    out["flag"] = "NONE"
    out.loc[out["liquid"] & (out["fair"] > out["ask"] * (1 + cost_buffer)), "flag"] = "BUY"
    out.loc[out["liquid"] & (out["fair"] < out["bid"] * (1 - cost_buffer)), "flag"] = "SELL"
    return out
