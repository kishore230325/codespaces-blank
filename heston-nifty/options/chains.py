"""Option chains: real-data gate + synthetic generator (Stage O1).

MEASURE SEPARATION (load-bearing): fair value uses RISK-NEUTRAL params
(implied calibration). Physical/historical params govern hedge-path dynamics
only. Mixing them is a category error this module refuses: fairvalue functions
take RN params explicitly.

Real NIFTY chains: unavailable in this repo (data/raw/ absent; loader raises
OptionalDataUnavailable). All detection/hedge results below are SYNTHETIC and
validate machinery, not markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_market_chain(snapshot_csv: str | None) -> dict:
    """Gate: returns {'status': 'UNAVAILABLE', ...} until real chains arrive."""
    if not snapshot_csv:
        return {"status": "UNAVAILABLE",
                "reason": "No NSE chain snapshot. Synthetic validation only."}
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data import data_loader as DL
    try:
        df, prov = DL.load_options_snapshot(snapshot_csv)
        return {"status": "MARKET", "chain": df, "provenance": prov.to_dict()}
    except DL.OptionalDataUnavailable as e:
        return {"status": "UNAVAILABLE", "reason": str(e)}


def synthetic_chain(S0: float, r: float, rn: dict, seed: int = 0,
                    markup: dict | None = None) -> pd.DataFrame:
    """Arbitrage-consistent synthetic chain from KNOWN RN params (COS prices).

    markup: {(T_days, moneyness, cp): rel_markup} injected mispricing, e.g.
    {(30, 1.0, 'CE'): 0.05} lifts 30d ATM call asks by 5% (overpriced market).
    Bid/ask = mid*(1 -/+ spread/2), spread 40bps; volume/OI sketched for filters.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from calibration.implied import heston_call_price

    rng = np.random.default_rng(seed)
    markup = markup or {}
    rows = []
    for T_days in (7, 30, 60):
        T = T_days / 365
        for m in (0.94, 0.97, 1.0, 1.03, 1.06):
            for cp in ("CE", "PE"):
                K = S0 * m
                c = heston_call_price(S0, K, T, r, rn["kappa"], rn["theta"],
                                      rn["epsilon"], rn["rho"], rn["V0"])
                fair = c if cp == "CE" else c - S0 + K * np.exp(-r * T)
                mid = fair * (1 + rng.normal(0, 0.002))  # micro noise, no bias
                mid *= 1 + markup.get((T_days, m, cp), 0.0)
                spr = 0.004
                rows.append({"expiry_T": T, "strike": K, "option_type": cp,
                             "fair": fair, "bid": mid * (1 - spr / 2),
                             "ask": mid * (1 + spr / 2), "ltp": mid,
                             "volume": int(rng.integers(500, 20000)),
                             "open_interest": int(rng.integers(10000, 500000)),
                             "S0": S0, "r": r})
    return pd.DataFrame(rows)
