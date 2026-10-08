"""QE Monte Carlo fair-value pricer (Europeans) + COS cross-check.

Fair value = discounted risk-neutral expectation via the Stage-4 QE simulator
(mode='risk_neutral', martingale correction ON). NIFTY index options are
European cash-settled, so no American adjustment (assumption logged; re-verify
on real NSE contract specs before any capital use).
"""

from __future__ import annotations

import numpy as np

from simulation.heston_simulator import HestonParams, HestonSimulator


def qe_fair_value(S0: float, K: float, T: float, r: float, rn: HestonParams,
                  cp: str = "CE", N: int = 50000, n_steps: int = 26,
                  seed: int = 0) -> dict:
    sim = HestonSimulator(rn, mode="risk_neutral", r=r)
    res = sim.simulate_terminal(S0, T, max(n_steps, 1), N, seed=seed)
    disc = np.exp(-r * T)
    if cp == "CE":
        pay = np.maximum(res.S_T - K, 0.0)
    else:
        pay = np.maximum(K - res.S_T, 0.0)
    price = float(disc * pay.mean())
    se = float(disc * pay.std(ddof=1) / np.sqrt(N))
    return {"price": price, "se": se, "N": N, "n_steps": max(n_steps, 1),
            "frac_exp": res.frac_exponential}


def cos_fair_value(S0: float, K: float, T: float, r: float,
                   rn: HestonParams, cp: str = "CE") -> float:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from calibration.implied import heston_call_price

    c = heston_call_price(S0, K, T, r, rn.kappa, rn.theta, rn.epsilon, rn.rho, rn.V0)
    return c if cp == "CE" else c - S0 + K * np.exp(-r * T)
