"""Model Greeks via QE Monte Carlo (CRN central differences) + COS cross-check.

Delta = dP/dS0 (bump S0 ±0.5%); Vega = dP/dσ with σ = sqrt(V0) (bump V0 ±5%
relative, chain-rule to vol points). Common random numbers: identical seed for
base/up/down paths so finite differences are not swamped by independent MC
noise. COS finite differences provide the independent benchmark.
"""

from __future__ import annotations

import numpy as np

from options.fairvalue import cos_fair_value, qe_fair_value
from simulation.heston_simulator import HestonParams


def _bump(rn: HestonParams, **kw) -> HestonParams:
    d = dict(kappa=rn.kappa, theta=rn.theta, epsilon=rn.epsilon, rho=rn.rho, V0=rn.V0)
    d.update(kw)
    return HestonParams(**d)


def mc_greeks(S0: float, K: float, T: float, r: float, rn: HestonParams,
              cp: str = "CE", N: int = 50000, seed: int = 0) -> dict:
    h_s = 0.005 * S0
    up = qe_fair_value(S0 + h_s, K, T, r, rn, cp, N, seed=seed)["price"]
    dn = qe_fair_value(S0 - h_s, K, T, r, rn, cp, N, seed=seed)["price"]
    delta = (up - dn) / (2 * h_s)
    h_v = 0.05 * rn.V0
    vu = qe_fair_value(S0, K, T, r, _bump(rn, V0=rn.V0 + h_v), cp, N, seed=seed)["price"]
    vd = qe_fair_value(S0, K, T, r, _bump(rn, V0=max(rn.V0 - h_v, 1e-6)), cp, N, seed=seed)["price"]
    sig = float(np.sqrt(rn.V0))
    vega = (vu - vd) / (2 * h_v) * (2 * sig) / 100  # per 1 vol point (/100 = per 1%)
    base = qe_fair_value(S0, K, T, r, rn, cp, N, seed=seed)["price"]
    return {"price": base, "delta": float(delta), "vega": float(vega)}


def cos_greeks(S0: float, K: float, T: float, r: float, rn: HestonParams,
               cp: str = "CE") -> dict:
    h_s = 0.005 * S0
    delta = (cos_fair_value(S0 + h_s, K, T, r, rn, cp)
             - cos_fair_value(S0 - h_s, K, T, r, rn, cp)) / (2 * h_s)
    h_v = 0.05 * rn.V0
    sig = float(np.sqrt(rn.V0))
    vega = (cos_fair_value(S0, K, T, r, _bump(rn, V0=rn.V0 + h_v), cp)
            - cos_fair_value(S0, K, T, r, _bump(rn, V0=max(rn.V0 - h_v, 1e-6)), cp)) \
        / (2 * h_v) * (2 * sig) / 100
    return {"price": cos_fair_value(S0, K, T, r, rn, cp),
            "delta": float(delta), "vega": float(vega)}
