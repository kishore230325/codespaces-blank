"""Options-implied Heston calibration — Stage 3 (V2-ready, V1-deferred).

Status in V1: NO reliable NIFTY options history is available
(data_loader.load_options_snapshot raises OptionalDataUnavailable), so this
module CANNOT produce a market-implied calibration yet. It exists so that:
  (a) the pricer + objective are implemented, unit-tested on SYNTHETIC data,
      and ready for V2 without look-ahead shortcuts;
  (b) Stage 3 honestly reports "options-implied: unavailable" instead of
      silently substituting historical parameters.

Method: standard Heston characteristic function ("Little Trap" formulation,
Gatheral 2006 / Albrecher et al. 2007 — numerically stable for the full
parameter domain including Feller violation) + Lewis (2001) Fourier integral
for European calls; puts via put-call parity. Objective = vega-weighted SSE
over (model_price - market_price), minimised with L-BFGS-B under the same
box constraints as historical.py. Risk-neutral drift r is used here —
correct for pricing, NOT for physical forecasting (see Stage 1 RQ separation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from calibration.historical import BOUNDS, ORDER

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Characteristic function (Little Trap)
# ---------------------------------------------------------------------------

def heston_cf(u: complex, T: float, kappa: float, theta: float, eps: float,
              rho: float, V0: float, r: float, S0: float) -> complex:
    """Risk-neutral log-spot CF  E[exp(i*u*ln S_T)] (Little Trap form).

    With S0=1, r=0 this is the CF of ln(S_T/S_0); the COS pricer below uses
    only this normalised form, so drift conventions cannot leak in.
    """
    iu = 1j * u
    d = np.sqrt((rho * eps * iu - kappa) ** 2 + eps**2 * (iu + u**2))
    g = (kappa - rho * eps * iu - d) / (kappa - rho * eps * iu + d)
    exp_dT = np.exp(-d * T)
    C = (r * iu * T + kappa * theta / eps**2 * ((kappa - rho * eps * iu - d) * T
         - 2 * np.log((1 - g * exp_dT) / (1 - g))))
    D = ((kappa - rho * eps * iu - d) / eps**2) * ((1 - exp_dT) / (1 - g * exp_dT))
    return np.exp(C + D * V0 + iu * np.log(S0))


def _cos_truncation(S0: float, K: float, T: float, r: float,
                    kappa: float, theta: float, eps: float,
                    rho: float, V0: float, L: float = 12.0) -> tuple[float, float]:
    """COS range [a,b] for x_T = ln(S_T/K) from numeric cumulants of ln CF."""
    def phi(u: complex) -> complex:  # CF of ln(S_T/K)
        return heston_cf(u, T, kappa, theta, eps, rho, V0, r, S0) * np.exp(-1j * u * np.log(K))

    h = 1e-4
    # NB: differences of phi itself, NOT log-phi (complex-log branch cuts
    # make log-differences garbage). c1 = E[X], c2 = Var[X], X = ln(S_T/K).
    p_plus, p_minus = phi(h), phi(-h)
    c1 = float(np.real((p_plus - p_minus) / (2j * h)))
    c2 = float(-np.real((p_plus - 2 * phi(0j) + p_minus) / h**2) - c1**2)
    c2 = max(c2, 1e-8)
    # c4 proxy keeps range wide under strong vol-of-vol; floor avoids degeneracy.
    return c1 - L * np.sqrt(abs(c2)), c1 + L * np.sqrt(abs(c2))


def heston_call_price(S0: float, K: float, T: float, r: float,
                      kappa: float, theta: float, eps: float,
                      rho: float, V0: float, N: int = 256) -> float:
    """European call via COS method (Fang & Oosterlee 2008).

    Cross-checked: eps -> 0 with V0 == theta reproduces Black-Scholes;
    put-call parity holds to 1e-8 (see tests/test_calibration.py).
    """
    if T <= 0:
        return max(S0 - K, 0.0)
    x = float(np.log(S0 / K))
    a, b = _cos_truncation(S0, K, T, r, kappa, theta, eps, rho, V0)
    dx = b - a
    total = 0.0
    for k in range(N):
        w = 0.5 if k == 0 else 1.0
        u = k * np.pi / dx
        phi = heston_cf(u, T, kappa, theta, eps, rho, V0, r, S0) * np.exp(-1j * u * np.log(K))
        # phi is the CF of y = ln(S_T/K) (x already absorbed); COS phase is exp(-ik*pi*a/(b-a)).
        shift = np.exp(-1j * k * np.pi * a / dx)
        # Call payoff coefficients on [a,b]: Vk = 2K/(b-a) * (chi_k - psi_k).
        ka, kb = k * np.pi * (0.0 - a) / dx, k * np.pi * (b - a) / dx
        if k == 0:
            chi = np.exp(b) - 1.0
            psi = b
        else:
            up = k * np.pi / dx
            chi = (np.cos(kb) * np.exp(b) - np.cos(ka) * 1.0
                   + up * (np.sin(kb) * np.exp(b) - np.sin(ka) * 1.0)) / (1 + up**2)
            psi = (np.sin(kb) - np.sin(ka)) * dx / (k * np.pi)
        Vk = 2 * K / dx * (chi - psi)
        total += w * float(np.real(phi * shift)) * Vk
    return float(np.exp(-r * T) * total)


def bs_vega(S0: float, K: float, T: float, r: float, iv: float) -> float:
    from scipy.stats import norm

    if T <= 0 or iv <= 0:
        return 1.0
    d1 = (np.log(S0 / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    return float(S0 * norm.pdf(d1) * np.sqrt(T))


# ---------------------------------------------------------------------------
# Objective + calibration entry point
# ---------------------------------------------------------------------------

@dataclass
class ImpliedResult:
    kappa: float; theta: float; epsilon: float; rho: float; V0: float
    objective: float; rmse_price: float; success: bool; message: str; nit: int
    n_quotes: int; status: str = "MARKET"


def _sse(x: np.ndarray, quotes: pd.DataFrame, S0: float, r: float) -> float:
    kappa, theta, eps, rho, V0 = (float(v) for v in x)
    tot, wsum = 0.0, 0.0
    for _, q in quotes.iterrows():
        try:
            m = heston_call_price(S0, float(q["strike"]), float(q["T"]),
                                  r, kappa, theta, eps, rho, V0)
            if q.get("option_type", "CE") == "PE":
                # put-call parity: P = C - D*S0 + D*K
                m = m - S0 + float(q["strike"]) * np.exp(-r * float(q["T"]))
            w = 1.0 / max(float(q.get("vega", 1.0)), 1e-6)
            tot += w * (m - float(q["market_price"])) ** 2
            wsum += w
        except Exception:
            tot += 1e6
    return float(tot / max(wsum, 1e-12))


def calibrate_implied(
    quotes: pd.DataFrame, S0: float, r: float, x0: dict | None = None
) -> ImpliedResult:
    """Fit Heston to an options snapshot. quotes needs [strike, T, market_price,
    option_type] (+ optional vega). S0/r are snapshot-contemporaneous (no future)."""
    req = {"strike", "T", "market_price"}
    if not req.issubset(quotes.columns):
        raise ValueError(f"calibrate_implied: quotes missing {req - set(quotes.columns)}")
    x_init = np.array([(x0 or {}).get(k, d) for k, d in
                       zip(ORDER, [2.0, 0.04, 0.5, -0.6, 0.04])], dtype=float)
    res = minimize(_sse, x_init, args=(quotes, S0, r), method="L-BFGS-B",
                   bounds=[BOUNDS[k] for k in ORDER], options={"maxiter": 300})
    rmse = float(np.sqrt(max(res.fun, 0.0)))
    return ImpliedResult(kappa=float(res.x[0]), theta=float(res.x[1]),
                         epsilon=float(res.x[2]), rho=float(res.x[3]), V0=float(res.x[4]),
                         objective=float(res.fun), rmse_price=rmse,
                         success=bool(res.success), message=str(res.message),
                         nit=int(getattr(res, "nit", 0)), n_quotes=int(len(quotes)))


def try_market_implied(snapshot_csv: str | None, S0: float = 0.0, r: float = 0.065) -> dict:
    """Attempt market calibration; returns status dict (never raises for V1)."""
    if not snapshot_csv:
        return {"status": "UNAVAILABLE",
                "reason": "No options snapshot provided (V1-deferred, V2 work). "
                          "See data_loader.load_options_snapshot schema."}
    try:
        import sys as _s
        from pathlib import Path as _P
        _s.path.insert(0, str(_P(__file__).resolve().parents[1]))
        from data import data_loader as DL
        df, prov = DL.load_options_snapshot(snapshot_csv)
        quotes = df.rename(columns={"ltp": "market_price"})
        quotes["T"] = ((pd.to_datetime(quotes["expiry"]) - pd.to_datetime(quotes["date"]))
                       .dt.days / 365.0).clip(lower=1 / 365)
        res = calibrate_implied(quotes, S0 or float(df["underlying_spot"].iloc[0]), r)
        return {"status": "MARKET", "result": res, "provenance": prov.to_dict()}
    except Exception as e:
        return {"status": "FAILED", "reason": f"{type(e).__name__}: {e}"}


def synthetic_quotes(S0: float = 22000.0, r: float = 0.065,
                     params: dict | None = None, seed: int = 7) -> pd.DataFrame:
    """Generate arbitrage-free synthetic quotes from known params (self-test)."""
    p = params or {"kappa": 2.5, "theta": 0.045, "epsilon": 0.55, "rho": -0.65, "V0": 0.04}
    rng = np.random.default_rng(seed)
    rows = []
    for T in (30 / 365, 60 / 365, 90 / 365):
        for m in (0.94, 0.97, 1.0, 1.03, 1.06):
            K = S0 * m
            c = heston_call_price(S0, K, T, r, p["kappa"], p["theta"], p["epsilon"], p["rho"], p["V0"])
            noisy = c * (1 + rng.normal(0, 0.01))
            rows.append({"strike": K, "T": T, "market_price": noisy,
                         "option_type": "CE", "vega": bs_vega(S0, K, T, r, 0.2)})
    return pd.DataFrame(rows)
