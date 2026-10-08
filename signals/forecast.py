"""Point-in-time forecast distributions (Stage 5).

PIT discipline (hard rules):
  P1. forecast_day receives ONLY df_le_t (rows with date <= t) and asserts
      t equals the slice's last index. V0, mu and S_t are derived INSIDE
      from that slice — the caller cannot inject future information.
  P2. Heston structural params (kappa, theta, epsilon, rho) come from a
      VintageProvider whose calibration window ends <= t (monthly vintages).
  P3. RNG seed is a deterministic function of (t, h) — reproducible, and
      independent of realised future returns by construction.

No thresholds, no positions, no PnL in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulation.heston_simulator import HestonParams, HestonSimulator

DT = 1 / 252


def trailing_state(df_le_t: pd.DataFrame, v0_span: int = 30,
                   mu_window: int = 504) -> dict:
    """Derive forecast state at t from data <= t only.

    V0: EWMA(span) of the annualised variance proxy (r-mean)^2/DT over the
        trailing slice (NaNs dropped; needs >= 30 obs).
    mu: trailing mean log-return / DT (physical drift; Stage-3 caveats apply).
    S_t: close at t. vix_t: India VIX level at t (may be NaN -> E-signal NaN).
    """
    if len(df_le_t) < 30:
        raise ValueError("trailing_state: need >=30 obs")
    r = df_le_t["log_ret"].dropna()
    mu_dt = float(r.tail(mu_window).mean())
    proxy = ((r - mu_dt) ** 2 / DT)
    V0 = float(proxy.ewm(span=v0_span, min_periods=1).mean().iloc[-1])
    V0 = float(np.clip(V0, 0.0025, 0.64))
    return {"S_t": float(df_le_t["close"].iloc[-1]),
            "V0": V0,
            "mu_ann": float(mu_dt / DT),
            "vix_t": float(df_le_t["india_vix_close"].iloc[-1])
            if "india_vix_close" in df_le_t else float("nan")}


@dataclass
class Vintage:
    vintage_date: pd.Timestamp  # calibration window ends HERE (<= any t using it)
    params: HestonParams
    mu_ann: float               # drift vintage (trailing mean at vintage_date)


class VintageProvider:
    """Maps each trade date t -> latest vintage with vintage_date <= t."""

    def __init__(self, vintages: list[Vintage]):
        self.vintages = sorted(vintages, key=lambda v: v.vintage_date)
        if not self.vintages:
            raise ValueError("VintageProvider: empty vintage list")

    def for_date(self, t: pd.Timestamp) -> Vintage:
        ok = [v for v in self.vintages if v.vintage_date <= t]
        if not ok:
            raise LookupError(f"No vintage available for {t} (earliest "
                              f"{self.vintages[0].vintage_date})")
        return ok[-1]


@dataclass
class Forecast:
    t: pd.Timestamp
    h: int
    S_t: float
    S_T: np.ndarray       # simulated terminal prices, shape (N,)
    Ibar: np.ndarray      # per-path integrated variance (variance*years)
    T_years: float
    seed: int
    vintage_date: pd.Timestamp
    psi_mean: float
    frac_exponential: float


def forecast_day(df_le_t: pd.DataFrame, t: pd.Timestamp, h: int,
                 vintage: Vintage, N: int, seed: int) -> Forecast:
    """One-day-ahead (h trading days) predictive distribution as of t's close."""
    assert df_le_t.index.max() == t, "PIT violation: slice must end exactly at t"
    assert vintage.vintage_date <= t, "PIT violation: vintage newer than t"
    st = trailing_state(df_le_t)
    p = HestonParams(kappa=vintage.params.kappa, theta=vintage.params.theta,
                     epsilon=vintage.params.epsilon, rho=vintage.params.rho,
                     V0=st["V0"])
    sim = HestonSimulator(p, mode="physical", mu=st["mu_ann"])
    T = h / 252
    res = sim.simulate_paths(st["S_t"], T, h, N, seed=seed, V0=st["V0"])
    assert res.V_path is not None
    dt = T / h
    Ibar = (((res.V_path[:-1] + res.V_path[1:]) / 2) * dt).sum(axis=0)
    return Forecast(t=t, h=h, S_t=st["S_t"], S_T=res.S_T, Ibar=Ibar,
                    T_years=T, seed=seed, vintage_date=vintage.vintage_date,
                    psi_mean=res.psi_mean, frac_exponential=res.frac_exponential)
