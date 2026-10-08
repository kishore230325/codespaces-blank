"""Historical (physical-measure) Heston calibration — Stage 3.

Methodological foundation (Andersen):
  dS/S = mu*dt + sqrt(V)*dW_S
  dV   = kappa*(theta - V)*dt + epsilon*sqrt(V)*dW_V,  corr = rho
Exact CIR conditional moments used for moment-matching and validation:
  m  = E[V_{t+dt}|V_t] = theta + (V_t - theta)*exp(-kappa*dt)
  s2 = Var[V_{t+dt}|V_t]
     = V_t*eps^2*exp(-kappa*dt)/kappa*(1-exp(-kappa*dt))
     + theta*eps^2/(2*kappa)*(1-exp(-kappa*dt))^2
Stationary moments: E[V]=theta, Var[V]=theta*eps^2/(2*kappa),
  Corr[V_{t+dt},V_t] = exp(-kappa*dt) (exact for CIR).

Two-step V1 procedure (NOT trading-optimised):
  Step 1 (MoM start): closed-form estimates from a realised-variance proxy.
  Step 2 (scipy refine): L-BFGS-B minimisation of a GMM moment-matching
         objective. The objective matches MODEL moments to EMPIRICAL moments
         of the variance proxy — never to trading PnL.

Variance proxy (documented limitation): daily annualised proxy
  v_raw,t = (r_t - mean(r))^2 / dt,  r_t = log_ret.
Because v_raw is extremely noisy, persistence (kappa) is estimated from a
21-day moving-average proxy v_smooth; level (theta) from the mean of v_raw.
This introduces smoothing bias (kappa understated); reported as limitation.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

log = logging.getLogger(__name__)

DT = 1 / 252  # one NSE trading day in years
SMOOTH = 21   # proxy smoothing window (trading days)

# Hard box constraints for L-BFGS-B: (kappa, theta, epsilon, rho, V0).
# theta/V0 bounds = 5%..80% vol in variance units. Feller NOT hard-enforced
# (Andersen's QE is built to handle violation); reported as diagnostic.
BOUNDS = {
    "kappa": (0.10, 20.0),
    "theta": (0.0025, 0.64),
    "epsilon": (0.05, 3.00),
    "rho": (-0.99, 0.20),
    "V0": (0.0025, 0.64),
}
ORDER = ["kappa", "theta", "epsilon", "rho", "V0"]


@dataclass
class CalibrationResult:
    kappa: float
    theta: float
    epsilon: float
    rho: float
    V0: float
    mu_ann: float            # annualised drift of log returns (physical)
    objective: float         # final GMM objective value
    calib_error_rmse: float   # RMSE over the 4 standardised moment errors
    success: bool
    message: str
    nit: int
    x0: tuple                # starting vector (sensitivity audit)
    window_start: str = ""
    window_end: str = ""
    n_obs: int = 0
    feller_stat: float = 0.0  # 2*kappa*theta / epsilon^2 (>=1 satisfies Feller)
    feller_ok: bool = False

    def params(self) -> dict:
        return {k: getattr(self, k) for k in ORDER}

    def as_row(self) -> dict:
        d = asdict(self)
        d["x0"] = str(self.x0)
        return d


# ---------------------------------------------------------------------------
# Exact Andersen/CIR moments
# ---------------------------------------------------------------------------

def cir_cond_mean(V: np.ndarray, kappa: float, theta: float, dt: float = DT) -> np.ndarray:
    return theta + (V - theta) * np.exp(-kappa * dt)


def cir_cond_var(V: np.ndarray, kappa: float, theta: float, eps: float, dt: float = DT) -> np.ndarray:
    e = np.exp(-kappa * dt)
    return V * eps**2 * e / kappa * (1 - e) + theta * eps**2 / (2 * kappa) * (1 - e) ** 2


def cir_stat_var(kappa: float, theta: float, eps: float) -> float:
    return theta * eps**2 / (2 * kappa)


# ---------------------------------------------------------------------------
# Proxy + empirical moments
# ---------------------------------------------------------------------------

def variance_proxy(logret: pd.Series, smooth: int = SMOOTH) -> pd.DataFrame:
    r = logret.dropna()
    mu_dt = float(r.mean())
    v_raw = ((r - mu_dt) ** 2 / DT).rename("v_raw")
    v_smooth = v_raw.rolling(smooth, min_periods=smooth).mean().rename("v_smooth")
    return pd.DataFrame({"logret": r, "v_raw": v_raw, "v_smooth": v_smooth})


def empirical_moments(prox: pd.DataFrame) -> dict:
    v = prox["v_raw"].dropna()
    vs = prox["v_smooth"].dropna()
    r = prox["logret"].reindex(vs.index)
    dV = vs.diff()
    return {
        "mean": float(v.mean()),
        "var": float(v.var()),
        "ac1": float(vs.autocorr(1)),
        "rho_lev": float(r.corr(dV)),
        "mu_ann": float(prox["logret"].mean() / DT),
        "n": int(len(v)),
    }


def mom_start(prox: pd.DataFrame) -> dict:
    """Closed-form MoM starting values (also the V1 fallback if scipy fails)."""
    m = empirical_moments(prox)
    vs = prox["v_smooth"].dropna()
    beta = float(vs.autocorr(1))
    beta = float(np.clip(beta, 0.50, 0.9995))  # stationarity guard
    kappa = float(-np.log(beta) / DT)
    kappa = float(np.clip(kappa, *BOUNDS["kappa"]))
    theta = float(np.clip(m["mean"], *BOUNDS["theta"]))
    resid = (vs - vs.shift(1)).dropna()
    Vbar = float(vs.mean())
    # Euler approx: Var(resid) ~= eps^2 * Vbar * DT  -> eps = std(resid/sqrt(V))/sqrt(DT)
    eps = float(np.nanstd(resid.values / np.sqrt(np.maximum(vs.shift(1).dropna().values, 1e-6))) / np.sqrt(DT))
    eps = float(np.clip(eps if np.isfinite(eps) else 0.5, *BOUNDS["epsilon"]))
    rho = float(np.clip(m["rho_lev"] if np.isfinite(m["rho_lev"]) else -0.5, *BOUNDS["rho"]))
    V0 = float(np.clip(vs.ewm(span=30, min_periods=1).mean().iloc[-1], *BOUNDS["V0"]))
    return {"kappa": kappa, "theta": theta, "epsilon": eps, "rho": rho, "V0": V0,
            "mu_ann": m["mu_ann"]}


# ---------------------------------------------------------------------------
# GMM objective (model-vs-empirical moment matching)
# ---------------------------------------------------------------------------

def gmm_objective(x: np.ndarray, emp: dict, weights: dict | None = None) -> float:
    kappa, theta, eps, rho = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    w = {"mean": 1.0, "var": 1.0, "ac1": 1.0, "rho": 0.5}
    if weights:
        w.update(weights)
    scale = max(emp["mean"], 1e-4)
    e_mean = (theta - emp["mean"]) / scale
    e_var = (cir_stat_var(kappa, theta, eps) - emp["var"]) / max(emp["var"], 1e-6)
    e_ac1 = (np.exp(-kappa * DT) - emp["ac1"]) / 0.1
    e_rho = (rho - emp["rho_lev"]) / 0.5
    return float(w["mean"] * e_mean**2 + w["var"] * e_var**2 + w["ac1"] * e_ac1**2 + w["rho"] * e_rho**2)


def calibrate_window(
    df_window: pd.DataFrame,
    x0: dict | None = None,
    weights: dict | None = None,
) -> CalibrationResult:
    """Calibrate ONE trailing window. Uses log_ret only (no future data)."""
    prox = variance_proxy(df_window["log_ret"])
    prox = prox.dropna()
    if len(prox) < 60:
        raise ValueError(f"calibrate_window: need >=60 obs, got {len(prox)}")
    emp = empirical_moments(prox)
    start = mom_start(prox) if x0 is None else {**mom_start(prox), **x0}
    x_init = np.array([start[k] for k in ORDER[:4]], dtype=float)
    bounds = [BOUNDS[k] for k in ORDER[:4]]
    res = minimize(gmm_objective, x_init, args=(emp, weights), method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": 500})
    if res.success:
        kappa, theta, eps, rho = (float(np.clip(v, *BOUNDS[k]))
                                  for v, k in zip(res.x, ORDER[:4]))
    else:
        log.warning("scipy refinement failed (%s); using MoM start.", res.message)
        kappa, theta, eps, rho = (start[k] for k in ORDER[:4])
    V0 = float(np.clip(prox["v_smooth"].ewm(span=30, min_periods=1).mean().iloc[-1], *BOUNDS["V0"]))
    # Standardised RMSE over the 4 moment errors at the solution.
    errs = np.array([
        (theta - emp["mean"]) / max(emp["mean"], 1e-4),
        (cir_stat_var(kappa, theta, eps) - emp["var"]) / max(emp["var"], 1e-6),
        (np.exp(-kappa * DT) - emp["ac1"]) / 0.1,
        (rho - emp["rho_lev"]) / 0.5,
    ])
    rmse = float(np.sqrt(np.mean(errs**2)))
    fell = float(2 * kappa * theta / max(eps**2, 1e-12))
    return CalibrationResult(
        kappa=kappa, theta=theta, epsilon=eps, rho=rho, V0=V0,
        mu_ann=emp["mu_ann"],
        objective=float(res.fun if res.success else gmm_objective(x_init, emp, weights)),
        calib_error_rmse=rmse, success=bool(res.success),
        message=str(res.message), nit=int(getattr(res, "nit", 0)),
        x0=tuple(round(float(v), 6) for v in x_init),
        window_start=str(prox.index.min().date()), window_end=str(prox.index.max().date()),
        n_obs=int(len(prox)), feller_stat=fell, feller_ok=bool(fell >= 1.0),
    )


def calibrate_rolling(
    df: pd.DataFrame,
    window: int = 504,
    step: int = 21,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Rolling point-in-time calibration. Window ending at t uses rows <= t only."""
    df = df.sort_index()
    out, k = [], 0
    for end in range(window, len(df) + 1, step):
        win = df.iloc[end - window:end]
        res = calibrate_window(win, weights=weights)
        row = res.as_row()
        row["t"] = win.index[-1]
        row["roll_id"] = k
        out.append(row)
        k += 1
    return pd.DataFrame(out).set_index("t").sort_index()


def multistart_sensitivity(
    df_window: pd.DataFrame, n_starts: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Sensitivity to starting values: re-optimise from log-uniform perturbations."""
    rng = np.random.default_rng(seed)
    prox = variance_proxy(df_window["log_ret"]).dropna()
    base = mom_start(prox)
    rows = []
    for i in range(n_starts):
        pert = {k: float(np.clip(base[k] * float(rng.uniform(0.5, 2.0)), *BOUNDS[k]))
                for k in ORDER[:4]}
        pert["V0"] = base["V0"]
        try:
            r = calibrate_window(df_window, x0=pert)
            rows.append({"start_id": i, **r.params(), "objective": r.objective,
                         "success": r.success})
        except Exception as e:  # record, don't hide fragility
            rows.append({"start_id": i, "error": str(e)})
    return pd.DataFrame(rows)
