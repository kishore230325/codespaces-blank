"""QE variance simulation — Andersen (2007) §3.1–3.2 (QE-M scheme).

Tags: [ANDERSEN] = formula taken directly from the paper.
      [EXTENSION] = implementation choice added for this project
      (guards, vectorisation, defaults), mathematically neutral.

[ANDERSEN] CIR conditional moments, given V(t), step dt:
    m  = E[V(t+dt)|V(t)] = theta + (V(t)-theta)*exp(-kappa*dt)
    s2 = Var[V(t+dt)|V(t)]
       = V(t)*eps^2*exp(-kappa*dt)/kappa*(1-exp(-kappa*dt))
       + theta*eps^2/(2*kappa)*(1-exp(-kappa*dt))^2
    psi = s2 / m^2   ("normalised variance"; controls branch selection)

[ANDERSEN] QE moment-matching law for V(t+dt):
    psi <= psi_c :  V = a*(b + Z_V)^2,  Z_V ~ N(0,1), with
        b^2 = 2/psi - 1 + sqrt(2/psi*(2/psi - 1)),  a = m/(1+b^2)
    psi > psi_c  :  atom-plus-exponential,
        P(V = 0) = p = (psi-1)/(psi+1);  given V > 0, V ~ Exp(rate beta),
        beta = (1-p)/m = 2/(m*(psi+1)).
    Sampling the tail: draw U ~ Unif[0,1]; V = 0 if U <= p,
        else V = ln((1-p)/(1-U))/beta.
    Both branches match E[V] = m and Var[V] = s2 exactly (weak/moment
    consistency — Andersen's design criterion, NOT pathwise accuracy).

[ANDERSEN] psi_c in [1,2]; Andersen's numerical work uses psi_c = 1.5.

[ANDERSEN] Integrated-variance proxy over [t, t+dt]:
    ∫V ds ≈ dt*(gamma1*V(t) + gamma2*V(t+dt)), gamma1 + gamma2 = 1,
    standard choice gamma1 = gamma2 = 1/2.
"""

from __future__ import annotations

import numpy as np

PSI_C_DEFAULT = 1.5
GAMMA1_DEFAULT = 0.5
GAMMA2_DEFAULT = 0.5
_M_TINY = 1e-12  # [EXTENSION] below this mean, return the atom (avoids 0/0)


def cond_mean(V: np.ndarray | float, kappa: float, theta: float, dt: float):
    """[ANDERSEN] Exact CIR conditional mean m."""
    V = np.asarray(V, dtype=float)
    return theta + (V - theta) * np.exp(-kappa * dt)


def cond_var(V: np.ndarray | float, kappa: float, theta: float, eps: float, dt: float):
    """[ANDERSEN] Exact CIR conditional variance s2."""
    V = np.asarray(V, dtype=float)
    e = np.exp(-kappa * dt)
    return V * eps**2 * e / kappa * (1 - e) + theta * eps**2 / (2 * kappa) * (1 - e) ** 2


def psi_stat(m: np.ndarray | float, s2: np.ndarray | float):
    """[ANDERSEN] psi = s2 / m^2 (inf where m == 0 -> exponential branch)."""
    m = np.asarray(m, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = s2 / np.maximum(m, 0.0) ** 2
    psi = np.where(m <= _M_TINY, np.inf, psi)
    return psi


def quadratic_params(m: float, psi: float) -> tuple[float, float]:
    """[ANDERSEN] (a, b) for the quadratic branch. Requires 0 < psi <= psi_c."""
    inv = 1.0 / psi
    b2 = 2 * inv - 1 + np.sqrt(max(2 * inv * (2 * inv - 1), 0.0))
    b = np.sqrt(max(b2, 0.0))
    a = m / (1 + b2)
    return float(a), float(b)


def exponential_params(m: float, psi: float) -> tuple[float, float]:
    """[ANDERSEN] (p, beta) for the exponential branch. Requires psi > psi_c."""
    p = (psi - 1) / (psi + 1)
    beta = (1 - p) / m  # == 2/(m*(psi+1))
    return float(p), float(beta)


def qe_step_variance(
    V: np.ndarray,
    kappa: float,
    theta: float,
    eps: float,
    dt: float,
    psi_c: float = PSI_C_DEFAULT,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[ANDERSEN] One QE variance step, vectorised over paths.

    Returns (V_next, psi, branch) with branch 0 = quadratic, 1 = exponential.
    V_next >= 0 by construction (square / atom-plus-exponential).
    [EXTENSION] m <= 1e-12 -> V_next = 0 (degenerate atom); float-error
    clipping at 0;psi_c parameterised (default Andersen's 1.5).
    """
    rng = np.random.default_rng() if rng is None else rng
    V = np.asarray(V, dtype=float)
    m = cond_mean(V, kappa, theta, dt)
    s2 = cond_var(V, kappa, theta, eps, dt)
    psi = psi_stat(m, s2)

    out = np.zeros_like(V)
    branch = np.ones_like(V, dtype=np.int8)  # default exponential
    # --- quadratic branch (scalar loop over the few distinct psi is wasteful;
    # elementwise closed forms are cheap, so vectorise with masks) ---
    q = (psi <= psi_c) & (m > _M_TINY)
    if np.any(q):
        inv = 1.0 / psi[q]
        b2 = 2 * inv - 1 + np.sqrt(np.maximum(2 * inv * (2 * inv - 1), 0.0))
        a = m[q] / (1 + b2)
        out[q] = a * (np.sqrt(np.maximum(b2, 0.0)) + rng.standard_normal(int(q.sum()))) ** 2
        branch[q] = 0
    # --- exponential branch ---
    e = (~q) & (m > _M_TINY)
    if np.any(e):
        p = (psi[e] - 1) / (psi[e] + 1)
        beta = (1 - p) / m[e]
        U = rng.random(int(e.sum()))
        tail = U > p
        Vn = np.zeros(int(e.sum()))
        # [EXTENSION] clip (1-p)/(1-U) away from 0 for log-stability
        Vn[tail] = np.log(np.maximum((1 - p[tail]) / (1 - U[tail]), 1e-300)) / beta[tail]
        out[e] = Vn
    # degenerate atom rows stay 0 with branch = exponential
    np.maximum(out, 0.0, out=out)  # [EXTENSION] float-error guard
    return out, psi, branch


def integrated_variance(
    V_t: np.ndarray,
    V_next: np.ndarray,
    dt: float,
    gamma1: float = GAMMA1_DEFAULT,
    gamma2: float = GAMMA2_DEFAULT,
) -> np.ndarray:
    """[ANDERSEN] Trapezoidal integrated-variance proxy dt*(g1*V_t + g2*V_next)."""
    return dt * (gamma1 * np.asarray(V_t, dtype=float) + gamma2 * np.asarray(V_next, dtype=float))
