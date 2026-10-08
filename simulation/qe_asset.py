"""QE log-price simulation — Andersen (2007) §3.3 (QE-M) + drift extension.

Derivation (Andersen's, reproduced here so the code is auditable).
Integrate the variance SDE over [t, t+dt]:
    V(t+dt) - V(t) = kappa*theta*dt - kappa*Ī + eps*∫√V dW^V
=>  ∫√V dW^V = (V(t+dt) - V(t) - kappa*theta*dt + kappa*Ī)/eps,
with Ī ≈ dt*(g1*V(t) + g2*V(t+dt)).

[ANDERSEN] Log-price increment conditional on the variance path:
    ln S(t+dt) - ln S(t)
      = drift*dt - Ī/2 + rho*∫√V dW^V + sqrt(1-rho^2)*sqrt(Ī)*Z,  Z ~ N(0,1),
    Correlation is embedded through the VARIANCE INCREMENT (rho/eps term),
    not through correlated normals — Z is independent of everything.
    Collecting terms with Ī = dt*(g1*V + g2*Vn):
    [ANDERSEN] K0 = drift*dt - rho*kappa*theta*dt/eps
               K1 = g1*dt*(kappa*rho/eps - 1/2) - rho/eps
               K2 = g2*dt*(kappa*rho/eps - 1/2) + rho/eps
               K3 = g1*dt*(1-rho^2)
               K4 = g2*dt*(1-rho^2)
    ln S(t+dt) = ln S(t) + K0 + K1*V + K2*Vn + sqrt(K3*V + K4*Vn)*Z.
    (Andersen states this for the forward with drift = 0; the drift*dt term
    is the generalisation.)

[ANDERSEN-concept / EXTENSION-implementation] Martingale correction M.
    Conditional on the variance path, E[exp(sqrt(..)*Z)] = exp((K3*V+K4*Vn)/2),
    so E[S(t+dt)|V(t)]/S(t) = exp(K0 + (K1+K3/2)*V) * A(V), where
    A(V) = E[exp(K2p*Vn)|V], K2p = K2 + K4/2, via the EXACT MGF of the QE law:
      quadratic branch (Vn = a(b+Zv)^2, non-central chi-square MGF):
          A = exp(a*b^2*K2p/(1-2a*K2p))/sqrt(1-2a*K2p), needs 2a*K2p < 1
      exponential branch (atom p + Exp(beta) tail):
          A = p + (1-p)*beta/(beta-K2p), needs K2p < beta.
    The correction M(V) = target*dt - (K0 + (K1+K3/2)*V) - log A makes
    E[S(t+dt)|V(t)] = S(t)*exp(target*dt) EXACTLY under the discretisation.
    [EXTENSION] target = mu*dt in physical mode (trading simulations),
    target = r*dt in risk-neutral mode (pricing / martingale validation).
    Andersen's paper is risk-neutral (target = r); the physical target and
    the closed-form MGF implementation are project additions.
    [EXTENSION] If an MGF existence condition fails, the correction is skipped
    for those paths (flagged) rather than producing NaNs.
"""

from __future__ import annotations

import numpy as np

from simulation.qe_variance import (
    PSI_C_DEFAULT,
    exponential_params,
    quadratic_params,
)


def k_coefficients(
    kappa: float,
    theta: float,
    eps: float,
    rho: float,
    dt: float,
    drift: float,
    gamma1: float = 0.5,
    gamma2: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """[ANDERSEN] K0..K4 constants for one log-price step."""
    K0 = drift * dt - rho * kappa * theta * dt / eps
    c = dt * (kappa * rho / eps - 0.5)
    K1 = gamma1 * c - rho / eps
    K2 = gamma2 * c + rho / eps
    K3 = gamma1 * dt * (1 - rho**2)
    K4 = gamma2 * dt * (1 - rho**2)
    return K0, K1, K2, K3, K4


def mgf_correction(
    V: np.ndarray,
    m: np.ndarray,
    psi: np.ndarray,
    K0: float,
    K1: float,
    K2: float,
    K3: float,
    K4: float,
    target_dt: float,
    psi_c: float = PSI_C_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """[ANDERSEN-concept/EXTENSION-implementation] Per-path M; returns (M, applied)."""
    V = np.asarray(V, dtype=float)
    K2p = K2 + 0.5 * K4
    M = np.full_like(V, np.nan)
    applied = np.zeros_like(V, dtype=bool)
    q = (psi <= psi_c) & (m > 1e-12)
    if np.any(q):
        inv = 1.0 / psi[q]
        b2 = 2 * inv - 1 + np.sqrt(np.maximum(2 * inv * (2 * inv - 1), 0.0))
        a = m[q] / (1 + b2)
        ok = (2 * a * K2p) < 1.0
        idx = np.where(q)[0][ok]
        if len(idx):
            logA = a[ok] * b2[ok] * K2p / (1 - 2 * a[ok] * K2p) - 0.5 * np.log(
                np.maximum(1 - 2 * a[ok] * K2p, 1e-300)
            )
            M[idx] = target_dt - (K0 + (K1 + 0.5 * K3) * V[idx]) - logA
            applied[idx] = True
    e = ~q & (m > 1e-12)
    if np.any(e):
        p = (psi[e] - 1) / (psi[e] + 1)
        beta = (1 - p) / m[e]
        ok = K2p < beta
        idx = np.where(e)[0][ok]
        if len(idx):
            A = p[ok] + (1 - p[ok]) * beta[ok] / (beta[ok] - K2p)
            M[idx] = target_dt - (K0 + (K1 + 0.5 * K3) * V[idx]) - np.log(np.maximum(A, 1e-300))
            applied[idx] = True
    # degenerate-atom rows (m ~ 0): Vn = 0 a.s., so A = 1 exactly.
    d = ~(q | e)
    if np.any(d):
        M[d] = target_dt - (K0 + (K1 + 0.5 * K3) * V[d])
        applied[d] = True
    return M, applied


def qe_step_price(
    logS: np.ndarray,
    V: np.ndarray,
    V_next: np.ndarray,
    m: np.ndarray,
    psi: np.ndarray,
    K0: float,
    K1: float,
    K2: float,
    K3: float,
    K4: float,
    M: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """[ANDERSEN] One log-price step given the variance path + correction M.

    Paths where M is NaN (MGF condition failed) fall back to the uncorrected
    step — [EXTENSION] counted in diagnostics, never silently zeroed.
    """
    Z = rng.standard_normal(logS.shape)
    disc_var = K3 * V + K4 * V_next
    step = K0 + K1 * V + K2 * V_next + np.sqrt(np.maximum(disc_var, 0.0)) * Z
    Mc = np.where(np.isfinite(M), M, 0.0)
    return logS + step + Mc
