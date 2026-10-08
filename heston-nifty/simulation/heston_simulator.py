"""Path simulator orchestrating qe_variance + qe_asset.

[EXTENSION] This orchestration layer (class API, RNG handling, diagnostics
book-keeping) is project code. The mathematics inside each step is Andersen's
(see qe_variance / qe_asset docstrings for the line-by-line attribution).

Conventions:
  mode = 'risk_neutral' : drift = r,   martingale target = r*dt  (Andersen's
                         original setting; used for pricing validation).
  mode = 'physical'     : drift = mu,  martingale target = mu*dt (trading
                         extension: keeps simulated forwards centred on the
                         physical drift so Stage-5 signals measure
                         distribution SHAPE, not a drift artefact).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation import qe_asset as QA
from simulation import qe_variance as QV


@dataclass
class HestonParams:
    kappa: float
    theta: float
    epsilon: float
    rho: float
    V0: float

    def feller(self) -> float:
        return 2 * self.kappa * self.theta / max(self.epsilon**2, 1e-12)


@dataclass
class SimulationResult:
    S_T: np.ndarray        # terminal prices, shape (N,)
    V_T: np.ndarray        # terminal variances, shape (N,)
    logS_path: np.ndarray | None  # (n_steps+1, N) if keep_paths
    V_path: np.ndarray | None
    psi_mean: float        # mean psi over all steps (branch diagnostic)
    frac_exponential: float  # fraction of steps in exponential branch
    frac_correction_skipped: float  # fraction of price steps without M
    n_steps: int
    seed: int


class HestonSimulator:
    def __init__(
        self,
        params: HestonParams,
        mode: str = "physical",
        mu: float = 0.0,
        r: float = 0.065,
        psi_c: float = QV.PSI_C_DEFAULT,
        gamma1: float = 0.5,
        gamma2: float = 0.5,
    ):
        assert mode in ("physical", "risk_neutral"), "mode must be physical|risk_neutral"
        self.p = params
        self.mode = mode
        self.mu = float(mu)
        self.r = float(r)
        self.psi_c = float(psi_c)
        self.g1, self.g2 = float(gamma1), float(gamma2)

    @property
    def drift(self) -> float:
        return self.mu if self.mode == "physical" else self.r

    def simulate_terminal(
        self,
        S0: float,
        T: float,
        n_steps: int,
        N: int,
        seed: int = 0,
        V0: float | None = None,
    ) -> SimulationResult:
        """Simulate N terminal values over horizon T with n_steps sub-steps."""
        rng = np.random.default_rng(seed)
        p, dt = self.p, T / n_steps
        V = np.full(N, p.V0 if V0 is None else V0)
        logS = np.full(N, np.log(S0))
        Ks = QA.k_coefficients(p.kappa, p.theta, p.epsilon, p.rho, dt,
                               self.drift, self.g1, self.g2)
        K0, K1, K2, K3, K4 = Ks
        psi_acc, exp_count, skip_count, total = 0.0, 0, 0, 0
        for _ in range(n_steps):
            m = QV.cond_mean(V, p.kappa, p.theta, dt)
            s2 = QV.cond_var(V, p.kappa, p.theta, p.epsilon, dt)
            psi = QV.psi_stat(m, s2)
            Vn, _, branch = QV.qe_step_variance(
                V, p.kappa, p.theta, p.epsilon, dt, self.psi_c, rng)
            M, applied = QA.mgf_correction(V, m, psi, K0, K1, K2, K3, K4,
                                           self.drift * dt, self.psi_c)
            logS = QA.qe_step_price(logS, V, Vn, m, psi, K0, K1, K2, K3, K4, M, rng)
            psi_acc += float(np.mean(psi[np.isfinite(psi)])) if np.any(np.isfinite(psi)) else 0.0
            exp_count += int((branch == 1).sum())
            skip_count += int((~applied).sum())
            total += N
            V = Vn
        return SimulationResult(
            S_T=np.exp(logS), V_T=V, logS_path=None, V_path=None,
            psi_mean=psi_acc / n_steps, frac_exponential=exp_count / total,
            frac_correction_skipped=skip_count / total, n_steps=n_steps, seed=seed)

    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int,
        N: int,
        seed: int = 0,
        V0: float | None = None,
    ) -> SimulationResult:
        """Same as simulate_terminal but retains full paths (memory: steps×N)."""
        rng = np.random.default_rng(seed)
        p, dt = self.p, T / n_steps
        V = np.full(N, p.V0 if V0 is None else V0)
        logS = np.full(N, np.log(S0))
        logS_path = np.empty((n_steps + 1, N))
        V_path = np.empty((n_steps + 1, N))
        logS_path[0], V_path[0] = logS, V
        Ks = QA.k_coefficients(p.kappa, p.theta, p.epsilon, p.rho, dt,
                               self.drift, self.g1, self.g2)
        K0, K1, K2, K3, K4 = Ks
        psi_acc, exp_count, skip_count = 0.0, 0, 0
        for i in range(n_steps):
            m = QV.cond_mean(V, p.kappa, p.theta, dt)
            s2 = QV.cond_var(V, p.kappa, p.theta, p.epsilon, dt)
            psi = QV.psi_stat(m, s2)
            Vn, _, branch = QV.qe_step_variance(
                V, p.kappa, p.theta, p.epsilon, dt, self.psi_c, rng)
            M, applied = QA.mgf_correction(V, m, psi, K0, K1, K2, K3, K4,
                                           self.drift * dt, self.psi_c)
            logS = QA.qe_step_price(logS, V, Vn, m, psi, K0, K1, K2, K3, K4, M, rng)
            logS_path[i + 1], V_path[i + 1] = logS, Vn
            psi_acc += float(np.mean(psi[np.isfinite(psi)])) if np.any(np.isfinite(psi)) else 0.0
            exp_count += int((branch == 1).sum())
            skip_count += int((~applied).sum())
            V = Vn
        return SimulationResult(
            S_T=np.exp(logS), V_T=V, logS_path=logS_path, V_path=V_path,
            psi_mean=psi_acc / n_steps, frac_exponential=exp_count / (n_steps * N),
            frac_correction_skipped=skip_count / (n_steps * N), n_steps=n_steps, seed=seed)
