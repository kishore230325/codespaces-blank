"""Fast unit tests for the QE simulator. Run: pytest tests/test_qe.py -q."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation import qe_asset as QA
from simulation import qe_variance as QV
from simulation.heston_simulator import HestonParams, HestonSimulator

P = HestonParams(kappa=2.5, theta=0.045, epsilon=0.55, rho=-0.65, V0=0.04)


def test_branch_selection_respects_psi_c():
    rng = np.random.default_rng(0)
    # tiny dt from moderate V -> small psi -> quadratic
    Vn, psi, br = QV.qe_step_variance(np.full(5000, 0.04), 2.5, 0.045, 0.55,
                                      1 / 252, rng=rng)
    assert (psi <= 1.5).all() and (br == 0).all() and (Vn >= 0).all()
    # huge vol-of-vol -> stationary psi = eps^2/(2*k*th) >> psi_c -> exponential
    Vn, psi, br = QV.qe_step_variance(np.full(20000, 0.04), 2.5, 0.045, 2.0,
                                      1.0, rng=rng)
    assert (psi > 1.5).all() and (br == 1).all()
    p = (float(psi[0]) - 1) / (float(psi[0]) + 1)
    assert abs((Vn == 0).mean() - p) < 0.02


def test_quadratic_params_match_andersen_formula():
    m, psi = 0.04, 0.5
    a, b = QV.quadratic_params(m, psi)
    inv = 1 / psi
    assert abs(b**2 - (2 * inv - 1 + np.sqrt(2 * inv * (2 * inv - 1)))) < 1e-12
    assert abs(a - m / (1 + b**2)) < 1e-12


def test_k_coefficients_sign_structure():
    K0, K1, K2, K3, K4 = QA.k_coefficients(2.5, 0.045, 0.55, -0.65, 1 / 252, 0.05)
    assert K3 > 0 and K4 > 0  # diffusion loadings positive
    # K2 - K1 = (g2-g1)*c + 2*rho/eps; with g1 == g2 the split is exactly 2*rho/eps
    assert abs((K2 - K1) - 2 * (-0.65) / 0.55) < 1e-12


def test_seed_reproducibility():
    sim = HestonSimulator(P, mode="risk_neutral", r=0.05)
    r1 = sim.simulate_terminal(100.0, 0.25, 10, 5000, seed=99)
    r2 = sim.simulate_terminal(100.0, 0.25, 10, 5000, seed=99)
    assert np.array_equal(r1.S_T, r2.S_T) and np.array_equal(r1.V_T, r2.V_T)


def test_martingale_approx_holds():
    from simulation.validation import check_martingale

    row = check_martingale(P, N=100000)
    assert row["passed"], row


def test_zero_V0_recovers():
    rng = np.random.default_rng(1)
    Vn, _, _ = QV.qe_step_variance(np.zeros(5000), 2.0, 0.04, 0.5, 1 / 252, rng=rng)
    assert (Vn >= 0).all() and Vn.mean() > 0  # mean reverts upward from boundary
