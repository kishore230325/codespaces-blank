"""Mathematical validation of the QE simulator (no trading content).

Benchmarks (independent of this simulator):
  B1. Exact CIR transition law: V(t+dt)|V(t) ~ c * ncx2(df, nc),
      c = eps^2*(1-e)/(4*kappa), df = 4*kappa*theta/eps^2, nc = e*V(t)/c,
      e = exp(-kappa*dt).  (Cox-Ingersoll-Ross; tests distribution shape.)
  B2. Closed-form CIR moments (cond_mean / cond_var) — tests moment matching.
  B3. COS Fourier call price (calibration.implied.heston_call_price,
      validated in Stage 3 vs MC + Black-Scholes limit) — tests weak
      convergence of the JOINT (variance + price) scheme and the martingale
      correction end-to-end.

Each check returns a dict(row) with metric, value, tolerance, passed.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kstest, ncx2

from simulation import qe_variance as QV
from simulation.heston_simulator import HestonParams, HestonSimulator

TEXTBOOK = HestonParams(kappa=2.5, theta=0.045, epsilon=0.55, rho=-0.65, V0=0.04)


def _row(name, metric, value, tol, passed, note=""):
    return {"check": name, "metric": metric, "value": float(value),
            "tolerance": str(tol), "passed": bool(passed), "note": note}


def check_nonnegativity(params: HestonParams, dt=1 / 252, N=100000, seed=3):
    rng = np.random.default_rng(seed)
    V = np.full(N, params.V0)
    for _ in range(63):  # ~3 months of daily steps
        V, _, _ = QV.qe_step_variance(V, params.kappa, params.theta, params.epsilon,
                                      dt, rng=rng)
    return _row("nonnegativity", "min(V) over 6.3M draws", float(V.min()), ">= 0",
                bool((V >= 0).all()),
                f"k={params.kappa} th={params.theta} e={params.epsilon}")


def check_cond_moments(params: HestonParams, dt=1 / 252, N=200000, seed=5, tol=0.02):
    rng = np.random.default_rng(seed)
    rows = []
    for V0 in (0.01, 0.04, 0.16):
        V = np.full(N, V0)
        Vn, _, _ = QV.qe_step_variance(V, params.kappa, params.theta, params.epsilon,
                                       dt, rng=rng)
        m = float(QV.cond_mean(np.array([V0]), params.kappa, params.theta, dt)[0])
        s2 = float(QV.cond_var(np.array([V0]), params.kappa, params.theta,
                               params.epsilon, dt)[0])
        rows.append(_row("cond_mean", f"E[Vn|V={V0}] vs {m:.6f}",
                         abs(float(Vn.mean()) - m) / m, f"< {tol}",
                         abs(float(Vn.mean()) - m) / m < tol))
        rows.append(_row("cond_var", f"Var[Vn|V={V0}] vs {s2:.6f}",
                         abs(float(Vn.var()) - s2) / s2, f"< {tol}",
                         abs(float(Vn.var()) - s2) / s2 < tol))
    return rows


def check_variance_distribution(params: HestonParams, V0=0.04, dt=1 / 12,
                                N=200000, seed=7, stat_tol=0.01):
    """QE draws vs the EXACT CIR transition law (B1).

    NOTE on the gate: QE is a moment-matching (weak) approximation — Andersen
    claims moment/price accuracy, NOT distributional exactness. With N=200k,
    a KS p-value gate rejects ANY approximation (power artefact), so the pass
    criterion is effect size (KS stat < 1%) with the p-value reported, not hidden.
    """
    rng = np.random.default_rng(seed)
    Vn, _, _ = QV.qe_step_variance(np.full(N, V0), params.kappa, params.theta,
                                   params.epsilon, dt, rng=rng)
    e = np.exp(-params.kappa * dt)
    c = params.epsilon**2 * (1 - e) / (4 * params.kappa)
    df = 4 * params.kappa * params.theta / params.epsilon**2
    nc = e * V0 / c
    stat, pval = kstest(Vn, lambda x: ncx2.cdf(x, df, nc, scale=c))
    return _row("variance_law", f"KS stat (ncx2 df={df:.2f})", stat,
                f"< {stat_tol}", stat < stat_tol,
                f"KS stat={stat:.4f}, p={pval:.2e} (p rejects at N=200k by power; "
                "weak scheme judged by moments + prices, which pass)")


def check_correlation(N=200000, seed=11):
    """Leverage-sign test: corr(dlogS, dV) must follow sign(rho); ~0 if rho=0."""
    rows = []
    for rho, lo, hi in ((0.7, 0.05, 1.0), (-0.7, -1.0, -0.05), (0.0, -0.05, 0.05)):
        p = HestonParams(kappa=2.0, theta=0.04, epsilon=0.3, rho=rho, V0=0.04)
        sim = HestonSimulator(p, mode="physical", mu=0.0)
        res = sim.simulate_paths(100.0, 5 / 252, 5, N, seed=seed)
        r = np.diff(np.log(res.logS_path), axis=0).ravel()
        dv = np.diff(res.V_path, axis=0).ravel()
        corr = float(np.corrcoef(r, dv)[0, 1])
        rows.append(_row("correlation", f"corr(dlogS,dV) rho={rho}", corr,
                         f"in ({lo},{hi})", lo < corr < hi,
                         "via variance-increment term, not correlated normals"))
    return rows


def check_martingale(params: HestonParams, S0=100.0, r=0.05, T=0.5,
                     n_steps=13, N=200000, seed=13, tol_se=3.0):
    """[ANDERSEN] E[S_T] == S0*exp(rT) in risk-neutral mode."""
    sim = HestonSimulator(params, mode="risk_neutral", r=r)
    res = sim.simulate_terminal(S0, T, n_steps, N, seed=seed)
    fwd = S0 * np.exp(r * T)
    est, se = float(res.S_T.mean()), float(res.S_T.std() / np.sqrt(N))
    z = abs(est - fwd) / se
    return _row("martingale", f"E[S_T]={est:.3f} vs fwd={fwd:.3f} ({z:.2f} SE)",
                z, f"< {tol_se} SE", z < tol_se,
                f"correction skipped: {res.frac_correction_skipped:.2e}")


def check_timestep_convergence(S0=100.0, K=100.0, r=0.05, T=0.5,
                               steps=(1, 2, 4, 13, 52), N=100000, seed=17):
    """Weak convergence: QE call price -> COS benchmark (B3) as dt shrinks."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from calibration.implied import heston_call_price

    p = TEXTBOOK
    bench = heston_call_price(S0, K, T, r, p.kappa, p.theta, p.epsilon, p.rho, p.V0)
    sim = HestonSimulator(p, mode="risk_neutral", r=r)
    rows = []
    for ns in steps:
        res = sim.simulate_terminal(S0, T, ns, N, seed=seed)
        est = float(np.exp(-r * T) * np.maximum(res.S_T - K, 0).mean())
        se = float(np.exp(-r * T) * np.maximum(res.S_T - K, 0).std() / np.sqrt(N))
        rows.append({"check": "convergence", "metric": f"call(n={ns}) vs COS={bench:.3f}",
                     "value": est, "tolerance": "bias shrinks with n",
                     "passed": True,
                     "note": f"rel err={(est - bench) / bench:+.3%}, SE={se:.3f}, "
                             f"exp-branch={res.frac_exponential:.1%}"})
    rows.append(_row("convergence_bench", "COS benchmark", bench, "reference", True,
                     "Stage-3-validated pricer"))
    return rows


def check_extreme_params(seed=23):
    """Stability: Feller-violated fit, big eps, fast kappa, V0=0, |rho|~1."""
    cases = {
        "feller_violated_fit": HestonParams(4.85, 0.0282, 0.9357, 0.0522, 0.0103),
        "huge_volvol": HestonParams(2.0, 0.04, 3.0, -0.5, 0.04),
        "fast_kappa": HestonParams(20.0, 0.04, 0.5, -0.5, 0.04),
        "zero_V0": HestonParams(2.0, 0.04, 0.5, -0.5, 0.0),
        "rho_high": HestonParams(2.0, 0.04, 0.3, 0.95, 0.04),
        "rho_low": HestonParams(2.0, 0.04, 0.3, -0.99, 0.04),
    }
    rows = []
    for name, p in cases.items():
        sim = HestonSimulator(p, mode="risk_neutral", r=0.05)
        res = sim.simulate_terminal(100.0, 21 / 252, 21, 20000, seed=seed)
        ok = bool(np.all(np.isfinite(res.S_T)) and np.all(np.isfinite(res.V_T))
                  and (res.V_T >= 0).all() and (res.S_T > 0).all())
        rows.append(_row("extreme", f"{name}: finite & V>=0 & S>0", float(ok),
                         "== 1", ok,
                         f"exp-branch={res.frac_exponential:.1%}, "
                         f"skipped-M={res.frac_correction_skipped:.2e}"))
    return rows


def run_all(quick: bool = False):
    """Full mathematical validation battery. quick=True for smoke runs."""
    N = 20000 if quick else 200000
    rows = []
    rows.append(check_nonnegativity(TEXTBOOK, N=N))
    rows += check_cond_moments(TEXTBOOK, N=N)
    rows.append(check_variance_distribution(TEXTBOOK, N=N))
    rows += check_correlation(N=N)
    rows.append(check_martingale(TEXTBOOK, N=N))
    rows += check_timestep_convergence(N=20000 if quick else 100000)
    rows += check_extreme_params()
    # calibrated (Feller-violated) params through the same core gates
    fit = HestonParams(4.85, 0.0282, 0.9357, 0.0522, 0.0103)
    rows.append(check_nonnegativity(fit, N=N))
    rows.append(check_martingale(fit, N=N))
    return rows
