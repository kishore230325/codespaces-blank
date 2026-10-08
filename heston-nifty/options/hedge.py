"""Discrete delta-hedge loop simulator (synthetic underlying, costed).

Answers one question: if the detector flags an underpriced option and we
delta-hedge it to expiry with the MODEL delta, does residual P&L ≈ injected
edge minus hedging frictions? A 'yes' closes the loop (detect -> hedge ->
capture); a 'no' means detection edge is an artefact of hedge error.

Setup (synthetic): underlying Heston path from TRUE physical params; hedge with
model delta evaluated at (S_t, V_true_t). Real-world V is latent (filtering is a
Stage-O2 item — flagged, not solved here). Cash accrues at r; stock trades pay
proportional cost bps. Option premium paid/received at MARKET (ask/bid), so the
injected mispricing enters P&L honestly.
"""

from __future__ import annotations

import numpy as np

from options.greeks import cos_greeks
from simulation.heston_simulator import HestonParams, HestonSimulator


def hedge_pnl(S_path: np.ndarray, V_path: np.ndarray, K: float, T: float, r: float,
              rn: HestonParams, cp: str, market_price: float, side: int,
              stock_cost_bps: float = 5.0, dt_years: float = 1 / 252) -> dict:
    """side=+1 long option / -1 short. Returns attribution dict."""
    dt = dt_years
    n = len(S_path) - 1
    assert abs(n * dt - T) < 0.01 * T, \
        f"path horizon {n * dt:.4f}y != option T {T:.4f}y (convention mismatch)"
    assert len(V_path) == len(S_path)
    cash = -side * market_price  # pay ask (long) / receive bid (short)
    shares = 0.0
    costs = 0.0
    for i in range(n):
        t_rem = T - i * dt
        # delta at CURRENT variance: synthetic oracle (real world needs a V filter — O2).
        rn_t = HestonParams(kappa=rn.kappa, theta=rn.theta, epsilon=rn.epsilon,
                            rho=rn.rho, V0=max(float(V_path[i]), 1e-6))
        # hedge shares = -side * delta_call: long call -> short stock, short call -> long stock
        tgt = -side * cos_greeks(S_path[i], K, max(t_rem, 1 / 365), r, rn_t, cp)["delta"]
        trade = tgt - shares
        c = abs(trade) * S_path[i] * stock_cost_bps / 1e4
        costs += c
        cash -= trade * S_path[i] + c
        cash *= np.exp(r * dt)
        shares = tgt
    # expiry settlement
    payoff = max(S_path[-1] - K, 0.0) if cp == "CE" else max(K - S_path[-1], 0.0)
    cash += side * payoff + shares * S_path[-1]
    return {"residual": float(cash), "hedge_costs": float(costs),
            "n_steps": n}


def demo_hedge_loop(seed: int = 0) -> dict:
    """One full loop on synthetic truth: fair RN == physical-vol dynamics."""
    from options.chains import synthetic_chain

    S0, r = 22000.0, 0.065
    true = {"kappa": 2.5, "theta": 0.045, "epsilon": 0.55, "rho": -0.65, "V0": 0.04}
    rn = HestonParams(**true)
    # market rich by 4% on 30d ATM call -> SELL flag expected
    ch = synthetic_chain(S0, r, true, seed=seed, markup={(30, 1.0, "CE"): 0.04})
    row = ch[(ch["expiry_T"] == 30 / 365) & (ch["strike"] == S0)].iloc[0]
    # physical DAILY path over the OPTION's life (calendar-day steps: T/30).
    # Trading-day vs calendar-day mixing previously biased this demo by ~130;
    # the assert in hedge_pnl now forbids horizon mismatch outright.
    sim = HestonSimulator(HestonParams(**true), mode="physical", mu=0.08)
    res = sim.simulate_paths(S0, 30 / 365, 30, 1, seed=seed + 1)
    assert res.logS_path is not None and res.V_path is not None
    S_path, V_path = np.exp(res.logS_path[:, 0]), res.V_path[:, 0]
    out = hedge_pnl(S_path, V_path, S0, 30 / 365, r, rn, "CE",
                    float(row["bid"]), side=-1, dt_years=(30 / 365) / 30)
    out.update({"market_bid": float(row["bid"]), "fair": float(row["fair"]),
                "injected_edge": float(row["bid"] - row["fair"])})
    return out


def loop_statistic(n_paths: int = 200, seed: int = 0,
                   stock_cost_bps: float = 1.0) -> dict:
    """Mean-closure test over many RN paths (mu = r: drift-neutral mechanics).

    Theory (continuous hedge, RN measure): E[residual] = edge - E[costs].
    Discrete daily hedging + unhedged vega add mean-zero noise (averages out),
    NOT bias. Assertion target: |mean_resid - (edge - mean_costs)| < 2.5*SE.
    Single paths prove nothing (hedge noise >> typical edge) — this is the
    economically honest version of 'the loop closes'.
    """
    from options.chains import synthetic_chain

    S0, r = 22000.0, 0.065
    true = {"kappa": 2.5, "theta": 0.045, "epsilon": 0.55, "rho": -0.65, "V0": 0.04}
    rn = HestonParams(**true)
    ch = synthetic_chain(S0, r, true, seed=seed, markup={(30, 1.0, "CE"): 0.04})
    row = ch[(ch["expiry_T"] == 30 / 365) & (ch["strike"] == S0)].iloc[0]
    edge = float(row["bid"] - row["fair"])
    sim = HestonSimulator(HestonParams(**true), mode="physical", mu=r)
    resids, costs = [], []
    for i in range(n_paths):
        res = sim.simulate_paths(S0, 30 / 365, 30, 1, seed=seed + 1000 + i)
        assert res.logS_path is not None and res.V_path is not None
        out = hedge_pnl(np.exp(res.logS_path[:, 0]), res.V_path[:, 0],
                        S0, 30 / 365, r, rn, "CE", float(row["bid"]), side=-1,
                        stock_cost_bps=stock_cost_bps, dt_years=(30 / 365) / 30)
        resids.append(out["residual"])
        costs.append(out["hedge_costs"])
    resids = np.array(resids)
    return {"mean_resid": float(resids.mean()), "se": float(resids.std(ddof=1) / np.sqrt(n_paths)),
            "mean_costs": float(np.mean(costs)), "edge": edge, "n": n_paths}
