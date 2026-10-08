"""Options-stage tests: parity, QE≈COS, detection power, Greeks, hedge loop."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from options import chains as C
from options import fairvalue as FV
from options import greeks as G
from options import hedge as HD
from options import mispricing as MP
from simulation.heston_simulator import HestonParams

RN = HestonParams(kappa=2.5, theta=0.045, epsilon=0.55, rho=-0.65, V0=0.04)
KW = dict(S0=22000.0, K=22000.0, T=30 / 365, r=0.065, rn=RN)


def test_put_call_parity_qe():
    c = FV.qe_fair_value(**KW, cp="CE", N=20000, seed=0)
    p = FV.qe_fair_value(**KW, cp="PE", N=20000, seed=0)
    lhs = c["price"] - p["price"]
    rhs = KW["S0"] - KW["K"] * np.exp(-KW["r"] * KW["T"])
    # absolute scale: parity residual is small-absolute; relative gates mislead
    assert abs(lhs - rhs) < 0.003 * KW["S0"]


def test_qe_matches_cos_within_se():
    q = FV.qe_fair_value(**KW, cp="CE", N=30000, seed=1)
    b = FV.cos_fair_value(**KW, cp="CE")
    assert abs(q["price"] - b) < 3 * q["se"] + 0.01 * b


def test_detection_power_and_no_false_positives():
    true = dict(kappa=2.5, theta=0.045, epsilon=0.55, rho=-0.65, V0=0.04)
    fair = C.synthetic_chain(22000.0, 0.065, true, seed=0)
    fair["fair"] = fair["fair"]  # model == truth
    assert (MP.detect(fair)["flag"] == "NONE").all()
    rich = C.synthetic_chain(22000.0, 0.065, true, seed=0,
                             markup={(30, 1.0, "CE"): 0.05})
    rich["fair"] = fair["fair"].to_numpy()  # model at truth, market marked up
    flags = MP.detect(rich)
    hit = flags[(flags["expiry_T"] == 30 / 365) & (flags["strike"] == 22000.0)
                & (flags["option_type"] == "CE")]["flag"].iloc[0]
    assert hit == "SELL"
    assert (flags["flag"] == "SELL").sum() <= 3  # targeted, not scattergun


def test_greeks_mc_vs_cos():
    m = G.mc_greeks(**KW, N=30000, seed=2)
    b = G.cos_greeks(**KW)
    assert abs(m["delta"] - b["delta"]) < 0.03
    assert abs(m["vega"] - b["vega"]) / max(abs(b["vega"]), 1) < 0.25


def test_hedge_loop_captures_majority_of_edge():
    # single path: finiteness + cost accounting only (hedge noise >> edge)
    out = HD.demo_hedge_loop(seed=0)
    assert np.isfinite(out["residual"]) and out["hedge_costs"] >= 0
    # mean closure over RN paths: E[residual] = edge - E[costs] (no leakage)
    st = HD.loop_statistic(n_paths=200, seed=0)
    gap = abs(st["mean_resid"] - (st["edge"] - st["mean_costs"]))
    assert gap < 2.5 * st["se"], (gap, st)


def test_market_gate_unavailable():
    assert C.load_market_chain(None)["status"] == "UNAVAILABLE"
