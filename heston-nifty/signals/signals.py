"""Distribution statistics + candidate signals (Stage 5).

All functions are pure transforms of a simulated terminal distribution.
Conventions (frozen, NOT tuned):
  R  = S_T / S_t - 1 (simple return over horizon h).
  c  = cost hurdle (default 0.001 = 2 x 5 bps/side) for Signal D.
  Signal E spread is in annualised decimal units (model vol - VIX/100).

Signal semantics (a priori, symmetric, unoptimised):
  A (probability): P(R > 0). High = bullish.
  B (expected return): E[R].
  C (risk-adjusted): E[R] / sd(R).
  D (cost-aware prob): LONG leg P(R > +c); SHORT leg P(R < -c).
  E (vol spread): model_vol - VIX-implied vol. NOT directional — predicts
      relative volatility richness; evaluated as a vol forecast (metrics.py).
"""

from __future__ import annotations

import numpy as np

QS = (0.05, 0.25, 0.50, 0.75, 0.95)


def dist_stats(S_T: np.ndarray, S_t: float, Ibar: np.ndarray,
               T_years: float, vix_t: float = float("nan"),
               hurdle: float = 0.001) -> dict:
    R = S_T / S_t - 1.0
    N = len(R)
    sd = float(R.std())
    p_pos = float((R > 0).mean())
    p_dn1 = float((R < -0.01).mean())
    p_up1 = float((R > 0.01).mean())
    q = np.quantile(R, QS)
    exp_vol = float(np.sqrt(max(Ibar.mean() / max(T_years, 1e-12), 0.0)))
    return {
        # required distribution statistics
        "exp_ret": float(R.mean()),
        "median_ret": float(q[2]),
        "p_pos": p_pos,
        "p_below_m1": p_dn1,
        "p_above_p1": p_up1,
        "q05": float(q[0]), "q25": float(q[1]), "q75": float(q[3]), "q95": float(q[4]),
        "exp_vol": exp_vol,
        "downside_prob": p_dn1,
        "up_down_ratio": float(p_up1 / max(p_dn1, 1e-6)),
        # MC standard errors (judge signal vs noise)
        "se_eret": float(sd / np.sqrt(N)),
        "se_prob": float(np.sqrt(max(p_pos * (1 - p_pos), 0)) / np.sqrt(N)),
        "sd_ret": sd,
        # candidate signals A–E
        "sigA_prob": p_pos,
        "sigB_eret": float(R.mean()),
        "sigC_sharpe": float(R.mean() / sd) if sd > 0 else float("nan"),
        "sigD_up": float((R > hurdle).mean()),
        "sigD_down": float((R < -hurdle).mean()),
        "sigE_spread": float(exp_vol - vix_t / 100.0) if np.isfinite(vix_t) else float("nan"),
        "vix_t": float(vix_t),
        "N": int(N),
    }


def candidate_direction(sigD_up: float, sigD_down: float,
                        thresh: float = 0.5) -> tuple[str, bool]:
    """A-priori trichotomy (LONG / SHORT / FLAT). Returns (direction, conflict)."""
    long = sigD_up > thresh
    short = sigD_down > thresh
    if long and short:
        return "FLAT", True  # contradictory tails — abstain and flag
    if long:
        return "LONG", False
    if short:
        return "SHORT", False
    return "FLAT", False
