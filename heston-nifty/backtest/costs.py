"""Frictional cost model (Stage 6).

Costs are deducted as fractions of traded notional, per side:
    cost_frac = (cost_bps_per_side + slippage_bps) / 1e4.
Slippage is modelled as symmetric frictional bps, NOT as price impact, because
V1 has no intraday data to estimate impact (documented limitation; Stage 9 item).
Direction-independent and additive, so the cost-accounting identity
net = signed_gross - cost_in - cost_out holds EXACTLY (test (b) enforces it).
"""

from __future__ import annotations


def side_cost_frac(cost_bps_per_side: float, slippage_bps: float) -> float:
    if cost_bps_per_side < 0 or slippage_bps < 0:
        raise ValueError("cost/slippage bps must be non-negative")
    return (cost_bps_per_side + slippage_bps) / 1e4
