"""CLI: run the QE mathematical validation battery and write the report.

Usage (from heston-nifty/):
  python -m simulation.run_validation --quick        # smoke (~30 s)
  python -m simulation.run_validation                # full (~5 min)
No trading content; no signals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation import validation as V

ATTRIBUTION = """\
| # | Andersen-derived (paper) | Project-added (extension) |
|---|---|---|
| 1 | CIR m, s2, psi formulae | m<=1e-12 atom guard; float clip at 0 |
| 2 | Quadratic law a(b+Z)^2 with b^2, a | vectorised masks; psi_c parameterised (default 1.5) |
| 3 | Exponential law p, beta, inversion | log-arg floor 1e-300 |
| 4 | Trapezoidal integrated variance, gamma1=gamma2=1/2 | gamma parameterised |
| 5 | K0..K4 log-price step with -/+rho/eps terms | drift*dt generalisation (Andersen: forward, drift 0) |
| 6 | Martingale-correction CONCEPT (M) | closed-form M via branch MGFs; physical target mu*dt; skip-with-flag fallback |
| 7 | Correlation via variance increment | corr(Z,Zv)=0 asserted in code; sign test in validation |
| 8 | Weak-consistency philosophy | COS/MC/ncx2 benchmark choices; tolerances; seeds |
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]

    rows = V.run_all(quick=args.quick)
    df = pd.DataFrame(rows)
    out_csv = root / "reports" / "simulation_checks.csv"
    df.to_csv(out_csv, index=False)

    figdir = root / "reports" / "figures"
    _plot_convergence(df, figdir)
    _plot_variance_law(figdir)

    n_pass = int(df["passed"].sum())
    body = [
        "# Stage 4 — QE simulator validation report",
        "",
        f"Mode: {'quick smoke' if args.quick else 'full battery'}; "
        f"{n_pass}/{len(df)} checks passed.",
        "",
        "## Attribution (Andersen vs extension)",
        ATTRIBUTION,
        "## Results",
        df.to_markdown(index=False),
        "",
        "## Figures",
        "- reports/figures/qe_convergence.png — call-price bias vs COS shrinks with n_steps",
        "- reports/figures/qe_variance_law.png — QE draws vs exact CIR (ncx2) transition density",
        "",
        "## Verdict for Stage 5",
        ("PASS: simulator is mathematically sound; Stage 5 may build forecast "
         "distributions on it." if n_pass == len(df) else
         f"CAUTION: {len(df) - n_pass} checks failed — see table; do not proceed "
         "to signals until resolved."),
    ]
    (root / "reports" / "simulation_validation.md").write_text("\n".join(body))
    print("\n".join(body))
    return 0 if n_pass == len(df) else 1


def _plot_convergence(df: pd.DataFrame, figdir: Path):
    sub = df[df["check"] == "convergence"].copy()
    if sub.empty:
        return
    ns = [int(s.split("n=")[1].split(")")[0]) for s in sub["metric"]]
    bench = float(df[df["check"] == "convergence_bench"]["value"].iloc[0])
    est = sub["value"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.semilogx(ns, (est - bench) / bench * 100, "o-")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("n_steps (T=0.5y)")
    ax.set_ylabel("rel bias vs COS (%)")
    ax.set_title("QE weak convergence: European call vs Fourier benchmark")
    fig.tight_layout()
    fig.savefig(figdir / "qe_convergence.png", dpi=120)
    plt.close(fig)


def _plot_variance_law(figdir: Path):
    from scipy.stats import ncx2

    from simulation import qe_variance as QV

    rng = np.random.default_rng(7)
    k, th, e, V0, dt = 2.5, 0.045, 0.55, 0.04, 1 / 12
    Vn, _, br = QV.qe_step_variance(np.full(200000, V0), k, th, e, dt, rng=rng)
    ed = np.exp(-k * dt)
    c = e**2 * (1 - ed) / (4 * k)
    dff, nc = 4 * k * th / e**2, ed * V0 / c
    x = np.linspace(max(Vn.min(), 1e-6), np.quantile(Vn, 0.999), 400)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(Vn, bins=120, density=True, alpha=0.5, label="QE draws")
    ax.plot(x, ncx2.pdf(x, dff, nc, scale=c), "r-", lw=1.5, label="exact CIR (ncx2)")
    ax.set_title(f"QE vs exact transition law (exp-branch={np.mean(br):.1%})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "qe_variance_law.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
