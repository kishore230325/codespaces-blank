"""Validation + visual diagnostics for Stage 3 (no trading content)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibration.historical import DT, cir_cond_mean, cir_cond_var, variance_proxy


def moment_validation_table(df: pd.DataFrame, res) -> pd.DataFrame:
    """One-step-ahead CIR moment check on the calibration window's own proxy."""
    prox = variance_proxy(df["log_ret"]).dropna()
    V = prox["v_smooth"].values[:-1]
    realised = prox["v_smooth"].values[1:]
    m = cir_cond_mean(V, res.kappa, res.theta)
    s2 = cir_cond_var(V, res.kappa, res.theta, res.epsilon)
    std_resid = (realised - m) / np.sqrt(np.maximum(s2, 1e-12))
    return pd.DataFrame([{
        "mean_std_resid": float(np.mean(std_resid)),
        "std_std_resid": float(np.std(std_resid)),
        "mae_mean": float(np.mean(np.abs(realised - m))),
        "coverage_2sigma": float(np.mean(np.abs(std_resid) < 2.0)),
        "n": int(len(V)),
    }])


def plot_variance_fit(df: pd.DataFrame, res, out: Path) -> Path:
    prox = variance_proxy(df["log_ret"]).dropna()
    V = prox["v_smooth"]
    m = cir_cond_mean(V.shift(1).bfill().values, res.kappa, res.theta)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(V.index, np.sqrt(V) * 100, lw=1, label="proxy vol (21d smooth, %)")
    ax.plot(V.index, np.sqrt(np.maximum(m, 0)) * 100, lw=1, ls="--",
            label="CIR 1-step cond-mean vol (%)")
    ax.set_title(f"Variance-proxy fit  {res.window_start}..{res.window_end}  "
                 f"(k={res.kappa:.2f} th={res.theta:.4f} e={res.epsilon:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_rolling(roll: pd.DataFrame, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for col, title in [
        ("kappa", "kappa (mean-reversion speed; half-life ln2/k in days = %.0f...)"),
        ("theta", "theta (long-run variance; sqrt = long-run vol)"),
        ("epsilon", "epsilon (vol-of-vol)"),
        ("rho", "rho (spot-vol correlation)"),
        ("V0", "V0 (current variance; sqrt = spot vol)"),
        ("feller_stat", "Feller statistic 2*k*th/e^2 (>=1 satisfies)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 2.6))
        ax.plot(roll.index, roll[col], marker=".", ms=3, lw=1)
        if col == "feller_stat":
            ax.axhline(1.0, color="r", ls="--", lw=1, label="Feller boundary")
            ax.legend(fontsize=8)
        ax.set_title(title)
        fig.autofmt_xdate()
        fig.tight_layout()
        p = outdir / f"rolling_{col}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    # convergence / error panel
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    axes[0].plot(roll.index, roll["objective"], marker=".", ms=3)
    axes[0].set_title("GMM objective (lower = tighter moment match; NOT PnL)")
    axes[1].plot(roll.index, roll["calib_error_rmse"], marker=".", ms=3, color="g")
    axes[1].set_title("Calibration RMSE over standardised moment errors")
    fig.autofmt_xdate()
    fig.tight_layout()
    p = outdir / "rolling_objective.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths
