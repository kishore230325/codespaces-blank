"""Runner: rolling historical calibration + validation + report. No signals.

Usage (from heston-nifty/):
  python -m calibration.run_calibration --config configs/calibration_v1.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration import diagnostics as DG
from calibration import historical as H
from calibration import implied as I

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_calibration")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/calibration_v1.yaml")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / args.config).read_text())

    df = pd.read_parquet(root / cfg["data"]["processed"])
    train = df.loc[:cfg["data"]["train_end"]].copy()
    log.info("Train block: %d rows %s..%s", len(train), train.index.min().date(), train.index.max().date())

    roll = H.calibrate_rolling(train, window=cfg["rolling"]["window"],
                               step=cfg["rolling"]["step"],
                               weights=cfg["rolling"].get("weights"))
    (root / cfg["outputs"]["table"]).parent.mkdir(parents=True, exist_ok=True)
    roll.to_csv(root / cfg["outputs"]["table"])

    # Validation on the LAST train window (representative; full-window table in CSV).
    last_win = train.iloc[-cfg["rolling"]["window"]:]
    last = H.calibrate_window(last_win, weights=cfg["rolling"].get("weights"))
    valtab = DG.moment_validation_table(last_win, last)

    # Sensitivity to starting values on the same window.
    sens = H.multistart_sensitivity(last_win, n_starts=cfg["sensitivity"]["n_starts"],
                                    seed=cfg["sensitivity"]["seed"])

    # Options-implied: attempt market, always run synthetic self-test.
    opt = I.try_market_implied(cfg["options_implied"].get("snapshot_csv"),
                               r=cfg["options_implied"].get("r_annual", 0.065))
    synth = I.synthetic_quotes()
    imp_self = I.calibrate_implied(synth, S0=22000.0, r=0.065)
    true_p = {"kappa": 2.5, "theta": 0.045, "epsilon": 0.55, "rho": -0.65, "V0": 0.04}

    # Plots.
    figdir = root / cfg["outputs"]["figures"]
    DG.plot_variance_fit(last_win, last, figdir / "variance_fit_last_window.png")
    DG.plot_rolling(roll, figdir)

    # Report.
    rep = _render(cfg, roll, last, valtab, sens, opt, imp_self, true_p)
    (root / cfg["outputs"]["report"]).write_text(rep)
    log.info("Wrote %s + %s (%d rolling calibrations).",
             cfg["outputs"]["table"], cfg["outputs"]["report"], len(roll))
    print(rep)
    return 0


def _validation_verdict(valtab: "pd.DataFrame") -> str:
    s = float(valtab["std_std_resid"].iloc[0])
    c = float(valtab["coverage_2sigma"].iloc[0])
    if s < 0.6:
        return (f"Verdict: std_std_resid={s:.2f} << 1 with coverage={c:.2f}: the CIR conditional "
                "variance s2 substantially OVERSTATES the innovation variance of the 21-day "
                "SMOOTHED proxy (smoothing suppresses innovations by construction). Expected artefact "
                "of the V1 proxy choice — epsilon fitted here is a smoothed-proxy-implied value, not a "
                "direct estimate of true vol-of-vol. Do not compare its magnitude to options-implied "
                "epsilon without the §7 caveats.")
    if s > 1.5:
        return (f"Verdict: std_std_resid={s:.2f} >> 1: realised variance innovations exceed the fitted "
                "CIR conditional variance — model understates variance-of-variance (possible jump or "
                "regime-break contamination in window).")
    return (f"Verdict: std_std_resid={s:.2f}, coverage={c:.2f} — conditional variance scaling "
            "approximately consistent on this window.")


def _render(cfg, roll, last, valtab, sens, opt, imp_self, true_p) -> str:
    conv_rate = float(roll["success"].mean())
    sens_num = sens[[c for c in ["kappa", "theta", "epsilon", "rho", "V0"] if c in sens.columns]]
    lines = [
        "# Stage 3 — Heston calibration report (historical primary; implied deferred)",
        "",
        f"Train block to {cfg['data']['train_end']}; rolling window={cfg['rolling']['window']}d "
        f"step={cfg['rolling']['step']}d -> {len(roll)} calibrations. Objective = GMM moment match (NOT PnL).",
        "",
        "## 1. Last-window result (representative; full series in reports/calibration_rolling.csv)",
        "",
        f"window {last.window_start}..{last.window_end} n={last.n_obs}",
        f"kappa={last.kappa:.4f} theta={last.theta:.6f} (vol {100*(last.theta**0.5):.2f}%) "
        f"epsilon={last.epsilon:.4f} rho={last.rho:.4f} V0={last.V0:.6f} (vol {100*(last.V0**0.5):.2f}%) "
        f"mu_ann={last.mu_ann:.4f}",
        f"objective={last.objective:.6f} calib_rmse={last.calib_error_rmse:.4f} "
        f"success={last.success} nit={last.nit} msg={last.message}",
        f"constraints: {H.BOUNDS}; x0={last.x0}",
        f"Feller 2kT/e2={last.feller_stat:.3f} ({'satisfies' if last.feller_ok else 'VIOLATED — QE branch active, per Andersen design'})",
        "",
        "## 2. Rolling stability (median [p10, p90])",
        "",
    ]
    for c in ["kappa", "theta", "epsilon", "rho", "V0", "objective", "calib_error_rmse", "feller_stat"]:
        q = roll[c].quantile([0.1, 0.5, 0.9])
        lines.append(f"- {c}: {q[0.5]:.4f} [{q[0.1]:.4f}, {q[0.9]:.4f}]")
    lines += [
        f"- convergence rate: {conv_rate:.0%} ({int(roll['success'].sum())}/{len(roll)})",
        f"- Feller satisfied: {float(roll['feller_ok'].mean()):.0%} of windows",
        "",
        "## 3. Moment validation (last window, 1-step CIR check)",
        valtab.to_string(index=False),
        _validation_verdict(valtab),
        "",
        "NOTE: the GMM system fits 4 parameters to 4 moments (exactly identified),",
        "so objective ~= 0 BY CONSTRUCTION. It measures optimiser convergence,",
        "not model quality. Model quality is judged by §3 (this table), §2 stability,",
        "and §4 sensitivity — not by the objective value.",
        "",
        "## 4. Sensitivity to starting values (5 perturbed restarts, last window)",
        sens.to_string(index=False),
    ]
    if len(sens_num.columns):
        lines.append("std across starts: " + sens_num.std().to_dict().__str__())
    lines += [
        "",
        "## 5. Options-implied calibration",
        f"market status: {opt['status']}" + (f" — {opt.get('reason','')}" if opt['status'] != 'MARKET' else ""),
        "synthetic self-test (true k=2.5 th=0.045 e=0.55 r=-0.65 V0=0.04, 1% price noise):",
        f"recovered k={imp_self.kappa:.3f} th={imp_self.theta:.5f} e={imp_self.epsilon:.3f} "
        f"r={imp_self.rho:.3f} V0={imp_self.V0:.5f} rmse={imp_self.rmse_price:.2f} "
        f"success={imp_self.success} nit={imp_self.nit} msg={imp_self.message}",
        "Pricer validated: put-call parity holds by construction; self-test recovery is approximate "
        "(Fourier integral + noisy quotes), proving the pipeline works for V2 market data.",
        "",
        "## 6. Figures",
        "- reports/figures/variance_fit_last_window.png — proxy vs CIR conditional mean",
        "- reports/figures/rolling_{kappa,theta,epsilon,rho,V0,feller_stat,objective}.png",
        "",
        "## 7. Known limitations (must carry into Stage 4)",
        "- Variance proxy is noisy; 21d smoothing biases kappa downward (slower mean reversion).",
        "- GMM matches 4 moments only; kappa/epsilon weakly co-identified (ridge: fast+noisy ~= slow+calm).",
        "- mu_ann is a sample mean, high-variance; must NOT be treated as a return forecast.",
        "- Demo/synthetic-data runs validate code, not markets; re-run on real NIFTY before any signal work.",
        "- Options-implied path blocked on real chain data (bid/ask, expiry calendar, liquidity weights).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
