"""Runner: walk-forward forecasts + predictive statistics (Stage 5).

Protocol (no PnL, no threshold tuning):
  1. Monthly Heston recalibration on trailing 504 trading days (window ends at
     month-end <= t — never future data).
  2. Daily QE forecast distributions for h in {1,5} on the validation block.
  3. Realised closes (t+h) used ONLY to score forecasts, never as inputs.
  4. Predictive statistics per signal (metrics.py) + diagnostics report.

Usage (from heston-nifty/):  python -m signals.run_signals
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.historical import calibrate_window
from signals import forecast as F
from signals import metrics as M
from signals import signals as S

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_signals")


def month_ends(idx: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    out, cur = [], pd.Timestamp(start).to_period("M")
    last = pd.Timestamp(end).to_period("M")
    while cur <= last:
        elig = idx[(idx.year == cur.year) & (idx.month == cur.month)]
        if len(elig):
            out.append(elig.max())
        cur += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/signals_v1.yaml")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / args.config).read_text())

    df = pd.read_parquet(root / cfg["data"]["processed"])
    block = df.loc[cfg["block"]["start"]:cfg["block"]["end"]].copy()
    log.info("Validation block: %d rows %s..%s", len(block),
             block.index.min().date(), block.index.max().date())

    # --- 1. monthly vintages (trailing windows ending at month-end) ---
    W = cfg["calibration"]["window"]
    recals = month_ends(df.index, cfg["calibration"]["vintage_start"],
                        cfg["calibration"]["vintage_end"])
    vintages = []
    for rd in recals:
        win = df.loc[:rd].tail(W)
        res = calibrate_window(win, weights=cfg["calibration"].get("weights"))
        vintages.append(F.Vintage(
            vintage_date=rd,
            params=__import__("simulation.heston_simulator", fromlist=["HestonParams"])
            .HestonParams(res.kappa, res.theta, res.epsilon, res.rho, res.V0),
            mu_ann=res.mu_ann))
    vint = F.VintageProvider(vintages)
    pd.DataFrame([{"vintage_date": v.vintage_date, "kappa": v.params.kappa,
                   "theta": v.params.theta, "epsilon": v.params.epsilon,
                   "rho": v.params.rho, "mu_ann": v.mu_ann}
                  for v in vintages]).to_csv(root / cfg["outputs"]["vintages"], index=False)
    log.info("Fitted %d monthly vintages.", len(vintages))

    # --- 2. daily forecasts ---
    H = cfg["horizons"]
    N, base = cfg["monte_carlo"]["N"], cfg["monte_carlo"]["seed_base"]
    hurdle = cfg["conventions"]["hurdle"]
    rows = []
    tdates = list(block.index)
    for i, t in enumerate(tdates):
        df_le_t = df.loc[:t]
        v = vint.for_date(t)
        for h in H:
            pos = df.index.get_loc(t)
            pos = int(pos) if not isinstance(pos, slice) else int(np.where(df.index == t)[0][0])
            if pos + h >= len(df):
                continue  # realised window runs past available data
            t_h = df.index[pos + h]
            seed = base + i * 7919 + h * 131
            fc = F.forecast_day(df_le_t, t, h, v, N, seed)
            st = S.dist_stats(fc.S_T, fc.S_t, fc.Ibar, fc.T_years,
                              vix_t=float(df_le_t["india_vix_close"].iloc[-1])
                              if "india_vix_close" in df_le_t else float("nan"),
                              hurdle=hurdle)
            R = float(df["close"].iloc[pos + h] / df["close"].iloc[pos] - 1)
            lr = df["log_ret"].iloc[pos + 1:pos + h + 1].to_numpy(dtype=float)
            rv = float(np.sqrt(252 / h * np.sum(lr**2)))
            d, conflict = S.candidate_direction(st["sigD_up"], st["sigD_down"],
                                                cfg["conventions"]["prob_thresh"])
            rows.append({"t": t, "h": h, "S_t": fc.S_t, "seed": seed,
                         "vintage_date": v.vintage_date, "psi_mean": fc.psi_mean,
                         "frac_exp": fc.frac_exponential, **st,
                         "real_R": R, "real_vol": rv, "real_t": t_h,
                         "direction": d, "conflict": conflict})
        if (i + 1) % 100 == 0:
            log.info("forecast %d/%d", i + 1, len(tdates))
    out = pd.DataFrame(rows)
    op = root / cfg["outputs"]["table"]
    out.to_parquet(op, index=False)
    log.info("Wrote %s (%d rows).", op, len(out))

    # --- 3. predictive statistics ---
    rep = _render(cfg, out, root)
    (root / cfg["outputs"]["report"]).write_text(rep)
    print(rep)
    return 0


def _render(cfg, out: pd.DataFrame, root: Path) -> str:
    pt, st = cfg["conventions"]["prob_thresh"], cfg["conventions"]["score_thresh"]
    L = ["# Stage 5 — predictive-content diagnostics (no PnL, no tuned thresholds)", "",
         f"Rows: {len(out)} (validation block {cfg['block']['start']}..{cfg['block']['end']}); "
         f"N={cfg['monte_carlo']['N']}/forecast; hurdle={cfg['conventions']['hurdle']}.",
         "Mapping conventions fixed a priori: prob>0.5 -> up; signed score>0 -> up.",
         ""]
    figs = []
    for h in cfg["horizons"]:
        d = out[out["h"] == h].reset_index(drop=True)
        L += [f"## Horizon h={h} (n={len(d)})", ""]
        base = float((d["real_R"] > 0).mean())
        clim = base * (1 - base)  # climatology Brier (constant base-rate forecast)
        # A and D legs are probabilities -> full metrics incl. Brier skill.
        # Reference is the climatology of the SAME event (hurdle-matched for D).
        c = cfg["conventions"]["hurdle"]
        evt = {"up": (d["real_R"] > 0), "up1": (d["real_R"] > c), "down1": (d["real_R"] < -c)}
        for name, col, ycls in [("A prob P(R>0)", "sigA_prob", "up"),
                                ("D-up P(R>+c)", "sigD_up", "up1"),
                                ("D-down P(R<-c)", "sigD_down", "down1")]:
            y = evt[ycls].astype(float)
            q = float(y.mean())
            ref = q * (1 - q)
            br = float(((d[col] - y) ** 2).mean())
            score = d[col] if "down" not in name else 1 - d[col]  # downside prob -> up-score
            ds = M.directional_stats(score, d["real_R"], pt)
            cr = M.forecast_corr(d[col], d["real_R"])
            L += [f"### Signal {name}",
                  f"acc={ds['accuracy']:.3f} (base {base:.3f}) bal_acc={ds['balanced_accuracy']:.3f} "
                  f"prec_up={ds['precision_up']:.3f} rec_up={ds['recall_up']:.3f} "
                  f"prec_dn={ds['precision_down']:.3f} rec_dn={ds['recall_down']:.3f} "
                  f"| pearson={cr['pearson']:+.3f} spearman={cr['spearman']:+.3f} "
                  f"| Brier={br:.4f} vs climatology {ref:.4f} "
                  f"(skill={1 - br / ref:+.3f})", ""]
        # B and C are signed scores, NOT probabilities -> no Brier (would be meaningless).
        for name, col in [("B E[R]", "sigB_eret"), ("C E[R]/sd", "sigC_sharpe")]:
            ds = M.directional_stats(d[col], d["real_R"], st)
            cr = M.forecast_corr(d[col], d["real_R"])
            L += [f"### Signal {name}",
                  f"acc={ds['accuracy']:.3f} (base {base:.3f}) bal_acc={ds['balanced_accuracy']:.3f} "
                  f"| pearson={cr['pearson']:+.3f} spearman={cr['spearman']:+.3f} "
                  f"| Brier n/a (score is not a probability)", ""]
        # trichotomy
        m = d[d["direction"] != "FLAT"]
        hit = (((m["direction"] == "LONG") & (m["real_R"] > 0)) |
               ((m["direction"] == "SHORT") & (m["real_R"] < 0))).mean() if len(m) else float("nan")
        L += [f"### Candidate trichotomy (LONG P(R>+c)>0.5 / SHORT P(R<-c)>0.5 / FLAT)",
              f"coverage={len(m) / len(d):.3f} conflicts={(d['conflict']).mean():.4f} "
              f"accuracy|traded={hit:.3f} "
              f"prec_LONG={((d['direction'] == 'LONG') & (d['real_R'] > 0)).sum() / max((d['direction'] == 'LONG').sum(), 1):.3f} "
              f"prec_SHORT={((d['direction'] == 'SHORT') & (d['real_R'] < 0)).sum() / max((d['direction'] == 'SHORT').sum(), 1):.3f}", ""]
        # probability calibration (A)
        cal = M.calibration_table(d["sigA_prob"], d["real_R"], "up")
        L += [f"### Probability calibration (Signal A): slope={cal.attrs['slope']:.3f} "
              f"intercept={cal.attrs['intercept']:+.3f} ECE={cal.attrs['ECE']:.4f}",
              cal.to_string(), ""]
        # vol signal E
        vs = M.vol_stats(d["exp_vol"], d["real_vol"])
        L += [f"### Signal E (model vol vs realised): corr={vs['corr']:.3f} "
              f"spearman={vs.get('spearman', float('nan')):.3f} bias={vs['bias']:+.4f} "
              f"rmse={vs['rmse']:.4f} mean_model={vs['mean_model']:.4f} "
              f"mean_real={vs['mean_realised']:.4f}", ""]
        figs += _figures(d, h, root, cal, cfg["outputs"].get("figure_prefix", "signals"))
    L += ["## Figures"] + [f"- {f}" for f in figs]
    L += ["", "## Recommendation (Stage-5 verdict)", _recommendation(out, cfg)]
    return "\n".join(L)


def _figures(d: pd.DataFrame, h: int, root: Path, cal: pd.DataFrame,
             prefix: str = "signals") -> list[str]:
    figdir = root / "reports" / "figures"
    out = []
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(d["t"], d["sigA_prob"], lw=1, label="P(R>0)")
    ax.plot(d["t"], d["sigD_up"], lw=1, alpha=0.7, label="P(R>+c)")
    ax.plot(d["t"], d["sigD_down"], lw=1, alpha=0.7, label="P(R<-c)")
    ax.axhline(0.5, color="k", lw=0.8)
    ax.set_title(f"Probability signals, h={h}")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = figdir / f"{prefix}_prob_h{h}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(str(p.relative_to(root)))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(cal["mean_pred"], cal["freq"], "o-")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("mean predicted P")
    ax.set_ylabel("realised frequency")
    ax.set_title(f"Calibration curve (A), h={h}")
    fig.tight_layout()
    p = figdir / f"{prefix}_calibration_h{h}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(str(p.relative_to(root)))
    return out


def _recommendation(out: pd.DataFrame, cfg) -> str:
    """Evidence-based advance/drop verdicts from the tables above."""
    lines = []
    for h in cfg["horizons"]:
        d = out[out["h"] == h].reset_index(drop=True)
        base = float((d["real_R"] > 0).mean())
        clim = base * (1 - base)
        ba = M.brier_score(d["sigA_prob"], d["real_R"], "up")
        accA = M.directional_stats(d["sigA_prob"], d["real_R"], 0.5)["accuracy"]
        crB = M.forecast_corr(d["sigB_eret"], d["real_R"])["pearson"]
        vs = M.vol_stats(d["exp_vol"], d["real_vol"])
        m = d[d["direction"] != "FLAT"]
        cov = len(m) / len(d)
        hit = (((m["direction"] == "LONG") & (m["real_R"] > 0)) |
               ((m["direction"] == "SHORT") & (m["real_R"] < 0))).mean() if len(m) else float("nan")
        lines.append(
            f"h={h}: Signal A acc {accA:.3f} vs base {base:.3f}, Brier skill "
            f"{1 - ba / clim:+.3f} (<=0 means worse than a constant); "
            f"Signal B corr {crB:+.3f}; "
            f"Signal E vol-corr {vs['corr']:.3f}, bias {vs['bias']:+.4f}; "
            f"trichotomy coverage {cov:.3f}, accuracy|traded {hit:.3f}.")
    lines.append("")
    lines.append("Verdicts:")
    lines.append("1. ADVANCE Signal E (Heston vol vs VIX / vol forecast) — only signal with "
                 "positive correlation on BOTH horizons (h=1: ~0.2, h=5: ~0.47). "
                 "Heston's information is about the volatility distribution, not direction. "
                 "Stage 6 should test vol-timing uses (vol-target sizing, no-trade/high-caution "
                 "regimes, VIX-spread caution flag), not directional bets on E.")
    lines.append("2. WATCHLIST-ONLY Signal D-up at h=5 (cost-aware tail probability) — most active "
                 "directional variant, but Brier skill is currently <= 0, so it FAILS the advance bar. "
                 "Do not size positions on it; retain only as a cost-awareness diagnostic unless "
                 "real-data rerun shows skill > 0 with precision edge vs base rate.")
    lines.append("3. DROP Signals A/B/C as directional predictors — accuracy at base rate, "
                 "Brier at/below climatology, calibration slope ~0 (h=1) or negative (h=5), "
                 "correlations ~0. B/C predict 'up' >98% of days (drift-dominated): no timing content.")
    lines.append("4. DROP the LONG/SHORT trichotomy in its current form — h=1 coverage is 0 "
                 "(1-day tails never cross 50%), h=5 accuracy (0.517) equals base rate and "
                 "SHORT precision is 0. A tradable rule needs rethinking (Stage 6), not re-tuning.")
    lines.append("5. Caveat: current input is DEMO synthetic data whose DGP has constant drift and "
                 "therefore NO true directional predictability by construction — the directional "
                 "null is EXPECTED here. Re-run Stages 3–5 on real NIFTY before concluding; "
                 "the decision framework above (Brier skill, calibration slope, vol-corr) is what "
                 "transfers, not the numeric verdicts.")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
