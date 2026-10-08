"""Predictive-accuracy metrics (Stage 5). No prices, no PnL — only forecasts vs realisations.

For directional signals the binary mapping is a MEASUREMENT convention fixed
a priori (prob > 0.5 -> up; signed score > 0 -> up), identical for every
signal. It is not a tuned trading threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def directional_stats(score: pd.Series, realised: pd.Series,
                      thresh: float) -> dict:
    """Accuracy / precision / recall for up- and down-predictions."""
    pred_up = (score > thresh)
    act_up = (realised > 0)
    tp = int((pred_up & act_up).sum())
    tn = int(((~pred_up) & (~act_up)).sum())
    fp = int((pred_up & (~act_up)).sum())
    fn = int(((~pred_up) & act_up).sum())
    n = len(score)
    return {
        "n": n,
        "frac_predicted_up": float(pred_up.mean()),
        "frac_actual_up": float(act_up.mean()),
        "accuracy": _safe_div(tp + tn, n),
        "precision_up": _safe_div(tp, tp + fp),
        "recall_up": _safe_div(tp, tp + fn),
        "precision_down": _safe_div(tn, tn + fn),
        "recall_down": _safe_div(tn, tn + fp),
        "balanced_accuracy": float(np.nanmean([_safe_div(tp, tp + fn), _safe_div(tn, tn + fp)])),
    }


def brier_score(prob: pd.Series, realised: pd.Series, positive: str = "up") -> float:
    """Mean squared error of a probability forecast. positive: 'up' (R>0),
    'down1' (R<-1%), 'up1' (R>+1%)."""
    if positive == "up":
        y = (realised > 0).astype(float)
    elif positive == "down1":
        y = (realised < -0.01).astype(float)
    elif positive == "up1":
        y = (realised > 0.01).astype(float)
    else:
        raise ValueError(positive)
    return float(((prob - y) ** 2).mean())


def forecast_corr(score: pd.Series, realised: pd.Series) -> dict:
    x = score.to_numpy(dtype=float)
    y = realised.to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10 or x.std() == 0:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": len(x)}
    return {"pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
            "n": len(x)}


def calibration_table(prob: pd.Series, realised: pd.Series,
                       positive: str = "up", n_bins: int = 10) -> pd.DataFrame:
    """Decile calibration: mean predicted prob vs realised frequency per bin."""
    if positive == "up":
        y = (realised > 0).astype(float)
    elif positive == "down1":
        y = (realised < -0.01).astype(float)
    else:
        y = (realised > 0.01).astype(float)
    df = pd.DataFrame({"p": prob.to_numpy(dtype=float), "y": y.to_numpy(dtype=float)})
    df = df[np.isfinite(df["p"])]
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), freq=("y", "mean"))
    # slope/intercept of realised ~ predicted (1/0 = perfect calibration)
    slope, intercept = (float("nan"), float("nan"))
    if df["p"].std() > 0 and len(df) > 20:
        slope, intercept = np.polyfit(df["p"], df["y"], 1)
    g.attrs["slope"] = slope
    g.attrs["intercept"] = intercept
    g.attrs["ECE"] = float((abs(g["mean_pred"] - g["freq"]) * g["n"] / g["n"].sum()).sum())
    return g


def vol_stats(model_vol: pd.Series, realised_vol: pd.Series) -> dict:
    d = pd.DataFrame({"m": model_vol.astype(float), "r": realised_vol.astype(float)}).dropna()
    if len(d) < 10:
        return {"n": len(d), "corr": float("nan"), "bias": float("nan"), "rmse": float("nan")}
    e = d["m"] - d["r"]
    return {"n": len(d), "corr": float(d["m"].corr(d["r"])),
            "spearman": float(d["m"].corr(d["r"], method="spearman")),
            "bias": float(e.mean()), "rmse": float(np.sqrt((e**2).mean())),
            "mean_model": float(d["m"].mean()), "mean_realised": float(d["r"].mean())}
