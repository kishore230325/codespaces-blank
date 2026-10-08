# Supplement — synthetic ground-truth assessment (DEMO DATA ONLY)

**Scope warning:** the current `data/processed/nifty_daily.parquet` was built with
`--demo` (seeded Heston-flavoured path, seed=42). Ground truth is therefore KNOWN
(build_demo_frames: kappa=2.0, theta=0.04, epsilon=0.30, rho=-0.65, mu_ann=0.08).
This note compares V1 estimates against truth to characterise estimator bias.
**Disregard this file after re-running on real NIFTY data; the calibration CODE is
unchanged, only the input data is synthetic.**

## Historical GMM (rolling medians over 51 train windows vs truth)

| param | truth | median est. | bias direction | verdict |
|---|---|---|---|---|
| kappa | 2.00 | 4.26 | ~2x high | persistence underestimated (noisy proxy + AR(1) attenuation) |
| theta | 0.0400 (20.0% vol) | 0.0260 (16.1% vol) | low | level under-recovered; demeaning + MA smoothing leak variance |
| epsilon | 0.30 | 0.75 | ~2.5x high | absorbs inconsistency of mixed proxies (raw var + smoothed ac1); see §3 verdict |
| rho | -0.65 | -0.08 | washed toward 0 | 21d smoothing destroys daily return-vs-dV leverage correlation |
| V0 | varies | tracks proxy | ok (by construction) | EWMA of proxy; tautological fit, not a model test |
| Feller 2kT/e² | 1.78 (satisfies) | 0.34 (violated) | regime flip | estimator pushes model into Feller-violation regime on data that satisfies it |

**Consequences for Stage 4 (QE):** the QE simulator MUST be exercised in the
exponential-branch (psi > psi_c) regime — which is exactly the regime Andersen
designed it for. Pass the estimated (violating) parameters through, do not
"fix" them to satisfy Feller.

**Remedies deferred (do not implement in Stage 3):** subsampled/realised-kernel
variance proxy; QMLE on raw returns via particle filter; joint estimation with
VIX term-structure; all are Stage 9+ robustness items.

## Options-implied self-test (synthetic quotes, 1% noise, truth k=2.5/th=0.045/e=0.55/r=-0.65/V0=0.04)

Recovered: k=1.04, th=0.0525, e=0.46, r=-0.71, V0=0.0394.
Vol level (theta), skew (rho) and spot vol (V0) recovered well; kappa is
essentially unidentified from 3 short maturities — textbook Heston ridge
(kappa/theta trade off along the term-structure slope). V2 market calibration
must use wider maturity span and report kappa CIs, not point values.
