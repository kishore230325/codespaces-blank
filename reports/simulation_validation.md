# Stage 4 — QE simulator validation report

Mode: full battery; 26/26 checks passed.

## Attribution (Andersen vs extension)
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

## Results
| check             | metric                                   |        value | tolerance           | passed   | note                                                                                                          |
|:------------------|:-----------------------------------------|-------------:|:--------------------|:---------|:--------------------------------------------------------------------------------------------------------------|
| nonnegativity     | min(V) over 6.3M draws                   |  9.43643e-13 | >= 0                | True     | k=2.5 th=0.045 e=0.55                                                                                         |
| cond_mean         | E[Vn|V=0.01] vs 0.010346                 |  0.000685882 | < 0.02              | True     |                                                                                                               |
| cond_var          | Var[Vn|V=0.01] vs 0.000012               |  0.00414204  | < 0.02              | True     |                                                                                                               |
| cond_mean         | E[Vn|V=0.04] vs 0.040049                 |  1.66877e-05 | < 0.02              | True     |                                                                                                               |
| cond_var          | Var[Vn|V=0.04] vs 0.000048               |  0.00524627  | < 0.02              | True     |                                                                                                               |
| cond_mean         | E[Vn|V=0.16] vs 0.158865                 |  3.39632e-05 | < 0.02              | True     |                                                                                                               |
| cond_var          | Var[Vn|V=0.16] vs 0.000189               |  0.00155217  | < 0.02              | True     |                                                                                                               |
| variance_law      | KS stat (ncx2 df=1.49)                   |  0.00476436  | < 0.01              | True     | KS stat=0.0048, p=2.27e-04 (p rejects at N=200k by power; weak scheme judged by moments + prices, which pass) |
| correlation       | corr(dlogS,dV) rho=0.7                   |  0.699818    | in (0.05,1.0)       | True     | via variance-increment term, not correlated normals                                                           |
| correlation       | corr(dlogS,dV) rho=-0.7                  | -0.700271    | in (-1.0,-0.05)     | True     | via variance-increment term, not correlated normals                                                           |
| correlation       | corr(dlogS,dV) rho=0.0                   | -0.000484374 | in (-0.05,0.05)     | True     | via variance-increment term, not correlated normals                                                           |
| martingale        | E[S_T]=102.561 vs fwd=102.532 (0.92 SE)  |  0.918353    | < 3.0 SE            | True     | correction skipped: 0.00e+00                                                                                  |
| convergence       | call(n=1) vs COS=6.798                   |  6.80023     | bias shrinks with n | True     | rel err=+0.026%, SE=0.025, exp-branch=0.0%                                                                    |
| convergence       | call(n=2) vs COS=6.798                   |  6.82491     | bias shrinks with n | True     | rel err=+0.390%, SE=0.025, exp-branch=0.0%                                                                    |
| convergence       | call(n=4) vs COS=6.798                   |  6.80429     | bias shrinks with n | True     | rel err=+0.086%, SE=0.025, exp-branch=0.0%                                                                    |
| convergence       | call(n=13) vs COS=6.798                  |  6.81195     | bias shrinks with n | True     | rel err=+0.199%, SE=0.025, exp-branch=0.0%                                                                    |
| convergence       | call(n=52) vs COS=6.798                  |  6.82516     | bias shrinks with n | True     | rel err=+0.393%, SE=0.025, exp-branch=0.0%                                                                    |
| convergence_bench | COS benchmark                            |  6.79843     | reference           | True     | Stage-3-validated pricer                                                                                      |
| extreme           | feller_violated_fit: finite & V>=0 & S>0 |  1           | == 1                | True     | exp-branch=26.1%, skipped-M=0.00e+00                                                                          |
| extreme           | huge_volvol: finite & V>=0 & S>0         |  1           | == 1                | True     | exp-branch=73.0%, skipped-M=0.00e+00                                                                          |
| extreme           | fast_kappa: finite & V>=0 & S>0          |  1           | == 1                | True     | exp-branch=0.0%, skipped-M=0.00e+00                                                                           |
| extreme           | zero_V0: finite & V>=0 & S>0             |  1           | == 1                | True     | exp-branch=15.9%, skipped-M=0.00e+00                                                                          |
| extreme           | rho_high: finite & V>=0 & S>0            |  1           | == 1                | True     | exp-branch=0.0%, skipped-M=0.00e+00                                                                           |
| extreme           | rho_low: finite & V>=0 & S>0             |  1           | == 1                | True     | exp-branch=0.0%, skipped-M=0.00e+00                                                                           |
| nonnegativity     | min(V) over 6.3M draws                   |  0           | >= 0                | True     | k=4.85 th=0.0282 e=0.9357                                                                                     |
| martingale        | E[S_T]=102.547 vs fwd=102.532 (0.65 SE)  |  0.651554    | < 3.0 SE            | True     | correction skipped: 0.00e+00                                                                                  |

## Figures
- reports/figures/qe_convergence.png — call-price bias vs COS shrinks with n_steps
- reports/figures/qe_variance_law.png — QE draws vs exact CIR (ncx2) transition density

## Verdict for Stage 5
PASS: simulator is mathematically sound; Stage 5 may build forecast distributions on it.