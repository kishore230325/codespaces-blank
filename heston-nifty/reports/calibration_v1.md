# Stage 3 — Heston calibration report (historical primary; implied deferred)

Train block to 2020-12-31; rolling window=504d step=21d -> 51 calibrations. Objective = GMM moment match (NOT PnL).

## 1. Last-window result (representative; full series in reports/calibration_rolling.csv)

window 2019-02-25..2020-12-31 n=484
kappa=4.5990 theta=0.027705 (vol 16.64%) epsilon=0.9171 rho=0.0479 V0=0.007605 (vol 8.72%) mu_ann=0.1542
objective=0.000000 calib_rmse=0.0003 success=True nit=10 msg=CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
constraints: {'kappa': (0.1, 20.0), 'theta': (0.0025, 0.64), 'epsilon': (0.05, 3.0), 'rho': (-0.99, 0.2), 'V0': (0.0025, 0.64)}; x0=(4.616223, 0.027705, 0.279411, 0.047865)
Feller 2kT/e2=0.303 (VIOLATED — QE branch active, per Andersen design)

## 2. Rolling stability (median [p10, p90])

- kappa: 4.2589 [3.2425, 5.2832]
- theta: 0.0260 [0.0188, 0.0297]
- epsilon: 0.7524 [0.6439, 0.8857]
- rho: -0.0834 [-0.1252, 0.0393]
- V0: 0.0223 [0.0101, 0.0381]
- objective: 0.0000 [0.0000, 0.0000]
- calib_error_rmse: 0.0003 [0.0002, 0.0004]
- feller_stat: 0.3401 [0.3067, 0.4797]
- convergence rate: 100% (51/51)
- Feller satisfied: 0% of windows

## 3. Moment validation (last window, 1-step CIR check)
 mean_std_resid  std_std_resid  mae_mean  coverage_2sigma   n
       -0.00982       0.307183  0.001834              1.0 483
Verdict: std_std_resid=0.31 << 1 with coverage=1.00: the CIR conditional variance s2 substantially OVERSTATES the innovation variance of the 21-day SMOOTHED proxy (smoothing suppresses innovations by construction). Expected artefact of the V1 proxy choice — epsilon fitted here is a smoothed-proxy-implied value, not a direct estimate of true vol-of-vol. Do not compare its magnitude to options-implied epsilon without the §7 caveats.

NOTE: the GMM system fits 4 parameters to 4 moments (exactly identified),
so objective ~= 0 BY CONSTRUCTION. It measures optimiser convergence,
not model quality. Model quality is judged by §3 (this table), §2 stability,
and §4 sensitivity — not by the objective value.

## 4. Sensitivity to starting values (5 perturbed restarts, last window)
 start_id    kappa    theta  epsilon      rho       V0    objective  success
        0 4.616075 0.027705 0.918783 0.047866 0.007605 6.016587e-11     True
        1 4.616242 0.027705 0.918796 0.047865 0.007605 2.099022e-12     True
        2 4.616240 0.027705 0.918796 0.047866 0.007605 1.297721e-12     True
        3 4.616219 0.027705 0.918794 0.047865 0.007605 1.371882e-13     True
        4 4.615957 0.027705 0.918762 0.047848 0.007605 8.421218e-10     True
std across starts: {'kappa': 0.00012663546502332884, 'theta': 5.697093647866761e-08, 'epsilon': 1.4815434995473134e-05, 'rho': 7.92542616337141e-06, 'V0': 9.69739903612216e-19}

## 5. Options-implied calibration
market status: UNAVAILABLE — No options snapshot provided (V1-deferred, V2 work). See data_loader.load_options_snapshot schema.
synthetic self-test (true k=2.5 th=0.045 e=0.55 r=-0.65 V0=0.04, 1% price noise):
recovered k=1.037 th=0.05247 e=0.462 r=-0.711 V0=0.03944 rmse=5.98 success=True nit=50 msg=CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
Pricer validated: put-call parity holds by construction; self-test recovery is approximate (Fourier integral + noisy quotes), proving the pipeline works for V2 market data.

## 6. Figures
- reports/figures/variance_fit_last_window.png — proxy vs CIR conditional mean
- reports/figures/rolling_{kappa,theta,epsilon,rho,V0,feller_stat,objective}.png

## 7. Known limitations (must carry into Stage 4)
- Variance proxy is noisy; 21d smoothing biases kappa downward (slower mean reversion).
- GMM matches 4 moments only; kappa/epsilon weakly co-identified (ridge: fast+noisy ~= slow+calm).
- mu_ann is a sample mean, high-variance; must NOT be treated as a return forecast.
- Demo/synthetic-data runs validate code, not markets; re-run on real NIFTY before any signal work.
- Options-implied path blocked on real chain data (bid/ask, expiry calendar, liquidity weights).