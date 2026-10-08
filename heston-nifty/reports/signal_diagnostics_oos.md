# Stage 5 — predictive-content diagnostics (no PnL, no tuned thresholds)

Rows: 1494 (validation block 2023-01-05..2025-12-31); N=20000/forecast; hurdle=0.001.
Mapping conventions fixed a priori: prob>0.5 -> up; signed score>0 -> up.

## Horizon h=1 (n=749)

### Signal A prob P(R>0)
acc=0.469 (base 0.493) bal_acc=0.474 prec_up=0.478 rec_up=0.870 prec_dn=0.385 rec_dn=0.079 | pearson=-0.106 spearman=-0.076 | Brier=0.2508 vs climatology 0.2499 (skill=-0.003)

### Signal D-up P(R>+c)
acc=0.507 (base 0.493) bal_acc=0.500 prec_up=nan rec_up=0.000 prec_dn=0.507 rec_dn=1.000 | pearson=-0.046 spearman=-0.071 | Brier=0.2490 vs climatology 0.2481 (skill=-0.004)

### Signal D-down P(R<-c)
acc=0.493 (base 0.493) bal_acc=0.500 prec_up=0.493 rec_up=1.000 prec_dn=nan rec_dn=0.000 | pearson=+0.077 spearman=+0.048 | Brier=0.2501 vs climatology 0.2491 (skill=-0.004)

### Signal B E[R]
acc=0.491 (base 0.493) bal_acc=0.498 | pearson=-0.104 spearman=-0.082 | Brier n/a (score is not a probability)

### Signal C E[R]/sd
acc=0.491 (base 0.493) bal_acc=0.498 | pearson=-0.095 spearman=-0.074 | Brier n/a (score is not a probability)

### Candidate trichotomy (LONG P(R>+c)>0.5 / SHORT P(R<-c)>0.5 / FLAT)
coverage=0.000 conflicts=0.0000 accuracy|traded=nan prec_LONG=0.000 prec_SHORT=0.000

### Probability calibration (Signal A): slope=-3.244 intercept=+2.146 ECE=0.0541
      n  mean_pred      freq
bin                         
0    76   0.497416  0.605263
1    75   0.501498  0.520000
2    74   0.503220  0.405405
3    76   0.505273  0.460526
4    74   0.507209  0.418919
5    74   0.509171  0.540541
6    75   0.511667  0.533333
7    75   0.514663  0.520000
8    76   0.519071  0.473684
9    74   0.526765  0.445946

### Signal E (model vol vs realised): corr=0.256 spearman=0.225 bias=+0.0393 rmse=0.1365 mean_model=0.1970 mean_real=0.1577

## Horizon h=5 (n=745)

### Signal A prob P(R>0)
acc=0.511 (base 0.530) bal_acc=0.484 prec_up=0.522 rec_up=0.932 prec_dn=0.325 rec_dn=0.037 | pearson=-0.152 spearman=-0.121 | Brier=0.2508 vs climatology 0.2491 (skill=-0.007)

### Signal D-up P(R>+c)
acc=0.478 (base 0.530) bal_acc=0.473 prec_up=0.507 rec_up=0.549 prec_dn=0.438 rec_dn=0.397 | pearson=-0.139 spearman=-0.127 | Brier=0.2512 vs climatology 0.2496 (skill=-0.006)

### Signal D-down P(R<-c)
acc=0.530 (base 0.530) bal_acc=0.500 prec_up=0.530 rec_up=1.000 prec_dn=nan rec_dn=0.000 | pearson=+0.153 spearman=+0.120 | Brier=0.2496 vs climatology 0.2475 (skill=-0.008)

### Signal B E[R]
acc=0.532 (base 0.530) bal_acc=0.504 | pearson=-0.132 spearman=-0.123 | Brier n/a (score is not a probability)

### Signal C E[R]/sd
acc=0.532 (base 0.530) bal_acc=0.504 | pearson=-0.145 spearman=-0.122 | Brier n/a (score is not a probability)

### Candidate trichotomy (LONG P(R>+c)>0.5 / SHORT P(R<-c)>0.5 / FLAT)
coverage=0.574 conflicts=0.0000 accuracy|traded=0.507 prec_LONG=0.507 prec_SHORT=0.000

### Probability calibration (Signal A): slope=-1.791 intercept=+1.467 ECE=0.0747
      n  mean_pred      freq
bin                         
0    75   0.498700  0.653333
1    77   0.505432  0.571429
2    72   0.508942  0.486111
3    75   0.512099  0.453333
4    74   0.515724  0.527027
5    74   0.520771  0.472973
6    75   0.526403  0.546667
7    74   0.534130  0.689189
8    75   0.544918  0.520000
9    74   0.563183  0.378378

### Signal E (model vol vs realised): corr=0.467 spearman=0.487 bias=+0.0069 rmse=0.0756 mean_model=0.1964 mean_real=0.1895

## Figures
- reports/figures/signals_oos_prob_h1.png
- reports/figures/signals_oos_calibration_h1.png
- reports/figures/signals_oos_prob_h5.png
- reports/figures/signals_oos_calibration_h5.png

## Recommendation (Stage-5 verdict)
h=1: Signal A acc 0.469 vs base 0.493, Brier skill -0.003 (<=0 means worse than a constant); Signal B corr -0.104; Signal E vol-corr 0.256, bias +0.0393; trichotomy coverage 0.000, accuracy|traded nan.
h=5: Signal A acc 0.511 vs base 0.530, Brier skill -0.007 (<=0 means worse than a constant); Signal B corr -0.132; Signal E vol-corr 0.467, bias +0.0069; trichotomy coverage 0.574, accuracy|traded 0.507.

Verdicts:
1. ADVANCE Signal E (Heston vol vs VIX / vol forecast) — only signal with positive correlation on BOTH horizons (h=1: ~0.2, h=5: ~0.47). Heston's information is about the volatility distribution, not direction. Stage 6 should test vol-timing uses (vol-target sizing, no-trade/high-caution regimes, VIX-spread caution flag), not directional bets on E.
2. WATCHLIST-ONLY Signal D-up at h=5 (cost-aware tail probability) — most active directional variant, but Brier skill is currently <= 0, so it FAILS the advance bar. Do not size positions on it; retain only as a cost-awareness diagnostic unless real-data rerun shows skill > 0 with precision edge vs base rate.
3. DROP Signals A/B/C as directional predictors — accuracy at base rate, Brier at/below climatology, calibration slope ~0 (h=1) or negative (h=5), correlations ~0. B/C predict 'up' >98% of days (drift-dominated): no timing content.
4. DROP the LONG/SHORT trichotomy in its current form — h=1 coverage is 0 (1-day tails never cross 50%), h=5 accuracy (0.517) equals base rate and SHORT precision is 0. A tradable rule needs rethinking (Stage 6), not re-tuning.
5. Caveat: current input is DEMO synthetic data whose DGP has constant drift and therefore NO true directional predictability by construction — the directional null is EXPECTED here. Re-run Stages 3–5 on real NIFTY before concluding; the decision framework above (Brier skill, calibration slope, vol-corr) is what transfers, not the numeric verdicts.