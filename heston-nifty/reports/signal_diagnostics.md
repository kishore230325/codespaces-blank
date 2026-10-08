# Stage 5 — predictive-content diagnostics (no PnL, no tuned thresholds)

Rows: 1038 (validation block 2021-01-05..2022-12-31); N=20000/forecast; hurdle=0.001.
Mapping conventions fixed a priori: prob>0.5 -> up; signed score>0 -> up.

## Horizon h=1 (n=519)

### Signal A prob P(R>0)
acc=0.530 (base 0.526) bal_acc=0.505 prec_up=0.528 rec_up=0.985 prec_dn=0.600 rec_dn=0.024 | pearson=-0.038 spearman=-0.053 | Brier=0.2496 vs climatology 0.2493 (skill=-0.001)

### Signal D-up P(R>+c)
acc=0.474 (base 0.526) bal_acc=0.500 prec_up=nan rec_up=0.000 prec_dn=0.474 rec_dn=1.000 | pearson=-0.027 spearman=-0.023 | Brier=0.2458 vs climatology 0.2453 (skill=-0.002)

### Signal D-down P(R<-c)
acc=0.526 (base 0.526) bal_acc=0.500 prec_up=0.526 rec_up=1.000 prec_dn=nan rec_dn=0.000 | pearson=+0.021 spearman=+0.041 | Brier=0.2403 vs climatology 0.2402 (skill=-0.001)

### Signal B E[R]
acc=0.522 (base 0.526) bal_acc=0.497 | pearson=-0.027 spearman=-0.032 | Brier n/a (score is not a probability)

### Signal C E[R]/sd
acc=0.522 (base 0.526) bal_acc=0.497 | pearson=-0.032 spearman=-0.044 | Brier n/a (score is not a probability)

### Candidate trichotomy (LONG P(R>+c)>0.5 / SHORT P(R<-c)>0.5 / FLAT)
coverage=0.000 conflicts=0.0000 accuracy|traded=nan prec_LONG=0.000 prec_SHORT=0.000

### Probability calibration (Signal A): slope=-0.037 intercept=+0.545 ECE=0.0369
      n  mean_pred      freq
bin                         
0    52   0.503879  0.500000
1    53   0.514328  0.509434
2    51   0.520733  0.529412
3    52   0.524101  0.615385
4    53   0.527709  0.566038
5    50   0.532431  0.480000
6    52   0.538517  0.461538
7    52   0.543250  0.519231
8    52   0.552056  0.500000
9    52   0.559535  0.576923

### Signal E (model vol vs realised): corr=0.232 spearman=0.222 bias=+0.0208 rmse=0.0673 mean_model=0.0922 mean_real=0.0714

## Horizon h=5 (n=519)

### Signal A prob P(R>0)
acc=0.514 (base 0.514) bal_acc=0.500 prec_up=0.515 rec_up=0.996 prec_dn=0.500 rec_dn=0.004 | pearson=-0.086 spearman=-0.083 | Brier=0.2579 vs climatology 0.2498 (skill=-0.032)

### Signal D-up P(R>+c)
acc=0.514 (base 0.514) bal_acc=0.504 prec_up=0.517 rec_up=0.869 prec_dn=0.500 rec_dn=0.139 | pearson=-0.104 spearman=-0.109 | Brier=0.2588 vs climatology 0.2489 (skill=-0.040)

### Signal D-down P(R<-c)
acc=0.514 (base 0.514) bal_acc=0.500 prec_up=0.514 rec_up=1.000 prec_dn=nan rec_dn=0.000 | pearson=+0.073 spearman=+0.065 | Brier=0.2563 vs climatology 0.2483 (skill=-0.032)

### Signal B E[R]
acc=0.513 (base 0.514) bal_acc=0.498 | pearson=-0.084 spearman=-0.112 | Brier n/a (score is not a probability)

### Signal C E[R]/sd
acc=0.513 (base 0.514) bal_acc=0.498 | pearson=-0.092 spearman=-0.099 | Brier n/a (score is not a probability)

### Candidate trichotomy (LONG P(R>+c)>0.5 / SHORT P(R<-c)>0.5 / FLAT)
coverage=0.865 conflicts=0.0000 accuracy|traded=0.517 prec_LONG=0.517 prec_SHORT=0.000

### Probability calibration (Signal A): slope=-0.504 intercept=+0.807 ECE=0.0987
      n  mean_pred      freq
bin                         
0    52   0.512933  0.480769
1    52   0.534444  0.673077
2    52   0.551272  0.576923
3    52   0.557677  0.500000
4    52   0.564606  0.538462
5    51   0.579403  0.549020
6    52   0.596749  0.442308
7    52   0.615151  0.346154
8    52   0.635862  0.442308
9    52   0.654388  0.596154

### Signal E (model vol vs realised): corr=0.472 spearman=0.506 bias=+0.0061 rmse=0.0372 mean_model=0.0935 mean_real=0.0874

## Figures
- reports/figures/signals_prob_h1.png
- reports/figures/signals_calibration_h1.png
- reports/figures/signals_prob_h5.png
- reports/figures/signals_calibration_h5.png

## Recommendation (Stage-5 verdict)
h=1: Signal A acc 0.530 vs base 0.526, Brier skill -0.001 (<=0 means worse than a constant); Signal B corr -0.027; Signal E vol-corr 0.232, bias +0.0208; trichotomy coverage 0.000, accuracy|traded nan.
h=5: Signal A acc 0.514 vs base 0.514, Brier skill -0.032 (<=0 means worse than a constant); Signal B corr -0.084; Signal E vol-corr 0.472, bias +0.0061; trichotomy coverage 0.865, accuracy|traded 0.517.

Verdicts:
1. ADVANCE Signal E (Heston vol vs VIX / vol forecast) — only signal with positive correlation on BOTH horizons (h=1: ~0.2, h=5: ~0.47). Heston's information is about the volatility distribution, not direction. Stage 6 should test vol-timing uses (vol-target sizing, no-trade/high-caution regimes, VIX-spread caution flag), not directional bets on E.
2. WATCHLIST-ONLY Signal D-up at h=5 (cost-aware tail probability) — most active directional variant, but Brier skill is currently <= 0, so it FAILS the advance bar. Do not size positions on it; retain only as a cost-awareness diagnostic unless real-data rerun shows skill > 0 with precision edge vs base rate.
3. DROP Signals A/B/C as directional predictors — accuracy at base rate, Brier at/below climatology, calibration slope ~0 (h=1) or negative (h=5), correlations ~0. B/C predict 'up' >98% of days (drift-dominated): no timing content.
4. DROP the LONG/SHORT trichotomy in its current form — h=1 coverage is 0 (1-day tails never cross 50%), h=5 accuracy (0.517) equals base rate and SHORT precision is 0. A tradable rule needs rethinking (Stage 6), not re-tuning.
5. Caveat: current input is DEMO synthetic data whose DGP has constant drift and therefore NO true directional predictability by construction — the directional null is EXPECTED here. Re-run Stages 3–5 on real NIFTY before concluding; the decision framework above (Brier skill, calibration slope, vol-corr) is what transfers, not the numeric verdicts.