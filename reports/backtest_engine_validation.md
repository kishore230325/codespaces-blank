# Stage 6 — backtest engine validation (mechanics only)

## mode=longshort: PASS
- signals=1038 traded=449 flat_signal=589 conflict_flat=0 dropped_no_execution=0
- (a) entry-after-signal=True, prices-match-prints=True
- (b) max|gross-costs-net|=0.00e+00 (< 1e-8)
- (c) B&H replication error=0.00e+00 (< 1e-10)

## mode=longflat: PASS
- signals=1038 traded=449 flat_signal=589 conflict_flat=0 dropped_no_execution=0
- (a) entry-after-signal=True, prices-match-prints=True
- (b) max|gross-costs-net|=0.00e+00 (< 1e-8)
- (c) B&H replication error=0.00e+00 (< 1e-10)

Verdict: PASS — engine cleared for Stage 7 strategy evaluation.