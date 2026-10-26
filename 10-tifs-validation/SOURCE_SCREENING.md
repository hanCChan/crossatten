# Source Screening

Decision: **STOP**.
Primary criterion frozen before fits: raw checkpoint minimizing source packet CE.
EMA/held-CE alternatives are not ranked or used to rescue a failed candidate.

| Dataset | Mean Gain (pp) | Positive Folds | Worst RX Delta (pp) |
|---|---:|---:|---:|
| ManyRx | -0.4881 | 1/4 | -3.1111 |
| ManySig | 0.5651 | 3/4 | -1.6667 |

Checks:
- manyrx_three_positive_folds: FAIL
- manyrx_mean_at_least_1pp: FAIL
- worst_rx_no_more_than_2pp_loss: FAIL
- manyrx_group_and_seed_consistent: FAIL
- manysig_no_systematic_degradation: FAIL

32 new soft fits; 32 exactly compatible historical hard controls reused.
Missing S09 weights do not prevent prediction/config pairing but preclude historical BN diagnosis.
Details: `decision_final.json`, `source_paired_receivers.csv`.
