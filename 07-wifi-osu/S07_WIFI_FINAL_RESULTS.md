# S07 WiSig Final Results

Completed all 72 frozen final fits. Four RX folds x three seeds. Source-selected configurations are disclosed in `decision_final.json`.

CRD is primary. The public targets were previously inspected; this is not a fresh blind test. S06 external fair baselines are frozen and use the same normalization and cell budgets. S07 seeds4101/4102/4103 are not paired with S06 training seeds. Receiver-clustered intervals condition on trained models and do not represent independent datasets. Intervals are descriptive and not corrected for multiple pipeline/baseline comparisons. `Packet acc` is the official four-fold mean of packet-weighted scores; the CSV additionally contains accuracy pooled over all test packets/folds/seeds.

| Dataset | Pipeline | Task | RX acc % | Packet acc % | Macro-F1 % | Seed SD pp | Worst RX % | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ManyRx | source_frontend | CR | 81.90 | 81.94 | 79.17 | 0.38 | 53.32 | 90680 |
| ManyRx | source_frontend | CRD | 77.24 | 77.22 | 73.51 | 0.37 | 50.95 | 90680 |
| ManyRx | source_training | CR | 89.87 | 89.87 | 87.90 | 0.49 | 69.00 | 90680 |
| ManyRx | source_training | CRD | 87.65 | 87.65 | 85.43 | 0.61 | 68.01 | 90680 |
| ManyRx | source_combination | CR | 89.87 | 89.87 | 87.90 | 0.49 | 69.00 | 90680 |
| ManyRx | source_combination | CRD | 87.65 | 87.65 | 85.43 | 0.61 | 68.01 | 90680 |
| ManySig | source_frontend | CR | 70.49 | 70.49 | 67.90 | 2.12 | 58.64 | 90420;142276 |
| ManySig | source_frontend | CRD | 65.98 | 65.98 | 63.20 | 1.56 | 43.75 | 90420;142276 |
| ManySig | source_training | CR | 87.61 | 87.61 | 87.01 | 0.95 | 59.44 | 90420;142276 |
| ManySig | source_training | CRD | 86.62 | 86.62 | 85.96 | 0.94 | 54.67 | 90420;142276 |
| ManySig | source_combination | CR | 87.61 | 87.61 | 87.01 | 0.95 | 59.44 | 90420;142276 |
| ManySig | source_combination | CRD | 86.62 | 86.62 | 85.96 | 0.94 | 54.67 | 90420;142276 |

## CRD Baseline Differences

| Dataset | Pipeline | Baseline | Mean delta pp | RX bootstrap 95% interval | Better RX |
|---|---|---|---:|---|---:|
| ManyRx | source_frontend | W1 | -6.32 | [-9.13, -3.70] | 3/32 |
| ManyRx | source_frontend | IB-RFF | -10.83 | [-14.47, -7.50] | 4/32 |
| ManyRx | source_frontend | WaveMLP | -10.81 | [-14.29, -7.61] | 3/32 |
| ManyRx | source_frontend | FAN_source_selected | -10.32 | [-13.99, -6.83] | 6/32 |
| ManyRx | source_training | W1 | +4.09 | [+2.71, +5.58] | 27/32 |
| ManyRx | source_training | IB-RFF | -0.42 | [-1.86, +1.06] | 14/32 |
| ManyRx | source_training | WaveMLP | -0.41 | [-1.76, +0.94] | 14/32 |
| ManyRx | source_training | FAN_source_selected | +0.09 | [-1.38, +1.64] | 18/32 |
| ManyRx | source_combination | W1 | +4.09 | [+2.72, +5.58] | 27/32 |
| ManyRx | source_combination | IB-RFF | -0.42 | [-1.87, +1.03] | 14/32 |
| ManyRx | source_combination | WaveMLP | -0.41 | [-1.74, +0.95] | 14/32 |
| ManyRx | source_combination | FAN_source_selected | +0.09 | [-1.44, +1.60] | 18/32 |
| ManySig | source_frontend | W1 | -6.35 | [-15.68, +1.67] | 4/12 |
| ManySig | source_frontend | IB-RFF | -25.11 | [-30.88, -19.38] | 0/12 |
| ManySig | source_frontend | WaveMLP | -25.06 | [-32.16, -18.30] | 0/12 |
| ManySig | source_frontend | FAN_source_selected | -22.29 | [-28.22, -16.57] | 0/12 |
| ManySig | source_training | W1 | +14.29 | [+6.15, +22.84] | 10/12 |
| ManySig | source_training | IB-RFF | -4.48 | [-9.21, -0.10] | 4/12 |
| ManySig | source_training | WaveMLP | -4.42 | [-11.00, +0.56] | 5/12 |
| ManySig | source_training | FAN_source_selected | -1.66 | [-6.08, +2.63] | 6/12 |
| ManySig | source_combination | W1 | +14.29 | [+6.26, +22.61] | 10/12 |
| ManySig | source_combination | IB-RFF | -4.48 | [-9.17, -0.11] | 4/12 |
| ManySig | source_combination | WaveMLP | -4.42 | [-11.34, +0.50] | 5/12 |
| ManySig | source_combination | FAN_source_selected | -1.66 | [-6.13, +2.69] | 6/12 |

## Historical Released-Weight References

Separate unchanged published-weight verification from S05, not fair S07 multi-seed fits. These CRD aggregates use official packet weighting; do not equate them with RX-equal scores.

| Method | ManyRx CRD % | ManySig CRD % |
|---|---:|---:|
| IB-RFF | 89.19 | 93.57 |
| WaveMLP | 88.68 | 93.13 |

## Frozen Per-Fold Configurations

| Dataset | Fold | Pipeline | Frontend | Recipe | Augmentation | Balanced | Noise | Checkpoint | Epoch budget |
|---|---:|---|---|---|---|---|---|---|---:|
| ManyRx | 0 | source_frontend | frequency | unified | none | False | default | packet | 200 |
| ManyRx | 0 | source_training | frequency | unified | none | False | mild | packet | 200 |
| ManyRx | 0 | source_combination | frequency | unified | none | False | mild | packet | 200 |
| ManyRx | 1 | source_frontend | frequency | unified | none | False | default | packet | 200 |
| ManyRx | 1 | source_training | frequency | official | none | False | default | packet | 200 |
| ManyRx | 1 | source_combination | frequency | official | none | False | default | packet | 200 |
| ManyRx | 2 | source_frontend | frequency | unified | none | False | default | packet | 200 |
| ManyRx | 2 | source_training | frequency | unified | none | False | mild | packet | 200 |
| ManyRx | 2 | source_combination | frequency | unified | none | False | mild | packet | 200 |
| ManyRx | 3 | source_frontend | frequency | unified | none | False | default | packet | 200 |
| ManyRx | 3 | source_training | frequency | official | none | False | default | packet | 200 |
| ManyRx | 3 | source_combination | frequency | official | none | False | default | packet | 200 |
| ManySig | 0 | source_frontend | frequency | unified | none | False | default | packet | 200 |
| ManySig | 0 | source_training | frequency | official | none | False | default | packet | 200 |
| ManySig | 0 | source_combination | frequency | official | none | False | default | packet | 200 |
| ManySig | 1 | source_frontend | frequency | unified | none | False | default | packet | 200 |
| ManySig | 1 | source_training | frequency | official | none | False | default | packet | 200 |
| ManySig | 1 | source_combination | frequency | official | none | False | default | packet | 200 |
| ManySig | 2 | source_frontend | frequency | unified | none | False | default | packet | 200 |
| ManySig | 2 | source_training | frequency | official | none | False | default | packet | 200 |
| ManySig | 2 | source_combination | frequency | official | none | False | default | packet | 200 |
| ManySig | 3 | source_frontend | split | unified | none | False | default | packet | 200 |
| ManySig | 3 | source_training | split | official | none | False | default | packet | 200 |
| ManySig | 3 | source_combination | split | official | none | False | default | packet | 200 |

Identical registered pipelines, not independent methods: `ManyRx:source_combination` = `ManyRx:source_training`; `ManySig:source_combination` = `ManySig:source_training`.

No method should be selected after seeing this table. All registered pipelines are reported. These results support only tested-protocol statements; SOTA requires a separate primary-source review.
