# Model Ablations

No new network efficacy or model ablation is established by S11. Only two hard source refits and two successful F1 engineering smoke fits were completed.
No-selection, No-phase and No-HSTU conditional formal fits were not admitted by the zero-fit gate.
Historical S09 phase ablations remain valid in their old configuration, not proof of phase/HSTU necessity under the new F2 distribution.
BN layer moments are real frozen-model diagnostics, not missing-checkpoint speculation.
| dataset | fold | group | seed | before_pct | after_fit_only_recalibration_pct | delta_pp | before_ce | after_ce | no_target_adaptation | no_fit_or_model_selection |
|---|---|---|---|---|---|---|---|---|---|---|
| ManyRx | 0 | last | 8101 | 89.3309 | 89.3671 | 0.0361 | 0.3897 | 0.3977 | True | True |
| ManySig | 0 | last | 8101 | 96.1667 | 96.2917 | 0.1250 | 0.1841 | 0.1903 | True | True |

BN->GN was NOT admitted: fold0-only coverage in both datasets cannot establish a systematic normalization bottleneck. Fit-only BN recalibration is not a GroupNorm ablation.
Layer/RX pre-BN moments and normalized DTX/DRX: `source_bn_layer_rx.csv`.
