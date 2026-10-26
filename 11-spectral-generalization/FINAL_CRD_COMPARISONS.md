# Final CRD Comparisons

**No S11 formal target evaluation was authorized or performed.** The following is historical S10-audited evidence, not newly obtained S11 scores.
| dataset | model | accuracy_pct | macro_f1_pct | worst_rx_pct | seed_sd_pp | fold_sd_pp |
|---|---|---|---|---|---|---|
| ManyRx | IB-RFF | 88.0702 | 85.8595 | 67.1781 | not reported | 2.8737 |
| ManyRx | S07_frequency_mild | 87.6506 | 85.4252 | 68.0071 | not reported | 4.6586 |
| ManyRx | S10_ask | 88.7697 | 86.5035 | 65.2028 | 0.1870 | 3.3275 |
| ManyRx | WaveMLP | 88.0562 | 85.6157 | 68.9594 | not reported | 2.3964 |
| ManySig | IB-RFF | 91.0926 | 90.8187 | 76.7778 | not reported | 2.7587 |
| ManySig | S08_hard | 93.0116 | 92.7970 | 84.6667 | not reported | 1.0800 |
| ManySig | S10_ask | 93.4838 | 93.1502 | 82.9167 | 0.1123 | 2.3339 |
| ManySig | WaveMLP | 91.0370 | 90.4086 | 78.7500 | not reported | 3.5879 |

Missing historical seed SD is marked not reported, never zero or newly estimated.
No candidate90%/93.5% or SOTA achievement is claimed. No per-fold target-specific settings were selected.
Receiver-cluster comparisons and uncertainty are inherited from `../10-tifs-validation/receiver_cluster_intervals.csv`; target data were not loaded again.
