# Final CR/CRD Comparisons

Reliability candidate decision: **STOP**.
RX-equal means. Historical controls are kept with their original recipes, never target-ranked.

| Dataset | Model | Task | Accuracy (%) | Macro-F1 (%) | Worst RX (%) | 95% RX CI |
|---|---|---|---:|---:|---:|---|
| ManyRx | IB-RFF | CR | 90.2163 | 88.1223 | 69.9000 | [87.910, 92.328] |
| ManyRx | IB-RFF | CRD | 88.0702 | 85.8595 | 67.1781 | [85.726, 90.233] |
| ManyRx | S07_frequency_mild | CR | 89.8685 | 87.9035 | 69.0000 | [87.302, 92.177] |
| ManyRx | S07_frequency_mild | CRD | 87.6506 | 85.4252 | 68.0071 | [84.814, 90.274] |
| ManyRx | S10_ask | CR | 91.1865 | 89.2254 | 72.2833 | [88.790, 93.335] |
| ManyRx | S10_ask | CRD | 88.7697 | 86.5035 | 65.2028 | [86.222, 90.997] |
| ManyRx | WaveMLP | CR | 90.8329 | 88.8491 | 70.1833 | [88.462, 93.021] |
| ManyRx | WaveMLP | CRD | 88.0562 | 85.6157 | 68.9594 | [85.855, 90.136] |
| ManySig | IB-RFF | CR | 91.1713 | 90.7645 | 79.0278 | [88.005, 93.965] |
| ManySig | IB-RFF | CRD | 91.0926 | 90.8187 | 76.7778 | [87.616, 94.102] |
| ManySig | S08_hard | CR | 94.1343 | 94.0030 | 87.7778 | [91.972, 96.065] |
| ManySig | S08_hard | CRD | 93.0116 | 92.7970 | 84.6667 | [90.090, 95.713] |
| ManySig | S09_hard_packet | CR | 93.7917 | 93.5528 | 84.1389 | [90.861, 96.319] |
| ManySig | S09_hard_packet | CRD | 92.8958 | 92.4762 | 83.7222 | [89.870, 95.644] |
| ManySig | S10_ask | CR | 94.4282 | 94.2713 | 91.3611 | [93.273, 95.625] |
| ManySig | S10_ask | CRD | 93.4838 | 93.1502 | 82.9167 | [91.007, 95.532] |
| ManySig | WaveMLP | CR | 92.7269 | 92.4750 | 83.6667 | [90.532, 94.606] |
| ManySig | WaveMLP | CRD | 91.0370 | 90.4086 | 78.7500 | [87.863, 93.787] |

Seed/fold dispersion: `final_run_metrics.csv`, `final_comparisons.csv`.
Paired receiver differences: `receiver_cluster_intervals.csv`.
Seeds for ASKNet2023-2025 differ from our seeds. Intervals pair RX, not training randomness.
Published/offered upstream logs are not mixed with fair multi-seed accuracy.
A larger mean does not prove stable superiority when the RX difference interval overlaps zero.
This bounded baseline set is not an exhaustive literature leaderboard; no global SOTA claim.
Shen97.48% remains historical evidence under its own LoRa frontend/protocol; no new Shen fit was made.
