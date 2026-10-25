# Paper Validation Results

Evidence updated: 2026-10-25T20:55:43+08:00. Registration: `bc30820`.
This report includes only finished, audited fits. Smoke results are not accuracy evidence.
**Status: in progress**; audited fits: 554.

## Completion

| Stage | Finished | Missing |
|---|---:|---:|
| source | 176 | 0 |
| explore | 100 | 0 |
| final | 238 | 2 |
| confirm | 40 | 32 |

Artifact audit issues: 0; recorded failure/interruption entries: 10.
A failed execution is not discarded; a subsequent successful retry may complete its registered fit.
Saved finite checkpoints checked: 554; prediction/confusion audit: 7488778 packets, 3804 stored-logit ties.
Accuracy uses original float32 logits. Stored float16 logits may merge top scores; all untied decisions and tie-compatible confusion counts are verified.

## Final Comparisons

All values are percent. RX accuracy SD across fits combines fold and seed variation.
Seed SD is computed from fold-averaged seed results; folds and packets are not extra independent RX.
Incomplete rows are provisional and cannot establish improvement.

| Dataset | Model | Task | Fits | RX Acc | Packet Acc | RX F1 | Seed SD | Fold SD | Worst RX | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ManyRx | C0 | CR | 12 | 84.57 +/- 5.61 | 84.59 | 82.46 | 1.00 | 6.05 | 68.03 | 48890 |
| ManyRx | C0 | CRD | 12 | 79.58 +/- 5.22 | 79.60 | 76.69 | 1.73 | 5.15 | 57.72 | 48890 |
| ManyRx | CF | CR | 12 | 79.46 +/- 7.84 | 79.44 | 76.76 | 0.69 | 8.57 | 49.98 | 52718 |
| ManyRx | CF | CRD | 12 | 74.97 +/- 6.94 | 74.98 | 71.82 | 0.50 | 7.58 | 49.90 | 52718 |
| ManyRx | CP | CR | 12 | 78.20 +/- 7.98 | 78.19 | 75.42 | 0.48 | 8.73 | 51.25 | 91716 |
| ManyRx | CP | CRD | 12 | 73.22 +/- 5.99 | 73.23 | 69.67 | 0.12 | 6.56 | 50.73 | 91716 |
| ManyRx | FAN_official | CR | 12 | 90.11 +/- 3.04 | 90.12 | 88.03 | 0.81 | 3.20 | 70.90 | 151761 |
| ManyRx | FAN_official | CRD | 12 | 88.26 +/- 3.29 | 88.26 | 86.03 | 0.27 | 3.31 | 65.43 | 151761 |
| ManyRx | FAN_source_selected | CR | 12 | 90.19 +/- 3.05 | 90.21 | 88.11 | 0.34 | 3.26 | 70.90 | 151761 |
| ManyRx | FAN_source_selected | CRD | 12 | 87.56 +/- 3.55 | 87.55 | 85.26 | 0.80 | 3.58 | 65.43 | 151761 |
| ManyRx | FAN_unified | CR | 12 | 89.13 +/- 3.29 | 89.17 | 86.94 | 0.23 | 3.55 | 71.80 | 151761 |
| ManyRx | FAN_unified | CRD | 12 | 86.39 +/- 3.30 | 86.39 | 83.84 | 0.54 | 3.19 | 63.74 | 151761 |
| ManyRx | IB-RFF | CR | 10 | 89.96 +/- 3.14 | 89.99 | 87.82 | 0.88 | 3.11 | 69.90 | 151761 |
| ManyRx | IB-RFF | CRD | 10 | 88.20 +/- 3.12 | 88.20 | 86.05 | 1.08 | 3.50 | 67.18 | 151761 |
| ManyRx | W0 | CR | 12 | 85.76 +/- 5.77 | 85.76 | 83.60 | 0.80 | 6.18 | 64.72 | 72232 |
| ManyRx | W0 | CRD | 12 | 83.19 +/- 4.82 | 83.19 | 80.86 | 0.79 | 5.05 | 62.20 | 72232 |
| ManyRx | W1 | CR | 12 | 86.74 +/- 6.41 | 86.75 | 84.57 | 0.52 | 6.95 | 64.17 | 91128 |
| ManyRx | W1 | CRD | 12 | 83.56 +/- 6.38 | 83.58 | 81.14 | 1.87 | 6.63 | 62.89 | 91128 |
| ManyRx | W2 | CR | 12 | 85.51 +/- 6.67 | 85.51 | 83.20 | 1.18 | 6.93 | 66.02 | 91128 |
| ManyRx | W2 | CRD | 12 | 80.55 +/- 5.69 | 80.54 | 77.44 | 0.99 | 5.66 | 60.59 | 91128 |
| ManyRx | WaveMLP | CR | 12 | 90.83 +/- 3.17 | 90.86 | 88.85 | 0.23 | 3.47 | 70.18 | 84490,85258 |
| ManyRx | WaveMLP | CRD | 12 | 88.06 +/- 2.27 | 88.05 | 85.62 | 0.35 | 2.40 | 68.96 | 84490,85258 |
| ManySig | C0 | CR | 12 | 72.10 +/- 9.68 | 72.10 | 70.44 | 1.48 | 9.26 | 46.14 | 48630 |
| ManySig | C0 | CRD | 12 | 68.96 +/- 9.54 | 68.96 | 66.95 | 0.52 | 9.17 | 39.58 | 48630 |
| ManySig | CF | CR | 12 | 67.52 +/- 8.45 | 67.52 | 65.70 | 1.63 | 9.08 | 33.78 | 52458 |
| ManySig | CF | CRD | 12 | 63.96 +/- 8.65 | 63.96 | 61.46 | 1.61 | 9.19 | 28.69 | 52458 |
| ManySig | CP | CR | 12 | 65.86 +/- 8.87 | 65.86 | 63.31 | 0.24 | 9.44 | 35.11 | 91456 |
| ManySig | CP | CRD | 12 | 61.28 +/- 9.24 | 61.28 | 58.33 | 0.45 | 9.65 | 28.72 | 91456 |
| ManySig | FAN_official | CR | 12 | 88.70 +/- 6.18 | 88.70 | 88.08 | 1.98 | 5.31 | 73.25 | 146605 |
| ManySig | FAN_official | CRD | 12 | 88.01 +/- 5.01 | 88.01 | 87.44 | 2.31 | 3.91 | 70.50 | 146605 |
| ManySig | FAN_source_selected | CR | 12 | 89.22 +/- 6.67 | 89.22 | 88.61 | 2.24 | 5.93 | 73.25 | 146605 |
| ManySig | FAN_source_selected | CRD | 12 | 88.27 +/- 5.24 | 88.27 | 87.76 | 2.56 | 4.28 | 70.50 | 146605 |
| ManySig | FAN_unified | CR | 12 | 89.70 +/- 3.96 | 89.70 | 89.08 | 0.84 | 3.83 | 81.58 | 146605 |
| ManySig | FAN_unified | CRD | 12 | 87.81 +/- 4.27 | 87.81 | 87.22 | 1.26 | 4.19 | 73.89 | 146605 |
| ManySig | IB-RFF | CR | 12 | 91.17 +/- 5.96 | 91.17 | 90.76 | 2.91 | 3.84 | 79.03 | 146605 |
| ManySig | IB-RFF | CRD | 12 | 91.09 +/- 5.02 | 91.09 | 90.82 | 1.69 | 2.76 | 76.78 | 146605 |
| ManySig | W0 | CR | 12 | 72.97 +/- 9.72 | 72.97 | 71.35 | 2.53 | 10.03 | 49.83 | 71972 |
| ManySig | W0 | CRD | 12 | 69.70 +/- 11.17 | 69.70 | 67.21 | 2.87 | 11.77 | 46.94 | 71972 |
| ManySig | W1 | CR | 12 | 75.90 +/- 8.63 | 75.90 | 74.03 | 0.81 | 9.28 | 49.39 | 90868 |
| ManySig | W1 | CRD | 12 | 72.33 +/- 8.22 | 72.33 | 69.99 | 0.82 | 8.70 | 53.14 | 90868 |
| ManySig | W2 | CR | 12 | 70.00 +/- 10.15 | 70.00 | 68.20 | 2.47 | 10.17 | 48.19 | 90868 |
| ManySig | W2 | CRD | 12 | 66.50 +/- 10.28 | 66.50 | 64.25 | 2.33 | 10.53 | 32.39 | 90868 |
| ManySig | WaveMLP | CR | 12 | 92.73 +/- 2.74 | 92.73 | 92.47 | 0.33 | 2.87 | 83.67 | 83974,84742 |
| ManySig | WaveMLP | CRD | 12 | 91.04 +/- 3.57 | 91.04 | 90.41 | 1.15 | 3.59 | 78.75 | 83974,84742 |
| Shen | FAN_full_official | dev_val | 6 | 73.62 +/- 3.12 | 73.62 | 72.60 | 2.13 | 2.18 | 64.63 | 310481 |
| Shen | FAN_full_official | confirm | 6 | 75.50 +/- 4.52 | 75.50 | 74.51 | 4.87 | 0.82 | 60.20 | 310481 |
| Shen | FAN_full_unified | dev_val | 4 | 83.44 +/- 1.82 | 83.44 | 82.23 | 1.31 | 1.04 | 74.72 | 310481 |
| Shen | FAN_full_unified | confirm | 4 | 83.93 +/- 2.58 | 83.93 | 83.08 | 0.15 | 3.14 | 72.50 | 310481 |
| Shen | FAN_ib_unified | dev_val | 5 | 89.56 +/- 4.02 | 89.56 | 89.26 | 1.48 | 0.75 | 87.48 | 310481 |
| Shen | FAN_ib_unified | confirm | 5 | 86.58 +/- 3.55 | 86.58 | 86.29 | 0.93 | 0.47 | 73.90 | 310481 |
| Shen | IB-RFF_a0.004_ib_unified | dev_val | 1 | 93.56 +/- 0.00 | 93.56 | 93.52 | 0.00 | 0.00 | 90.65 | 310481 |
| Shen | IB-RFF_a0.004_ib_unified | confirm | 1 | 90.90 +/- 0.00 | 90.90 | 90.72 | 0.00 | 0.00 | 81.20 | 310481 |
| Shen | IB-RFF_a0.04_full_official | dev_val | 6 | 73.55 +/- 3.84 | 73.55 | 72.57 | 2.93 | 2.31 | 64.38 | 310481 |
| Shen | IB-RFF_a0.04_full_official | confirm | 6 | 76.30 +/- 4.57 | 76.30 | 75.39 | 4.43 | 0.16 | 59.85 | 310481 |
| Shen | S_base | dev_val | 3 | 95.19 +/- 1.00 | 95.19 | 95.16 | 0.74 | 1.21 | 92.38 | 580136 |
| Shen | S_base | confirm | 3 | 93.67 +/- 0.55 | 93.67 | 93.54 | 0.28 | 0.39 | 82.80 | 580136 |
| Shen | S_cnnflops | dev_val | 3 | 86.26 +/- 0.56 | 86.26 | 85.91 | 0.07 | 0.56 | 77.50 | 655358 |
| Shen | S_cnnflops | confirm | 3 | 84.68 +/- 3.48 | 84.68 | 84.48 | 2.00 | 4.26 | 65.50 | 655358 |
| Shen | S_cnnonly | dev_val | 3 | 94.01 +/- 0.83 | 94.01 | 93.86 | 0.44 | 1.01 | 82.35 | 100154 |
| Shen | S_cnnonly | confirm | 3 | 93.05 +/- 1.29 | 93.05 | 92.90 | 0.80 | 1.58 | 82.10 | 100154 |
| Shen | S_cnnparam | dev_val | 4 | 86.10 +/- 0.93 | 86.10 | 85.95 | 0.69 | 0.64 | 78.53 | 138898 |
| Shen | S_cnnparam | confirm | 4 | 83.50 +/- 1.03 | 83.50 | 83.36 | 0.86 | 0.87 | 66.58 | 138898 |
| Shen | S_hstem | dev_val | 3 | 97.25 +/- 0.46 | 97.25 | 97.25 | 0.48 | 0.48 | 94.80 | 142392 |
| Shen | S_hstem | confirm | 3 | 97.07 +/- 0.26 | 97.07 | 96.98 | 0.30 | 0.04 | 91.38 | 142392 |
| Shen | WaveMLP_scaled_full_official | dev_val | 2 | 34.75 +/- 34.68 | 34.75 | 32.22 | 34.68 | 34.68 | 10.00 | 594442 |
| Shen | WaveMLP_scaled_full_official | confirm | 2 | 34.22 +/- 34.09 | 34.22 | 30.88 | 34.09 | 34.09 | 9.65 | 594442 |

## Training Cost

Observed wall-clock training under concurrent scheduling; not isolated speed benchmarks.
IB-RFF total includes its matching FAN initialization and HSIC fine-tuning. Model-selection search is excluded.

| Dataset | Model | Fits | This Stage s | Initialization s | Total s |
|---|---|---:|---:|---:|---:|
| ManyRx | C0 | 12 | 52.5 | 0.0 | 52.5 |
| ManyRx | CF | 12 | 62.5 | 0.0 | 62.5 |
| ManyRx | CP | 12 | 65.9 | 0.0 | 65.9 |
| ManyRx | FAN_official | 12 | 129.8 | 0.0 | 129.8 |
| ManyRx | FAN_source_selected | 12 | 367.2 | 0.0 | 367.2 |
| ManyRx | FAN_unified | 12 | 1083.8 | 0.0 | 1083.8 |
| ManyRx | IB-RFF | 10 | 279.8 | 240.2 | 520.0 |
| ManyRx | W0 | 12 | 122.0 | 0.0 | 122.0 |
| ManyRx | W1 | 12 | 102.6 | 0.0 | 102.6 |
| ManyRx | W2 | 12 | 119.3 | 0.0 | 119.3 |
| ManyRx | WaveMLP | 12 | 564.5 | 0.0 | 564.5 |
| ManySig | C0 | 12 | 72.4 | 0.0 | 72.4 |
| ManySig | CF | 12 | 61.3 | 0.0 | 61.3 |
| ManySig | CP | 12 | 61.4 | 0.0 | 61.4 |
| ManySig | FAN_official | 12 | 26.0 | 0.0 | 26.0 |
| ManySig | FAN_source_selected | 12 | 84.2 | 0.0 | 84.2 |
| ManySig | FAN_unified | 12 | 257.2 | 0.0 | 257.2 |
| ManySig | IB-RFF | 12 | 134.9 | 84.2 | 219.0 |
| ManySig | W0 | 12 | 123.3 | 0.0 | 123.3 |
| ManySig | W1 | 12 | 131.8 | 0.0 | 131.8 |
| ManySig | W2 | 12 | 142.0 | 0.0 | 142.0 |
| ManySig | WaveMLP | 12 | 96.5 | 0.0 | 96.5 |
| Shen | FAN_full_official | 6 | 65.1 | 0.0 | 65.1 |
| Shen | FAN_full_unified | 4 | 475.4 | 0.0 | 475.4 |
| Shen | FAN_ib_unified | 5 | 478.5 | 0.0 | 478.5 |
| Shen | IB-RFF_a0.004_ib_unified | 1 | 716.4 | 581.9 | 1298.3 |
| Shen | IB-RFF_a0.04_full_official | 6 | 47.5 | 65.1 | 112.6 |
| Shen | S_base | 3 | 777.8 | 0.0 | 777.8 |
| Shen | S_cnnflops | 3 | 604.6 | 0.0 | 604.6 |
| Shen | S_cnnonly | 3 | 723.9 | 0.0 | 723.9 |
| Shen | S_cnnparam | 4 | 526.2 | 0.0 | 526.2 |
| Shen | S_hstem | 3 | 944.6 | 0.0 | 944.6 |
| Shen | WaveMLP_scaled_full_official | 2 | 647.9 | 0.0 | 647.9 |

## Registered Ablations

| Dataset | Task | Pair | Fits | Delta pp | Positive | Worst RX Delta | Decision |
|---|---|---|---:|---:|---:|---:|---|
| ManyRx | CR | W1-W0 | 12 | +0.99 | 11/12 | -3.55 | unclear |
| ManyRx | CRD | W1-W0 | 12 | +0.37 | 7/12 | -5.95 | unclear |
| ManyRx | CR | W2-W1 | 12 | -1.23 | 4/12 | -8.18 | unclear |
| ManyRx | CRD | W2-W1 | 12 | -3.02 | 3/12 | -11.78 | unclear |
| ManyRx | CR | W1-C0 | 12 | +2.17 | 10/12 | -7.00 | unclear |
| ManyRx | CRD | W1-C0 | 12 | +3.98 | 10/12 | -4.33 | unclear |
| ManyRx | CR | W1-CP | 12 | +8.55 | 12/12 | -0.73 | improves |
| ManyRx | CRD | W1-CP | 12 | +10.34 | 12/12 | -4.03 | unclear |
| ManyRx | CR | W1-CF | 12 | +7.29 | 12/12 | -4.52 | unclear |
| ManyRx | CRD | W1-CF | 12 | +8.59 | 12/12 | -5.83 | unclear |
| ManySig | CR | W1-W0 | 12 | +2.93 | 8/12 | -12.00 | unclear |
| ManySig | CRD | W1-W0 | 12 | +2.63 | 8/12 | -7.42 | unclear |
| ManySig | CR | W2-W1 | 12 | -5.90 | 2/12 | -26.53 | harms |
| ManySig | CRD | W2-W1 | 12 | -5.83 | 2/12 | -32.94 | harms |
| ManySig | CR | W1-C0 | 12 | +3.80 | 11/12 | -0.83 | improves |
| ManySig | CRD | W1-C0 | 12 | +3.37 | 8/12 | -5.94 | unclear |
| ManySig | CR | W1-CP | 12 | +10.04 | 11/12 | -4.28 | unclear |
| ManySig | CRD | W1-CP | 12 | +11.05 | 12/12 | -3.75 | unclear |
| ManySig | CR | W1-CF | 12 | +8.37 | 10/12 | -2.72 | unclear |
| ManySig | CRD | W1-CF | 12 | +8.37 | 11/12 | -4.19 | unclear |
| Shen | dev_val | S_hstem-S_base | 3 | pending | -/3 | pending | incomplete |
| Shen | confirm | S_hstem-S_base | 3 | pending | -/3 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnonly | 3 | pending | -/3 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnonly | 3 | pending | -/3 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnparam | 3 | pending | -/3 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnparam | 3 | pending | -/3 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnflops | 3 | pending | -/3 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnflops | 3 | pending | -/3 | pending | incomplete |

## Measured Cost

Measurements precede formal training on an otherwise idle A100. Random weights, FP32, device IQ to logits.
Batch latency excludes host transfer. Repeated timing samples measure system variability, not accuracy seeds.

| Architecture | Params | MFLOPs/packet | GPU b1 p50 ms | p95 ms | b512 packets/s | CPU4t p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| W0_ManyRx_official | 72232 | 7.7 | 2.015 | 2.836 | 242215 | 1.59 |
| W1_ManyRx_official | 91128 | 10.8 | 2.295 | 3.059 | 200982 | 1.87 |
| W2_ManyRx_official | 91128 | 10.8 | 2.494 | 3.446 | 185488 | 2.02 |
| C0_ManyRx_official | 48890 | 7.7 | 1.567 | 1.9 | 311151 | 1.26 |
| CP_ManyRx_official | 91716 | 23.6 | 1.329 | 1.757 | 247952 | 1.31 |
| CF_ManyRx_official | 52718 | 10.8 | 1.26 | 1.456 | 303505 | 1.02 |
| FAN_ManyRx_official | 151761 | 10.6 | 1.664 | 2.595 | 331868 | 1.83 |
| WaveMLP_default_ManyRx_official | 85258 | 1.4 | 5.521 | 7.296 | 75297 | 3.32 |
| WaveMLP_alternate_ManyRx_official | 84490 | 1.3 | 4.34 | 4.413 | 116966 | 2.7 |
| W0_ManySig_official | 71972 | 7.7 | 1.949 | 1.984 | 245934 | 1.46 |
| W1_ManySig_official | 90868 | 10.8 | 2.209 | 2.262 | 210251 | 1.8 |
| W2_ManySig_official | 90868 | 10.8 | 2.42 | 2.467 | 190376 | 1.93 |
| C0_ManySig_official | 48630 | 7.7 | 1.549 | 1.576 | 316597 | 1.22 |
| CP_ManySig_official | 91456 | 23.6 | 1.313 | 1.34 | 250440 | 1.33 |
| CF_ManySig_official | 52458 | 10.8 | 1.261 | 1.358 | 307799 | 0.99 |
| FAN_ManySig_official | 146605 | 4.7 | 1.496 | 1.969 | 333587 | 1.37 |
| WaveMLP_default_ManySig_official | 83974 | 1.3 | 4.412 | 6.327 | 114524 | 2.69 |
| WaveMLP_alternate_ManySig_official | 84742 | 1.4 | 5.291 | 5.775 | 96140 | 3.4 |
| FAN_full_official | 310481 | 18.5 | 1.711 | 2.842 | 204657 | 2.66 |
| WaveMLP_absolute_full_official | 85258 | 49.9 | 5.897 | 8.055 | 45503 | 14.17 |
| WaveMLP_scaled_full_official | 594442 | 16.6 | 10.789 | 14.273 | 34282 | 16.93 |
| FAN_ib_official | 310481 | 18.5 | 1.868 | 1.955 | 176435 | 2.57 |
| WaveMLP_absolute_ib_official | 85258 | 49.9 | 5.777 | 6.001 | 44375 | 12.74 |
| WaveMLP_scaled_ib_official | 594442 | 16.6 | 10.276 | 10.667 | 35276 | 14.05 |
| S_base | 580136 | 139.2 | 2.164 | 2.237 | 42602 | 4.34 |
| S_hstem | 142392 | 190.6 | 2.533 | 3.806 | 35867 | 5.02 |
| S_cnnonly | 100154 | 187.4 | 1.838 | 1.927 | 37895 | 4.88 |
| S_cnnparam | 138898 | 18.5 | 1.889 | 3.19 | 97639 | 2.6 |
| S_cnnflops | 655358 | 188.1 | 1.55 | 2.058 | 31735 | 6.73 |

## Source And Exploration

All candidate configurations are retained below. Shen dev_val is report-only; selection uses source CE.
Source RX WiSig scores use only internally held-out source receivers on days 1-2 and are not CRD test scores.

| Dataset | Arm | Fits | Source CE | Source Packet Acc | Internal RX Acc | Shen dev_val |
|---|---|---:|---:|---:|---:|---:|
| ManyRx | C0_ManyRx_official | 4 | 0.1900 | 94.82 | 83.76 | - |
| ManyRx | C0_ManyRx_unified | 4 | 0.1429 | 95.83 | 71.05 | - |
| ManyRx | CF_ManyRx_official | 4 | 0.2033 | 94.63 | 78.25 | - |
| ManyRx | CF_ManyRx_unified | 4 | 0.1639 | 95.46 | 66.12 | - |
| ManyRx | CP_ManyRx_official | 4 | 0.1914 | 95.08 | 76.14 | - |
| ManyRx | CP_ManyRx_unified | 4 | 0.1614 | 95.58 | 65.28 | - |
| ManyRx | FAN_ManyRx_official | 4 | 0.2125 | 94.16 | 87.83 | - |
| ManyRx | FAN_ManyRx_unified | 4 | 0.2037 | 94.56 | 87.55 | - |
| ManyRx | IB-RFF_a0.004_ManyRx_official | 4 | 0.1998 | 94.42 | 88.32 | - |
| ManyRx | IB-RFF_a0.004_ManyRx_unified | 4 | 0.1974 | 94.68 | 87.37 | - |
| ManyRx | IB-RFF_a0.04_ManyRx_official | 4 | 0.2034 | 93.52 | 87.47 | - |
| ManyRx | IB-RFF_a0.04_ManyRx_unified | 4 | 0.1748 | 94.64 | 88.08 | - |
| ManyRx | W0_ManyRx_official | 4 | 0.1635 | 94.77 | 84.98 | - |
| ManyRx | W0_ManyRx_unified | 4 | 0.1138 | 96.47 | 58.53 | - |
| ManyRx | W1_ManyRx_official | 4 | 0.1503 | 95.10 | 86.44 | - |
| ManyRx | W1_ManyRx_unified | 4 | 0.1143 | 96.40 | 56.17 | - |
| ManyRx | W2_ManyRx_official | 4 | 0.1537 | 95.05 | 84.35 | - |
| ManyRx | W2_ManyRx_unified | 4 | 0.1172 | 96.27 | 66.51 | - |
| ManyRx | WaveMLP_alternate_ManyRx_official | 4 | 0.1904 | 94.58 | 89.43 | - |
| ManyRx | WaveMLP_alternate_ManyRx_unified | 4 | 0.1810 | 94.76 | 86.77 | - |
| ManyRx | WaveMLP_default_ManyRx_official | 4 | 0.1999 | 94.23 | 88.80 | - |
| ManyRx | WaveMLP_default_ManyRx_unified | 4 | 0.1825 | 94.52 | 87.24 | - |
| ManySig | C0_ManySig_official | 4 | 0.0375 | 99.03 | 71.85 | - |
| ManySig | C0_ManySig_unified | 4 | 0.0255 | 99.52 | 59.10 | - |
| ManySig | CF_ManySig_official | 4 | 0.0329 | 99.15 | 69.59 | - |
| ManySig | CF_ManySig_unified | 4 | 0.0319 | 99.49 | 51.97 | - |
| ManySig | CP_ManySig_official | 4 | 0.0314 | 99.32 | 65.45 | - |
| ManySig | CP_ManySig_unified | 4 | 0.0330 | 99.57 | 52.15 | - |
| ManySig | FAN_ManySig_official | 4 | 0.1386 | 97.51 | 91.46 | - |
| ManySig | FAN_ManySig_unified | 4 | 0.1031 | 98.03 | 91.69 | - |
| ManySig | IB-RFF_a0.004_ManySig_official | 4 | 0.1361 | 97.51 | 91.95 | - |
| ManySig | IB-RFF_a0.004_ManySig_unified | 4 | 0.0993 | 98.15 | 91.70 | - |
| ManySig | IB-RFF_a0.04_ManySig_official | 4 | 0.0863 | 97.48 | 92.91 | - |
| ManySig | IB-RFF_a0.04_ManySig_unified | 4 | 0.0663 | 98.08 | 90.61 | - |
| ManySig | W0_ManySig_official | 4 | 0.0500 | 98.81 | 73.70 | - |
| ManySig | W0_ManySig_unified | 4 | 0.0221 | 99.54 | 59.26 | - |
| ManySig | W1_ManySig_official | 4 | 0.0547 | 98.72 | 80.49 | - |
| ManySig | W1_ManySig_unified | 4 | 0.0226 | 99.55 | 69.99 | - |
| ManySig | W2_ManySig_official | 4 | 0.0561 | 98.68 | 73.01 | - |
| ManySig | W2_ManySig_unified | 4 | 0.0237 | 99.45 | 67.12 | - |
| ManySig | WaveMLP_alternate_ManySig_official | 4 | 0.0716 | 98.49 | 95.57 | - |
| ManySig | WaveMLP_alternate_ManySig_unified | 4 | 0.0529 | 98.59 | 88.70 | - |
| ManySig | WaveMLP_default_ManySig_official | 4 | 0.0769 | 98.39 | 94.92 | - |
| ManySig | WaveMLP_default_ManySig_unified | 4 | 0.0551 | 98.66 | 89.77 | - |
| Shen | FAN_full_official | 4 | 0.3060 | 89.91 | - | 75.08 |
| Shen | FAN_full_unified | 4 | 0.1763 | 93.56 | - | 81.81 |
| Shen | FAN_ib_official | 4 | 0.4201 | 82.13 | - | 81.00 |
| Shen | FAN_ib_unified | 4 | 0.2252 | 91.45 | - | 89.90 |
| Shen | IB-RFF_a0.004_full_official | 4 | 0.3004 | 90.53 | - | 75.10 |
| Shen | IB-RFF_a0.004_full_unified | 4 | 0.1390 | 95.05 | - | 82.93 |
| Shen | IB-RFF_a0.004_ib_official | 4 | 0.3768 | 84.53 | - | 83.41 |
| Shen | IB-RFF_a0.004_ib_unified | 4 | 0.1239 | 95.69 | - | 94.30 |
| Shen | IB-RFF_a0.04_full_official | 4 | 0.2991 | 90.59 | - | 75.05 |
| Shen | IB-RFF_a0.04_full_unified | 4 | 0.1374 | 95.08 | - | 82.88 |
| Shen | IB-RFF_a0.04_ib_official | 4 | 0.3768 | 84.54 | - | 83.44 |
| Shen | IB-RFF_a0.04_ib_unified | 4 | 0.1241 | 95.59 | - | 94.31 |
| Shen | S_base | 4 | 0.1143 | 96.65 | - | 95.30 |
| Shen | S_cnnflops | 4 | 0.3222 | 87.98 | - | 87.62 |
| Shen | S_cnnonly | 4 | 0.1180 | 96.34 | - | 94.05 |
| Shen | S_cnnparam | 4 | 0.3846 | 85.68 | - | 86.00 |
| Shen | S_hstem | 4 | 0.0620 | 97.99 | - | 97.26 |
| Shen | WaveMLP_absolute_full_official | 4 | 0.9487 | 58.70 | - | 61.20 |
| Shen | WaveMLP_absolute_full_unified | 4 | 0.6442 | 75.43 | - | 77.01 |
| Shen | WaveMLP_absolute_ib_official | 4 | 1.0840 | 51.78 | - | 52.24 |
| Shen | WaveMLP_absolute_ib_unified | 4 | 0.7599 | 70.05 | - | 72.36 |
| Shen | WaveMLP_scaled_full_official | 4 | 0.4127 | 85.19 | - | 71.06 |
| Shen | WaveMLP_scaled_full_unified | 4 | 0.2545 | 89.63 | - | 81.36 |
| Shen | WaveMLP_scaled_ib_official | 4 | 0.7567 | 68.80 | - | 70.25 |
| Shen | WaveMLP_scaled_ib_unified | 4 | 0.3746 | 85.89 | - | 86.93 |

## Paper Route

**pending**. Complete the registered matrix before choosing the paper scope.
A route based on completed comparisons is not a claim of sufficient novelty or submission readiness.
External IB-RFF uses two training stages (FAN initialization plus HSIC); both stages and their time are retained.

## Interpretation Limits

- CRD is the primary WiSig task. W0/W1/W2 remain separate; no target-based winner selection.
- Final configuration choices use only source RX days 1-2; source evidence and decisions are committed before target access.
- W2's fixed cutoff comes from the earlier fold-0 source PSD audit, not independent per-fold estimation. Those RX can be targets in other folds; W2 is not pristine per-fold source-only preprocessing evidence. W0/W1 do not use this cutoff.
- Shen confirm RX were inspected repeatedly and are report-only, not independent blind-test evidence.
- Published weights in `05-wisig/RESULTS.md` are historical references, not fair multiseed retraining.
- HSTU attribution requires W1 vs C0 and matched CNN controls; parameter efficiency does not imply faster inference.
- Cost timings use raw device-resident IQ to logits, not host transfer; FFT is timed but incompletely counted in FLOPs.
- FAN upstream CUDA median issues a nondeterminism warning; fixed seeds do not guarantee bit-identical upstream training.
- No new Loss, feature branch, architecture or target-selected hyperparameter is added after this registration.

Full per-fit metrics, configurations, source choices and failures: `summary/report.json`; costs: `summary/cost_*/bench.md`.
The bounded experiment matrix must finish before choosing a general RF, LoRa-only, or engineering-only paper route.
