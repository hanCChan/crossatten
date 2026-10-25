# Paper Validation Results

Evidence updated: 2026-10-25T19:34:46+08:00. Registration: `bc30820`.
This report includes only finished, audited fits. Smoke results are not accuracy evidence.
**Status: in progress**; audited fits: 194.

## Completion

| Stage | Finished | Missing |
|---|---:|---:|
| source | 163 | 13 |
| explore | 31 | 69 |
| final | 0 | awaiting source freeze |
| confirm | 0 | awaiting source freeze |

Artifact audit issues: 0; recorded failure/interruption entries: 10.
A failed execution is not discarded; a subsequent successful retry may complete its registered fit.

## Final Comparisons

All values are percent. RX accuracy SD across fits combines fold and seed variation.
Seed SD is computed from fold-averaged seed results; folds and packets are not extra independent RX.
Incomplete rows are provisional and cannot establish improvement.

| Dataset | Model | Task | Fits | RX Acc | Packet Acc | RX F1 | Seed SD | Fold SD | Worst RX | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Registered Ablations

| Dataset | Task | Pair | Fits | Delta pp | Positive | Worst RX Delta | Decision |
|---|---|---|---:|---:|---:|---:|---|
| ManyRx | CR | W1-W0 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W1-W0 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CR | W2-W1 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W2-W1 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CR | W1-C0 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W1-C0 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CR | W1-CP | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W1-CP | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CR | W1-CF | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W1-CF | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W1-W0 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W1-W0 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W2-W1 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W2-W1 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W1-C0 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W1-C0 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W1-CP | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W1-CP | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W1-CF | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W1-CF | 0 | pending | -/0 | pending | incomplete |
| Shen | dev_val | S_hstem-S_base | 0 | pending | -/0 | pending | incomplete |
| Shen | confirm | S_hstem-S_base | 0 | pending | -/0 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnonly | 0 | pending | -/0 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnonly | 0 | pending | -/0 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnparam | 0 | pending | -/0 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnparam | 0 | pending | -/0 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnflops | 0 | pending | -/0 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnflops | 0 | pending | -/0 | pending | incomplete |

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
| ManyRx | IB-RFF_a0.04_ManyRx_official | 4 | 0.2034 | 93.52 | 87.47 | - |
| ManyRx | W0_ManyRx_official | 4 | 0.1635 | 94.77 | 84.98 | - |
| ManyRx | W0_ManyRx_unified | 4 | 0.1138 | 96.47 | 58.53 | - |
| ManyRx | W1_ManyRx_official | 4 | 0.1503 | 95.10 | 86.44 | - |
| ManyRx | W1_ManyRx_unified | 4 | 0.1143 | 96.40 | 56.17 | - |
| ManyRx | W2_ManyRx_official | 4 | 0.1537 | 95.05 | 84.35 | - |
| ManyRx | W2_ManyRx_unified | 4 | 0.1172 | 96.27 | 66.51 | - |
| ManyRx | WaveMLP_alternate_ManyRx_official | 4 | 0.1904 | 94.58 | 89.43 | - |
| ManyRx | WaveMLP_alternate_ManyRx_unified | 2 | 0.1557 | 95.66 | 86.88 | - |
| ManyRx | WaveMLP_default_ManyRx_official | 4 | 0.1999 | 94.23 | 88.80 | - |
| ManyRx | WaveMLP_default_ManyRx_unified | 1 | 0.2310 | 93.09 | 88.18 | - |
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
| Shen | FAN_full_unified | 3 | 0.1704 | 93.83 | - | 83.30 |
| Shen | FAN_ib_official | 3 | 0.4509 | 80.75 | - | 79.16 |
| Shen | FAN_ib_unified | 3 | 0.2052 | 92.50 | - | 90.69 |
| Shen | IB-RFF_a0.004_full_official | 1 | 0.3173 | 89.75 | - | 75.93 |
| Shen | IB-RFF_a0.004_ib_official | 1 | 0.3402 | 85.65 | - | 85.00 |
| Shen | IB-RFF_a0.04_full_official | 1 | 0.3164 | 89.65 | - | 75.99 |
| Shen | IB-RFF_a0.04_ib_official | 1 | 0.3396 | 85.80 | - | 85.03 |
| Shen | S_base | 1 | 0.1068 | 96.85 | - | 94.49 |
| Shen | S_cnnflops | 2 | 0.3096 | 88.10 | - | 86.72 |
| Shen | S_cnnonly | 2 | 0.1008 | 96.95 | - | 93.17 |
| Shen | S_cnnparam | 1 | 0.3562 | 86.60 | - | 85.60 |
| Shen | S_hstem | 2 | 0.0581 | 97.98 | - | 97.08 |
| Shen | WaveMLP_absolute_full_official | 1 | 1.0902 | 52.15 | - | 57.56 |
| Shen | WaveMLP_absolute_ib_official | 2 | 1.0055 | 56.13 | - | 54.05 |
| Shen | WaveMLP_scaled_full_official | 1 | 0.3726 | 86.45 | - | 65.20 |
| Shen | WaveMLP_scaled_ib_official | 1 | 1.0567 | 55.55 | - | 54.72 |
| Shen | WaveMLP_scaled_ib_unified | 1 | 0.1993 | 94.25 | - | 93.02 |

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
