# S08 Source-Only Diagnostics

S07 source training artifacts only; no historical CR/CRD reads or target-derived decisions. Two disjoint source-held groups are pre-registered by RX-list order, not target weakness.

| Dataset | Factor | Runs | Median selected epoch | Mean held RX % |
|---|---|---:|---:|---:|
| ManySig | official | 8 | 12.5 | 87.61 |
| ManySig | unified | 8 | 76.5 | 77.20 |
| ManySig | mild_noise | 4 | 87.0 | 78.93 |
| ManySig | held_epoch | 7 | 5.0 | 82.97 |
| ManyRx | official | 8 | 5.5 | 89.05 |
| ManyRx | unified | 8 | 28.0 | 80.53 |
| ManyRx | mild_noise | 7 | 16.0 | 88.87 |
| ManyRx | held_epoch | 5 | 5.0 | 86.75 |

These historical rows mix selected frontends (ManySig fold3 uses split) and selective repeat coverage. They are descriptive, not paired frequency-only ablations. Raw minimum packet-CE frequently occurs early; that does not establish that EMA or longer training generalizes better.

## Spectral Audit

Each fold uses its own source RX days1-2, first four packets/class/RX, the frozen normalization and three complex-spectrum channels. Reliability rule is amplitude/RMS>=0.05 at both adjacent bins; no wrap-around pair.

| Dataset | Fold | Low-energy bins | Median reliable pairs | Re/Im/Log RMS |
|---|---:|---:|---:|---|
| ManySig | 0 | 0.1023 | 0.8706 | 0.704/0.710/0.998 |
| ManySig | 1 | 0.0902 | 0.8706 | 0.711/0.703/0.998 |
| ManySig | 2 | 0.1055 | 0.8667 | 0.710/0.704/0.998 |
| ManySig | 3 | 0.1021 | 0.8706 | 0.706/0.708/0.998 |
| ManyRx | 0 | 0.1163 | 0.8471 | 0.707/0.707/0.998 |
| ManyRx | 1 | 0.1055 | 0.8627 | 0.705/0.709/0.998 |
| ManyRx | 2 | 0.1185 | 0.8471 | 0.706/0.708/0.998 |
| ManyRx | 3 | 0.1230 | 0.8392 | 0.706/0.708/0.998 |

The mask is a conservative fixed numerical/energy threshold, not a noise-floor estimate or proof of receiver invariance. Phase features are conditional on training gate failure and median reliable fraction>=0.5 in all four ManySig source folds.

## Gate Scale

Frozen S07 frequency-only unified feature-fit probes remain near-unit token RMS with some abs(score)>6 saturation. They are not the new official-training checkpoints. Saturation alone does not prove causal failure; no gate redesign is authorized.

Exact rows: `historical_source_diagnostics.csv`, `source_spectral_diagnostics.csv`, `historical_source_gates.csv`. Diagnostic inputs are not used to rank new candidates.
