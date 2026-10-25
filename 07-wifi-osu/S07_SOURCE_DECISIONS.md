# S07 Source-Only Decision Trace

All choices were frozen before final target evaluation. This explanation reads only registered source fits and final configuration declarations; no CR/CRD metrics.

## Training Factor Replication

Two positive seed-matched held-RX deltas, mean >=1 pp and worst individual held-RX delta >=-2 pp are required for combination support. A factor with the highest source rank can remain the single control even if it fails this stricter gate.

| Dataset | Fold | Factor | Delta seed3101 pp | Delta seed3102 pp | Mean pp | Worst RX delta pp | Supported |
|---|---:|---|---:|---:|---:|---:|---|
| ManyRx | 0 | mild_noise | +4.46 | +2.26 | +3.36 | +0.65 | True |
| ManyRx | 0 | official | +4.01 | +2.11 | +3.06 | -2.25 | False |
| ManyRx | 1 | held_epoch | +10.00 | +19.61 | +14.80 | +2.32 | True |
| ManyRx | 1 | official | +11.26 | +22.10 | +16.68 | +4.11 | True |
| ManyRx | 2 | mild_noise | +4.53 | +3.22 | +3.88 | -1.42 | True |
| ManyRx | 2 | official | +3.84 | +2.70 | +3.27 | -0.70 | True |
| ManyRx | 3 | mild_noise | +8.57 | +12.05 | +10.31 | -0.10 | True |
| ManyRx | 3 | official | +10.61 | +11.51 | +11.06 | +1.80 | True |
| ManySig | 0 | held_epoch | +2.04 | +9.54 | +5.79 | -4.00 | False |
| ManySig | 0 | official | +10.04 | +14.83 | +12.44 | +5.08 | True |
| ManySig | 1 | held_epoch | +6.54 | +8.50 | +7.52 | +0.67 | True |
| ManySig | 1 | official | +5.33 | +17.46 | +11.40 | +1.08 | True |
| ManySig | 2 | official | +1.75 | +8.00 | +4.88 | +1.50 | True |
| ManySig | 3 | held_epoch | +7.33 | +9.21 | +8.27 | +5.67 | True |
| ManySig | 3 | official | +15.38 | +10.54 | +12.96 | +10.42 | True |

## Frozen Pipelines

| Dataset | Fold | Frontend | Supported factors | Strongest single | Final training pipeline | Final combination pipeline |
|---|---:|---|---|---|---|---|
| ManyRx | 0 | frequency | mild_noise | mild_noise | unified; AWGN10-50 | unified; AWGN10-50 |
| ManyRx | 1 | frequency | official, held_epoch | official | official | official |
| ManyRx | 2 | frequency | mild_noise, official | mild_noise | unified; AWGN10-50 | unified; AWGN10-50 |
| ManyRx | 3 | frequency | official, mild_noise | official | official | official |
| ManySig | 0 | frequency | official | official | official | official |
| ManySig | 1 | frequency | official, held_epoch | official | official | official |
| ManySig | 2 | frequency | official | official | official | official |
| ManySig | 3 | split | official, held_epoch | official | official | official |

No official/unified recipe mixing, stacked physical augmentations or target-specific configuration changes are allowed. Gate is available only for the source-selected split frontend. All three pipelines must be reported; identical configurations are not independent methods. The first-pass factor table contains additional negative candidates that were not selected for replication.

Frozen final decision SHA256: `80c219b324045b39c6fcaa7db6ac73cef90c9b7e2ffa24b8846ad452a3b9d8b4`.
