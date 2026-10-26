# S09 Measured Costs

NVIDIA A100-SXM4-40GB; Random initialized weights, synthetic resident IQ to logits. Includes features/FFT in time but not all scalar/FFT FLOPs. Excludes packet detection and transfers. Runtime variance prevents speedup claims from parameter count alone.

| Dataset | Feature | Params | MFLOPs | CPU b1 ms | GPU b1 p50 ms | GPU b1 p95 ms | GPU b256 packet/s |
|---|---|---:|---:|---:|---:|---:|---:|
| ManySig | frequency | 90420 | 10.568 | 1.579 | 3.104 | 3.166 | 82170 |
| ManySig | null | 90868 | 10.798 | 1.766 | 3.320 | 3.409 | 76572 |
| ManySig | first | 90868 | 10.798 | 1.823 | 3.356 | 4.285 | 73585 |
| ManySig | centered | 90868 | 10.798 | 1.930 | 4.002 | 5.674 | 61193 |
| ManySig | curvature | 90868 | 10.798 | 1.858 | 3.518 | 4.998 | 71667 |
| ManyRx | frequency | 90680 | 10.569 | 1.692 | 3.754 | 4.627 | 81103 |
| ManyRx | null | 91128 | 10.798 | 1.799 | 3.687 | 5.358 | 70680 |
| ManyRx | first | 91128 | 10.798 | 1.897 | 3.830 | 4.958 | 51729 |
| ManyRx | centered | 91128 | 10.798 | 2.309 | 3.632 | 4.577 | 68537 |
| ManyRx | curvature | 91128 | 10.798 | 1.899 | 3.545 | 5.011 | 65936 |
