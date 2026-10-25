# S08 Measured Costs

NVIDIA A100-SXM4-40GB; PyTorch 2.5.0+cu124; CPU4threads. Random weights, synthetic resident IQ to logits; excludes transfer and detection. FFT/scalar work included in timing but not completely counted in FLOPs. EMA and source checkpoint rules do not change inference graphs. Costs are this runtime's measurements, not universal deployment claims.

| Dataset | Feature | Params | MFLOPs | CPU b1 ms | GPU b1 p50 ms | GPU b1 p95 ms | GPU b256 packet/s |
|---|---|---:|---:|---:|---:|---:|---:|
| ManySig | frequency | 90420 | 10.568 | 1.785 | 3.521 | 5.602 | 82556 |
| ManySig | phase | 90868 | 10.798 | 1.720 | 3.441 | 5.776 | 63113 |
| ManySig | phase_null | 90868 | 10.798 | 2.019 | 3.999 | 5.749 | 56360 |
| ManyRx | frequency | 90680 | 10.569 | 1.572 | 3.126 | 3.822 | 82166 |
| ManyRx | phase | 91128 | 10.798 | 2.074 | 3.912 | 5.482 | 54208 |
| ManyRx | phase_null | 91128 | 10.798 | 1.990 | 3.361 | 4.556 | 72081 |

Parameter count alone does not imply speed. Full200, clean BN recalibration and evaluating multiple source rules increase training/selection cost; the final inference model remains a single checkpoint, not an ensemble.
