# Confirmation interventions

RX-equal mean; all six fits; delta relative to unchanged batch64 evaluation.

| arm | role | intervention | accuracy (%) | delta (pp) |
|---|---|---|---:|---:|
| s03_resid | dev_val | original | 88.008 | +0.000 |
| s03_resid | dev_val | shuffle_residual | 38.613 | -49.395 |
| s03_resid | dev_val | zero_residual | 43.568 | -44.440 |
| s03_resid | confirm | original | 87.206 | +0.000 |
| s03_resid | confirm | shuffle_residual | 37.690 | -49.517 |
| s03_resid | confirm | zero_residual | 42.267 | -44.940 |
| s03_symstack | dev_val | mean_chirp | 34.637 | -55.637 |
| s03_symstack | dev_val | misalign_symbols | 12.845 | -77.428 |
| s03_symstack | dev_val | original | 90.273 | +0.000 |
| s03_symstack | dev_val | reverse_symbols | 29.882 | -60.392 |
| s03_symstack | dev_val | shuffle_symbol_residual | 27.825 | -62.448 |
| s03_symstack | confirm | mean_chirp | 32.310 | -56.708 |
| s03_symstack | confirm | misalign_symbols | 12.790 | -76.229 |
| s03_symstack | confirm | original | 89.019 | +0.000 |
| s03_symstack | confirm | reverse_symbols | 28.183 | -60.835 |
| s03_symstack | confirm | shuffle_symbol_residual | 27.923 | -61.096 |

The registered cyclic shift changes donor TX for 0.80-0.80% of packets across RX; the remainder keep a same-TX residual donor. Do not treat a weak shuffle effect as proof that identity residuals are unused. The maximum unchanged batch64 accuracy difference from saved batch500 evaluation is 0.0000 pp.
