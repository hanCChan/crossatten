# 03-chirp-structure explore: auto-generated tables

run table sha256: `77130e0159b2d3c993ddebaffaf5bcc964f2be7ad12573cb059539914085eee8`

## Arms (selected checkpoint; %, RX-equal mean over the role's receivers)

| arm | n_fits | params | dev_val_acc_mean | dev_val_acc_sd | dev_val_fold0 | dev_val_fold1 | dev_val_f1_mean | dev_val_worst_rx_seedmean | dev_val_worst_rx_name | dev_val_within_fold_seed_sd | src_val_acc | best_epoch_median | train_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s03_hstem | 4 | 142392 | 97.27 | 0.55 | 97.12 | 97.41 | 97.26 | 94.6 | pluto_1 | 0.64 | 97.96 | 182.0 | 1237.2 |
| s03_preln_d4 | 4 | 622374 | 96.03 | 0.85 | 95.3 | 96.76 | 96.01 | 92.85 | pluto_1 | 0.14 | 96.56 | 177.0 | 1475.0 |
| s03_ifreq | 4 | 580360 | 95.93 | 0.73 | 95.32 | 96.53 | 95.91 | 91.95 | pluto_1 | 0.24 | 96.69 | 175.5 | 1245.3 |
| s03_nofft | 4 | 579912 | 95.77 | 0.81 | 95.17 | 96.37 | 95.75 | 93.1 | pluto_1 | 0.48 | 96.54 | 158.5 | 1048.7 |
| s03_preln | 4 | 580136 | 95.3 | 0.53 | 95.02 | 95.59 | 95.27 | 92.3 | pluto_1 | 0.4 | 96.66 | 160.5 | 1462.4 |
| s02_ep200 | 4 | 580008 | 94.27 | 1.83 | 93.65 | 94.89 | 94.24 | 89.7 | b200_mini_1 | 1.85 | 94.15 | 181.5 | 1466.4 |
| s03_base | 4 | 580008 | 94.27 | 1.83 | 93.65 | 94.89 | 94.24 | 89.7 | b200_mini_1 | 1.85 | 94.15 | 181.5 | 1462.3 |
| s03_symstack | 4 | 579912 | 91.81 | 0.85 | 91.46 | 92.16 | 91.74 | 86.2 | pluto_1 | 0.87 | 92.31 | 178.5 | 1227.6 |
| s03_resid | 4 | 580584 | 88.09 | 0.75 | 88.41 | 87.78 | 87.93 | 85.85 | b200_mini_1 | 0.8 | 89.64 | 81.5 | 1458.6 |

## Pre-registered comparisons (delta = B - A, pp)

| Q | role | B | A | factor | pairs | mean | min | max | +pairs | +RX | worst-RX delta | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | dev_val | s03_preln | s03_base | Pre-LN engineering control | 4 | 1.04 | -1.07 | 3.82 | 3/4 | 8/10 | 2.6 | improves |
| P1 | dev_val | s03_preln_d4 | s03_preln | depth 4 after Pre-LN | 4 | 0.73 | -0.14 | 1.36 | 3/4 | 8/10 | 0.55 | unclear |
| P2 | dev_val | s03_resid | s03_preln | unit-RMS complex residual channels | 4 | -7.21 | -8.39 | -6.6 | 0/4 | 0/10 | -6.45 | harms |
| P2 | dev_val | s03_nofft | s03_preln | remove whole-packet FFT channel | 4 | 0.47 | -0.11 | 1.16 | 3/4 | 7/10 | 0.8 | unclear |
| P2 | dev_val | s03_symstack | s03_nofft | equal-parameter symbol-aligned tokenizer | 4 | -3.96 | -5.05 | -2.64 | 0/4 | 0/10 | -6.9 | harms |
| ref_structure | dev_val | s03_symstack | s03_preln | system reference (also removes FFT) | 4 | -3.5 | -4.89 | -2.24 | 0/4 | 0/10 | -6.1 | reference (no verdict) |
| P3 | dev_val | s03_ifreq | s03_preln | instantaneous-frequency deviation channel | 4 | 0.62 | -0.09 | 1.25 | 3/4 | 8/10 | -0.35 | unclear |
| P4 | dev_val | s03_hstem | s03_preln | hierarchical tokenizer | 4 | 1.96 | 1.1 | 3.1 | 4/4 | 10/10 | 2.3 | improves |
| ref_determinism | dev_val | s03_base | s02_ep200 | bit-deterministic baseline rerun | 4 | 0.0 | 0.0 | 0.0 | 0/4 | 0/10 | 0.0 | reference (no verdict) |

## Per-receiver seed-mean accuracy (%)

### dev_val

| arm | f0:b200_mini_1 | f0:pluto_1 | f0:rtl_2 | f0:rtl_5 | f0:rtl_7 | f1:b210_1 | f1:rtl_1 | f1:rtl_3 | f1:rtl_6 | f1:rtl_9 |
|---|---|---|---|---|---|---|---|---|---|---|
| s03_hstem | 97.2 | 94.6 | 97.5 | 97.8 | 98.6 | 97.8 | 97.2 | 96.3 | 97.6 | 98.2 |
| s03_preln_d4 | 94.7 | 92.8 | 95.6 | 96.0 | 97.4 | 96.9 | 96.0 | 96.2 | 97.1 | 97.7 |
| s03_ifreq | 94.3 | 92.0 | 96.4 | 96.6 | 97.3 | 96.2 | 96.3 | 95.7 | 96.9 | 97.6 |
| s03_nofft | 94.4 | 93.1 | 95.8 | 96.0 | 96.5 | 96.3 | 95.9 | 95.8 | 96.6 | 97.2 |
| s03_preln | 93.6 | 92.3 | 96.3 | 96.1 | 96.8 | 96.2 | 95.4 | 94.6 | 95.0 | 96.7 |
| s02_ep200 | 89.7 | 93.2 | 94.3 | 94.4 | 96.6 | 94.7 | 94.1 | 94.0 | 94.8 | 96.8 |
| s03_base | 89.7 | 93.2 | 94.3 | 94.4 | 96.6 | 94.7 | 94.1 | 94.0 | 94.8 | 96.8 |
| s03_symstack | 89.6 | 86.2 | 93.2 | 93.4 | 94.8 | 91.0 | 91.7 | 91.8 | 92.7 | 93.7 |
| s03_resid | 85.9 | 86.5 | 88.6 | 90.0 | 91.1 | 88.9 | 86.8 | 86.1 | 87.4 | 89.6 |
