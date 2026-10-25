# 03-chirp-structure confirm: auto-generated tables

run table sha256: `50c91aee6160814ba3d979e7269c6a200f5dc729246d680bbaee52d2f6ad4470`

## Arms (selected checkpoint; %, RX-equal mean over the role's receivers)

| arm | n_fits | params | dev_val_acc_mean | dev_val_acc_sd | dev_val_fold0 | dev_val_fold1 | dev_val_f1_mean | dev_val_worst_rx_seedmean | dev_val_worst_rx_name | dev_val_within_fold_seed_sd | confirm_acc_mean | confirm_acc_sd | confirm_f1_mean | confirm_worst_rx_seedmean | confirm_worst_rx_name | src_val_acc | best_epoch_median | train_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s03_hstem | 6 | 142392 | 97.48 | 0.41 | 97.17 | 97.8 | 97.48 | 95.57 | pluto_1 | 0.24 | 96.84 | 0.58 | 96.78 | 90.22 | b200_1 | 98.25 | 165.0 | 1401.9 |
| s02_ep200 | 6 | 580008 | 95.84 | 0.69 | 95.31 | 96.38 | 95.82 | 92.65 | pluto_1 | 0.37 | 93.89 | 0.48 | 93.74 | 83.92 | b200_1 | 96.61 | 180.0 | 1503.8 |
| s03_base | 6 | 580008 | 95.84 | 0.69 | 95.31 | 96.38 | 95.82 | 92.65 | pluto_1 | 0.37 | 93.89 | 0.48 | 93.74 | 83.92 | b200_1 | 96.61 | 180.0 | 1336.7 |
| s03_preln_d4 | 6 | 622374 | 95.58 | 0.79 | 95.47 | 95.69 | 95.56 | 93.98 | b200_mini_1 | 0.74 | 93.73 | 0.75 | 93.61 | 82.67 | b200_1 | 96.52 | 166.0 | 1398.9 |
| s03_ifreq | 6 | 580360 | 95.5 | 1.23 | 94.46 | 96.54 | 95.46 | 90.95 | pluto_1 | 0.52 | 93.94 | 0.91 | 93.82 | 81.13 | b200_1 | 96.6 | 164.5 | 1360.3 |
| s03_nofft | 6 | 579912 | 95.48 | 0.95 | 94.67 | 96.28 | 95.46 | 90.63 | pluto_1 | 0.38 | 93.65 | 0.39 | 93.52 | 83.15 | b200_1 | 96.43 | 147.0 | 1155.8 |
| s03_preln | 6 | 580136 | 95.48 | 0.91 | 94.66 | 96.3 | 95.45 | 90.98 | pluto_1 | 0.14 | 93.79 | 0.59 | 93.66 | 83.4 | b200_1 | 96.42 | 156.5 | 1360.4 |
| s03_symstack | 6 | 579912 | 90.27 | 0.75 | 89.87 | 90.68 | 90.17 | 81.35 | pluto_1 | 0.67 | 89.02 | 2.03 | 88.83 | 77.33 | b200_1 | 91.37 | 166.5 | 1201.3 |
| s03_resid | 6 | 580584 | 88.01 | 1.68 | 88.69 | 87.32 | 87.85 | 85.35 | rtl_3 | 1.37 | 87.21 | 1.7 | 87.07 | 79.35 | b200_1 | 89.63 | 89.5 | 1402.6 |

## Pre-registered comparisons (delta = B - A, pp)

| Q | role | B | A | factor | pairs | mean | min | max | +pairs | +RX | worst-RX delta | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | dev_val | s03_preln | s03_base | Pre-LN engineering control | 6 | -0.36 | -0.8 | 0.29 | 2/6 | 3/10 | -1.67 | unclear |
| P1 | confirm | s03_preln | s03_base | Pre-LN engineering control | 6 | -0.1 | -1.12 | 0.32 | 4/6 | 3/8 | -0.52 | unclear |
| P1 | dev_val | s03_preln_d4 | s03_preln | depth 4 after Pre-LN | 6 | 0.1 | -2.04 | 0.99 | 4/6 | 4/10 | 3.0 | unclear |
| P1 | confirm | s03_preln_d4 | s03_preln | depth 4 after Pre-LN | 6 | -0.06 | -1.69 | 1.82 | 2/6 | 3/8 | -0.73 | unclear |
| P2 | dev_val | s03_resid | s03_preln | unit-RMS complex residual channels | 6 | -7.47 | -11.1 | -5.44 | 0/6 | 0/10 | -5.63 | harms |
| P2 | confirm | s03_resid | s03_preln | unit-RMS complex residual channels | 6 | -6.58 | -9.89 | -4.78 | 0/6 | 0/8 | -4.05 | harms |
| P2 | dev_val | s03_nofft | s03_preln | remove whole-packet FFT channel | 6 | -0.0 | -0.48 | 0.48 | 3/6 | 5/10 | -0.35 | unclear |
| P2 | confirm | s03_nofft | s03_preln | remove whole-packet FFT channel | 6 | -0.14 | -0.61 | 0.22 | 3/6 | 4/8 | -0.25 | unclear |
| P2 | dev_val | s03_symstack | s03_nofft | equal-parameter symbol-aligned tokenizer | 6 | -5.2 | -6.12 | -4.35 | 0/6 | 0/10 | -9.28 | harms |
| P2 | confirm | s03_symstack | s03_nofft | equal-parameter symbol-aligned tokenizer | 6 | -4.63 | -8.12 | -2.47 | 0/6 | 0/8 | -5.82 | harms |
| ref_structure | dev_val | s03_symstack | s03_preln | system reference (also removes FFT) | 6 | -5.21 | -6.6 | -4.11 | 0/6 | 0/10 | -9.63 | reference (no verdict) |
| ref_structure | confirm | s03_symstack | s03_preln | system reference (also removes FFT) | 6 | -4.77 | -8.45 | -2.25 | 0/6 | 0/8 | -6.07 | reference (no verdict) |
| P3 | dev_val | s03_ifreq | s03_preln | instantaneous-frequency deviation channel | 6 | 0.02 | -0.6 | 0.77 | 3/6 | 8/10 | -0.03 | unclear |
| P3 | confirm | s03_ifreq | s03_preln | instantaneous-frequency deviation channel | 6 | 0.15 | -1.59 | 1.81 | 4/6 | 5/8 | -2.27 | unclear |
| P4 | dev_val | s03_hstem | s03_preln | hierarchical tokenizer | 6 | 2.0 | 1.29 | 2.66 | 6/6 | 10/10 | 4.58 | improves |
| P4 | confirm | s03_hstem | s03_preln | hierarchical tokenizer | 6 | 3.06 | 1.66 | 4.26 | 6/6 | 8/8 | 6.82 | improves |
| ref_determinism | dev_val | s03_base | s02_ep200 | bit-deterministic baseline rerun | 6 | 0.0 | 0.0 | 0.0 | 0/6 | 0/10 | 0.0 | reference (no verdict) |
| ref_determinism | confirm | s03_base | s02_ep200 | bit-deterministic baseline rerun | 6 | 0.0 | 0.0 | 0.0 | 0/6 | 0/8 | 0.0 | reference (no verdict) |

## Per-receiver seed-mean accuracy (%)

### dev_val

| arm | f0:b200_mini_1 | f0:pluto_1 | f0:rtl_2 | f0:rtl_5 | f0:rtl_7 | f1:b210_1 | f1:rtl_1 | f1:rtl_3 | f1:rtl_6 | f1:rtl_9 |
|---|---|---|---|---|---|---|---|---|---|---|
| s03_hstem | 97.1 | 95.6 | 97.3 | 97.7 | 98.2 | 98.0 | 97.6 | 96.9 | 98.0 | 98.5 |
| s02_ep200 | 94.1 | 92.6 | 96.0 | 96.4 | 97.3 | 96.7 | 95.8 | 95.4 | 96.5 | 97.5 |
| s03_base | 94.1 | 92.6 | 96.0 | 96.4 | 97.3 | 96.7 | 95.8 | 95.4 | 96.5 | 97.5 |
| s03_preln_d4 | 94.0 | 94.5 | 95.5 | 96.0 | 97.3 | 96.1 | 95.2 | 94.9 | 95.6 | 96.6 |
| s03_ifreq | 92.6 | 91.0 | 95.6 | 96.0 | 97.1 | 96.5 | 96.3 | 95.6 | 96.9 | 97.3 |
| s03_nofft | 93.4 | 90.6 | 95.9 | 96.5 | 97.0 | 96.6 | 95.8 | 95.6 | 96.0 | 97.4 |
| s03_preln | 94.0 | 91.0 | 95.6 | 96.0 | 96.7 | 96.1 | 96.0 | 95.6 | 96.6 | 97.2 |
| s03_symstack | 89.7 | 81.4 | 91.7 | 92.4 | 94.2 | 89.6 | 90.3 | 90.2 | 90.4 | 93.0 |
| s03_resid | 88.1 | 85.6 | 89.0 | 89.9 | 90.8 | 88.9 | 85.6 | 85.3 | 88.5 | 88.3 |

### confirm

| arm | f0:b200_1 | f0:n210_1 | f0:rtl_4 | f0:rtl_8 | f1:b200_1 | f1:n210_1 | f1:rtl_4 | f1:rtl_8 |
|---|---|---|---|---|---|---|---|---|
| s03_hstem | 90.2 | 99.2 | 98.7 | 98.8 | 94.0 | 97.8 | 97.8 | 98.2 |
| s02_ep200 | 83.9 | 98.1 | 96.6 | 97.7 | 85.4 | 96.3 | 96.0 | 97.1 |
| s03_base | 83.9 | 98.1 | 96.6 | 97.7 | 85.4 | 96.3 | 96.0 | 97.1 |
| s03_preln_d4 | 82.7 | 98.5 | 96.2 | 97.9 | 87.4 | 95.5 | 95.8 | 96.0 |
| s03_ifreq | 81.1 | 97.9 | 96.6 | 97.5 | 88.5 | 96.4 | 96.6 | 96.8 |
| s03_nofft | 83.1 | 97.8 | 96.4 | 97.8 | 85.8 | 95.5 | 96.0 | 96.8 |
| s03_preln | 83.4 | 97.8 | 96.4 | 97.7 | 85.6 | 96.4 | 96.1 | 97.0 |
| s03_symstack | 79.7 | 94.1 | 93.2 | 95.7 | 77.3 | 90.2 | 90.7 | 91.2 |
| s03_resid | 79.3 | 91.4 | 90.5 | 90.4 | 81.0 | 89.3 | 87.1 | 88.7 |
