# 02-single-factor-screen confirm: auto-generated tables

run table sha256: `abedf0bc7e66c433540ba6bf3c6d5fb35e70acf1fccf03d601b4c7a6e7b039ff`

## Arms (selected checkpoint; %, RX-equal mean over the role's receivers)

| arm | n_fits | params | dev_val_acc_mean | dev_val_acc_sd | dev_val_fold0 | dev_val_fold1 | dev_val_f1_mean | dev_val_worst_rx_seedmean | dev_val_worst_rx_name | dev_val_within_fold_seed_sd | confirm_acc_mean | confirm_acc_sd | confirm_f1_mean | confirm_worst_rx_seedmean | confirm_worst_rx_name | src_val_acc | best_epoch_median | train_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s02_ep200 | 6 | 580008 | 95.84 | 0.69 | 95.31 | 96.38 | 95.82 | 92.65 | pluto_1 | 0.37 | 93.89 | 0.48 | 93.74 | 83.92 | b200_1 | 96.61 | 180.0 | 1503.8 |
| xa_trueib_cos_awgn | 6 | 580008 | 94.74 | 1.33 | 93.87 | 95.6 | 94.69 | 91.62 | pluto_1 | 1.03 | 92.11 | 0.85 | 91.9 | 79.97 | b200_1 | 94.4 | 91.5 | 839.4 |
| s02_base | 6 | 580008 | 94.74 | 1.33 | 93.87 | 95.6 | 94.69 | 91.62 | pluto_1 | 1.03 | 92.11 | 0.85 | 91.9 | 79.97 | b200_1 | 94.4 | 91.5 | 770.6 |
| s02_awgn10 | 6 | 580008 | 94.72 | 2.37 | 93.18 | 96.26 | 94.68 | 90.1 | b200_mini_1 | 1.5 | 92.14 | 1.88 | 91.97 | 78.13 | b200_1 | 94.97 | 93.0 | 730.6 |
| s02_cisoob | 6 | 619816 | 94.11 | 0.83 | 93.67 | 94.56 | 94.03 | 89.92 | pluto_1 | 0.75 | 91.96 | 0.88 | 91.75 | 82.87 | b200_1 | 97.19 | 93.5 | 753.8 |
| cis_flat | 6 | 13442954 | 93.12 | 1.62 | 91.81 | 94.44 | 92.89 | 81.98 | pluto_1 | 0.8 | 95.2 | 0.87 | 95.09 | 90.48 | b200_1 | 97.28 | 226.0 | 1725.8 |
| s02_tfsplit | 6 | 583294 | 93.05 | 1.98 | 92.26 | 93.84 | 92.98 | 88.8 | b200_mini_1 | 1.78 | 90.42 | 2.17 | 90.19 | 77.25 | b200_1 | 93.4 | 93.5 | 919.5 |
| s02_cis_rank16 | 6 | 658314 | 92.36 | 2.08 | 90.86 | 93.87 | 92.13 | 80.8 | pluto_1 | 1.16 | 94.75 | 1.23 | 94.62 | 89.67 | rtl_4 | 97.09 | 257.0 | 1527.5 |
| s02_dechirp | 6 | 580008 | 90.98 | 5.66 | 88.7 | 93.26 | 90.86 | 83.63 | b200_mini_1 | 5.37 | 88.66 | 4.94 | 88.46 | 72.48 | b200_1 | 90.46 | 96.0 | 729.6 |
| s02_combfft | 6 | 580008 | 88.99 | 11.94 | 83.15 | 94.84 | 88.57 | 78.42 | b200_mini_1 | 8.59 | 86.48 | 11.05 | 85.93 | 69.07 | b200_1 | 88.64 | 97.5 | 789.1 |
| s02_rawoob_nearL | 6 | 625896 | 84.25 | 4.06 | 80.69 | 87.81 | 82.96 | 70.02 | rtl_7 | 1.25 | 83.43 | 1.55 | 82.37 | 62.83 | b200_1 | 93.93 | 74.0 | 864.2 |
| s02_rawoob_near | 6 | 625896 | 84.11 | 3.87 | 83.9 | 84.31 | 83.23 | 66.95 | rtl_6 | 4.32 | 83.95 | 7.88 | 82.87 | 65.48 | rtl_8 | 95.63 | 83.0 | 878.1 |
| s02_rawoob_full | 6 | 625896 | 83.44 | 2.88 | 82.03 | 84.85 | 82.26 | 62.32 | rtl_7 | 2.16 | 83.63 | 6.43 | 82.64 | 62.27 | rtl_8 | 95.63 | 89.5 | 859.3 |
| s02_meanchirp | 6 | 580008 | 81.09 | 1.85 | 80.57 | 81.61 | 80.76 | 66.68 | pluto_1 | 1.59 | 80.4 | 2.83 | 80.26 | 65.27 | b200_1 | 82.15 | 97.0 | 785.3 |
| s02_depth4 | 6 | 622246 | 67.08 | 30.42 | 65.12 | 69.04 | 64.92 | 59.42 | b200_mini_1 | 33.93 | 65.59 | 29.08 | 63.46 | 53.47 | b200_1 | 67.44 | 95.5 | 777.0 |

## Pre-registered comparisons (delta = B - A, pp)

| Q | role | B | A | factor | pairs | mean | min | max | +pairs | +RX | worst-RX delta | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | dev_val | s02_meanchirp | s02_base | coherent 8-symbol mean chirp (drops residual/transient) | 6 | -13.64 | -16.56 | -11.75 | 0/6 | 0/10 | -24.93 | harms |
| B | confirm | s02_meanchirp | s02_base | coherent 8-symbol mean chirp (drops residual/transient) | 6 | -11.71 | -15.82 | -9.44 | 0/6 | 0/8 | -14.7 | harms |
| B | dev_val | s02_combfft | s02_base | FFT view: comb bins only | 6 | -5.74 | -28.43 | 0.51 | 1/6 | 0/10 | -13.2 | harms |
| B | confirm | s02_combfft | s02_base | FFT view: comb bins only | 6 | -5.63 | -27.85 | -0.04 | 0/6 | 0/8 | -10.9 | harms |
| B | dev_val | s02_dechirp | s02_base | dechirped input | 6 | -3.76 | -13.12 | 2.49 | 2/6 | 0/10 | -7.98 | unclear |
| B | confirm | s02_dechirp | s02_base | dechirped input | 6 | -3.45 | -12.54 | 3.18 | 1/6 | 0/8 | -7.48 | harms |
| A | dev_val | s02_rawoob_full | s02_base | + raw OOB keys/values, all |f|>62.5 kHz | 6 | -11.3 | -14.39 | -7.4 | 0/6 | 1/10 | -29.3 | harms |
| A | confirm | s02_rawoob_full | s02_base | + raw OOB keys/values, all |f|>62.5 kHz | 6 | -8.48 | -16.6 | -1.85 | 0/6 | 2/8 | -17.7 | harms |
| A | dev_val | s02_rawoob_near | s02_base | + raw OOB keys/values, 62.5-200 kHz | 6 | -10.63 | -14.02 | -4.18 | 0/6 | 2/10 | -24.67 | harms |
| A | confirm | s02_rawoob_near | s02_base | + raw OOB keys/values, 62.5-200 kHz | 6 | -8.16 | -18.74 | 1.18 | 1/6 | 3/8 | -14.48 | harms |
| A | dev_val | s02_rawoob_nearL | s02_base | + raw OOB keys/values, left near band only | 6 | -10.48 | -14.52 | -5.73 | 0/6 | 1/10 | -21.6 | harms |
| A | confirm | s02_rawoob_nearL | s02_base | + raw OOB keys/values, left near band only | 6 | -8.68 | -10.81 | -5.5 | 0/6 | 2/8 | -17.13 | harms |
| A | dev_val | s02_cisoob | s02_base | + CIS-quotient OOB keys/values | 6 | -0.62 | -2.22 | 1.24 | 1/6 | 2/10 | -1.7 | unclear |
| A | confirm | s02_cisoob | s02_base | + CIS-quotient OOB keys/values | 6 | -0.15 | -1.09 | 0.64 | 3/6 | 3/8 | 2.9 | unclear |
| C | dev_val | s02_depth4 | s02_base | 4 mixer blocks | 6 | -27.65 | -66.21 | 2.12 | 2/6 | 0/10 | -32.2 | unclear |
| C | confirm | s02_depth4 | s02_base | 4 mixer blocks | 6 | -26.52 | -64.57 | 2.15 | 1/6 | 0/8 | -26.5 | harms |
| C | dev_val | s02_tfsplit | s02_base | time/frequency split tokens (strict IB) | 6 | -1.68 | -3.38 | -0.13 | 0/6 | 0/10 | -2.82 | harms |
| C | confirm | s02_tfsplit | s02_base | time/frequency split tokens (strict IB) | 6 | -1.69 | -3.87 | 0.75 | 2/6 | 0/8 | -2.72 | unclear |
| D | dev_val | s02_ep200 | s02_base | 200-epoch horizon | 6 | 1.11 | -0.11 | 1.96 | 5/6 | 10/10 | 1.03 | improves |
| D | confirm | s02_ep200 | s02_base | 200-epoch horizon | 6 | 1.78 | 0.62 | 2.32 | 6/6 | 7/8 | 3.95 | improves |
| D | dev_val | s02_awgn10 | s02_base | AWGN U[10,50) instead of U[0,50) | 6 | -0.01 | -3.09 | 1.65 | 4/6 | 6/10 | -1.52 | unclear |
| D | confirm | s02_awgn10 | s02_base | AWGN U[10,50) instead of U[0,50) | 6 | 0.03 | -3.24 | 1.62 | 3/6 | 4/8 | -1.83 | unclear |
| eff | dev_val | s02_cis_rank16 | cis_flat | rank-16 head vs Flatten-Dense (stage-01 fits, same data streams) | 6 | -0.76 | -2.36 | 1.22 | 2/6 | 1/10 | -1.18 | unclear |
| eff | confirm | s02_cis_rank16 | cis_flat | rank-16 head vs Flatten-Dense (stage-01 fits, same data streams) | 6 | -0.46 | -1.41 | 0.81 | 2/6 | 1/8 | -0.82 | unclear |
| ref | dev_val | s02_base | xa_trueib_cos_awgn | determinism check: re-run of stage-01 arm (expected 0.00) | 6 | 0.0 | 0.0 | 0.0 | 0/6 | 0/10 | 0.0 | reference (no verdict) |
| ref | confirm | s02_base | xa_trueib_cos_awgn | determinism check: re-run of stage-01 arm (expected 0.00) | 6 | 0.0 | 0.0 | 0.0 | 0/6 | 0/8 | 0.0 | reference (no verdict) |
| ref | dev_val | s02_base | s02_cis_rank16 | SYSTEM: CrossAttn-TrueIB vs rank-16 CIS | 6 | 2.37 | 0.02 | 4.17 | 6/6 | 7/10 | 10.82 | reference (no verdict) |
| ref | confirm | s02_base | s02_cis_rank16 | SYSTEM: CrossAttn-TrueIB vs rank-16 CIS | 6 | -2.64 | -5.01 | -1.25 | 0/6 | 3/8 | -9.7 | reference (no verdict) |

## Per-receiver seed-mean accuracy (%)

### dev_val

| arm | f0:b200_mini_1 | f0:pluto_1 | f0:rtl_2 | f0:rtl_5 | f0:rtl_7 | f1:b210_1 | f1:rtl_1 | f1:rtl_3 | f1:rtl_6 | f1:rtl_9 |
|---|---|---|---|---|---|---|---|---|---|---|
| s02_ep200 | 94.1 | 92.6 | 96.0 | 96.4 | 97.3 | 96.7 | 95.8 | 95.4 | 96.5 | 97.5 |
| xa_trueib_cos_awgn | 92.1 | 91.6 | 94.5 | 94.8 | 96.4 | 95.0 | 95.1 | 94.6 | 95.9 | 97.5 |
| s02_base | 92.1 | 91.6 | 94.5 | 94.8 | 96.4 | 95.0 | 95.1 | 94.6 | 95.9 | 97.5 |
| s02_awgn10 | 90.1 | 94.7 | 92.0 | 94.0 | 95.0 | 96.4 | 95.8 | 95.2 | 96.2 | 97.7 |
| s02_cisoob | 96.6 | 89.9 | 93.1 | 93.1 | 95.6 | 98.7 | 90.9 | 93.8 | 94.3 | 95.2 |
| cis_flat | 98.6 | 82.0 | 89.5 | 91.2 | 97.8 | 98.9 | 89.4 | 90.7 | 97.5 | 95.6 |
| s02_tfsplit | 88.8 | 90.6 | 92.8 | 93.9 | 95.1 | 92.5 | 93.1 | 93.4 | 94.4 | 95.8 |
| s02_cis_rank16 | 98.4 | 80.8 | 88.5 | 91.0 | 95.5 | 99.0 | 88.4 | 89.6 | 97.2 | 95.1 |
| s02_dechirp | 83.6 | 89.4 | 89.0 | 89.8 | 91.7 | 90.9 | 92.6 | 93.1 | 94.2 | 95.6 |
| s02_combfft | 78.4 | 82.0 | 84.0 | 85.2 | 86.1 | 93.5 | 94.6 | 94.4 | 95.1 | 96.7 |
| s02_rawoob_nearL | 82.4 | 87.6 | 81.6 | 81.8 | 70.0 | 96.1 | 88.2 | 85.9 | 78.9 | 90.0 |
| s02_rawoob_near | 91.6 | 92.7 | 82.6 | 84.5 | 68.1 | 95.6 | 89.1 | 85.6 | 66.9 | 84.2 |
| s02_rawoob_full | 89.9 | 86.3 | 84.8 | 86.8 | 62.3 | 96.2 | 90.8 | 83.7 | 69.4 | 84.2 |
| s02_meanchirp | 77.3 | 66.7 | 85.0 | 86.4 | 87.5 | 77.0 | 81.4 | 82.8 | 84.4 | 82.4 |
| s02_depth4 | 59.4 | 66.5 | 66.3 | 65.5 | 67.9 | 68.6 | 67.7 | 67.5 | 70.6 | 70.8 |

### confirm

| arm | f0:b200_1 | f0:n210_1 | f0:rtl_4 | f0:rtl_8 | f1:b200_1 | f1:n210_1 | f1:rtl_4 | f1:rtl_8 |
|---|---|---|---|---|---|---|---|---|
| s02_ep200 | 83.9 | 98.1 | 96.6 | 97.7 | 85.4 | 96.3 | 96.0 | 97.1 |
| xa_trueib_cos_awgn | 80.0 | 97.5 | 94.7 | 96.8 | 80.2 | 96.5 | 95.2 | 96.0 |
| s02_base | 80.0 | 97.5 | 94.7 | 96.8 | 80.2 | 96.5 | 95.2 | 96.0 |
| s02_awgn10 | 78.1 | 96.6 | 93.7 | 95.5 | 83.6 | 97.0 | 95.9 | 96.7 |
| s02_cisoob | 82.9 | 95.4 | 92.8 | 97.9 | 84.8 | 95.8 | 90.6 | 95.4 |
| cis_flat | 90.5 | 96.5 | 92.1 | 99.4 | 95.2 | 96.8 | 93.5 | 97.8 |
| s02_tfsplit | 77.3 | 96.3 | 94.1 | 95.9 | 77.2 | 94.7 | 93.3 | 94.4 |
| s02_cis_rank16 | 90.4 | 96.0 | 89.7 | 98.9 | 94.7 | 98.2 | 92.8 | 97.4 |
| s02_dechirp | 72.5 | 94.4 | 89.2 | 92.5 | 78.5 | 95.8 | 92.4 | 93.9 |
| s02_combfft | 69.1 | 87.4 | 84.2 | 86.6 | 78.7 | 96.4 | 94.4 | 95.2 |
| s02_rawoob_nearL | 62.8 | 90.7 | 78.8 | 98.4 | 86.9 | 96.1 | 71.9 | 81.7 |
| s02_rawoob_near | 81.2 | 95.1 | 85.0 | 97.7 | 83.9 | 95.3 | 67.9 | 65.5 |
| s02_rawoob_full | 77.4 | 93.1 | 86.9 | 97.7 | 85.4 | 95.4 | 70.8 | 62.3 |
| s02_meanchirp | 67.0 | 88.4 | 85.4 | 89.0 | 65.3 | 82.1 | 83.1 | 82.8 |
| s02_depth4 | 53.5 | 69.2 | 65.1 | 68.1 | 59.7 | 70.6 | 68.6 | 70.0 |
