# Stage 01 confirm: auto-generated tables
run table sha256: `f5a7ea2111f6bc4cb9a21201d43f91df0884a757b6602dfaeb1f8fa72619eb8e`

## Arms (selected checkpoint; accuracy in %, RX-equal mean over the role's receivers)
| arm | n_fits | params | dev_val_acc_mean | dev_val_acc_sd | dev_val_fold0 | dev_val_fold1 | dev_val_f1_mean | dev_val_worst_rx_seedmean | dev_val_worst_rx_name | dev_val_within_fold_seed_sd | confirm_acc_mean | confirm_acc_sd | confirm_f1_mean | confirm_worst_rx_seedmean | confirm_worst_rx_name | src_val_acc | best_epoch_median | train_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xa_trueib_cos_awgn | 6 | 580008 | 94.74 | 1.33 | 93.87 | 95.6 | 94.69 | 91.62 | pluto_1 | 1.03 | 92.11 | 0.85 | 91.9 | 79.97 | b200_1 | 94.4 | 91.5 | 839.4 |
| cis_ib_flat | 6 | 13442954 | 94.62 | 3.03 | 91.86 | 97.39 | 94.16 | 72.0 | pluto_1 | 0.14 | 95.72 | 0.12 | 95.5 | 87.78 | b200_1 | 97.22 | 231.0 | 1782.6 |
| cis_flat | 6 | 13442954 | 93.12 | 1.62 | 91.81 | 94.44 | 92.89 | 81.98 | pluto_1 | 0.8 | 95.2 | 0.87 | 95.09 | 90.48 | b200_1 | 97.28 | 226.0 | 1725.8 |
| cis_flat_noawgn | 6 | 13442954 | 84.33 | 3.26 | 81.52 | 87.15 | 83.52 | 66.52 | pluto_1 | 1.19 | 88.51 | 1.44 | 88.23 | 78.1 | rtl_4 | 95.08 | 110.0 | 875.4 |
| xa_trueib_cos | 6 | 580008 | 81.75 | 13.71 | 86.15 | 77.34 | 81.67 | 73.83 | rtl_3 | 14.28 | 80.05 | 12.83 | 79.83 | 65.8 | b200_1 | 82.61 | 56.0 | 844.0 |
| cis_gap_noawgn | 6 | 270218 | 69.54 | 4.4 | 73.02 | 66.06 | 68.31 | 55.43 | rtl_1 | 2.26 | 71.92 | 5.57 | 70.89 | 59.72 | b200_1 | 83.37 | 233.5 | 1381.2 |
| cis_gap | 6 | 270218 | 58.67 | 8.96 | 61.24 | 56.11 | 55.81 | 40.3 | b210_1 | 9.14 | 54.84 | 9.6 | 51.46 | 27.75 | b200_1 | 55.46 | 273.5 | 1735.9 |
| xa_lp200_cos_awgn | 6 | 625896 | 52.38 | 2.68 | 51.03 | 53.73 | 49.37 | 36.45 | rtl_6 | 2.44 | 57.17 | 6.28 | 54.98 | 32.75 | rtl_8 | 82.91 | 51.5 | 839.9 |
| xa_oobnear_cos_awgn | 6 | 625896 | 52.32 | 3.89 | 51.4 | 53.24 | 48.85 | 35.6 | rtl_6 | 4.19 | 57.14 | 7.09 | 55.36 | 29.9 | rtl_8 | 83.23 | 40.5 | 754.3 |
| tf_cos_awgn | 6 | 629182 | 50.7 | 7.44 | 49.98 | 51.42 | 47.79 | 30.72 | rtl_6 | 7.77 | 56.12 | 9.66 | 54.22 | 26.65 | rtl_8 | 81.68 | 42.5 | 1008.5 |
| xa_cos_awgn | 6 | 625896 | 50.46 | 4.56 | 47.61 | 53.31 | 46.85 | 40.25 | rtl_6 | 3.52 | 56.75 | 4.4 | 54.71 | 37.38 | rtl_4 | 83.17 | 46.0 | 760.0 |
| xa_nooob_cos_awgn | 6 | 580008 | 49.34 | 11.08 | 47.64 | 51.04 | 46.38 | 45.67 | rtl_6 | 12.05 | 49.35 | 10.73 | 46.49 | 39.38 | rtl_4 | 57.78 | 96.5 | 660.4 |
| xa_lp200_cos | 6 | 625896 | 45.46 | 4.33 | 48.97 | 41.96 | 41.37 | 20.1 | rtl_1 | 2.19 | 54.02 | 4.19 | 51.19 | 21.17 | rtl_4 | 82.19 | 15.0 | 867.5 |
| xa_const | 6 | 625896 | 44.49 | 5.04 | 47.76 | 41.22 | 41.33 | 21.0 | rtl_1 | 3.97 | 51.82 | 7.35 | 48.53 | 18.08 | rtl_4 | 79.97 | 11.5 | 775.5 |
| xa_cos | 6 | 625896 | 44.08 | 3.48 | 46.34 | 41.81 | 39.33 | 23.83 | rtl_1 | 2.65 | 53.7 | 6.06 | 50.24 | 25.78 | rtl_4 | 81.02 | 10.0 | 783.9 |
| xa_oobnear_cos | 6 | 625896 | 43.38 | 3.63 | 45.68 | 41.08 | 38.96 | 21.83 | rtl_1 | 2.43 | 52.65 | 4.4 | 48.92 | 20.38 | rtl_4 | 82.05 | 12.5 | 765.6 |
| xa_p1fixed | 6 | 625896 | 40.43 | 5.3 | 44.78 | 36.09 | 37.08 | 28.52 | rtl_1 | 2.54 | 41.46 | 7.92 | 37.85 | 14.75 | rtl_4 | 59.78 | 4.0 | 30.2 |
| tf_cos | 6 | 629182 | 38.62 | 1.69 | 37.58 | 39.66 | 34.71 | 27.17 | rtl_2 | 1.34 | 45.87 | 6.08 | 42.67 | 19.4 | rtl_4 | 76.54 | 6.5 | 1017.4 |
| xa_nooob_cos | 6 | 580008 | 36.8 | 18.16 | 34.0 | 39.61 | 32.26 | 24.07 | rtl_2 | 19.89 | 39.96 | 19.13 | 36.08 | 22.48 | rtl_4 | 53.82 | 16.0 | 720.6 |

## Pre-registered one-factor comparisons (delta = B - A, pp)

| Q | role | B | A | factor | pairs | mean | min | max | +pairs | +RX | worst-RX delta | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | dev_val | xa_const | xa_p1fixed | horizon 100ep + source-val checkpoint | 6 | 4.06 | 0.89 | 10.87 | 6/6 | 8/10 | -7.52 | unclear |
| Q1 | confirm | xa_const | xa_p1fixed | horizon 100ep + source-val checkpoint | 6 | 10.35 | 6.08 | 17.97 | 6/6 | 8/8 | 3.33 | improves |
| Q1 | dev_val | xa_cos | xa_const | warmup+cosine schedule | 6 | -0.41 | -6.66 | 8.92 | 2/6 | 4/10 | 2.83 | unclear |
| Q1 | confirm | xa_cos | xa_const | warmup+cosine schedule | 6 | 1.88 | -6.13 | 14.89 | 4/6 | 6/8 | 7.7 | unclear |
| Q1 | dev_val | xa_cos_awgn | xa_cos | AWGN U[0,50) dB | 6 | 6.38 | -2.53 | 17.51 | 5/6 | 5/10 | 16.42 | improves |
| Q1 | confirm | xa_cos_awgn | xa_cos | AWGN U[0,50) dB | 6 | 3.06 | -4.89 | 10.95 | 4/6 | 4/8 | 11.6 | unclear |
| Q2 | dev_val | cis_gap | cis_flat | GAP head vs Flatten-Dense (AWGN) | 6 | -34.45 | -43.16 | -16.55 | 0/6 | 0/10 | -41.68 | harms |
| Q2 | confirm | cis_gap | cis_flat | GAP head vs Flatten-Dense (AWGN) | 6 | -40.36 | -49.76 | -23.38 | 0/6 | 0/8 | -62.73 | harms |
| Q2 | dev_val | cis_gap_noawgn | cis_flat_noawgn | GAP head vs Flatten-Dense (no AWGN) | 6 | -14.79 | -22.52 | -4.64 | 0/6 | 1/10 | -11.08 | harms |
| Q2 | confirm | cis_gap_noawgn | cis_flat_noawgn | GAP head vs Flatten-Dense (no AWGN) | 6 | -16.59 | -23.01 | -8.44 | 0/6 | 0/8 | -18.38 | harms |
| Q2 | dev_val | cis_flat | cis_flat_noawgn | AWGN on CIS (flatten) | 6 | 8.79 | 6.47 | 11.51 | 6/6 | 10/10 | 15.47 | improves |
| Q2 | confirm | cis_flat | cis_flat_noawgn | AWGN on CIS (flatten) | 6 | 6.69 | 5.79 | 8.02 | 6/6 | 8/8 | 12.38 | improves |
| Q2 | dev_val | cis_gap | cis_gap_noawgn | AWGN on CIS (GAP) | 6 | -10.87 | -25.43 | 2.85 | 1/6 | 1/10 | -15.13 | harms |
| Q2 | confirm | cis_gap | cis_gap_noawgn | AWGN on CIS (GAP) | 6 | -17.07 | -34.3 | -5.93 | 0/6 | 0/8 | -31.97 | harms |
| Q2 | dev_val | cis_ib_flat | cis_flat | strict-IB CIS vs Shen crop (-203..+195 kHz) | 6 | 1.5 | -0.89 | 3.49 | 5/6 | 6/10 | -9.98 | unclear |
| Q2 | confirm | cis_ib_flat | cis_flat | strict-IB CIS vs Shen crop (-203..+195 kHz) | 6 | 0.52 | -0.78 | 1.58 | 4/6 | 4/8 | -2.7 | unclear |
| Q2* | dev_val | cis_flat | xa_cos_awgn | SYSTEM-LEVEL: Shen-CIS vs CrossAttn (different input/model/recipe) | 6 | 42.66 | 38.22 | 46.86 | 6/6 | 10/10 | 41.73 | system-level (no verdict) |
| Q2* | confirm | cis_flat | xa_cos_awgn | SYSTEM-LEVEL: Shen-CIS vs CrossAttn (different input/model/recipe) | 6 | 38.45 | 34.1 | 46.81 | 6/6 | 8/8 | 53.1 | system-level (no verdict) |
| Q3 | dev_val | xa_oobnear_cos | xa_cos | OOB pathway near band only (no AWGN) | 6 | -0.69 | -5.23 | 2.74 | 3/6 | 4/10 | -2.0 | unclear |
| Q3 | confirm | xa_oobnear_cos | xa_cos | OOB pathway near band only (no AWGN) | 6 | -1.05 | -8.08 | 7.19 | 2/6 | 2/8 | -5.4 | unclear |
| Q3 | dev_val | xa_oobnear_cos_awgn | xa_cos_awgn | OOB pathway near band only (AWGN) | 6 | 1.86 | -1.83 | 6.25 | 5/6 | 6/10 | -4.65 | unclear |
| Q3 | confirm | xa_oobnear_cos_awgn | xa_cos_awgn | OOB pathway near band only (AWGN) | 6 | 0.39 | -4.5 | 4.96 | 3/6 | 4/8 | -7.48 | unclear |
| Q3 | dev_val | xa_lp200_cos | xa_cos | input band-limited |f|<=200k (no AWGN) | 6 | 1.39 | -4.58 | 4.88 | 4/6 | 7/10 | -3.73 | unclear |
| Q3 | confirm | xa_lp200_cos | xa_cos | input band-limited |f|<=200k (no AWGN) | 6 | 0.32 | -4.21 | 4.94 | 4/6 | 4/8 | -4.62 | unclear |
| Q3 | dev_val | xa_lp200_cos_awgn | xa_cos_awgn | input band-limited |f|<=200k (AWGN) | 6 | 1.92 | -1.99 | 7.77 | 4/6 | 6/10 | -3.8 | unclear |
| Q3 | confirm | xa_lp200_cos_awgn | xa_cos_awgn | input band-limited |f|<=200k (AWGN) | 6 | 0.42 | -4.15 | 4.01 | 4/6 | 5/8 | -4.63 | unclear |
| Q3 | dev_val | xa_nooob_cos | xa_cos | remove explicit OOB pathway (no AWGN) | 6 | -7.27 | -24.47 | 19.24 | 2/6 | 2/10 | 0.23 | unclear |
| Q3 | confirm | xa_nooob_cos | xa_cos | remove explicit OOB pathway (no AWGN) | 6 | -13.73 | -35.17 | 11.84 | 2/6 | 0/8 | -3.3 | unclear |
| Q3 | dev_val | xa_nooob_cos_awgn | xa_cos_awgn | remove explicit OOB pathway (AWGN) | 6 | -1.12 | -17.68 | 14.52 | 4/6 | 4/10 | 5.42 | unclear |
| Q3 | confirm | xa_nooob_cos_awgn | xa_cos_awgn | remove explicit OOB pathway (AWGN) | 6 | -7.4 | -27.05 | 6.49 | 1/6 | 3/8 | 2.0 | harms |
| Q3 | dev_val | xa_trueib_cos | xa_nooob_cos | strict IB input vs full input, both without OOB pathway (no AWGN) | 6 | 44.94 | 3.49 | 72.73 | 6/6 | 10/10 | 49.77 | improves |
| Q3 | confirm | xa_trueib_cos | xa_nooob_cos | strict IB input vs full input, both without OOB pathway (no AWGN) | 6 | 40.09 | 2.39 | 68.4 | 6/6 | 8/8 | 43.32 | improves |
| Q3 | dev_val | xa_trueib_cos_awgn | xa_nooob_cos_awgn | strict IB input vs full input, both without OOB pathway (AWGN) | 6 | 45.39 | 33.08 | 62.55 | 6/6 | 10/10 | 45.95 | improves |
| Q3 | confirm | xa_trueib_cos_awgn | xa_nooob_cos_awgn | strict IB input vs full input, both without OOB pathway (AWGN) | 6 | 42.76 | 30.07 | 59.49 | 6/6 | 8/8 | 40.58 | improves |
| Q3 | dev_val | xa_trueib_cos | xa_cos | TrueIB vs full CrossAttn-OOB (no AWGN) | 6 | 37.67 | 22.73 | 56.44 | 6/6 | 10/10 | 50.0 | improves |
| Q3 | confirm | xa_trueib_cos | xa_cos | TrueIB vs full CrossAttn-OOB (no AWGN) | 6 | 26.35 | 10.16 | 40.27 | 6/6 | 8/8 | 40.02 | improves |
| Q3 | dev_val | xa_trueib_cos_awgn | xa_cos_awgn | TrueIB vs full CrossAttn-OOB (AWGN) | 6 | 44.27 | 38.77 | 48.59 | 6/6 | 10/10 | 51.37 | improves |
| Q3 | confirm | xa_trueib_cos_awgn | xa_cos_awgn | TrueIB vs full CrossAttn-OOB (AWGN) | 6 | 35.36 | 31.71 | 43.99 | 6/6 | 8/8 | 42.58 | improves |
| Q4 | dev_val | tf_cos | xa_cos | time/frequency split tokens (no AWGN) | 6 | -5.46 | -10.01 | -0.3 | 0/6 | 3/10 | 3.33 | harms |
| Q4 | confirm | tf_cos | xa_cos | time/frequency split tokens (no AWGN) | 6 | -7.82 | -14.49 | -2.1 | 0/6 | 0/8 | -6.38 | harms |
| Q4 | dev_val | tf_cos_awgn | xa_cos_awgn | time/frequency split tokens (AWGN) | 6 | 0.24 | -10.65 | 11.78 | 2/6 | 6/10 | -9.53 | unclear |
| Q4 | confirm | tf_cos_awgn | xa_cos_awgn | time/frequency split tokens (AWGN) | 6 | -0.63 | -15.06 | 9.64 | 3/6 | 4/8 | -10.73 | unclear |

## Per-receiver seed-mean accuracy (%)

### dev_val

| arm | f0:b200_mini_1 | f0:pluto_1 | f0:rtl_2 | f0:rtl_5 | f0:rtl_7 | f1:b210_1 | f1:rtl_1 | f1:rtl_3 | f1:rtl_6 | f1:rtl_9 |
|---|---|---|---|---|---|---|---|---|---|---|
| xa_trueib_cos_awgn | 92.1 | 91.6 | 94.5 | 94.8 | 96.4 | 95.0 | 95.1 | 94.6 | 95.9 | 97.5 |
| cis_ib_flat | 96.0 | 72.0 | 96.4 | 97.1 | 97.9 | 97.5 | 97.0 | 96.4 | 97.4 | 98.6 |
| cis_flat | 98.6 | 82.0 | 89.5 | 91.2 | 97.8 | 98.9 | 89.4 | 90.7 | 97.5 | 95.6 |
| cis_flat_noawgn | 93.5 | 66.5 | 78.0 | 81.5 | 88.0 | 95.7 | 78.4 | 82.7 | 93.2 | 85.7 |
| xa_trueib_cos | 83.3 | 89.1 | 83.1 | 85.2 | 90.0 | 77.6 | 74.4 | 73.8 | 81.5 | 79.3 |
| cis_gap_noawgn | 81.9 | 66.9 | 69.5 | 65.0 | 81.8 | 80.0 | 55.4 | 60.5 | 68.5 | 66.0 |
| cis_gap | 45.6 | 63.8 | 64.5 | 64.6 | 67.6 | 40.3 | 57.6 | 58.2 | 61.8 | 62.6 |
| xa_lp200_cos_awgn | 54.9 | 55.3 | 44.4 | 54.7 | 45.9 | 73.5 | 51.7 | 60.8 | 36.5 | 46.1 |
| xa_oobnear_cos_awgn | 52.6 | 57.5 | 48.6 | 50.2 | 48.0 | 73.8 | 53.1 | 57.6 | 35.6 | 46.1 |
| tf_cos_awgn | 52.2 | 55.1 | 47.7 | 49.1 | 45.8 | 65.7 | 55.3 | 58.8 | 30.7 | 46.5 |
| xa_cos_awgn | 46.7 | 52.7 | 46.7 | 51.2 | 40.7 | 66.4 | 52.7 | 58.2 | 40.2 | 49.0 |
| xa_nooob_cos_awgn | 47.5 | 48.4 | 46.2 | 47.8 | 48.3 | 60.9 | 49.0 | 49.4 | 45.7 | 50.2 |
| xa_lp200_cos | 53.5 | 57.0 | 35.1 | 49.4 | 49.8 | 73.5 | 20.1 | 44.5 | 42.9 | 28.7 |
| xa_const | 57.6 | 60.2 | 32.9 | 43.0 | 45.1 | 64.1 | 21.0 | 46.9 | 40.4 | 33.7 |
| xa_cos | 53.2 | 58.6 | 34.7 | 40.4 | 44.9 | 69.8 | 23.8 | 45.3 | 41.8 | 28.3 |
| xa_oobnear_cos | 51.1 | 54.8 | 31.4 | 43.2 | 48.0 | 64.7 | 21.8 | 47.8 | 41.0 | 30.1 |
| xa_p1fixed | 50.1 | 49.9 | 32.3 | 42.6 | 49.0 | 46.6 | 28.5 | 38.0 | 36.7 | 30.6 |
| tf_cos | 41.7 | 47.0 | 27.2 | 32.3 | 39.8 | 53.3 | 27.5 | 49.6 | 36.2 | 31.7 |
| xa_nooob_cos | 38.1 | 37.7 | 24.1 | 28.7 | 41.4 | 48.1 | 34.4 | 38.0 | 39.1 | 38.5 |

### confirm

| arm | f0:b200_1 | f0:n210_1 | f0:rtl_4 | f0:rtl_8 | f1:b200_1 | f1:n210_1 | f1:rtl_4 | f1:rtl_8 |
|---|---|---|---|---|---|---|---|---|
| xa_trueib_cos_awgn | 80.0 | 97.5 | 94.7 | 96.8 | 80.2 | 96.5 | 95.2 | 96.0 |
| cis_ib_flat | 87.8 | 99.0 | 97.9 | 98.2 | 90.0 | 98.2 | 97.0 | 97.7 |
| cis_flat | 90.5 | 96.5 | 92.1 | 99.4 | 95.2 | 96.8 | 93.5 | 97.8 |
| cis_flat_noawgn | 80.5 | 92.3 | 78.1 | 98.6 | 87.9 | 94.5 | 81.6 | 94.6 |
| xa_trueib_cos | 72.2 | 91.4 | 84.6 | 88.9 | 65.8 | 82.0 | 74.5 | 81.0 |
| cis_gap_noawgn | 66.4 | 83.5 | 68.7 | 88.0 | 59.7 | 78.7 | 68.8 | 61.5 |
| cis_gap | 31.9 | 67.8 | 62.5 | 66.0 | 27.8 | 64.3 | 57.6 | 61.0 |
| xa_lp200_cos_awgn | 49.1 | 55.9 | 53.4 | 86.4 | 63.0 | 79.7 | 37.0 | 32.8 |
| xa_oobnear_cos_awgn | 42.7 | 59.9 | 55.0 | 91.5 | 63.7 | 77.9 | 36.5 | 29.9 |
| tf_cos_awgn | 47.7 | 57.8 | 56.3 | 90.0 | 62.2 | 71.8 | 36.5 | 26.7 |
| xa_cos_awgn | 43.4 | 54.2 | 51.9 | 91.0 | 59.4 | 79.2 | 37.4 | 37.6 |
| xa_nooob_cos_awgn | 44.1 | 50.3 | 48.0 | 52.5 | 55.7 | 61.8 | 39.4 | 43.0 |
| xa_lp200_cos | 51.4 | 61.1 | 28.8 | 88.0 | 67.4 | 74.3 | 21.2 | 39.9 |
| xa_const | 53.2 | 60.4 | 31.6 | 85.1 | 59.9 | 64.4 | 18.1 | 41.7 |
| xa_cos | 48.8 | 63.2 | 30.6 | 87.9 | 61.0 | 67.6 | 25.8 | 44.5 |
| xa_oobnear_cos | 48.6 | 58.1 | 30.2 | 87.2 | 62.3 | 73.4 | 20.4 | 40.9 |
| xa_p1fixed | 51.6 | 49.0 | 28.8 | 60.0 | 44.8 | 46.9 | 14.8 | 36.0 |
| tf_cos | 43.0 | 52.4 | 27.1 | 82.2 | 54.1 | 56.8 | 19.4 | 32.1 |
| xa_nooob_cos | 41.8 | 40.3 | 29.1 | 53.8 | 42.2 | 52.8 | 22.5 | 37.2 |
