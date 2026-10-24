# Stage 01 explore: auto-generated tables
run table sha256: `50aa2dc75bb566e5ad1d9cd940739a3ac3f10f28f6698661ca5bb8f1f0ede208`

## Arms (selected checkpoint; accuracy in %, RX-equal mean over the role's receivers)
| arm | n_fits | params | dev_val_acc_mean | dev_val_acc_sd | dev_val_fold0 | dev_val_fold1 | dev_val_f1_mean | dev_val_worst_rx_seedmean | dev_val_worst_rx_name | dev_val_within_fold_seed_sd | src_val_acc | best_epoch_median | train_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cis_ib_flat | 4 | 13442954 | 94.06 | 2.89 | 91.64 | 96.49 | 93.59 | 71.97 | pluto_1 | 0.84 | 96.81 | 207.5 | 1651.5 |
| cis_flat | 4 | 13442954 | 93.2 | 1.47 | 92.03 | 94.38 | 92.98 | 85.07 | pluto_1 | 0.58 | 97.61 | 255.0 | 1815.4 |
| xa_trueib_cos_awgn | 4 | 580008 | 91.48 | 5.71 | 87.75 | 95.22 | 91.29 | 80.38 | b200_mini_1 | 3.24 | 91.36 | 94.5 | 721.6 |
| cis_flat_noawgn | 4 | 13442954 | 84.46 | 3.56 | 81.45 | 87.47 | 83.55 | 68.78 | pluto_1 | 0.95 | 95.39 | 119.5 | 894.1 |
| xa_trueib_cos | 4 | 580008 | 80.98 | 13.15 | 81.55 | 80.41 | 80.79 | 77.28 | rtl_3 | 16.02 | 81.55 | 59.5 | 724.4 |
| cis_gap_noawgn | 4 | 270218 | 69.01 | 2.36 | 70.32 | 67.69 | 68.12 | 60.05 | rtl_1 | 2.1 | 81.09 | 250.0 | 1594.0 |
| xa_cos_awgn | 4 | 625896 | 56.73 | 3.65 | 54.67 | 58.79 | 54.43 | 46.75 | rtl_6 | 3.28 | 84.0 | 69.5 | 719.9 |
| cis_gap | 4 | 270218 | 56.22 | 10.65 | 62.2 | 50.23 | 52.56 | 36.05 | b210_1 | 8.29 | 52.84 | 246.5 | 1840.3 |
| xa_lp200_cos_awgn | 4 | 625896 | 54.14 | 4.04 | 52.7 | 55.59 | 51.23 | 40.25 | rtl_6 | 4.33 | 83.44 | 56.0 | 750.6 |
| xa_oobnear_cos_awgn | 4 | 625896 | 48.86 | 4.3 | 50.52 | 47.2 | 46.05 | 24.83 | rtl_6 | 4.09 | 81.46 | 53.0 | 715.2 |
| tf_cos_awgn | 4 | 629182 | 48.8 | 3.91 | 49.7 | 47.91 | 45.97 | 28.85 | rtl_6 | 3.35 | 81.66 | 28.0 | 854.8 |
| xa_lp200_cos | 4 | 625896 | 46.95 | 7.03 | 52.78 | 41.11 | 43.57 | 15.15 | rtl_1 | 2.26 | 81.36 | 12.5 | 753.9 |
| xa_oobnear_cos | 4 | 625896 | 46.05 | 6.6 | 51.76 | 40.35 | 42.05 | 13.68 | rtl_1 | 0.35 | 81.66 | 22.5 | 698.5 |
| xa_nooob_cos | 4 | 580008 | 45.58 | 18.57 | 37.84 | 53.31 | 41.42 | 26.18 | rtl_2 | 19.39 | 56.8 | 35.5 | 640.7 |
| xa_cos | 4 | 625896 | 45.38 | 6.38 | 50.07 | 40.69 | 41.74 | 14.35 | rtl_1 | 4.12 | 82.08 | 13.5 | 694.4 |
| xa_const | 4 | 625896 | 45.37 | 4.93 | 48.19 | 42.54 | 41.58 | 21.55 | rtl_1 | 4.49 | 81.0 | 18.0 | 671.4 |
| tf_cos | 4 | 629182 | 39.79 | 1.03 | 39.58 | 40.0 | 35.13 | 24.52 | rtl_1 | 1.05 | 77.74 | 6.5 | 856.0 |
| xa_nooob_cos_awgn | 4 | 580008 | 37.55 | 19.18 | 22.15 | 52.96 | 33.99 | 21.68 | b200_mini_1 | 7.13 | 39.59 | 85.0 | 658.8 |
| xa_p1fixed | 4 | 625896 | 37.07 | 5.72 | 38.36 | 35.77 | 32.56 | 25.68 | rtl_1 | 5.63 | 60.74 | 4.0 | 28.4 |

## Pre-registered one-factor comparisons (delta = B - A, pp)

| Q | role | B | A | factor | pairs | mean | min | max | +pairs | +RX | worst-RX delta | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q1 | dev_val | xa_const | xa_p1fixed | horizon 100ep + source-val checkpoint | 4 | 8.3 | 2.64 | 12.89 | 4/4 | 9/10 | -4.12 | unclear |
| Q1 | dev_val | xa_cos | xa_const | warmup+cosine schedule | 4 | 0.01 | -1.96 | 2.29 | 2/4 | 5/10 | -7.2 | unclear |
| Q1 | dev_val | xa_cos_awgn | xa_cos | AWGN U[0,50) dB | 4 | 11.35 | 4.4 | 19.08 | 4/4 | 7/10 | 32.4 | improves |
| Q2 | dev_val | cis_gap | cis_flat | GAP head vs Flatten-Dense (AWGN) | 4 | -36.99 | -45.99 | -19.43 | 0/4 | 0/10 | -49.02 | harms |
| Q2 | dev_val | cis_gap_noawgn | cis_flat_noawgn | GAP head vs Flatten-Dense (no AWGN) | 4 | -15.45 | -20.96 | -9.6 | 0/4 | 1/10 | -8.73 | harms |
| Q2 | dev_val | cis_flat | cis_flat_noawgn | AWGN on CIS (flatten) | 4 | 8.74 | 6.25 | 10.71 | 4/4 | 10/10 | 16.3 | improves |
| Q2 | dev_val | cis_gap | cis_gap_noawgn | AWGN on CIS (GAP) | 4 | -12.79 | -17.47 | 0.62 | 1/4 | 2/10 | -24.0 | harms |
| Q2 | dev_val | cis_ib_flat | cis_flat | strict-IB CIS vs Shen crop (-203..+195 kHz) | 4 | 0.86 | -0.64 | 2.73 | 2/4 | 6/10 | -13.1 | unclear |
| Q2* | dev_val | cis_flat | xa_cos_awgn | SYSTEM-LEVEL: Shen-CIS vs CrossAttn (different input/model/recipe) | 4 | 36.47 | 34.03 | 39.63 | 4/4 | 10/10 | 38.32 | system-level (no verdict) |
| Q3 | dev_val | xa_oobnear_cos | xa_cos | OOB pathway near band only (no AWGN) | 4 | 0.68 | -2.62 | 4.94 | 2/4 | 8/10 | -0.68 | unclear |
| Q3 | dev_val | xa_oobnear_cos_awgn | xa_cos_awgn | OOB pathway near band only (AWGN) | 4 | -7.87 | -14.44 | -2.45 | 0/4 | 1/10 | -21.92 | harms |
| Q3 | dev_val | xa_lp200_cos | xa_cos | input band-limited |f|<=200k (no AWGN) | 4 | 1.57 | -1.29 | 3.64 | 3/4 | 7/10 | 0.8 | improves |
| Q3 | dev_val | xa_lp200_cos_awgn | xa_cos_awgn | input band-limited |f|<=200k (AWGN) | 4 | -2.59 | -3.68 | -0.98 | 0/4 | 3/10 | -6.5 | harms |
| Q3 | dev_val | xa_nooob_cos | xa_cos | remove explicit OOB pathway (no AWGN) | 4 | 0.2 | -32.39 | 20.35 | 3/4 | 4/10 | 11.82 | unclear |
| Q3 | dev_val | xa_nooob_cos_awgn | xa_cos_awgn | remove explicit OOB pathway (AWGN) | 4 | -19.18 | -34.08 | 1.16 | 1/4 | 1/10 | -25.07 | harms |
| Q3 | dev_val | xa_trueib_cos | xa_nooob_cos | strict IB input vs full input, both without OOB pathway (no AWGN) | 4 | 35.4 | 6.31 | 73.0 | 4/4 | 10/10 | 51.1 | improves |
| Q3 | dev_val | xa_trueib_cos_awgn | xa_nooob_cos_awgn | strict IB input vs full input, both without OOB pathway (AWGN) | 4 | 53.92 | 33.56 | 71.56 | 4/4 | 10/10 | 58.7 | improves |
| Q3 | dev_val | xa_trueib_cos | xa_cos | TrueIB vs full CrossAttn-OOB (no AWGN) | 4 | 35.6 | 22.35 | 52.78 | 4/4 | 10/10 | 62.93 | improves |
| Q3 | dev_val | xa_trueib_cos_awgn | xa_cos_awgn | TrueIB vs full CrossAttn-OOB (AWGN) | 4 | 34.75 | 25.55 | 40.59 | 4/4 | 10/10 | 33.62 | improves |
| Q4 | dev_val | tf_cos | xa_cos | time/frequency split tokens (no AWGN) | 4 | -5.59 | -13.36 | 0.79 | 1/4 | 2/10 | 10.17 | harms |
| Q4 | dev_val | tf_cos_awgn | xa_cos_awgn | time/frequency split tokens (AWGN) | 4 | -7.93 | -17.2 | -1.92 | 0/4 | 1/10 | -17.9 | harms |

## Per-receiver seed-mean accuracy (%)

### dev_val

| arm | f0:b200_mini_1 | f0:pluto_1 | f0:rtl_2 | f0:rtl_5 | f0:rtl_7 | f1:b210_1 | f1:rtl_1 | f1:rtl_3 | f1:rtl_6 | f1:rtl_9 |
|---|---|---|---|---|---|---|---|---|---|---|
| cis_ib_flat | 95.8 | 72.0 | 96.2 | 96.6 | 97.6 | 96.6 | 95.8 | 95.3 | 96.8 | 98.0 |
| cis_flat | 98.7 | 85.1 | 88.4 | 91.1 | 96.9 | 98.9 | 88.5 | 91.2 | 97.8 | 95.5 |
| xa_trueib_cos_awgn | 80.4 | 88.9 | 89.0 | 90.0 | 90.4 | 94.8 | 94.5 | 94.4 | 95.5 | 96.8 |
| cis_flat_noawgn | 92.0 | 68.8 | 77.8 | 79.2 | 89.5 | 95.5 | 81.0 | 84.2 | 92.3 | 84.4 |
| xa_trueib_cos | 77.7 | 84.1 | 78.8 | 81.3 | 85.8 | 79.1 | 78.6 | 77.3 | 82.8 | 84.3 |
| cis_gap_noawgn | 78.6 | 69.8 | 64.7 | 64.7 | 73.8 | 81.3 | 60.0 | 60.7 | 68.1 | 68.2 |
| xa_cos_awgn | 55.7 | 59.7 | 48.9 | 59.5 | 49.6 | 79.8 | 53.5 | 63.8 | 46.8 | 50.0 |
| cis_gap | 47.0 | 65.5 | 65.6 | 65.0 | 67.9 | 36.0 | 52.8 | 54.0 | 53.2 | 55.2 |
| xa_lp200_cos_awgn | 57.1 | 61.1 | 47.1 | 56.9 | 41.3 | 76.6 | 49.9 | 64.1 | 40.2 | 47.1 |
| xa_oobnear_cos_awgn | 53.7 | 57.0 | 50.0 | 53.0 | 39.0 | 64.7 | 46.9 | 53.4 | 24.8 | 46.1 |
| tf_cos_awgn | 51.5 | 56.7 | 48.9 | 49.2 | 42.2 | 59.0 | 54.8 | 54.0 | 28.8 | 43.0 |
| xa_lp200_cos | 60.5 | 61.2 | 41.8 | 47.6 | 52.7 | 73.2 | 15.1 | 48.1 | 38.5 | 30.6 |
| xa_oobnear_cos | 59.1 | 63.0 | 35.1 | 48.4 | 53.2 | 67.5 | 13.7 | 44.7 | 40.3 | 35.5 |
| xa_nooob_cos | 43.0 | 42.4 | 26.2 | 31.6 | 46.0 | 56.2 | 49.3 | 55.2 | 51.7 | 54.1 |
| xa_cos | 57.2 | 59.9 | 34.7 | 46.8 | 51.7 | 64.5 | 14.4 | 49.9 | 39.3 | 35.4 |
| xa_const | 60.1 | 54.8 | 32.8 | 45.9 | 47.5 | 71.0 | 21.6 | 50.0 | 34.5 | 35.6 |
| tf_cos | 41.7 | 55.0 | 28.2 | 33.7 | 39.2 | 54.5 | 24.5 | 46.7 | 41.0 | 33.2 |
| xa_nooob_cos_awgn | 21.7 | 22.8 | 21.7 | 21.8 | 22.9 | 57.3 | 49.9 | 52.8 | 54.8 | 49.9 |
| xa_p1fixed | 40.4 | 42.4 | 29.5 | 34.2 | 45.3 | 50.9 | 25.7 | 38.7 | 34.3 | 29.3 |
