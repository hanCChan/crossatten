# 02-single-factor-screen smoke: auto-generated tables

run table sha256: `9c3b25ba595a4cec0b5c87ccadf9dd3d40739011081536bf34902c8fc31a58fc`

## Arms (selected checkpoint; %, RX-equal mean over the role's receivers)

| arm | n_fits | params | dev_val_acc_mean | dev_val_acc_sd | dev_val_fold0 | dev_val_fold1 | dev_val_f1_mean | dev_val_worst_rx_seedmean | dev_val_worst_rx_name | dev_val_within_fold_seed_sd | src_val_acc | best_epoch_median | train_s_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s02_cis_rank16 | 1 | 658314 | 11.87 | 0.0 | 11.87 | None | 4.06 | 9.5 | rtl_5 | 0.0 | 12.6 | 2.0 | 1.3 |
| s02_depth4 | 1 | 622246 | 10.54 | 0.0 | 10.54 | None | 3.58 | 10.2 | rtl_7 | 0.0 | 9.55 | 2.0 | 1.8 |
| xa_trueib_cos_awgn | 1 | 580008 | 10.16 | 0.0 | 10.16 | None | 2.29 | 9.95 | rtl_5 | 0.0 | 10.5 | 2.0 | 3.3 |
| s02_awgn10 | 1 | 580008 | 10.16 | 0.0 | 10.16 | None | 2.29 | 9.95 | rtl_5 | 0.0 | 10.5 | 2.0 | 1.5 |
| s02_base | 1 | 580008 | 10.16 | 0.0 | 10.16 | None | 2.29 | 9.95 | rtl_5 | 0.0 | 10.5 | 2.0 | 1.5 |
| s02_ep200 | 1 | 580008 | 10.16 | 0.0 | 10.16 | None | 2.29 | 9.95 | rtl_5 | 0.0 | 10.5 | 2.0 | 1.7 |
| s02_rawoob_full | 1 | 625896 | 10.1 | 0.0 | 10.1 | None | 4.22 | 9.65 | pluto_1 | 0.0 | 10.8 | 1.0 | 2.3 |
| s02_combfft | 1 | 580008 | 10.09 | 0.0 | 10.09 | None | 2.05 | 10.0 | rtl_5 | 0.0 | 10.15 | 2.0 | 2.3 |
| s02_cisoob | 1 | 619816 | 10.08 | 0.0 | 10.08 | None | 4.4 | 9.45 | rtl_7 | 0.0 | 10.15 | 2.0 | 2.2 |
| s02_tfsplit | 1 | 583294 | 10.04 | 0.0 | 10.04 | None | 2.02 | 9.9 | rtl_2 | 0.0 | 10.0 | 2.0 | 2.9 |
| s02_meanchirp | 1 | 580008 | 10.0 | 0.0 | 10.0 | None | 1.82 | 10.0 | b200_mini_1 | 0.0 | 10.0 | 2.0 | 1.6 |
| s02_rawoob_nearL | 1 | 625896 | 9.98 | 0.0 | 9.98 | None | 3.26 | 9.55 | rtl_5 | 0.0 | 10.2 | 1.0 | 2.5 |
| cis_flat | 1 | 13442954 | 9.93 | 0.0 | 9.93 | None | 2.42 | 9.2 | rtl_2 | 0.0 | 10.65 | 2.0 | 1.7 |
| s02_dechirp | 1 | 580008 | 9.89 | 0.0 | 9.89 | None | 1.82 | 9.8 | pluto_1 | 0.0 | 9.95 | 2.0 | 2.1 |
| s02_rawoob_near | 1 | 625896 | 9.76 | 0.0 | 9.76 | None | 4.32 | 9.45 | rtl_5 | 0.0 | 11.1 | 1.0 | 2.6 |

## Pre-registered comparisons (delta = B - A, pp)

| Q | role | B | A | factor | pairs | mean | min | max | +pairs | +RX | worst-RX delta | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | dev_val | s02_meanchirp | s02_base | coherent 8-symbol mean chirp (drops residual/transient) | 1 | -0.16 | -0.16 | -0.16 | 0/1 | 1/5 | 0.05 | unclear |
| B | dev_val | s02_combfft | s02_base | FFT view: comb bins only | 1 | -0.07 | -0.07 | -0.07 | 0/1 | 2/5 | 0.05 | unclear |
| B | dev_val | s02_dechirp | s02_base | dechirped input | 1 | -0.27 | -0.27 | -0.27 | 0/1 | 1/5 | -0.15 | unclear |
| A | dev_val | s02_rawoob_full | s02_base | + raw OOB keys/values, all |f|>62.5 kHz | 1 | -0.06 | -0.06 | -0.06 | 0/1 | 2/5 | -0.3 | unclear |
| A | dev_val | s02_rawoob_near | s02_base | + raw OOB keys/values, 62.5-200 kHz | 1 | -0.4 | -0.4 | -0.4 | 0/1 | 0/5 | -0.5 | unclear |
| A | dev_val | s02_rawoob_nearL | s02_base | + raw OOB keys/values, left near band only | 1 | -0.18 | -0.18 | -0.18 | 0/1 | 1/5 | -0.4 | unclear |
| A | dev_val | s02_cisoob | s02_base | + CIS-quotient OOB keys/values | 1 | -0.08 | -0.08 | -0.08 | 0/1 | 2/5 | -0.5 | unclear |
| C | dev_val | s02_depth4 | s02_base | 4 mixer blocks | 1 | 0.38 | 0.38 | 0.38 | 1/1 | 5/5 | 0.25 | unclear |
| C | dev_val | s02_tfsplit | s02_base | time/frequency split tokens (strict IB) | 1 | -0.12 | -0.12 | -0.12 | 0/1 | 1/5 | -0.05 | unclear |
| D | dev_val | s02_ep200 | s02_base | 200-epoch horizon | 1 | 0.0 | 0.0 | 0.0 | 0/1 | 0/5 | 0.0 | unclear |
| D | dev_val | s02_awgn10 | s02_base | AWGN U[10,50) instead of U[0,50) | 1 | 0.0 | 0.0 | 0.0 | 0/1 | 0/5 | 0.0 | unclear |
| eff | dev_val | s02_cis_rank16 | cis_flat | rank-16 head vs Flatten-Dense (stage-01 fits, same data streams) | 1 | 1.94 | 1.94 | 1.94 | 1/1 | 4/5 | 0.3 | unclear |
| ref | dev_val | s02_base | xa_trueib_cos_awgn | determinism check: re-run of stage-01 arm (expected 0.00) | 1 | 0.0 | 0.0 | 0.0 | 0/1 | 0/5 | 0.0 | reference (no verdict) |
| ref | dev_val | s02_base | s02_cis_rank16 | SYSTEM: CrossAttn-TrueIB vs rank-16 CIS | 1 | -1.71 | -1.71 | -1.71 | 0/1 | 1/5 | 0.45 | reference (no verdict) |

## Per-receiver seed-mean accuracy (%)

### dev_val

| arm | f0:b200_mini_1 | f0:pluto_1 | f0:rtl_2 | f0:rtl_5 | f0:rtl_7 |
|---|---|---|---|---|---|
| s02_cis_rank16 | 11.9 | 10.3 | 12.3 | 9.5 | 15.2 |
| s02_depth4 | 10.5 | 10.5 | 10.4 | 11.0 | 10.2 |
| xa_trueib_cos_awgn | 10.2 | 10.2 | 10.3 | 10.0 | 10.1 |
| s02_awgn10 | 10.2 | 10.2 | 10.3 | 10.0 | 10.1 |
| s02_base | 10.2 | 10.2 | 10.3 | 10.0 | 10.1 |
| s02_ep200 | 10.2 | 10.2 | 10.3 | 10.0 | 10.1 |
| s02_rawoob_full | 10.1 | 9.7 | 10.5 | 10.2 | 10.1 |
| s02_combfft | 10.1 | 10.1 | 10.2 | 10.0 | 10.2 |
| s02_cisoob | 9.6 | 10.5 | 10.1 | 10.7 | 9.4 |
| s02_tfsplit | 10.0 | 10.1 | 9.9 | 10.1 | 10.1 |
| s02_meanchirp | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| s02_rawoob_nearL | 9.7 | 10.7 | 10.2 | 9.6 | 9.8 |
| cis_flat | 10.0 | 10.1 | 9.2 | 10.1 | 10.3 |
| s02_dechirp | 9.8 | 9.8 | 10.0 | 10.0 | 9.8 |
| s02_rawoob_near | 9.6 | 9.8 | 10.0 | 9.4 | 10.1 |
