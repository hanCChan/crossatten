# S03 Round 1: Complete Results

Protocol P-H5x160, all eight preregistered arms, 200 epochs with strict IB + AWGN. 32 exploration fits and 48 new-seed confirmation fits. No blind RX read; only source validation CE selected checkpoints. Confirm RX are report-only and have been repeatedly inspected.

## Decision

Confirmation dev_val verdicts: `{"s03_hstem": "improves", "s03_ifreq": "unclear", "s03_nofft": "unclear", "s03_preln": "unclear", "s03_preln_d4": "unclear", "s03_resid": "harms", "s03_symstack": "harms"}`.

Factors eligible for a user-approved next round: `['s03_hstem']`. Eligibility is not authorization to combine factors; stop after this report.

Structural hypothesis verdicts: s03_resid=harms, s03_symstack=harms, s03_ifreq=unclear. Symstack's clean single-factor comparison is against nofft; symstack vs preln also removes FFT and is only a system reference. Pre-LN is an engineering change, not claimed as the method contribution.

## Confirmation Accuracy

RX-equal mean and fit SD (%), selected checkpoint; all six fits per arm. Fit SD includes fold differences; within-fold seed SD is reported separately. Seeds are training repeats, not additional independent receivers.

| arm | n_fits | params | dev_val_acc_mean | dev_val_acc_sd | dev_val_within_fold_seed_sd | dev_val_worst_rx_seedmean | dev_val_f1_mean | confirm_acc_mean | confirm_acc_sd | confirm_worst_rx_seedmean | src_val_acc | best_epoch_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s03_hstem | 6 | 142392 | 97.48 | 0.41 | 0.24 | 95.57 | 97.48 | 96.84 | 0.58 | 90.22 | 98.25 | 165.0 |
| s03_base | 6 | 580008 | 95.84 | 0.69 | 0.37 | 92.65 | 95.82 | 93.89 | 0.48 | 83.92 | 96.61 | 180.0 |
| s03_preln_d4 | 6 | 622374 | 95.58 | 0.79 | 0.74 | 93.98 | 95.56 | 93.73 | 0.75 | 82.67 | 96.52 | 166.0 |
| s03_ifreq | 6 | 580360 | 95.5 | 1.23 | 0.52 | 90.95 | 95.46 | 93.94 | 0.91 | 81.13 | 96.6 | 164.5 |
| s03_nofft | 6 | 579912 | 95.48 | 0.95 | 0.38 | 90.63 | 95.46 | 93.65 | 0.39 | 83.15 | 96.43 | 147.0 |
| s03_preln | 6 | 580136 | 95.48 | 0.91 | 0.14 | 90.98 | 95.45 | 93.79 | 0.59 | 83.4 | 96.42 | 156.5 |
| s03_symstack | 6 | 579912 | 90.27 | 0.75 | 0.67 | 81.35 | 90.17 | 89.02 | 2.03 | 77.33 | 91.37 | 166.5 |
| s03_resid | 6 | 580584 | 88.01 | 1.68 | 1.37 | 85.35 | 87.85 | 87.21 | 1.7 | 79.35 | 89.63 | 89.5 |

## Exploration Versus Confirmation

All four exploration fits per arm, dev_val only; none were used to remove arms from confirmation. Compare against the six new-seed fits above.

| arm | n_fits | dev_val_acc_mean | dev_val_acc_sd | dev_val_within_fold_seed_sd | dev_val_worst_rx_seedmean |
|---|---|---|---|---|---|
| s03_hstem | 4 | 97.27 | 0.55 | 0.64 | 94.6 |
| s03_preln_d4 | 4 | 96.03 | 0.85 | 0.14 | 92.85 |
| s03_ifreq | 4 | 95.93 | 0.73 | 0.24 | 91.95 |
| s03_nofft | 4 | 95.77 | 0.81 | 0.48 | 93.1 |
| s03_preln | 4 | 95.3 | 0.53 | 0.4 | 92.3 |
| s03_base | 4 | 94.27 | 1.83 | 1.85 | 89.7 |
| s03_symstack | 4 | 91.81 | 0.85 | 0.87 | 86.2 |
| s03_resid | 4 | 88.09 | 0.75 | 0.8 | 85.85 |

| arm | exploration | confirmation |
|---|---|---|
| s03_hstem | improves | improves |
| s03_ifreq | unclear | unclear |
| s03_nofft | unclear | unclear |
| s03_preln | improves | unclear |
| s03_preln_d4 | unclear | unclear |
| s03_resid | harms | harms |
| s03_symstack | harms | harms |

## Paired Comparisons

Delta is B minus A, pp. Decisions below use dev_val only. Confirm comparisons and every per-RX mean are in [confirm_tables.md](summary/confirm_tables.md).

| q | b | a | n_pairs | delta_mean | delta_min | delta_max | pos_pairs | worst_rx_delta | verdict |
|---|---|---|---|---|---|---|---|---|---|
| P1 | s03_preln | s03_base | 6 | -0.36 | -0.8 | 0.29 | 2 | -1.67 | unclear |
| P1 | s03_preln_d4 | s03_preln | 6 | 0.1 | -2.04 | 0.99 | 4 | 3.0 | unclear |
| P2 | s03_resid | s03_preln | 6 | -7.47 | -11.1 | -5.44 | 0 | -5.63 | harms |
| P2 | s03_nofft | s03_preln | 6 | -0.0 | -0.48 | 0.48 | 3 | -0.35 | unclear |
| P2 | s03_symstack | s03_nofft | 6 | -5.2 | -6.12 | -4.35 | 0 | -9.28 | harms |
| ref_structure | s03_symstack | s03_preln | 6 | -5.21 | -6.6 | -4.11 | 0 | -9.63 | reference (no verdict) |
| P3 | s03_ifreq | s03_preln | 6 | 0.02 | -0.6 | 0.77 | 3 | -0.03 | unclear |
| P4 | s03_hstem | s03_preln | 6 | 2.0 | 1.29 | 2.66 | 6 | 4.58 | improves |
| ref_determinism | s03_base | s02_ep200 | 6 | 0.0 | 0.0 | 0.0 | 0 | 0.0 | reference (no verdict) |

## Per-Receiver Accuracy

Seed-mean accuracy (%) within each fold; all three confirmation seeds. Confirm receivers repeat across folds and are report-only.

### dev_val

| arm | f0:b200_mini_1 | f0:pluto_1 | f0:rtl_2 | f0:rtl_5 | f0:rtl_7 | f1:b210_1 | f1:rtl_1 | f1:rtl_3 | f1:rtl_6 | f1:rtl_9 |
|---|---|---|---|---|---|---|---|---|---|---|
| s03_base | 94.13 | 92.65 | 96.05 | 96.43 | 97.27 | 96.67 | 95.82 | 95.40 | 96.52 | 97.48 |
| s03_hstem | 97.13 | 95.57 | 97.28 | 97.67 | 98.20 | 97.98 | 97.60 | 96.93 | 97.95 | 98.52 |
| s03_ifreq | 92.55 | 90.95 | 95.63 | 96.03 | 97.13 | 96.48 | 96.33 | 95.65 | 96.92 | 97.32 |
| s03_nofft | 93.35 | 90.63 | 95.88 | 96.48 | 97.00 | 96.58 | 95.78 | 95.58 | 96.02 | 97.43 |
| s03_preln | 94.02 | 90.98 | 95.62 | 96.00 | 96.68 | 96.08 | 96.00 | 95.58 | 96.57 | 97.25 |
| s03_preln_d4 | 93.98 | 94.47 | 95.53 | 96.05 | 97.32 | 96.10 | 95.18 | 94.90 | 95.60 | 96.65 |
| s03_resid | 88.13 | 85.62 | 89.05 | 89.85 | 90.82 | 88.90 | 85.58 | 85.35 | 88.50 | 88.28 |
| s03_symstack | 89.68 | 81.35 | 91.68 | 92.45 | 94.17 | 89.57 | 90.28 | 90.23 | 90.35 | 92.97 |

### confirm

| arm | f0:b200_1 | f0:n210_1 | f0:rtl_4 | f0:rtl_8 | f1:b200_1 | f1:n210_1 | f1:rtl_4 | f1:rtl_8 |
|---|---|---|---|---|---|---|---|---|
| s03_base | 83.92 | 98.05 | 96.58 | 97.68 | 85.43 | 96.33 | 96.00 | 97.12 |
| s03_hstem | 90.22 | 99.22 | 98.65 | 98.78 | 94.02 | 97.82 | 97.82 | 98.23 |
| s03_ifreq | 81.13 | 97.90 | 96.57 | 97.52 | 88.53 | 96.43 | 96.63 | 96.80 |
| s03_nofft | 83.15 | 97.82 | 96.42 | 97.78 | 85.82 | 95.45 | 96.02 | 96.75 |
| s03_preln | 83.40 | 97.77 | 96.38 | 97.67 | 85.63 | 96.40 | 96.10 | 96.95 |
| s03_preln_d4 | 82.67 | 98.45 | 96.20 | 97.93 | 87.37 | 95.48 | 95.77 | 95.97 |
| s03_resid | 79.35 | 91.37 | 90.47 | 90.38 | 81.02 | 89.27 | 87.10 | 88.70 |
| s03_symstack | 79.70 | 94.08 | 93.15 | 95.73 | 77.33 | 90.23 | 90.72 | 91.20 |

## Cost

NVIDIA A100-SXM4-40GB, PyTorch 2.5.0+cu124; random weights. Input is raw IQ already on device; preprocessing and all model execution are included in end-to-end timing. Convolution/matrix FLOPs are torch-counted; FFT and elementwise operations are not fully counted. Benchmark ran before formal training, without competing GPU experiments.

| arm | params | mflops_per_packet | gpu_b1_ms_p50 | gpu_b1_ms_p95 | gpu_b512_pkts_per_s | cpu4t_b1_ms_p50 | cpu4t_b1_ms_p95 |
|---|---|---|---|---|---|---|---|
| s03_base | 580008 | 139.2 | 2.321 | 3.469 | 41375 | 6.44 | 8.17 |
| s03_preln | 580136 | 139.2 | 2.276 | 3.533 | 41234 | 7.12 | 9.05 |
| s03_preln_d4 | 622374 | 142.3 | 3.075 | 4.524 | 39055 | 8.34 | 10.61 |
| s03_resid | 580584 | 146.5 | 2.328 | 3.744 | 40470 | 6.53 | 8.52 |
| s03_nofft | 579912 | 135.5 | 1.657 | 1.793 | 46354 | 4.64 | 5.43 |
| s03_symstack | 579912 | 135.5 | 1.726 | 1.794 | 46011 | 3.6 | 4.17 |
| s03_ifreq | 580360 | 142.9 | 2.45 | 3.457 | 40532 | 5.76 | 8.2 |
| s03_hstem | 142392 | 190.6 | 2.53 | 2.99 | 36101 | 6.08 | 7.57 |

Observed maximum allocated GPU memory in confirmation training/evaluation: 3.36 GB per fit (includes saved-checkpoint batch500 evaluation). This exceeds the specification's 3.1-GB estimate; no tasks were omitted because of it. Training itself remains batch64.


Fewer parameters do not necessarily mean less compute: hstem removes most tokenizer weights but executes extra convolutions at higher resolution. Single-packet GPU times include host dispatch variability; use distributions, not isolated p50 differences, for efficiency claims.

## Post-Hoc Token Scale

A CPU-only diagnostic compares initialization and selected exploration weights on balanced source-validation packets, without target data or decision changes. All arms, both folds, and both exploration seeds are reported in [token_scale.md](summary/token_scale.md). This diagnostic was added after the exploration freeze and is not a preregistered selection criterion.

| arm | trained_rms_range | saturated_pct_range |
|---|---|---|
| s03_base | 106.504-176.560 | 99.94-100.00 |
| s03_hstem | 0.989-1.016 | 82.40-96.77 |
| s03_ifreq | 0.977-1.009 | 82.83-99.78 |
| s03_nofft | 0.951-1.035 | 79.09-99.36 |
| s03_preln | 0.947-1.013 | 72.98-95.35 |
| s03_preln_d4 | 1.013-1.063 | 94.51-99.94 |
| s03_resid | 0.995-1.023 | 75.70-98.06 |
| s03_symstack | 0.974-1.031 | 88.61-99.56 |

Pre-LN controls token amplitude but does not eliminate large learned q/k scores or sigmoid saturation. These measurements do not establish a causal explanation of historical depth4 failures.

## Mechanisms


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

Interventions use each of the six confirmed checkpoints, without fitting or selection. Unchanged and intervened evaluation share batch64 and identical RX packets. Packet-shuffle within a batch is a limited diagnostic if packets share identity; zero-residual and symbol misalignment are also reported. A performance drop is evidence that the representation is used, not proof that the model uses only that representation or that the signal has a unique causal interpretation. No second-round architecture is selected here.

## Verification And Failures

Legacy core/model2/bit-regression tests passed before training. New tests verify exact legacy weights/logits, equal-parameter symstack/nofft shared initialization, finite gradients, unit-RMS residual/reconstruction/periodic-zero behavior, instantaneous-frequency reference, and symbol-local token receptive field.

Saved-result checks passed for every smoke, exploration, and confirmation fit: arm/recipe, protocol hash, seed, parameters, full epoch budget, source-CE-minimum checkpoint, per-RX confusion denominators, macro-F1, RX-equal summaries, finite weights, and frozen table hashes. See [verification.json](summary/verification.json).

| stage | total | skipped | completed | failures |
|---|---|---|---|---|
| smoke | 8 | 0 | 8 | 0 |
| explore | 32 | 0 | 32 | 0 |
| confirm | 48 | 0 | 48 | 0 |

Smoke numbers are execution checks, not scientific accuracy results. Baseline rerun determinism is recorded as the s03_base vs s02_ep200 reference comparison, with selected weight and prediction equality checked separately. See exploration/confirmation tables and individual config/history/metrics for all seeds, folds, receiver scores, budgets, and checkpoint selection.

## Limitations And Stop

Exploration is not confirmation; report all negative and unclear factors. Only confirmation dev_val determines eligibility. Confirm RX have been reused many times and cannot support an independent generalization claim. Neither SOTA-on-Shen training nor WiSig fair multi-seed retraining was launched. WiSig preparation and released-weight validation are reported separately in [05-wisig/RESULTS.md](../05-wisig/RESULTS.md), including the 25-Msps audit and incomplete LTF crop. Literature and numerical-scale diagnostics do not establish novelty or prove the cause of historical depth4 instability.

Round 1 is complete. User decides whether to proceed with combinations, stage04, and WiSig training. On 2026-10-25 the user's latest request authorized organizing and publishing existing results to GitHub, overriding the earlier no-push instruction without authorizing more training. See the repository-wide [RESULTS.md](../RESULTS.md). S03 model.pt files remain local and are not included in Git or the earlier stage00-02 checkpoint release.
