# S09 Trained Source Mechanism Probes

Fixed16 packets/TX, held-source RX only, raw minimum held-CE weights. These are post-training diagnostics, not independent validation or selection. Extra-only intervention keeps original Re/Im/log magnitude unchanged and changes only engineered channels. All-channel perturbation changes the full observed packet.

| Dataset | Feature | Perturbation | Scope | Acc delta pp | Prediction disagreement % |
|---|---|---|---|---:|---:|
| ManyRx | first | circular_shift9 | all_channels | -3.85 | 8.88 |
| ManyRx | first | circular_shift9 | extra_only | -3.11 | 5.88 |
| ManyRx | first | crop_shift9 | all_channels | -8.11 | 14.31 |
| ManyRx | first | crop_shift9 | extra_only | -5.85 | 10.15 |
| ManyRx | first | cfo10khz | all_channels | -33.03 | 38.26 |
| ManyRx | first | cfo10khz | extra_only | -3.81 | 7.75 |
| ManyRx | first | smooth_channel | all_channels | -0.21 | 1.90 |
| ManyRx | first | smooth_channel | extra_only | -0.09 | 0.83 |
| ManyRx | centered | circular_shift9 | all_channels | -0.46 | 4.78 |
| ManyRx | centered | circular_shift9 | extra_only | -0.03 | 0.26 |
| ManyRx | centered | crop_shift9 | all_channels | -5.98 | 12.15 |
| ManyRx | centered | crop_shift9 | extra_only | -4.80 | 9.35 |
| ManyRx | centered | cfo10khz | all_channels | -33.87 | 38.63 |
| ManyRx | centered | cfo10khz | extra_only | -2.98 | 6.52 |
| ManyRx | centered | smooth_channel | all_channels | -0.07 | 1.52 |
| ManyRx | centered | smooth_channel | extra_only | -0.10 | 0.68 |
| ManyRx | curvature | circular_shift9 | all_channels | -0.51 | 5.69 |
| ManyRx | curvature | circular_shift9 | extra_only | +0.00 | 0.00 |
| ManyRx | curvature | crop_shift9 | all_channels | -10.91 | 18.58 |
| ManyRx | curvature | crop_shift9 | extra_only | -9.35 | 15.54 |
| ManyRx | curvature | cfo10khz | all_channels | -41.17 | 47.22 |
| ManyRx | curvature | cfo10khz | extra_only | -11.51 | 17.03 |
| ManyRx | curvature | smooth_channel | all_channels | -0.01 | 1.81 |
| ManyRx | curvature | smooth_channel | extra_only | -0.09 | 0.72 |
| ManySig | first | circular_shift9 | all_channels | -10.09 | 14.81 |
| ManySig | first | circular_shift9 | extra_only | -6.12 | 9.05 |
| ManySig | first | crop_shift9 | all_channels | -21.22 | 26.07 |
| ManySig | first | crop_shift9 | extra_only | -13.05 | 16.31 |
| ManySig | first | cfo10khz | all_channels | -41.24 | 45.02 |
| ManySig | first | cfo10khz | extra_only | -10.42 | 13.09 |
| ManySig | first | smooth_channel | all_channels | -1.01 | 2.77 |
| ManySig | first | smooth_channel | extra_only | +0.07 | 0.55 |
| ManySig | centered | circular_shift9 | all_channels | -3.78 | 8.59 |
| ManySig | centered | circular_shift9 | extra_only | -0.03 | 0.07 |
| ManySig | centered | crop_shift9 | all_channels | -20.54 | 25.68 |
| ManySig | centered | crop_shift9 | extra_only | -11.62 | 15.33 |
| ManySig | centered | cfo10khz | all_channels | -40.27 | 43.78 |
| ManySig | centered | cfo10khz | extra_only | -9.80 | 12.99 |
| ManySig | centered | smooth_channel | all_channels | -0.88 | 2.64 |
| ManySig | centered | smooth_channel | extra_only | -0.00 | 1.01 |
| ManySig | curvature | circular_shift9 | all_channels | -2.86 | 7.81 |
| ManySig | curvature | circular_shift9 | extra_only | +0.00 | 0.00 |
| ManySig | curvature | crop_shift9 | all_channels | -24.84 | 30.18 |
| ManySig | curvature | crop_shift9 | extra_only | -16.11 | 20.74 |
| ManySig | curvature | cfo10khz | all_channels | -43.33 | 47.69 |
| ManySig | curvature | cfo10khz | extra_only | -19.92 | 24.15 |
| ManySig | curvature | smooth_channel | all_channels | -0.39 | 2.05 |
| ManySig | curvature | smooth_channel | extra_only | +0.07 | 0.55 |

Full-classifier shift invariance is not implied by invariant engineered channels. Synthetic perturbations are not measured receiver time offsets; they cannot establish that timing causes the remaining WiSig error.
