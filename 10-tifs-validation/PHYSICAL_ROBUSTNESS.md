# Physical Robustness

Frozen diagnostics only; zero additional fits. Conditions registered before source outcomes.
Source phase encoder comparisons use identical packets and draws, common source-only RMS scaling.
DTX=TX-centroid separation; DRX=TX-conditioned RX displacement, both divided by feature dimension.

| encoder | perturbation | value | feature_mse | d_tx | d_rx | d_rx_over_d_tx |
|---|---|---|---|---|---|---|
| hard | awgn | 0 | 1.723710 | 0.018555 | 0.045294 | 2.528665 |
| hard | awgn | 10 | 1.040494 | 0.063292 | 0.072466 | 1.168922 |
| hard | awgn | 20 | 0.421741 | 0.112024 | 0.083755 | 0.765264 |
| hard | awgn | 30 | 0.109474 | 0.132641 | 0.090249 | 0.696790 |
| hard | awgn | 5 | 1.397414 | 0.037185 | 0.060089 | 1.654498 |
| hard | clean | 0 | 0.000000 | 0.136934 | 0.093822 | 0.701310 |
| hard | multipath | [2, 0.3] | 0.061295 | 0.141555 | 0.094565 | 0.685308 |
| hard | multipath | [2, 0.6] | 0.138904 | 0.140674 | 0.094762 | 0.692234 |
| hard | multipath | [4, 0.3] | 0.066948 | 0.135423 | 0.093386 | 0.706519 |
| hard | multipath | [4, 0.6] | 0.150820 | 0.133915 | 0.093671 | 0.718471 |
| hard | multipath | [8, 0.3] | 0.081846 | 0.139591 | 0.092781 | 0.685715 |
| hard | multipath | [8, 0.6] | 0.195907 | 0.139898 | 0.092787 | 0.688982 |
| hard | shift | -2 | 0.177160 | 0.142161 | 0.094803 | 0.685642 |
| hard | shift | -4 | 0.246674 | 0.139162 | 0.097494 | 0.722041 |
| hard | shift | 2 | 0.236088 | 0.144362 | 0.093304 | 0.662810 |
| hard | shift | 4 | 0.281505 | 0.143775 | 0.090020 | 0.644871 |
| soft | awgn | 0 | 1.465683 | 0.018090 | 0.043462 | 2.485774 |
| soft | awgn | 10 | 0.756553 | 0.060629 | 0.068210 | 1.148093 |
| soft | awgn | 20 | 0.200806 | 0.101428 | 0.075595 | 0.762295 |
| soft | awgn | 30 | 0.025440 | 0.114517 | 0.077595 | 0.693488 |
| soft | awgn | 5 | 1.127038 | 0.036112 | 0.057324 | 1.624332 |
| soft | clean | 0 | 0.000000 | 0.116603 | 0.078122 | 0.685726 |
| soft | multipath | [2, 0.3] | 0.011166 | 0.118758 | 0.078257 | 0.675030 |
| soft | multipath | [2, 0.6] | 0.047543 | 0.114788 | 0.076698 | 0.684680 |
| soft | multipath | [4, 0.3] | 0.014036 | 0.115363 | 0.076955 | 0.683124 |
| soft | multipath | [4, 0.6] | 0.054363 | 0.112871 | 0.075060 | 0.681828 |
| soft | multipath | [8, 0.3] | 0.022478 | 0.118854 | 0.077870 | 0.674938 |
| soft | multipath | [8, 0.6] | 0.084585 | 0.116828 | 0.077088 | 0.682973 |
| soft | shift | -2 | 0.058734 | 0.117399 | 0.078824 | 0.688731 |
| soft | shift | -4 | 0.107081 | 0.116050 | 0.080869 | 0.716757 |
| soft | shift | 2 | 0.085737 | 0.119774 | 0.077737 | 0.664427 |
| soft | shift | 4 | 0.121482 | 0.119350 | 0.075837 | 0.652209 |
| unweighted | awgn | 0 | 1.921430 | 0.018583 | 0.045514 | 2.537519 |
| unweighted | awgn | 10 | 1.251257 | 0.063460 | 0.073151 | 1.176825 |
| unweighted | awgn | 20 | 0.649864 | 0.113559 | 0.086390 | 0.778890 |
| unweighted | awgn | 30 | 0.203531 | 0.138878 | 0.095523 | 0.703974 |
| unweighted | awgn | 5 | 1.599300 | 0.037264 | 0.060451 | 1.661576 |
| unweighted | clean | 0 | 0.000000 | 0.148216 | 0.101129 | 0.698054 |
| unweighted | multipath | [2, 0.3] | 0.069851 | 0.153753 | 0.103215 | 0.688392 |
| unweighted | multipath | [2, 0.6] | 0.191188 | 0.155093 | 0.105651 | 0.699649 |
| unweighted | multipath | [4, 0.3] | 0.074153 | 0.148174 | 0.099549 | 0.688018 |
| unweighted | multipath | [4, 0.6] | 0.199769 | 0.147188 | 0.097402 | 0.680555 |
| unweighted | multipath | [8, 0.3] | 0.093888 | 0.151701 | 0.101025 | 0.686053 |
| unweighted | multipath | [8, 0.6] | 0.257353 | 0.154266 | 0.101939 | 0.686859 |
| unweighted | shift | -2 | 0.258946 | 0.165839 | 0.100501 | 0.626945 |
| unweighted | shift | -4 | 0.340599 | 0.148380 | 0.107889 | 0.746371 |
| unweighted | shift | 2 | 0.364564 | 0.163135 | 0.100192 | 0.632095 |
| unweighted | shift | 4 | 0.423442 | 0.163418 | 0.100059 | 0.629981 |

Accuracy curves: `physical_accuracy_probes.csv` (all surviving S10 formal checkpoints, CRD only).
These are controlled distortions added to recorded IQ, not independently acquired electromagnetic environments.
No claim of arbitrary-channel robustness, authentication security or causal RX independence is supported.
Historical S07-S09 weights are missing. Consequently, soft-v-hard CLASSIFIER robustness cannot be supplied if the soft candidate fails its source gate. Encoder-only differences do not replace it.
No target soft-model diagnostics are run after a STOP gate.

At20dB AWGN, source normalized phase-feature MSE (hard -> soft): ManyRx: 0.432325 -> 0.205187; ManySig: 0.411156 -> 0.196424.
Lower feature MSE is consistent with attenuation of uncertain phase, but is NOT sufficient to preserve identity discrimination: source ManyRx classification declined under the same candidate.
Amplitude shrinkage itself lowers MSE; DTX and DRX/DTX are reported to avoid equating feature stability with useful RX invariance.

RX/seed-equal frozen classifier accuracy:

| dataset | model | perturbation | value | accuracy_pct |
|---|---|---|---|---|
| ManyRx | ask | awgn | 0 | 26.5474 |
| ManyRx | ask | awgn | 10 | 69.3574 |
| ManyRx | ask | awgn | 20 | 86.6841 |
| ManyRx | ask | awgn | 30 | 88.4400 |
| ManyRx | ask | awgn | 5 | 46.8831 |
| ManyRx | ask | clean | 0 | 88.6934 |
| ManyRx | ask | multipath | [2, 0.3] | 89.0109 |
| ManyRx | ask | multipath | [2, 0.6] | 89.0754 |
| ManyRx | ask | multipath | [4, 0.3] | 89.3309 |
| ManyRx | ask | multipath | [4, 0.6] | 89.3376 |
| ManyRx | ask | multipath | [8, 0.3] | 87.4148 |
| ManyRx | ask | multipath | [8, 0.6] | 84.6433 |
| ManyRx | ask | shift | -2 | 88.0836 |
| ManyRx | ask | shift | -4 | 86.4219 |
| ManyRx | ask | shift | 2 | 87.8067 |
| ManyRx | ask | shift | 4 | 85.3239 |
| ManySig | ask | awgn | 0 | 31.6435 |
| ManySig | ask | awgn | 10 | 71.6898 |
| ManySig | ask | awgn | 20 | 90.8912 |
| ManySig | ask | awgn | 30 | 92.8704 |
| ManySig | ask | awgn | 5 | 50.8102 |
| ManySig | ask | clean | 0 | 92.9861 |
| ManySig | ask | multipath | [2, 0.3] | 93.2523 |
| ManySig | ask | multipath | [2, 0.6] | 93.2407 |
| ManySig | ask | multipath | [4, 0.3] | 93.3565 |
| ManySig | ask | multipath | [4, 0.6] | 92.2917 |
| ManySig | ask | multipath | [8, 0.3] | 92.2685 |
| ManySig | ask | multipath | [8, 0.6] | 84.9769 |
| ManySig | ask | shift | -2 | 90.2315 |
| ManySig | ask | shift | -4 | 79.5833 |
| ManySig | ask | shift | 2 | 89.2940 |
| ManySig | ask | shift | 4 | 74.3171 |
