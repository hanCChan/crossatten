# OSU Post-Hoc Failure Diagnostics

This analysis occurs after all 24 registered scores. It is descriptive only, cannot select methods, and authorizes no additional fits or compensation.

All metadata transmitter IDs agree with filenames for both RXs. Original crop parity, labels and saved confusion matrices have passed checks.

| Model | Same RX future time block % | Other RX future time block % |
|---|---:|---:|
| patch_hstu | 81.28 | 1.41 |
| hstem_hstu | 91.24 | 1.42 |
| hstem_cnn | 89.90 | 13.31 |
| cis_rank16 | 91.96 | 21.00 |

Median RX2-minus-RX1 lag-phase implied modulo-CFO difference: 149.5 Hz. This statistic is aliased modulo 976.5625 Hz; it is not an unambiguous oscillator estimate. A systematic RX-dependent phase/relative-frequency offset is observed. It does not prove that correcting it will recover accuracy or that the detector is unbiased.

Unlike Shen's published preprocessed inputs, this pipeline does not compensate CFO. Negative OSU results apply to the registered detector and nominal strict-IB pipeline, not all possible OSU methods. Sparse per-class packets, a single capture, synchronization uncertainty and receiver transfer remain limitations. No OSU SOTA is established.
