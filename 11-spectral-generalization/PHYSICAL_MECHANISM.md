# Physical Mechanism

Source-only observations support useful spectral discrimination/stability structure:
| route | dataset | centroid_gain_pp | tx_scatter_retention | nuisance_ratio_change | reproducibility |
|---|---|---|---|---|---|
| F1 | ManyRx | 7.9024 | 0.3331 | 0.6575 | fixed deterministic |
| F1 | ManySig | 1.4271 | 0.3310 | 0.6239 | fixed deterministic |
| F2 | ManyRx | 4.1702 | 0.6528 | 0.8013 | 0.9980 |
| F2 | ManySig | 2.3646 | 0.6152 | 0.8315 | 0.9936 |

F2 weights reproducibly alter frequency emphasis and reduce the source nuisance/scatter ratio. This is observational discrimination, not a causal proof of RX-independent hardware identity.
Cartesian real/imaginary phase means avoid phase unwrap. Missing cells require complete day pairs and sufficient fit packets; mean estimation uncertainty appears in the denominator.
The registered raw TX-scatter retention gate is amplitude dependent. Higher probe accuracy and lower scatter can coexist. Both results must be reported; raw scatter loss does not prove TX information loss.
Controlled AWGN/multipath/noncircular-shift classifier tests were not run for an unadmitted candidate. S10 frozen perturbation evidence is retained as background, not reused as proof of S11 robustness.
No new target IQ or attack/open-set measurements were taken.
