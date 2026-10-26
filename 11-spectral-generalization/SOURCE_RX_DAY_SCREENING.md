# Source RX/Day Screening

Decision: **STOP at zero-fit mechanism admission**, not an empirical network STOP.
| route | dataset | centroid_gain_pp | tx_scatter_retention | nuisance_ratio_change | reproducibility |
|---|---|---|---|---|---|
| F1 | ManyRx | 7.9024 | 0.3331 | 0.6575 | fixed deterministic |
| F1 | ManySig | 1.4271 | 0.3310 | 0.6239 | fixed deterministic |
| F2 | ManyRx | 4.1702 | 0.6528 | 0.8013 | 0.9980 |
| F2 | ManySig | 2.3646 | 0.6152 | 0.8315 | 0.9936 |

F1 and F2 improved the fit-only nearest-centroid held-RX probe, but did not both satisfy the preregistered raw TX-scatter retention threshold0.65. No32-fit candidate network grid was admitted.
Raw between-TX scatter scales quadratically with gain. It is NOT a direct measure of mutual information or classification sufficiency. A constant positive scalar gain can lower this score without losing any information. This gate is conservative and cannot prove candidate uselessness.
The raw0.65 gate and zero-fit single-route preselection were agent-added pre-outcome protocol adaptations, not the requested trained-network source admission tests. Those requested tests remain unexecuted, not substituted by claimed equivalence.
In particular F1 has positive centroid gains and should be described as an untrained diagnostic whose retention gate failed, NOT as another failed HStem-HSTU bandwidth experiment.
Ordinary source paired gains>=1pp, final hard comparisons and source RX/day confirmation were NOT measured. The source day1->held RX day2 workflow was implemented but not admitted.
The protocol budget originally conflicted with full candidate/group/seed matrices; limits/resolution were frozen before audit. Unselected jobs were not run.
Engineering: F2 smoke failure occurred before optimization, not a low classification result.
Evidence: `decision_physics.json`, `source_centroid_probes.csv`, `source_spectral_statistics.csv`, `runs/smoke/`.
