# S12 Status

Status: **STOP_SOURCE_GATE**, 2026-10-26.

Preregistration: `f538477`. Pilot decision: `a7d41de`.
Completed: 8 source diagnostic partitions, 4/4 smoke fits, 16/16 pilot fits
and 14 synthetic regression tests. No fit failed or was retried.
Historical results are accepted; no old weight was located, loaded or audited.

## Trained Source Results

Accuracy is held-source RX-equal, then fold-equal, using `raw_packet`.
These are **not target CR/CRD results**.

| Dataset | Historical Hard | F1 | F2 | F1 Gain (pp) | F2 Gain (pp) |
|---|---:|---:|---:|---:|---:|
| ManyRx | 91.3201% | 89.5801% | 91.3951% | -1.7400 | +0.0750 |
| ManySig | 89.1250% | 91.2917% | 92.5313% | +2.1667 | +3.4062 |

F1 fails ManyRx mean/fold requirements and worst-RX protection.
F2 improves only 2/4 ManyRx folds, falls short of the +1 pp requirement,
and has a receiver-mean delta of -3.5333 pp (single-pair minimum -6.95 pp).
Neither candidate is admitted. `jobs_confirmation.json` is empty.

Not run: independent confirmation, source RX/day transfer and target CR/CRD.
No new target threshold attainment, statistical significance or universal
failure of spectral methods is claimed. Pilot uses first group/seed8101 only.

## Mechanism And Engineering

F2 smoke and pilot fold0 priors and full-fit partition seals match for both
datasets. Priors use 24,267 ManyRx / 5,880 ManySig full-fit packets;
smoke networks use 512 training packets. Held smoke coverage is stratified.

Source centroid means (descriptive, not network accuracy):

| Dataset | Hard | F1 | F2 | Uniform | Shuffled | TX-Permuted |
|---|---:|---:|---:|---:|---:|---:|
| ManyRx | 60.574% | 67.159% | 64.289% | 60.574% | 60.987% | 60.574% |
| ManySig | 69.562% | 71.354% | 71.396% | 69.562% | 69.625% | 69.562% |

The improved ManyRx centroid probe did not translate to a stable trained
network gain. This supports treating the probe as descriptive, not a veto
or a substitute for paired deep-network evidence.

Next research should examine source RX/day error mechanisms before adding
network complexity. ManySig F2 is a dataset-specific exploratory lead, not
a post-hoc exemption from the joint gate; any narrower continuation requires
a new prospective protocol and independent paired seeds.

Full checks: `decision_pilot.json`. Per-RX accuracy/F1/CE: `paired_pilot.json`.
Generated run evidence: `evidence/`, sealed by `evidence_manifest.json`.
New checkpoints remain local in `runs/`; no old or new weight audit is needed.
