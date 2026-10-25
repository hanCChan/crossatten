# Paper Validation Results

Evidence date: 2026-10-25. Registration: `bc30820`.
This report includes only finished, audited fits. Smoke results are not accuracy evidence.
**Status: in progress**; audited fits: 18.

## Completion

| Stage | Finished | Missing |
|---|---:|---:|
| source | 16 | 160 |
| explore | 2 | 98 |
| final | 0 | 0 |
| confirm | 0 | 0 |

Artifact audit issues: 0; recorded failure entries: 0.
A failed execution is not discarded; a subsequent successful retry may complete its registered fit.

## Final Comparisons

All values are percent. RX accuracy SD across fits combines fold and seed variation.
Seed SD is computed from fold-averaged seed results; folds and packets are not extra independent RX.
Incomplete rows are provisional and cannot establish improvement.

| Dataset | Model | Task | Fits | RX Acc | Packet Acc | RX F1 | Seed SD | Fold SD | Worst RX | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Registered Ablations

| Dataset | Task | Pair | Fits | Delta pp | Positive | Worst RX Delta | Decision |
|---|---|---|---:|---:|---:|---:|---|
| ManyRx | CR | W1-W0 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W1-W0 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CR | W2-W1 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W2-W1 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CR | W1-C0 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W1-C0 | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CR | W1-CP | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W1-CP | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CR | W1-CF | 0 | pending | -/0 | pending | incomplete |
| ManyRx | CRD | W1-CF | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W1-W0 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W1-W0 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W2-W1 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W2-W1 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W1-C0 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W1-C0 | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W1-CP | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W1-CP | 0 | pending | -/0 | pending | incomplete |
| ManySig | CR | W1-CF | 0 | pending | -/0 | pending | incomplete |
| ManySig | CRD | W1-CF | 0 | pending | -/0 | pending | incomplete |
| Shen | dev_val | S_hstem-S_base | 0 | pending | -/0 | pending | incomplete |
| Shen | confirm | S_hstem-S_base | 0 | pending | -/0 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnonly | 0 | pending | -/0 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnonly | 0 | pending | -/0 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnparam | 0 | pending | -/0 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnparam | 0 | pending | -/0 | pending | incomplete |
| Shen | dev_val | S_hstem-S_cnnflops | 0 | pending | -/0 | pending | incomplete |
| Shen | confirm | S_hstem-S_cnnflops | 0 | pending | -/0 | pending | incomplete |

## Interpretation Limits

- CRD is the primary WiSig task. W0/W1/W2 remain separate; no target-based winner selection.
- Final configuration choices use only source RX days 1-2; source evidence and decisions are committed before target access.
- Shen confirm RX were inspected repeatedly and are report-only, not independent blind-test evidence.
- Published weights in `05-wisig/RESULTS.md` are historical references, not fair multiseed retraining.
- HSTU attribution requires W1 vs C0 and matched CNN controls; parameter efficiency does not imply faster inference.
- Cost timings use raw device-resident IQ to logits, not host transfer; FFT is timed but incompletely counted in FLOPs.
- FAN upstream CUDA median issues a nondeterminism warning; fixed seeds do not guarantee bit-identical upstream training.
- No new Loss, feature branch, architecture or target-selected hyperparameter is added after this registration.

Full per-fit metrics, configurations, source choices and failures: `summary/report.json`; costs: `summary/cost_*/bench.md`.
The bounded experiment matrix must finish before choosing a general RF, LoRa-only, or engineering-only paper route.
