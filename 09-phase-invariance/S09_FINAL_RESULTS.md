# S09 Formal Matched Ablations

Fresh seeds9101/9102/9103; four folds; RX-equal means. Shared weights, packet partitions and training shuffle explicitly matched. All target reads occurred after source-only mechanism selection was committed. Public RX are not new blind tests.

| Dataset | Phase | Feature | Policy | Task | Acc % | Macro-F1 % | Seed SD pp | Worst RX % |
|---|---|---|---|---|---:|---:|---:|---:|
| ManySig | attribution | first | fixed | CR | 92.88 | 92.41 | 1.94 | 81.69 |
| ManySig | attribution | first | fixed | CRD | 92.36 | 91.90 | 2.36 | 79.75 |
| ManySig | attribution | first | packet | CR | 93.79 | 93.55 | 0.47 | 84.14 |
| ManySig | attribution | first | packet | CRD | 92.90 | 92.48 | 0.53 | 83.72 |
| ManySig | attribution | frequency | fixed | CR | 88.34 | 87.70 | 2.81 | 66.06 |
| ManySig | attribution | frequency | fixed | CRD | 86.69 | 85.67 | 2.36 | 62.11 |
| ManySig | attribution | null | fixed | CR | 88.34 | 87.70 | 2.81 | 66.06 |
| ManySig | attribution | null | fixed | CRD | 86.69 | 85.67 | 2.36 | 62.11 |
| ManySig | attribution | null | packet | CR | 90.10 | 89.56 | 1.38 | 73.75 |
| ManySig | attribution | null | packet | CRD | 88.51 | 87.56 | 0.49 | 69.25 |

## Paired Receiver Contrasts

| Dataset | Phase | Task | Contrast | Delta pp | Conditional RX95% interval | Improved RX |
|---|---|---|---|---:|---|---:|
| ManySig | attribution | CR | first:fixed-null:fixed | +4.53 | [+1.98, +7.34] | 11/12 |
| ManySig | attribution | CRD | first:fixed-null:fixed | +5.68 | [+2.41, +9.12] | 10/12 |
| ManySig | attribution | CR | first:fixed-frequency:fixed | +4.53 | [+1.98, +7.34] | 11/12 |
| ManySig | attribution | CRD | first:fixed-frequency:fixed | +5.68 | [+2.41, +9.12] | 10/12 |
| ManySig | attribution | CR | null:fixed-frequency:fixed | +0.00 | [+0.00, +0.00] | 0/12 |
| ManySig | attribution | CRD | null:fixed-frequency:fixed | +0.00 | [+0.00, +0.00] | 0/12 |
| ManySig | attribution | CR | first:packet-null:packet | +3.69 | [+1.37, +6.53] | 9/12 |
| ManySig | attribution | CRD | first:packet-null:packet | +4.39 | [+1.71, +7.53] | 9/12 |
| ManySig | attribution | CR | first:fixed-first:packet | -0.92 | [-2.74, +0.65] | 5/12 |
| ManySig | attribution | CRD | first:fixed-first:packet | -0.53 | [-2.30, +1.16] | 7/12 |
| ManySig | attribution | CR | null:fixed-null:packet | -1.76 | [-3.53, -0.04] | 3/12 |
| ManySig | attribution | CRD | null:fixed-null:packet | -1.82 | [-3.57, -0.06] | 4/12 |

Intervals condition on trained seeds and cluster by receiver. They are descriptive, not multiplicity corrected, and exclude full seed/dataset uncertainty. Fixed-versus-packet changes joint EMA/stopping policy; it does not isolate EMA alone. Null and three-channel models are initially function-equivalent; small GPU kernel rounding differences can diverge during optimization.

## Factorial Interaction

- CR: difference of phase-content gains, fixed minus packet policy +0.84pp [-1.11, +2.81].
- CRD: difference of phase-content gains, fixed minus packet policy +1.29pp [-0.82, +3.48].

## Frozen Fair Baselines

| Dataset | Variant | Baseline | Candidate % | Baseline % | Delta pp | Conditional RX95% interval |
|---|---|---|---:|---:|---:|---|
| ManySig | attribution:first:fixed | IB-RFF | 92.36 | 91.09 | +1.27 | [-2.24, +4.56] |
| ManySig | attribution:first:fixed | WaveMLP | 92.36 | 91.04 | +1.32 | [-1.56, +3.91] |
| ManySig | attribution:first:fixed | FAN_source_selected | 92.36 | 88.27 | +4.09 | [+0.01, +7.95] |
| ManySig | attribution:first:packet | IB-RFF | 92.90 | 91.09 | +1.80 | [-1.78, +5.06] |
| ManySig | attribution:first:packet | WaveMLP | 92.90 | 91.04 | +1.86 | [+0.03, +3.77] |
| ManySig | attribution:first:packet | FAN_source_selected | 92.90 | 88.27 | +4.62 | [+0.25, +8.85] |

External baseline seeds are unpaired. S05 released single-seed references are separate: ManySig IB-RFF93.57/WaveMLP93.13, ManyRx89.19/88.68. No global-SOTA claim: ASKNet is a newly located official competitor not fairly retrained here; source verification is in LITERATURE_CHECK.md. Mechanism selection aggregates source folds: RX source/target roles overlap across different folds, so it is not strictly nested fold-local hyperparameter selection. No day3-4 source selection or target outcome ranking occurred.
