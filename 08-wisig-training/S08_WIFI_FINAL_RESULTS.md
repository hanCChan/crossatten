# S08 Formal WiSig Results

All 24 frozen final fits complete. Four official folds x three fresh seeds for candidate and paired fixed-frequency official control. CRD primary; all algorithms selected from source RX days1-2. Public targets are not new blind tests.

| Dataset | Variant | Task | RX acc % | Packet acc % | Macro-F1 % | Seed SD pp | Worst RX % | Params |
|---|---|---|---:|---:|---:|---:|---:|---|
| ManySig | candidate | CR | 94.13 | 94.13 | 94.00 | 1.27 | 87.78 | 90868 |
| ManySig | candidate | CRD | 93.01 | 93.01 | 92.80 | 1.32 | 84.67 | 90868 |
| ManySig | control | CR | 88.87 | 88.87 | 88.34 | 1.87 | 77.14 | 90420 |
| ManySig | control | CRD | 87.56 | 87.56 | 86.67 | 1.94 | 70.50 | 90420 |

## Receiver-Matched CRD Differences

| Dataset | Baseline | Delta pp | Conditional RX95% interval | Improved RX |
|---|---|---:|---|---:|
| ManySig | S08-paired-control | +5.45 | [+2.09, +8.81] | 11/12 |
| ManySig | IB-RFF | +1.92 | [-1.33, +5.04] | 8/12 |
| ManySig | WaveMLP | +1.97 | [-0.19, +4.03] | 8/12 |
| ManySig | FAN_source_selected | +4.74 | [+0.49, +8.84] | 9/12 |
| ManySig | W1 | +20.68 | [+14.19, +27.32] | 12/12 |
| ManySig | S07-source-training | +6.40 | [+2.32, +11.74] | 10/12 |

RX-cluster bootstrap conditions on seed-averaged trained models; it does not capture full seed/dataset uncertainty and is not multiplicity corrected. S06/S07 seeds are unpaired; only new S08 control seeds are paired. No target-ranked pipeline or global-SOTA claim.

## Frozen Source Decisions

| Dataset | Fold | Feature | Recipe | Checkpoint rule | Exact refit epochs |
|---|---:|---|---|---|---:|
| ManySig | 0 | phase | official | ema_held | 10 |
| ManySig | 1 | phase | official | raw_held | 6 |
| ManySig | 2 | phase | official | ema_held | 4 |
| ManySig | 3 | phase | official | raw_held | 13 |

One pre-registered fold-local selection algorithm, not four target-selected models. The upper median of two held-source groups x two seeds determined exact refit epochs before final fitting. Final fits use all legal source RX with the same 70/30 packet budget. The same held RX are not independent final validation. The candidate uses no AWGN on any final fold. Rules retain raw weights on folds1/3 and use EMA on folds0/2; this is disclosed source selection.

## CRD By Fold

| Dataset | Fold | S08 candidate % | S08 control % | S07 % | IB-RFF % | WaveMLP % |
|---|---:|---:|---:|---:|---:|---:|
| ManySig | 0 | 92.10 | 85.13 | 87.67 | 92.83 | 86.10 |
| ManySig | 1 | 92.38 | 82.03 | 89.11 | 91.16 | 90.74 |
| ManySig | 2 | 94.52 | 95.40 | 92.12 | 93.20 | 94.15 |
| ManySig | 3 | 93.05 | 87.69 | 77.56 | 87.18 | 93.16 |

## Paired CRD Seeds

| Dataset | Seed | Candidate % | Control % | Delta pp |
|---|---:|---:|---:|---:|
| ManySig | 7101 | 93.04 | 87.58 | +5.47 |
| ManySig | 7102 | 94.31 | 89.50 | +4.81 |
| ManySig | 7103 | 91.68 | 85.61 | +6.07 |

Pairing means the same new data split, seed identifiers and training shuffle. The phase architecture has extra input weights and a different initialization graph; pairing does not imply identical shared initial parameters.

## Released-Weight Reference

Frozen S05 packet-weighted official-weight results, separate from fair multi-seed retraining: IB-RFF ManyRx/ManySig89.19/93.57%, WaveMLP88.68/93.13%. S08 ManySig packet and RX weighting coincide. Its93.01% is below both released ManySig reference means; this is not global-SOTA evidence.

## Attribution And Limits

ManySig phase input adds448 weights to90420 (90868 total), keeping32x64 tokens and two HSTU blocks. New features are reliability-masked adjacent complex-spectrum products, not receiver-invariant fingerprints. Source phase/null fits suggest information beyond extra input capacity, but there is no new formal target phase-only, null, or fixed-budget three-channel ablation. The final gain is joint evidence for representation plus source-selected epoch/EMA policy; it cannot be assigned independently to one component. The local phase gate requires at least0.5pp gain over its matched null rule in addition to the control gate.

Held-RX source CE minima are used for both checkpoint selection and research ranking on those same groups. The source gains therefore include selection optimism, not independent source confirmation. Fresh-seed final target results remain a retest of an already examined public benchmark.

Full200 AWGN training ran to completion but selected early checkpoints for most rules. It did not support a general 'more epochs is better' conclusion. ManyRx supported only2/4 source folds and was not admitted to new final evaluation. No target-based retry or unregistered model/loss was added after final scores.
