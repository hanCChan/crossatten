# Stage 07: WiSig Frontends And OSU Validation

Authorized 2026-10-25. Baseline `ebe322507d9d274265572d4f86545e93540a565d`.
Independent branch `exp/s07-wifi-osu`; no edits to stages00–06, their protocols or training rules.
Goal: improve strict source-only WiSig CRD and evaluate LoRa on OSU.
SOTA is a research target, not a promised result or an unlimited-search stopping rule.

## Bounded Sequence

1. Read-only S06 diagnostics; per-fold source-only feature audit; OSU metadata inventory.
2. Freeze seven frontends: R0 legacy, R1 time, R2 phase difference, R3 separate time/frequency,
   R4 prefix lag-20 product, R5 bounded identity-initialized complex filter, RF frequency-only.
   All use 32x64 tokens, two HSTU blocks and unified200. Four source folds, seed3101.
3. Replicate the two highest internal-source-RX frontends plus R0 at seed3102.
   Freeze the per-fold frontend using only its own source RX. No cross-fold global target leakage.
4. Fixed selected frontend: official/unified, RX-balanced sampling, held-source-RX epoch selection,
   and single phase/shift/CFO/channel/AWGN-strength perturbations. Seed3101; top-two plus control
   repeat3102. No GRL, SupCon, HSIC, target pseudo-labels, UDA or distillation.
5. At most two source-supported factors combine; at most four configurations per dataset/fold.
   Gate fusion is tested only for a selected split frontend; no forced attention.
6. Freeze source decisions and allowed final jobs in Git. At most three reported source-selected
   pipelines, four RX folds x seeds4101/4102/4103, CR/CRD separately. Reuse S06 fair baselines
   if data normalization/splits/budgets remain identical. Never select on target accuracy.
7. Report and stop, including failures, costs and receiver-clustered uncertainty.

Max WiSig formal fits: frontend56+replication24+training72+replication24+combination32+final72=280.
Adaptive stages may run fewer fits when no source-supported factors exist.
Unfinished jobs, source/target leakage, or changed evidence block final evaluation.
Small structural parameter differences are measured rather than described as exactly matched.

## Selection

All architecture, recipe and augmentation choices are per dataset/fold using only its source RX
days1–2. Internal RX: source-list last6 (ManyRx) or last2 (ManySig). Other source RX retain70/30
packet train/checkpoint validation. S06 target results are historical diagnostics, not tuning inputs.
First-pass ranks mean RX accuracy, ties source CE then identifier. Replication requires two
positive source-RX deltas, mean>=1pp and worst held-source RX drop>=-2pp before factor combination.
If no factor satisfies the gate, retain the strongest source-ranked single model without declaring improvement.
Final ordinary runs use minimum source packet CE; RX-budget runs use a source-frozen median
held-RX best epoch and last checkpoint. The scheduler horizon remains200.
One source-selected algorithm may choose different registered configurations per fold; disclose them.

## Physical Bounds

Phase rotation[-pi,pi] is a nuisance hypothesis, not an assertion that all phase is receiver-only.
Zero-padded time shifts <=2 samples (80ns), extra CFO <=2kHz (<one WiFi subcarrier), smooth
log-gain and phase <=0.1 are conservative experiment bounds, not estimated channel ground truth.
Single-factor experiments do not combine all perturbations. No default WiFi hard band crop.
Lag20 prefix[20,160) requires the source audit to show median coherence>=0.7 in that fold;
otherwise R4 is marked inadmissible, not silently re-tuned.
R5 reuses the general idea of learnable spectral filtering, not a novel FAN replacement.

## OSU

Local50 records (25TX x2RX) are inventoried without evaluating models.
Do not reuse G1's arbitrary8192-point periodic crops as synchronized8-chirp packets.
Register a fixed detector using only source pilot records; report all detection failures and counts.
Train/test capture IDs are disjoint across RX. Source validation uses purged temporal blocks
because only one capture/TX/RX is local; this is not independent-capture validation.
Use fixed temporal train/validation/target blocks to reduce shared-transmission leakage.
OSU results are separate25-class retraining, not Shen ten-class checkpoint evaluation.
Cross-day data is a separate gate: metadata, download budget and session split before training.
No full TB-scale download. No automatic push or release of new model weights.

## Evidence

Registration, source access, source choices, final predictions, fit curves and failed attempts
live in this stage; cached inputs/weights are ignored by Git.
Independent data does not make a previously examined public benchmark a new blind test.
At least FAN, IB-RFF, WaveMLP are WiSig references; OSU compares ordinary/HStem HSTU,
CNN-only and CIS plus available source-frozen external adapters.
Only claim protocol-specific measured superiority after receiver-level uncertainty and primary-source review.
