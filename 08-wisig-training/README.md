# Stage 08: WiSig Training And Source Selection

Authorized 2026-10-26. Frozen baseline `74f594312949f18613030d59743dba88fc20eca7`;
branch `exp/s08-wisig-training`. S00-S07, shared library code, old weights and
predictions remain unchanged. No automatic push, merge or weight publication.

## Completed Outcome

All admitted jobs completed: 136 formal fits (48 training controls, 32 phase/null
fits, 32 ManyRx source fits and 24 ManySig final fits), plus five smoke fits.
All sixteen extended-budget controls ran the full200 epochs.

ManySig source-selected phase/epoch/EMA achieved **CR94.13%, CRD93.01%**,
Macro-F1 92.80%, worst RX84.67%, with90,868 parameters. Relative to the new
paired frequency-only official control: **+5.45pp CRD**, improving11/12 RX.
It is +6.40pp above S07 and +1.92/+1.97pp above the frozen fair IB-RFF/WaveMLP
means. The external-baseline RX intervals cross zero; released-weight references
remain higher (93.57%/93.13%). No stable-SOTA claim.

ManyRx supported only2/4 source folds, below the registered3/4 gate. No new
ManyRx target evaluation was run. Do not combine its historical S07 score with
the new ManySig score as if both belonged to a confirmed S08 method.
AWGN/full200 did not yield a general source improvement. Final ManySig folds
selected phase features and source-frozen epochs, without AWGN; folds0/2 use
EMA, folds1/3 raw weights. The joint gain is not isolated component causality.

Read [Chinese results](RESULTS.md), [source results](S08_SOURCE_RESULTS.md),
[formal comparison](S08_WIFI_FINAL_RESULTS.md), [execution](S08_EXECUTION.md)
and [costs](S08_COSTS.md). [Acceptance](acceptance.json) verifies all completed
artifacts, source roles, committed final freeze and historical preservation.
The registered exploration is complete and stopped.

## Registered Sequence

1. CPU-only audit of S07 source histories, held-source predictions, spectral
   scales and gate diagnostics. No historical target scores select candidates.
2. ManySig: four official RX folds, two seeds, two disjoint held-source groups;
   paired official, official+AWGN10-50, and official+AWGN10-50 full200 fits.
   **48 fits.** Full200 disables early stopping but retains Adam/Plateau and
   source checkpoint selection, directly testing extended training.
3. Maintain raw and EMA parameters in each fit. EMA has one-epoch half-life,
   uses PyTorch AveragedModel, does not average BatchNorm buffers, and recalibrates
   BatchNorm each epoch on the clean fit-training subset only.
   Save raw/EMA minimum packet-CE and minimum held-RX-CE checkpoints.
4. Only if training candidates lack credible source gains and source spectral
   diagnostics support reliable adjacent bins, test spectral relative phase
   and a matched-capacity zero-channel control. **At most32 extra ManySig fits.**
   No new losses, large architecture or unrestricted combinations.
5. Expand to ManyRx only if ManySig passes the registered source gate.
   Compare the original official control with the pre-registered official+AWGN
   recipe and all four checkpoint rules, using four folds, two groups, two seeds.
   **At most32 fits.** ManySig results determine whether to proceed, not the
   ManyRx candidate menu; no cross-dataset RX information selects its recipe.
6. Freeze the same fold-local selection algorithm on each dataset: choose among
   source-supported registered recipes/checkpoint rules using only that fold's
   two source groups; otherwise fall back to official/raw packet-CE.
   All folds keep the frequency architecture; no target-tailored split branch.
   Rules may produce different source-selected recipes/epochs per fold, which
   are disclosed. Other folds' source scores cannot select this fold's model.
7. Confirm only admitted algorithms: four folds x three fresh paired seeds,
   candidate plus freshly trained fixed-frequency official control.
   **At most48 final fits.** Report CR/CRD separately and stop.

Formal cap: **160 fits** (48+32+32+48). Smoke tests use tiny source-only subsets,
are separately labeled, and never counted as formal evidence.

## Source Roles And Gate

ManySig holds first2 or last2 source RX; ManyRx first6 or last6. Each group
is excluded from its own fitting and packet-checkpoint subset. Fit packets
split70/30 with paired seed; source dates1-2 only.

Against paired official/raw packet-CE, each candidate's **fold-local** gate
requires both seed mean gains over the two groups positive, overall mean gain
>=1pp, neither group's seed-mean gain negative, and worst held-RX seed-mean
drop>=-3pp. Rank accepted candidates by mean RX accuracy, then worst RX,
source CE, identifier. An algorithm is admitted to final confirmation only if
at least three of four folds have a supported candidate. Unsupported folds use
the control. This global gate determines whether to run, never which candidate
to choose on another fold.

Packet-CE rules retain their frozen patience policy. Held-CE rules use the
upper median of the four source-selected epochs, re-fit all legal source RX
for exactly that budget, and select the final epoch. Held RX later included
in final fitting are not described as independent final validation.

## Interpretation

Official and full200 budgets are deliberately different; gains are not
architecture-only causal effects. Held-CE is a checkpoint-selection experiment,
not another independent trained model. EMA has training cost but the same
inference graph. Source gates are descriptive research criteria, not unbiased
significance tests. Public WiSig targets are not new blind tests. Reuse S06/S07
frozen fair baselines only as historical, unpaired references. Do not claim
global SOTA from two reference methods or from early-stop/noise tuning.

All jobs, source evidence, decisions and final configurations are committed
before target reads. All low scores, failures and interrupted attempts stay.
