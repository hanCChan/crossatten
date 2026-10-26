# S09 — Receiver-general phase mechanisms and attribution

Base commit: `c0aceeb703de6d1a08b1b5b0fc14b1200062f0a8`. Independent experimental branch. No S00–S08 files or results changed. **Only synthetic CPU tests were run in the drafting environment; no dataset or GPU training was executed.**

## Why

S08 showed ManySig CRD 93.01% with 90,868 parameters. The +5.45pp versus its paired official-frequency control jointly changes phase input and source-selected epoch/Raw–EMA checkpoint rule. Formal matched-training phase/null/frequency ablations are still missing. The phase input `p[k]=X[k]*conj(X[k-1])/(abs(X[k])*abs(X[k-1]))` cancels global phase but rotates with ideal time shifts by `exp(-j*2*pi*tau/N)`, so receiver timing remains a plausible nuisance, **not a proven mechanism behind the reported error**.

## Code

`phase_features.py` accepts existing S08 `frequency_views(x)` of shape [B,3,256] and returns:

| kind | channels | reason |
|---|---:|---|
| `frequency` | 3 | original frequency-only baseline |
| `null` | 5 | same extra input capacity but zero phase |
| `first` | 5 | exact S08 masked first phase difference |
| `centered` | 5 | circular-mean correction of first phase differences |
| `curvature` | 5 | second phase difference: `q[k]=p[k]*conj(p[k-1])` |

Centering and curvature are invariant to **ideal circular shifts**, not general truncation, multipath, CFO, or hardware-specific changes. Curvature may delete useful TX group delay and amplify phase noise. Centering falls back if its circular mean is weak. The 0.05 normalized magnitude cutoff is S08's rule, not an estimated receiver noise floor.

Run `cd 09-phase-invariance && pytest -q test_phase_features.py`. Six CPU tests cover S08 parity, circular shifts, phase rotation, finite values, differentiable gradient, zero controls and explicit channel counterexample.

## Mandatory experiment A: separate representation from source selection

For **ManySig only**, run three fresh arms under the **same frozen S08 per-fold training selection**: `first`, `null`, `frequency`. Fixed epochs [10,6,4,13]; final rules [EMA,raw,EMA,raw]. All use the original official Adam/Plateau recipe, no AWGN; same train packets and three fresh paired seeds. 4 folds × 3 seeds × 3 arms = **36 fits**. Do not reuse historical different-seed weights as matched controls. Preserve separate phase-null/frequency and capacity comparisons; report fold/receiver deltas and paired statistical intervals.

This isolates phase-content gain **conditional on** the frozen training policy; it does **not** independently identify EMA/epoch selection's contribution. If useful, follow up with a 2×2 feature (first/null) × globally frozen checkpoint policy (S08 rule vs raw packet CE) factorial. No per-target RX choices.

## Hypothesis B: remove timing nuisance without losing TX identity

First use **source-only RX** in both ManySig and ManyRx, official folds/days1–2, two held-source groups and two paired seeds to compare `first`, `centered`, `curvature` using equal five-channel HStem+HSTU. Freeze source gate and budget before inspecting any new target metrics. Where possible, reproduce or verify parity with existing S08 source `first` runs before reusing them. Source statistics: mean/worst held-RX Accuracy, macro-F1, CE, per-TX separation and low-spectral-reliability fraction. CPU diagnostics should include ideal circular shift, ordinary zero-padded shift, small CFO and smooth multipath; test whether correction removes identity cues.

Only carry forward a universally specified mechanism if it improves source-held receiver generalization in **both** datasets; otherwise report as dataset-limited. Source validation is not independent if used for hyperparameter selection.

## Broadness and honest ranking

**No OSU work.** A mechanism designed from Fourier shift symmetry should not be tuned to historical ManySig fold 3 or to target day3–4. Assess ManyRx and ManySig symmetrically through source folds, preserve all failed arms. Existing public test receivers have been repeatedly examined; new seeds are not new blind receivers. Claims of global SOTA require same-protocol strong baselines, receiver-cluster uncertainty and a broader review. If cross-LoRa usefulness is claimed, test Shen using physically appropriate LoRa protocol as a distinct validation; do not silently transfer WiSig preprocessing.

## Integration

Keep the existing `08-wisig-training/training.py` untouched. In a new S09 runner, use:

```python
from phase_features import phase_features_from_base
base = frequency_views(rms_normalize(iq))
views = phase_features_from_base(base, kind=job["feature"])
# 3-channel frequency has different stem; four other variants share 5 channels.
```

Reuse the existing S08 source/target role guard, manifests, exact packet partitions, official optimizer, batch size, BN recalibration and deterministic run IDs. First verify exact numeric parity between S09 `first` and S08 phase features using a fixed real source IQ minibatch and matching model weights. Implement registered paired controls before new training. Never overwrite historical outputs.

## Local execution registration (2026-10-26)

- A: 36 fixed-policy fits plus 24 first/null official raw-packet early-stop fits, enabling a matched 2x2 feature/policy comparison.
- B: 96 source-only fits, both datasets, four folds, first/last held-source groups, two fresh seeds, first/centered/curvature. All train exactly 40 epochs without early stopping; raw/EMA held-CE checkpoints are retained.
- Final: at most 48 fresh fits, one admitted mechanism per dataset and its first-order control, using identical source-frozen epochs and raw/EMA rule. The same feature/rule requires at least 3/4 supported source folds in each dataset to claim universal support. Otherwise any admitted confirmation is explicitly dataset-limited.
- Source gate: mean paired gain at least 0.5pp, both seeds positive, both group means nonnegative, no held RX seed-mean loss worse than 3pp. Rank accepted configurations only on source gains, worst-RX difference and lexical tie-break.
- Fresh source seeds 8101/8102; final and attribution seeds 9101/9102/9103.
- Shared initial parameters are copied from one canonical five-channel model. The three-channel model uses the identical first three convolution slices and all other identical parameters. Null and true phase have exactly identical complete initialization.
- No target read until B finishes and its decision is committed. A is attribution, not candidate selection; its target scores cannot change B, the final recipe or the number of trials.
- Centering weak-resultant fallback is retained exactly as in the CPU probe and reported explicitly. Only extra channels have the ideal invariance; the retained Re/Im channels and full classifier do not.
- All weights stay local and ignored. No automatic merge or push.

## Completed Outcome

All156 formal fits and five GPU smoke fits completed without failed/interrupted
training attempts. Source decision was committed as `96d52db` at
`2026-10-26T09:24:10+08:00`, before any new target read.

ManySig fixed-policy CRD: first92.36%, null86.69%, frequency86.69%.
The strict phase-content contrast is +5.68pp with conditional receiver95%
interval [+2.41,+9.12],10/12 receivers improved. Null and frequency produce
exactly identical target logits across all24 CR/CRD prediction comparisons.
The registered raw-packet early-stop contrast is also positive: first92.90%
versus null88.51%, +4.39pp [+1.71,+7.53].

Fixed S08 stopping/EMA policy did not improve over packet early-stop for
these fresh seeds. No centered/curvature feature passed the3/4 source-fold
gate in either dataset, so no conditional final mechanism fit was run.
Ideal circular-shift robustness improved, but did not deliver consistent
held-source RX gains; crop/CFO diagnostics retain important failure modes.

See `RESULTS.md` for the Chinese conclusion, `S09_FINAL_RESULTS.md` for full
paired/fold/seed comparisons, `SOURCE_RESULTS.md` and `MECHANISM_PROBES.md`
for negative mechanisms, `S09_COSTS.md` for measured costs and
`S09_EXECUTION.md` for runtime scope. The visual summary is
`S09_MECHANISM_SUMMARY.png`/`.pdf`.

S08 historical93.01% remains separate. S09 did not establish global SOTA,
universal timing-invariant TX identity, or a standalone EMA contribution.
Registered experiments are complete and stopped; no automatic search,
merge, GitHub push or weight publication.
