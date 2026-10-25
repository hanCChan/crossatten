# S03 Round 1: Chirp Structure

Preregistered on 2026-10-25, following
[the implementation specification](../docs/S03_S05_IMPLEMENTATION_SPEC.md).
This document is a prediction and analysis plan, not evidence of success.

## Protocol

P-H5x160 is unchanged. All arms use raw packet AWGN U[0,50), packet RMS
normalization, strict |f| <= 62.5 kHz, AdamW warmup/cosine 200 epochs, and
source-RX-equal validation CE checkpoint selection. No blind receivers are read.
Exploration: seeds 101,102 x folds 0,1, 32 fits. Confirmation: seeds
201,202,203 x folds 0,1, 48 fits. All eight arms enter confirmation.

## Arms And Predictions

| Arm | Control | Single change | Parameters | Prediction |
|---|---|---|---:|---|
| s03_base | s02_ep200 | rerun only | 580008 | exact deterministic equality |
| s03_preln | base | LayerNorm before mixer 1 | 580136 | small mean change; lower within-fold seed SD |
| s03_preln_d4 | preln | depth 4 | 622374 | less collapse; gain unknown |
| s03_resid | preln | two unit-RMS complex residual channels | 580584 | improves (main hypothesis) |
| s03_nofft | preln | remove FFT input, keep IQ/log amplitude/phase | 579912 | unknown |
| s03_symstack | nofft | symbol-aligned token coverage | 579912 | improves (second hypothesis) |
| s03_ifreq | preln | one instantaneous-frequency deviation channel | 580360 | unknown |
| s03_hstem | preln | hierarchical stride convolutions | 142392 | unknown; about 75% fewer parameters |

Residual = complex IB packet minus its coherent eight-symbol mean, normalized
to unit RMS with epsilon 1e-12. Compute centered differences relative to symbol
one (mathematically equivalent) so float32 mean-roundoff is not amplified in
exactly periodic inputs, which have strictly zero residual.
Symstack shares the same k7/k5 convolutions between eight chirps, then stacks
their features and tokenizes 32 aligned positions per token. Its convolution
and tokenizer parameter counts equal nofft exactly; flattened initial weights
and downstream initialization also match. Position and relative-position
encoding now index within-chirp position, equivalently instantaneous frequency,
not chronological position across the packet. Symbol-local boundary padding
and training BatchNorm grouping follow the specified tokenizer.

Instantaneous frequency is padded adjacent phase difference minus the ideal
SF7 upchirp sweep, clamped to +/- 2BW then divided by 2BW. Unlike Bothereau's
ICC 2026 phase-difference + ERT/LDA pipeline, it is a single input channel in
the fixed deep backbone evaluated across receivers. It is not claimed novel
without the required comparison. Pre-LN is an engineering intervention;
existing saturation measurements do not establish why depth4 collapsed.

## Frozen Decisions

Comparisons and controls are fixed in `03-3-aggregate.py`. Symstack vs preln
and base vs s02_ep200 are report-only references. The primary clean symstack
comparison is vs nofft. Delta is B minus A, percentage points.
Use `xa.aggregate.verdict`: improves requires mean >= +1, >=3/4 positive
exploration pairs (>=5/6 confirmation), and difference in minimum receiver
seed-mean accuracy >= -2. Harms requires mean <= -1 and the corresponding
number of negative pairs; otherwise unclear.

Only confirmation dev_val verdicts can make factors eligible for a later round.
Confirm RX accuracy is report-only because those receivers have been repeatedly
inspected. Exploration must be complete, aggregated, frozen, and committed
before confirmation. Report every fit and failure; never silently omit failed
arms or select seeds. Stop after round 1 for the user's decision.

## Verification And Mechanisms

Before training: all legacy tests, new CPU invariants, then eight GPU smoke
fits. Legacy5 weights/output must be bit-identical to IBVariantPipeline for
the same seed; symstack/nofft must match parameter counts and flattened initial
weights. Verify finite gradients, residual reconstruction, token locality,
ideal-chirp instantaneous frequency, and 32x64 token shapes.

After confirmation, evaluate all six resid checkpoints with residual channels
unchanged, zeroed, or packet-shuffled (one-packet cyclic shift within evaluation
batches; batch 64). Evaluate all six symstack checkpoints with original,
reversed symbol order, or different circular shifts per symbol (64*s samples).
These are diagnostic interventions, not additional selection criteria or
evidence that no other information is used. Their effect is recorded per RX
on dev_val and report-only confirm, without fitting new weights.

Benchmark all arms end-to-end: raw IQ on device to logits, counted convolution/
matrix FLOPs (FFT and scalar operations not fully counted), GPU batch1
50 warmup/300 trials p50,p95,p99, batch512 5/20 throughput, CPU four threads
20/100 trials p50,p95. Run benchmarks without competing GPU training.
