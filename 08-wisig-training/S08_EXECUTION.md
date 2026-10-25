# S08 Execution Record

Completed locally on 2026-10-26, Asia/Shanghai; branch `exp/s08-wisig-training`.
Baseline `74f594312949f18613030d59743dba88fc20eca7`. No new target-driven
training was launched after final evaluation.

## Environment

- Python3.12.8: `/data/miniforge/envs/env-3.12/bin/python`.
- PyTorch2.5.0+cu124, CUDA12.4, cuDNN90501.
- NumPy1.26.4, SciPy1.15.1, scikit-learn1.6.1.
- Four NVIDIA A100-SXM4-40GB GPUs; CUDA MPS concurrent small fits.
- Source queues ran up to eight processes/GPU; final queue six/GPU.
  Trainers used two CPU threads; inference benchmarks four CPU threads.
- Frozen cached first100 non-equalized packets/TX/RX/day, normalized float32
  IQ,256 samples. Official four-fold RX lists and source dates1-2 unchanged.

## Actual Registered Runs

Values are computed from each canonical `jobs_<phase>.json` and its
`runs/<phase>/<id>/metrics.json`. Field `seconds` records training and final
checkpoint/prediction evaluation, excluding earlier loading/setup.

| Phase | Fits | Sum elapsed hours | Min seconds | Max seconds |
|---|---:|---:|---:|---:|
| ManySig training controls | 48 | 1.467373 | 21.400 | 264.451 |
| ManySig phase/null controls | 32 | 0.265543 | 21.728 | 39.326 |
| ManyRx source controls | 32 | 1.194211 | 92.837 | 182.682 |
| ManySig final candidate/control | 24 | 0.148756 | 8.432 | 40.583 |
| Source smoke tests | 5 | 0.002242 | 1.465 | 1.758 |

**136 formal fits within the160-fit cap; five separate smoke fits.**
All registered fits completed without failed or interrupted training attempts.
All16 `official_awgn_full200` source fits ran200 epochs. Other conditions
retained the official minimum raw-packet-CE early-stop patience.
Raw/EMA and packet/held checkpoints are correlated products of the same
fit, not extra independent training seeds.

**Summed elapsed durations are not dedicated GPU-hours or total wall-clock
time.** MPS fits share GPUs concurrently. Source calibration, multiple
checkpoint evaluations and variable early-stop budgets complicate timing
comparisons. CPU diagnostics, acceptance/report generation and benchmarking
are excluded. These values cannot establish architectural training efficiency.

## Final Training Budgets

ManySig final candidate/control each had12 fits (four folds, three seeds).
Candidate mean elapsed duration15.171 seconds; control29.456 seconds.
They have different training budgets and selection work, not matched compute.

| Fold | Candidate feature | Checkpoint | Source-frozen refit epochs |
|---|---|---|---:|
| 0 | adjacent spectral phase | EMA last | 10 |
| 1 | adjacent spectral phase | raw last | 6 |
| 2 | adjacent spectral phase | EMA last | 4 |
| 3 | adjacent spectral phase | raw last | 13 |

All final candidates use Adam/Plateau without AWGN and the full legal source
pool split70/30. Budgets are the upper median of four source-group/seed held-CE
epochs, frozen before final fitting. Final control uses the three-channel
frequency model and minimum source packet CE; actual stop epochs13-30.
Source-held groups later incorporated into fitting are not independent final
validation. No final early/late epoch was picked from target performance.

## EMA And Cost Scope

EMA uses PyTorch AveragedModel, one-epoch half-life, parameter averaging only.
BatchNorm buffers are not averaged; clean fit-training inputs recalibrate
their running statistics each epoch. Validation/held/target inputs never
enter calibration. Integer counters retain integer semantics.

Six inference graphs were benchmarked after all trainers exited and two
idle-GPU snapshots passed. Inputs were synthetic resident IQ with random
weights. Timing includes FFT/features but excludes transfer/detection.
FLOPs only partially count FFT/scalar work. Results show runtime variation;
small parameter differences or lower median latency do not prove a speedup.
Final inference uses one selected checkpoint, not an EMA/raw ensemble.

## Freeze And Acceptance

Final whitelist committed as `7a5e4f8` at **2026-10-26T02:06:13+08:00**.
`decision_final.json` SHA256:
`cea438591865a8adba7b4ba2bce4253275e1fcbe5683cd23f714158d6db8702d`.

All source evidence/configurations, predictions, saved local weight hashes,
RX/date roles and target-read timestamps passed acceptance. S00-S07 tracked
files and shared library are unchanged; only additive S08 outputs and
`.gitignore` rules were added. Checkpoints stay ignored and local.

ManyRx failed the dataset admission gate (two supported folds, required three)
and has no S08 final target score. ManySig passed and finished formal confirmation.
Registered exploration stopped; no automatic merge, push or weight publication.
