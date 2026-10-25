# Stage 07 Execution Record

Local close-out: 2026-10-26, Asia/Shanghai. Baseline `ebe3225`; independent
branch `exp/s07-wifi-osu`. All registered experiments finished before the final
cost measurements and acceptance audit. No additional training was launched
after inspecting final target results.

## Environment

- Python 3.12.8, `/data/miniforge/envs/env-3.12/bin/python`, `PYTHONPATH=lib`.
- PyTorch 2.5.0+cu124, CUDA 12.4, cuDNN 90501.
- NumPy 1.26.4, SciPy 1.15.1, scikit-learn 1.6.1.
- Four NVIDIA A100-SXM4-40GB GPUs. Concurrent small fits used CUDA MPS;
  the staged executor default was up to 16 processes per GPU.
- Inference costs were measured after training stopped, with two idle-GPU
  snapshots, CPU four threads and resident synthetic IQ. See `S07_COSTS.md`
  and `costs.json` for measurement scope and limitations.

## Registered Fits

Numbers below are recomputed from `jobs_<phase>.json` and the corresponding
`runs/<phase>/<id>/metrics.json` field `seconds`. They count each registered
run once, including identical final configurations from separately reported
pipelines. Smoke fits are excluded from the formal WiSig cap.

| Phase | Completed fits | Sum of run elapsed hours | Min run seconds | Max run seconds |
|---|---:|---:|---:|---:|
| features | 56 | 16.613528 | 432.30 | 1915.07 |
| replicate | 23 | 5.783478 | 315.79 | 1707.33 |
| training | 72 | 19.694469 | 22.42 | 1676.71 |
| training_replicate | 23 | 3.497672 | 18.97 | 1284.02 |
| combine | 9 | 0.720839 | 12.21 | 1203.97 |
| final | 72 | 14.464322 | 29.11 | 1824.46 |
| WiSig smoke | 14 | 0.007906 | 1.58 | 2.61 |
| OSU formal | 24 | 0.475978 | 54.54 | 100.04 |

Total formal WiSig fits: **255/280**; OSU formal fits: **24/24**.
Four OSU smoke fits and retained detector-pilot/interrupted-launcher evidence
are separate, not additional successful formal models. Zero registered formal
fits failed. Failed preparation and interrupted attempts remain available.

**Summed run elapsed durations are not dedicated GPU-hours, total wall-clock
time, or compute-normalized model comparisons.** Fits ran concurrently under
MPS; `seconds` is each trainer's recorded elapsed duration. It excludes some
loading/preparation and does not measure actual device occupancy. CPU detection,
raw-data auditing, diagnostic/report generation and inference benchmarking are
not included. Official early-stop and fixed-200-epoch recipes have different
budgets, so their durations cannot establish architectural training efficiency.

## Final Fit Durations

Each cell averages twelve registered fits (four folds, three seeds).

| Dataset | Pipeline | Mean run seconds |
|---|---|---:|
| ManyRx | source_frontend | 1794.958 |
| ManyRx | source_training | 966.432 |
| ManyRx | source_combination | 967.062 |
| ManySig | source_frontend | 517.991 |
| ManySig | source_training | 46.458 |
| ManySig | source_combination | 46.397 |

Source-selected training and combination have the same final configurations.
ManySig selected the official early-stop recipe on all folds; ManyRx selected
it on folds 1 and 3 and unified mild-AWGN training on folds 0 and 2.
Selection used only the relevant fold's source RX on days 1-2.

## Provenance And Stop

Final jobs were frozen in commit `663ef69` at **2026-10-26T00:47:00+08:00**,
before formal target reads. `decision_final.json` SHA256:
`80c219b324045b39c6fcaa7db6ac73cef90c9b7e2ffa24b8846ad452a3b9d8b4`.

`acceptance.json` passed all registered prediction/configuration checks,
source/target access-role checks, final-freeze timestamps and historical-file
preservation. `wifi_final_report_evidence.json` records exact final report
hashes. Model weights and cached raw inputs remain local and ignored by Git.

Registered exploration is complete and stopped. No global or dual-dataset SOTA,
new blind-test claim, automatic merge, GitHub push or weight publication.
