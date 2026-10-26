# S09 Execution Record

Completed locally on2026-10-26, Asia/Shanghai. Four NVIDIA A100-SXM4-40GB GPUs with CUDA MPS; source queue used eight fits/GPU across GPUs0-2, attribution six fits on GPU3. Every trainer used two CPU threads.

Python3.12.8, PyTorch2.5.0+cu124, CUDA12.4. Existing cached first100 non-equalized packets/TX/RX/day; normalized float32 IQ,256 samples. Official four-fold receiver lists and source dates1-2 unchanged.

| Phase | Fits | Sum fit elapsed hours | Min seconds | Max seconds |
|---|---:|---:|---:|---:|
| smoke | 5 | 0.001387 | 0.801 | 1.141 |
| attribution | 60 | 0.354413 | 6.687 | 98.212 |
| source | 96 | 4.084547 | 53.168 | 270.629 |
| final | 0 | 0.000000 | 0.000 | 0.000 |

Summed durations are not dedicated GPU-hours or total wall time: fits share GPUs concurrently. They exclude loading/setup, target-only evaluation, CPU diagnostics and inference benchmarking. No architectural training-speed claim.

All156 formal fits and five smoke fits completed with zero failed/interrupted training attempts. All96 source fits ran exactly40 epochs. Formal attribution had36 fixed-budget and24 raw-packet early-stop fits. No new centered/curvature mechanism passed the source gate, so zero conditional final fits were run. Sixty registered attribution models each evaluated CR and CRD.

Source decision committed at2026-10-26T09:24:10+08:00. No target reads preceded that decision. Shared initialization, train/validation partitions, source/target roles, weights and predictions passed acceptance. Raw/EMA checkpoints from one fit are correlated outputs, not extra seeds.

Only model parameters are EMA averaged; clean fit-training inputs recalibrate BatchNorm each epoch. No held, packet-validation or target BN calibration. Final inference is a single checkpoint, not an ensemble.

Historical S00-S08 and shared library remain unchanged. Weights stay ignored/local; configs, histories, predictions and weight hashes are versioned. No target-selected retries, new losses, automatic merge or push.
