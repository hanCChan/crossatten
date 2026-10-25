# M2: WiSig Data And Released-Weight Validation

Observed on 2026-10-25. **Data preparation passed; WaveMLP released weights
passed; IB-RFF/FAN released-weight gate failed and its pipeline is stopped.**
No WiSig retraining or target-guided hyperparameter selection has been run.

## Data

ManyRx reused the existing user archive under `gai`; ManySig was downloaded
from the UCLA page's official Google Drive link. Conversion preserves RX names,
chronological days, non-equalized index 0, and the first 100 packets per cell.
Our cache and both upstream loaders give **exactly equal float32 inputs and
labels across every RX/day** (maximum input difference 0.0).

| Dataset | Fold | Training | Validation | Pool | Matches official log |
|---|---:|---:|---:|---:|---|
| ManyRx | 0 | 32527 | 13941 | 46468 | yes |
| ManyRx | 1 | 32527 | 13941 | 46468 | yes |
| ManyRx | 2 | 32667 | 14001 | 46668 | yes |
| ManyRx | 3 | 33040 | 14160 | 47200 | yes |
| ManySig | 0-3 (each) | 7560 | 3240 | 10800 | yes |

Spec corrections based on inspected upstream code/data:

- ManySig `max_sig` is 1000, not 200; selection still uses the first 100.
- Official layout stores `(N,256,2)` raw float64; loader transposes and
  normalizes. Our own cache is normalized float32 `(N,2,256)`.
- Existing real upstream `dataset/` directories are preserved; dataset-specific
  symlinks point to our ignored official layout.

ManySig official ZIP SHA256:
`5c3d6f5aece87a86ecd8cf5a0f6bdc56535f34cfd3e8a32e8ce6dbc448c85bac`.
Extracted pickle SHA256:
`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`.
Detailed counts and source hashes are in `out/05-1-*-manifest.json`.

## Source-Only Audit

Scope: fold-0 training RX, days 1 and 2 only (46468 ManyRx packets; 10800
ManySig packets). No target packets were used to choose the cutoff.

| Dataset | lag16 correlation | lag20 correlation | Fs | -10dB PSD edges | Frozen cutoff |
|---|---:|---:|---:|---|---|
| ManyRx | 0.2247 | 0.9208 | 25 MHz | +/-7.519531 MHz | 7.67578125 MHz |
| ManySig | 0.2216 | 0.9553 | 25 MHz | +/-7.519531 MHz | 7.67578125 MHz |

STF period is 20 samples, not the specification's illustrative 16-sample
20-Msps assumption. Frozen cutoff = source average PSD -10dB edge plus half
the 312.5kHz subcarrier spacing. The measured STF-dominated PSD is not a
measurement of every possible data subcarrier's support.

The lag20 sliding-correlation trace is strongly periodic early in the packet,
then falls sharply near the transition (see `out/05-2-audit.json`). Ideal
25-Msps preamble positions would be STF [0,200), LTF CP [200,240), first
LTF [240,320). These are **candidates, not verified alignment**: the 256-sample
crop does not contain the full LTF or its CP-matching tail, so the requested
CP/tail equality check cannot be completed on these samples. WiFi residual/
symstack design must respect this limitation and must not use LoRa lengths.

## Released Weights

All evaluation uses days 3,4 on target RX, unchanged published weights, batch
32, official normalization and packet-weighted accuracy. No training occurred.

| Method | Dataset | Fold0 | Fold1 | Fold2 | Fold3 | Four-fold mean | Official mean |
|---|---|---:|---:|---:|---:|---:|---:|
| WaveMLP | ManyRx | 88.5696 | 88.0570 | 91.9177 | 86.1593 | 88.6759 | 88.68 |
| WaveMLP | ManySig | 90.8611 | 89.6389 | 95.9722 | 96.0278 | 93.1250 | 93.13 |

WaveMLP means round to the official README values. Per-fold upstream logs
are not included in that repository, so the acceptance comparison uses its
published aggregate, with all our folds and per-RX counts reported in
`out/05-3-released-WaveMLP-RFF.json`.

FAN (IB-RFF stage1), ManyRx fold0: **89.9367% vs log 90.6600%, delta
-0.7233 pp**, outside the preregistered +/-0.5pp tolerance. Stop before
evaluating subsequent FAN/IB-RFF weights or retraining this pipeline.
This does not establish why they differ: count equality, loader equality, and
WaveMLP equality validate important parts of the pipe but do not prove that
the FAN released checkpoint corresponds to the provided log or that every
legacy serialization behavior is identical. Evidence is preserved in
`out/05-3-released-IB-RFF.json`.

The historical IB-RFF whole-model pickle could not directly forward on
PyTorch 2.5 because old GELU instances lacked `approximate`. For compatibility,
reconstruct the **unchanged inspected official FAN** and strictly load its
entire published state; every state tensor matches exactly. No upstream
source, architecture, weight, input, or training recipe was edited.

## Sources And Limits

- UCLA CORES official WiSig compact-subset page, inspected 2026-10-25.
- IB-RFF commit `5a03a04da86c4bbf5cf90f41a55c268bf2152203`:
  `utils/load_data.py`, `backbones/FAN.py`, stage1 logs, published weights.
- WaveMLP-RFF commit `1ade72d0082428db2087633a44ae533f1022ca22`:
  loader, model, weights, README. No license file was found; code remains
  local and ignored, not copied into this repository.
- Data is CC BY-NC-SA 4.0 per UCLA. The Kaggle ManyRx archive is not claimed
  byte-identical to the UCLA original despite matching counts and WaveMLP
  accuracy. Download provenance and hashes are retained.

M2 is **not fully passed**. Resolve the FAN checkpoint/log mismatch and LTF
alignment limitation before running WiSig training milestones.
