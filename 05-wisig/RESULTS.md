# M2: WiSig Data And Released-Weight Validation

Observed on 2026-10-25. **Compact count and released-weight gates passed
for both datasets and all inspected methods.** LTF alignment remains limited
by the 256-sample crop, as described below.
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

Evaluation uses the days matching each official log: FAN stage1 logs are CR
(days1,2); IB-RFF stage2 CRD and WaveMLP use days3,4. All use target RX,
unchanged published weights, batch32, official normalization and packet-weighted
accuracy. No training occurred.

| Method | Dataset | Fold0 | Fold1 | Fold2 | Fold3 | Four-fold mean | Official mean |
|---|---|---:|---:|---:|---:|---:|---:|
| WaveMLP | ManyRx | 88.5696 | 88.0570 | 91.9177 | 86.1593 | 88.6759 | 88.68 |
| WaveMLP | ManySig | 90.8611 | 89.6389 | 95.9722 | 96.0278 | 93.1250 | 93.13 |
| FAN (CR) | ManyRx | 90.6709 | 85.5506 | 92.8013 | 89.7133 | 89.6840 | 89.68 |
| FAN (CR) | ManySig | 95.6389 | 87.9444 | 96.6111 | 92.0556 | 93.0625 | 93.06 |
| IB-RFF (CRD) | ManyRx | 91.1456 | 87.8671 | 92.1456 | 85.5935 | 89.1879 | 89.19 |
| IB-RFF (CRD) | ManySig | 95.8889 | 88.5278 | 96.2778 | 93.5833 | 93.5694 | 93.57 |

WaveMLP means round to the official README values. Per-fold upstream logs
are not included in that repository, so the acceptance comparison uses its
published aggregate, with all our folds and per-RX counts reported in
`out/05-3-released-WaveMLP-RFF.json`.

The initial FAN fold0 comparison incorrectly compared CRD (89.9367%) with
the stage1 **CR** log (90.6600%). Its apparent -0.7233pp mismatch was an
evaluation-script error, not evidence against the released model. This was
caught by inspecting the log's `CR Testing` marker; the evaluation now uses
days1,2 for stage1. The corrected matched-protocol verification passed all
16 FAN/IB-RFF fits; the largest absolute log difference is 0.0109pp.
The script now parses each log's explicit CR/CRD marker and tests cover this
distinction. This correction is retained here for transparency, not presented
as a model failure.

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

M2's data and published-weight checks are **passed**, with the source signal
audit and frozen cutoff recorded. Full LTF alignment cannot be validated from
the crop; keep that restriction explicit when planning WiFi components.
No stage05 training follows automatically: the S03 first-round stop still applies.
