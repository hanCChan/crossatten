# WiSig Preparation (M2 Only)

Execution date: 2026-10-25. Follow the data-preparation, source-only signal
audit, and released-weight validation gates in
[the specification](../docs/S03_S05_IMPLEMENTATION_SPEC.md).
No WiSig training or target-domain hyperparameter selection is authorized in
this milestone. Stop on any count mismatch or >0.5 pp released-weight mismatch.

ManyRx reuses the user's `gai` Kaggle v1 archive, SHA256
`b0eca705816b03a47290286fd23ca1a10e02c6f9bd29b5461cfd92c13a86fbdd`.
ManySig is downloaded only from the Google Drive file linked by the UCLA
official WiSig page. Archive and extracted pickle SHA256 values are recorded
in the conversion manifests. Archives, pickles, and packet caches are ignored.

## Layout Correction

The specification's loader layout description is inconsistent with both
upstream loaders. Their actual code transposes the stored arrays from
`(N,256,2)` to `(N,2,256)` and then power-normalizes them. Our official layout
therefore preserves source dtype/scale in `(N,256,2)`; our own cache is
normalized float32 `(N,2,256)`. Test these outputs against the upstream loader,
including all labels, before evaluating weights. Do not transpose twice or
normalize the official files and then normalize them again in its loader.

RX order is the upstream name list (not compact pickle indices). Dates follow
chronological compact metadata. Select equalization index 0 and the first 100
signals per TX/RX/day. The source training pool is the 3/4 training RX on days
1 and 2, with official 70/30 packet split. All four folds must exactly match
the stage-1 log training and validation denominators for each dataset.

`external/IB-RFF/dataset` already has a real directory containing `link.txt`.
Preserve it and place dataset-specific symlinks inside it, without replacing
any real directories.

## Source Evidence

- UCLA CORES WiSig page (compact subsets, data license).
- IB-RFF inspected at `5a03a04da86c4bbf5cf90f41a55c268bf2152203`.
- WaveMLP-RFF inspected at `1ade72d0082428db2087633a44ae533f1022ca22`.
- Both `utils/load_data.py` implementations and IB-RFF stage-1 logs.
- Third-party code remains ignored in `external/`; no WaveMLP code is copied.
