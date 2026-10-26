# ASKNet Fair Reproduction

Official commit: `f00846d68425315b68ae69b8c6b712e08559c2fe` (2026-08-12).
24 official-path formal fits completed. No target score was consulted to change a default.
Evidence levels: model/loader/training code and our execution verified; upstream checkpoints unavailable.
Loader acceptance covered every cached cell with exactly zero float32 difference against official float64 normalization-then-cast. TX/RX/date counts and quarter partitions agree.
Official training functions are called directly with the original CPU DataLoader/RNG, Adam1e-3/wd0/batch128, Plateau factor.1/patience5/min1e-6, early-stop10 and200 epoch limit.
LSG Gumbel cutoff and complex unitary QR/SKO are unchanged. No our phase encoder is imported into ASKNet.

| Dataset | Official Single-Run Log Mean (%) | Fair RX-Equal CRD (%) | Fair Macro-F1 (%) |
|---|---:|---:|---:|
| ManyRx | 88.5300 | 88.7697 | 86.5035 |
| ManySig | 93.3475 | 93.4838 | 93.1502 |

Upstream log metric is packet accuracy, averaged across four single runs; fair metric is RX equal across4folds/3seeds. Raw/fair packet and RX metrics are retained in target files.
Environment differs: official1.11/CUDA11.3/3080Ti vs2.5/CUDA12.4/A100. Complex QR and random Gumbel selection can yield different runs; failure to exactly match a published number does not itself justify retuning.
Baidu pretrained-share access was attempted and not downloaded. No verification of original checkpoints or numerical paper reproduction is claimed; code-path acceptance is narrower.
Official LoRa uses a different dataset/spectrogram recipe, not Shen P-H5x160. Its number is excluded.
No license file was found in the pinned repository. Third-party source remains external and is not redistributed.

Primary sources:
- [Official model](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/f00846d68425315b68ae69b8c6b712e08559c2fe/WiSig/backbones/PatchNet.py)
- [Official training](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/f00846d68425315b68ae69b8c6b712e08559c2fe/WiSig/main.py)
- [Official loader](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/f00846d68425315b68ae69b8c6b712e08559c2fe/WiSig/utils/load_data.py)
- [Official logs](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/f00846d68425315b68ae69b8c6b712e08559c2fe/WiSig/logs/ASKNet/logs)
- [Publisher DOI](https://doi.org/10.1109/LCOMM.2026.3718894). Crossref metadata was retrieved on2026-10-26 after proxy fallback.
