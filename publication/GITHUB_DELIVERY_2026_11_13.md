# GitHub delivery: W1/W2, LoRa manuscript, and W3-A/B

Repository: [hanCChan/crossatten](https://github.com/hanCChan/crossatten).

The original research commits are preserved on three branches:

- [W1/W2 quality study](https://github.com/hanCChan/crossatten/tree/exp/wisig-quality-preservation), through `a666166`.
- [LoRa IEEE manuscript preparation](https://github.com/hanCChan/crossatten/tree/writing/lora-submission-preparation), through `6ce59ff`.
- [W3-A/B and all preceding deliverables](https://github.com/hanCChan/crossatten/tree/research/wisig-w3-transfer), including the frozen W3 outcome and this delivery commit.

The W3 branch contains the complete preceding code, reports, protocols, CSV evidence, manuscript sources, main and supplementary PDFs, and the following portable review archives:

| Archive | Contents | SHA256 |
|---|---|---|
| [LoRa LaTeX source package](../output/review/lora_hstem_submission_source.zip) | IEEE main and supplement sources, PDFs, tables, figures, bibliography and evidence | `30eb3559cd2d9a41e55aae58b8ae7c72d7ad8c69d7f26f505874de5882ee09eb` |
| [WiSig W1/W2 closure audit](../output/review/wisig_closure_audit.zip) | Portable predictions and audit inputs, prior STOP decisions, and NumPy verification | `dae59f81c1f86f15d4b61e92a219f2e0f7fb7c3b216bd929a61fba0a0ca7fa6f` |
| [WiSig W3-A/B review](../output/review/wisig_w3_ab_review.zip) | Frozen features, all shallow operators, complete metrics, registration and NumPy replay | `19beebd5d97aeaf8d55845fe523c8b2778f19d67dffb5d9eac7714854f0d71b7` |

The W3 branch also tracks `research/wisig_w3/artifacts/B_FEATURES.npz` and `B_OPERATORS.npz`, so its independent feature-level audit can run directly after cloning:

```powershell
python research/wisig_w3/audit_b.py --portable
```

That verifier needs Python and NumPy. It does not load raw IQ or neural weights. For W1/W2, extract the closure archive and follow its portable audit instructions. For LoRa, see [the build instructions](lora_hstem/README.md).

The registered W3 inputs are stored byte-for-byte in Git, including their original line endings. A fresh Git export also passed the complete NumPy audit; see [checkout verification](GITHUB_CHECKOUT_QA.json). This storage-only preservation does not change any code statement, registration, result or original commit identity.

The verified archives are uploaded byte-for-byte; publication does not regenerate or alter their scientific contents. The archive-level manifest is [GITHUB_DELIVERY_FILES.json](GITHUB_DELIVERY_FILES.json).

Raw official WiSig downloads, repeated download parts, temporary extraction directories, render previews and intermediate local training/optimizer checkpoints are not part of these review deliverables. Original dataset provenance, access boundaries and artifact hashes remain documented. This is a GitHub research archive upload, not a journal submission, and it does not change any STOP decision or authorize W3-C.
