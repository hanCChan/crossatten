# W3-A/B review package

This archive supports feature-level review with Python 3.10+ and NumPy. Extract the whole archive, preserving paths, then run from its root:

```powershell
python research/wisig_w3/audit_b.py --portable
```

The audit verifies all registered file digests, the source-only cohort and TX/day roles, every shallow fit's normal equations, all 69,120 held-TX metric rows and 1,440 geometry rows, and the prespecified TX-cluster bootstrap intervals. It reads frozen features and saved shallow parameters. It does not read raw waveforms, import PyTorch, refit operators, or train networks.

The original repository audit additionally checks that the registration and bound sources existed in commit `5165f11` before the recorded first official IQ readout. Historical Git LF versus Windows CRLF conversion is accounted for in text-blob comparisons; exact working-file hashes remain checked separately.

The archive deliberately does not contain raw WiSig waveforms or neural checkpoints. It therefore verifies feature-to-result arithmetic, not the entire raw-IQ-to-feature pipeline. The original data and weight identities are documented in the acquisition, manifest, preflight and feature receipts. Absolute paths in historical receipts identify the original environment; the portable verifier does not open those paths.

Read `research/wisig_w3/README.zh-CN.md` for the scientific interpretation. The decision is to stop this prespecified rank8 implementation. This does not establish that all receiver-associated changes are nontransferable. No W3-C neural training is authorized or performed.

WiSig source: [UCLA dataset](https://cores.ee.ucla.edu/downloads/datasets/wisig/), [official subset creation](https://github.com/WiSig-dataset/wisig-subset-creation). Data-derived features are supplied for this research review under the source data's CC BY-NC-SA 4.0 conditions; they are not a new public dataset release. Preserve attribution and consult the original terms before redistribution.
