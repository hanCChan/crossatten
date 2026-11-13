# W4 portable review

From the archive root or a complete checkout, with Python 3.10+ and NumPy:

```powershell
python research/wisig_w4/audit_w4.py --portable
```

The verifier resolves every historical absolute artifact reference through `REPLAY_FILES.json` to the supplied relative files, even if the original D: drive still exists. It checks registered code/evidence hashes, exact saved predictions, original source metadata, data roles, paired training histories, selected checkpoint hashes and validation CE, all final classification/risk tables, conditional RX intervals, the five preregistered gates, the E oracle/fixed fusion outputs and the 24-TX prototype diagnostic.

It imports no PyTorch, reads no IQ, refits no network and changes no estimator. The separate local `verify_training.py` audit loaded the actual tensor states to verify exact encoder/BN transfer, A0-matched random heads, optimizer/scheduler reset and training orders before held readout. Its receipt and metadata are included. The portable review does not independently retrain the raw-IQ pipeline or reproduce GPU training/latency.

The archive includes 15 selected checkpoints: 12 from the new auxiliary/downstream fits and three reused A0 models. E uses original saved HStem/ASKNet predictions and does not include ASKNet model weights. The scope excludes raw official datasets, raw training caches and optimizer/epoch-state intermediates. Data-derived artifacts retain the original WiSig dataset's [CC BY-NC-SA 4.0 conditions](https://cores.ee.ucla.edu/downloads/datasets/wisig/); implementation snapshots retain their original provenance and licenses.

Scientific outcome: mean A2 accuracy improves, but only 2/3 seeds improve over A1, so the prespecified relation implementation is not promoted. Neither 93.05% on this internal source protocol nor the labeled auxiliary diagnostic is a new unseen-target/SOTA result.
