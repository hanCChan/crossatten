# S11 Identity-Preserving Spectral Generalization

Base: `exp/s10-tifs-reliability@a4a792f`. Branch: `exp/s11-spectral-generalization`.
Only this directory is edited; historical S00-S10/shared libraries remain read-only.

1. Freeze `registration.json` and code before source outcomes/new fits.
2. Verify all24 ASKNet saved cutoff logits and hashes. Locate original checkpoints;
   if unavailable, run only2 registered source-only hard refits for BN diagnosis.
3. Compute fit-packet-only TX/RX/day spectra and zero-fit F1/F2 source probes.
4. Admit at most one route into32 paired source fits, reuse exactly matched S09 hard predictions.
5. Only after ordinary GO, narrower16-fit source RX/day paired confirmation; then48 fresh-seed formal fits.
6. Only confirmed methods enter bounded ablations/physical tests. No target-tuned extension.

The supplied estimates conflict with the full matrix: both candidates x2datasets x4folds x2groups x2seeds=64.
To honor the32 ordinary-fit cap, zero-fit mechanism probes select at most one route before network fitting.
The16 day-transfer fit cap supports only last group/seed8101; two-group/two-seed validation is not claimed.
Unselected/unadmitted experiments are labeled not run, not silently assigned synthetic outcomes.

F1 is fixed63 center bins, a diagnostic not a new contribution.
F2 is a fixed fit-only variance-ratio prior, g in[.5,1]. No trainable spectral gate,
new loss, extra depth or Soft reliability-threshold search.
Targets90%/93.5% and novelty are conditional hypotheses, not promised outcomes.

## Execution

S11 stopped at the frozen zero-fit mechanism gate; see `RESULTS.md`. The agent-added
raw scatter retention gate is amplitude dependent and is not a proof of identity
information loss. No formal candidate network comparison or target fit ran.
Two source hard refits reproduced all eight S09 checkpoint hashes exactly.
Two F1 smoke fits passed; two F2 smoke attempts failed before training and were not retried.
The eight reports distinguish completed diagnostics from unexecuted conditional experiments.
Checkpoints stay in local `runs/` with SHA256 seals; `.pt` files are excluded from ordinary Git.

```bash
python -m unittest discover -s 11-spectral-generalization -p 'test*.py' -v
python 11-spectral-generalization/verify.py
```
