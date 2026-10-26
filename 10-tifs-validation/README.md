# S10: Reliability-Aware Cross-Receiver Validation

Evidence cutoff: 2026-10-26. Historical stages and shared libraries are read-only.
Branch: `exp/s10-tifs-reliability`, base `1cb9c8f`.

One candidate replaces the S09 hard adjacent-frequency phase mask by a fixed
continuous product reliability, tau=0.05. No architecture/loss sweep.
The primary source criterion is **raw minimum packet-validation CE**, fixed
before source fits. EMA and held-CE selections are descriptive only.
Failure of any registered gate means STOP for the candidate, not new variants.

Execution order:
1. `python prepare.py`; deterministic cache reconstruction and official loader parity.
2. `python test_stage.py`; `python register.py`; commit registration and code.
3. `python launch.py --jobs jobs_smoke.json`; accept inspected official ASKNet path.
4. Concurrent source screening (32 fits) and ASKNet (24 fits).
5. Commit all source evidence; `python source_gate.py`; commit the source decision.
6. Train only admitted formal jobs; commit checkpoint seals before target scoring.
7. Generate the nine requested reports, robustness, intervals and cost evidence.

All outputs live here; data caches live in shared `data/wisig`.
No automatic fit retries, no target reads during training, no target-driven tuning.
Official ASKNet is imported at pinned commit, not copied/reimplemented.
Goals of 90%/93% and a SOTA claim are conditional scientific outcomes, not guarantees.

Completed: 32 source fits +24 ASKNet fits +4 smoke fits; no failed training.
Candidate verdict **STOP**. ManyRx source gain-0.4881pp; only1/4fold positive.
No conditional soft target fits or new trained ablations were admitted.
ASKNet CRD88.7697%/93.4838% (ManyRx/ManySig). See `RESULTS.md`.
Eight tests and independent prediction/access/checkpoint verification passed.

`advance.py` is intentionally fail-closed: the attachment's24-ablation-fit ceiling
cannot cover both arms x2datasets x4folds x3seeds (48). No such job set is generated.
The source STOP makes that conditional ambiguity irrelevant to this completed execution;
future re-registration must resolve it before any ablation fitting, never after target-driven tuning.
