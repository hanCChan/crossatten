# S12 Status

Status: **AUTHORIZED_PENDING_EXECUTION**, 2026-10-26.

New stage-owned paths implement full-fit F2 priors, class/domain-covering smoke
sampling and an explicit source-only pilot matrix. Historical results remain
accepted; no old checkpoint is accessed and no historical stop is relabeled.

Completed: implementation, write-once protocol preparation and 14 passing
CPU synthetic regression tests (including full-fit prior consistency, smoke
coverage, deterministic negative controls, pairing and target-access guards).
Not run: real-IQ diagnostics, GPU smoke fits, 16 pilot fits, independent
confirmation, source RX/day transfer or target CR/CRD evaluation.
No new network accuracy or significance claim is available.

The user authorized source-only training on 2026-10-26. Preregistration and
frozen code are committed before diagnostics or training. No target evaluation
is authorized by the S12-A/B protocol; confirmation is conditional on pilot GO.
