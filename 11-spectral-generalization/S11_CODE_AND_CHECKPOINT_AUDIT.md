# S11 Code And Checkpoint Audit

Evidence cutoff:2026-10-26.
24 actual S10 ASKNet checkpoint hashes verified, saved cutoff logits read directly.
Each saved k and last-trace k are separately recorded in `ask_checkpoint_bands.csv`.
k32:63bins, +/-3.02734375MHz; k24:47bins, +/-2.24609375MHz at25MHz/256.
No claim that ASKNet gain is due solely to bandwidth: SKO and backbone also differ.
Original S07-S09 weights were not found locally/in the previously audited two releases; only S00-S06 were distributed. Two registered source-only S09 hard refits will be stored under S11 and labeled refits unless all original tensor/checkpoint evidence matches.
S10 soft weights exist but are not silently treated as hard-phase weights for BN attribution.
Historical S00-S10 files remain read-only; S11 imports independently loaded frozen training modules.
Budget ambiguity: two complete F1/F2 grids require64 not32. Implement both and conduct zero-fit source-only route admission first, train at most one32-fit grid.
Day-transfer16 limit permits only last group/seed8101, not replicated two-group/two-seed confirmation.
Evidence: `registration.json`, `checkpoint_audit.json`, `ask_checkpoint_bands.csv`.
