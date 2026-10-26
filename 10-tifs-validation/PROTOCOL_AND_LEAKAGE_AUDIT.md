# Protocol And Leakage Audit

Evidence cutoff: 2026-10-26.
Candidate gate: **STOP**. Only source-held evidence selected candidate admission.
New formal fits: 56 = source32 + ASKNet24 + final0 + ablation0. Four bounded smoke fits are separately disclosed. No silent retry or replacement.
S00-S09 and shared libraries remained read-only. Pinned code and controls are hash-sealed.

| Item | Frozen Rule |
|---|---|
| IQ | Non-equalized, 256 samples, first100/TX/RX/day, per-packet power normalization |
| RX roles | Official fixed order, quarters; target never in fit/validation |
| Dates | March1/8/15/23,2021; source days1-2, CR target1-2, CRD target3-4 |
| Source screening | first/last held6 ManyRx or2 ManySig, seeds8101/8102, 40epochs |
| Primary rule | Raw minimum source packet-validation CE, precommitted |
| Candidate | Only continuous reliability tau=.05; no new architecture or loss |
| Source gate | >=3/4 ManyRx folds positive, >=1pp, minRX>=-2pp, both groups/seeds positive; ManySig no systematic loss |
| Formal candidate | Same packet-CE rule for both datasets; final9101/9102/9103 only if GO |
| ASKNet | Official default structure/Adam/200max/plateau/ES10; independent seeds2023-2025 |
| Target release | Training complete; committed checkpoint hash/metrics before separate scoring |

The same physical RX appears as source in other folds under published cross-validation. This is not a new external blind deployment; historical target outcomes were already seen.
Raw pickle reconstruction creates all cached roles inertly and verifies counts/normalization; it does not use target performance for fitting. Access logs classify subsequent fit/held/target reads.
Historical target diagnostics are descriptive only and were run after algorithm/gate preregistration.
Evidence: `registration.json`, `preparation.json`, `access.jsonl`, immutable jobs and run seals.
Uncertainty: RX-cluster bootstrap conditions on fixed fold models; no population guarantee.
