# TIFS Novelty Assessment

Final candidate verdict: **STOP**.
Do not submit S10 as a proven new SOTA algorithm unless the independent source gate, final paired results and needed mechanism ablations all support it.

| Claimed Contribution | Ownership/Status | Required Boundary |
|---|---|---|
| Physical phase encoding | S08/S09 existing adjacent complex phase; S10 fixed soft reliability is the only new candidate | Continuity and variance motivation are established; incremental classification utility is experimentally conditional |
| Hierarchical RF tokens + HSTU | Existing framework, not newly invented in S10 | Shen ablations do not automatically prove same-config WiSig necessity |
| Cross-RX/date/physics evaluation | S10 evidence contribution: audit, ASKNet, pairing, clustered intervals, controlled diagnostics | Benchmark protocol and analysis are not a new neural architecture |

ASKNet differs by trainable spectral cutoff plus unitary low-frequency phase evolution. Our reliability is fixed local attenuation of adjacent complex phase; no learned Koopman compensation.
Ma et al., *Towards Channel-Robust and Receiver-Independent Radio Frequency Fingerprint Identification* (arXiv2512.12070v1,2025-12-12), uses spectrogram, contrastive pretraining and Siamese classification. Our current simple encoder does not subsume that method or establish priority over receiver-independent RFFI.
The unnamed Liverpool2026 high-order spectral-difference paper remains unverified after targeted search and previous audit. No invented title/DOI or first-invention claim is made.

The manuscript must distinguish a shared hierarchical-token framework from identical signal frontends: Shen LoRa and WiSig need different frontend adaptations.
No unknown-TX rejection, spoofing or false-accept-rate experiments were run; hardware identity classification is not complete open-world authentication or attack resistance.

Scientific goals of ManyRx>=90% and ManySig>=93% are not guarantees. RX confidence intervals and a finite comparator set prevent a universal SOTA claim.

**STOP the soft reliability candidate.** Preserve S08/S09 results and use S10's negative paired evidence to constrain the paper's claims. No target-driven extension was started.
Only the original framework and validation/evidence contributions survive; the new soft reliability must be described as a rejected candidate, not a core confirmed innovation.

Primary source index:
- [ASKNet pinned README](https://github.com/BeechburgPieStar/ASKNet-RFF/blob/f00846d68425315b68ae69b8c6b712e08559c2fe/README.md)
- [Ma et al. primary preprint](https://arxiv.org/abs/2512.12070v1)
- Historical fair IB-RFF/WaveMLP: `../06-paper-validation/summary/paper/receiver_metrics.csv`.
- Frozen original literature scope: `../09-phase-invariance/LITERATURE_CHECK.md`.
