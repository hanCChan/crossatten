# S09 Literature And Ranking Boundary

Evidence cutoff: 2026-10-26, Asia/Shanghai. This is a targeted primary-source check, not an exhaustive SOTA survey. No newly discovered competitor was added to the registered experiment matrix.

## Confirmed Official Competitor

**ASKNet: Adaptive Spectral Koopman Network for Cross-Domain Radio Frequency Fingerprinting** is an additional official competitor. The publisher metadata returned by Crossref identifies DOI `10.1109/LCOMM.2026.3718894`, authors Zhengyang Liu, Zheyu Wang and Yu Wang, publication year 2026. The official repository README states IEEE Communications Letters acceptance.

- Official repository: <https://github.com/BeechburgPieStar/ASKNet-RFF>.
- Inspected commit: `f00846d68425315b68ae69b8c6b712e08559c2fe`, committer timestamp `2026-08-12T03:54:21Z`.
- Immutable README: <https://github.com/BeechburgPieStar/ASKNet-RFF/blob/f00846d68425315b68ae69b8c6b712e08559c2fe/README.md>.
- Publisher DOI: <https://doi.org/10.1109/LCOMM.2026.3718894>.
- Evidence: official README and publisher metadata; **not reproduced in S09**. This check did not inspect its training implementation or evaluate its released weights.

The official README describes learnable spectral gating and a unitary spectral Koopman operator addressing low-frequency uniform gain and linear phase distortions. It lists four-fold WiSig ManySig/ManyRx, source days1-2, 256-sample input and first100 packets/TX/RX/day. Those descriptions make it relevant to S09's phase-slope hypothesis. They do not establish protocol-equivalent numeric superiority without checking code, partitions, model selection and repeated-seed results.

The official GitHub license endpoint returned404. No third-party implementation was copied into this repository or executed. ASKNet is therefore a follow-up comparison requirement before any global-SOTA claim, not an additional result in S09.

## Unverified Higher-Order Difference Reference

The user mentioned a2026 Liverpool higher-order spectral-difference RFFI work without an exact title, authors or DOI. Targeted searches were attempted:

- `higher order spectral difference radio frequency fingerprint 2026`;
- `radio frequency fingerprinting higher order spectral difference`;
- `channel independent fingerprint higher order spectral difference Zhang Marshall 2026`.

The Liverpool institutional repository search returned403. Crossref queries did not identify that exact paper. A2022 SPIE record, *Multiple order difference based radio frequency (RF) signal processing*, DOI `10.1117/12.2619066`, appeared but was not established to be the claimed fingerprinting method.

**The exact2026 work remains unverified.** It must not be cited with invented title, numbers or novelty comparison. This negative search result does not prove the work is absent. S09 makes no first-invention claim for spectral second differences.

## Ranking Scope

S09 compares with frozen, receiver-matched IB-RFF and WaveMLP fair retraining results, keeping released single-seed references separate. These two comparisons, even if positive, do not establish global SOTA because the candidate set is incomplete and ASKNet is not fairly retrained here.

S09 investigates an engineered inductive bias from existing complex FFT information; it does not add information to the raw packet. Ideal circular-shift invariance is not arbitrary-channel or real-window invariance. A paper contribution must rely on matched content/capacity ablation, receiver-level consistency and measured costs, with this novelty boundary retained.
