# Novelty And SOTA Evidence Status

Evidence cutoff: 2026-10-25. Scope: strict source-only WiSig and the registered
OSU Diff Receivers / Indoor / SameTx comparison. This is a scoped evidence review,
not an exhaustive contemporaneous leaderboard.

## Existing Ideas

Time/frequency separation, complex phase products, learnable frequency filters,
receiver-balanced sampling and physical signal augmentation are existing signal
processing or representation ideas. None should be claimed as original merely
because it appears before RF-HSTU. The split model globally pools independently
encoded frequency tokens; it does not pretend time and frequency indices align.
The lag-20 prefix audit supports periodicity, not exact full-preamble synchronization.

FAN/IB-RFF already implements adaptive frequency processing; its local checked
revision is `5a03a04da86c4bbf5cf90f41a55c268bf2152203`.
WaveMLP implements learnable complex wavelets; its checked revision is
`1ade72d0082428db2087633a44ae533f1022ca22`.
Its upstream code stays local because no license was found. It is not copied into
this repository. S06 fair results remain frozen controls.

## Reference Levels

S06 fair three-seed WiSig CRD, RX-equal:

| Method | ManyRx % | ManySig % |
|---|---:|---:|
| W1 | 83.56 | 72.33 |
| IB-RFF | 88.07 | 91.09 |
| WaveMLP | 88.06 | 91.04 |

Historical official-weight reference: IB-RFF 89.19/93.57, WaveMLP 88.68/93.13.
Those are separate single-seed historical artifacts, not substitutes for fair
multi-seed uncertainty. S07 uses different seeds and cannot claim paired-seed
significance against S06. Receiver-matched, seed-averaged contrasts are available.
Existing public test data have been examined, so S07 is not a fresh blind test.

## OSU

All 24 registered independent 25-class fits completed. HStem-HSTU did not transfer
under the frozen detector/strict-IB protocol; this is a negative result. No matching
published OSU leaderboard protocol has been verified, and there are no local
cross-day captures. Comparing four models on a new protocol cannot establish
global OSU SOTA, even if one achieved the best local score.

## Access Failures And Scope Limits

Official OSU directory and arXiv attempts on 2026-10-25 returned proxy
`403 ERR_ACCESS_DENIED`; the web tool also returned no usable contents.
Thus the latest literature, official transmitter settings and additional OSU
scenarios could not be independently rechecked in this environment. The review
uses local upstream revisions and committed S06 evidence, and cannot certify
that no newer or higher same-protocol result exists.

## Primary Evidence Index

- `lib/xa/explore_models.py`: registered frontend implementations.
- `source_signal_audit.json`: source-only WiSig physical measurements.
- `registration.json`, `decision_*.json`: budget, selection and job lineage.
- `../06-paper-validation/summary/paper/main_metrics.csv`: frozen fair references.
- `../docs/LITERATURE_2025_2026.md`: literature evidence levels and upstream links.
- `S07_OSU_DATA_CARD.md`, `osu_data_manifest.json`, `osu_cache_audit.json`: raw
  data provenance, frozen detector and exact-crop/time-block validation.
- `S07_OSU_RESULTS.md`, `S07_OSU_POSTHOC.md`: full negative results and limitations.

Next decision must use all registered source/final outcomes, including failures.
Only measured superiority in the tested protocol may be asserted. Stronger
novelty or global SOTA claims require accessible primary papers, equivalent
protocols and independent corroboration.
