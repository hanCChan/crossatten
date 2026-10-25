# Stage 07 Results

Generated 2026-10-26T01:25:41+08:00. Complete registered exploration; STOP, no extra target-directed search.

Baseline `ebe3225`, branch `exp/s07-wifi-osu`. S00-S06 are unchanged. No automatic merge, GitHub push, model-weight publication or target-driven extension.

## Registered Execution

| Phase | Registered | Completed | Failed | Archived interrupted attempts |
|---|---:|---:|---:|---:|
| features | 56 | 56 | 0 | 0 |
| replicate | 23 | 23 | 0 | 0 |
| training | 72 | 72 | 0 | 0 |
| training_replicate | 23 | 23 | 0 | 0 |
| combine | 9 | 9 | 0 | 0 |
| final | 72 | 72 | 0 | 0 |
| smoke | 14 | 14 | 0 | 0 |
| osu | 24 | 24 | 0 | 0 |

Interrupted attempts and detector pilot failure are retained; they are not additional successful models. The WiSig formal cap is 280 fits.

## WiSig

Four official RX folds, three final seeds; CRD primary. Selections use each fold's source RX days1-2 only. S06 public targets are not new blind tests. All three pipelines are reported, with fold-specific source decisions disclosed; their configurations may coincide. Reused S06 baseline seeds are unpaired with S07.

| Dataset | Pipeline | CRD RX acc % | Macro-F1 % | Seed SD pp | Worst RX % |
|---|---|---:|---:|---:|---:|
| ManyRx | source_frontend | 77.24 | 73.51 | 0.37 | 50.95 |
| ManyRx | source_training | 87.65 | 85.43 | 0.61 | 68.01 |
| ManyRx | source_combination | 87.65 | 85.43 | 0.61 | 68.01 |
| ManySig | source_frontend | 65.98 | 63.20 | 1.56 | 43.75 |
| ManySig | source_training | 86.62 | 85.96 | 0.94 | 54.67 |
| ManySig | source_combination | 86.62 | 85.96 | 0.94 | 54.67 |

Full CR/CRD, packet/RX weightings, baseline differences and conditional RX bootstrap intervals: `S07_WIFI_FINAL_RESULTS.md`. No target-ranked pipeline is designated the method.

Single-seed first-pass source evidence, descriptive only:

| Dataset | Legacy held RX % | Frequency-only % | Time/frequency split % |
|---|---:|---:|---:|
| ManyRx | 60.25 | 81.83 | 71.75 |
| ManySig | 67.57 | 80.07 | 71.50 |

Removing all frequency information did not solve generalization. Pre-LN keeps token RMS near unit scale, but trained sigmoid scores can still saturate. Read-only masking is out-of-distribution sensitivity, not causal feature importance.

## OSU

Registered comparison complete: 25 TX, RX1/RX2 both directions, three seeds. 50 local raw records; guarded packet extraction instead of arbitrary windows. Nominal 125 kHz/SF7 is source-pilot supported, not certified by metadata. Only one continuous capture per TX/RX; buffered temporal train/validation/test blocks. Both RX pilot records were checked during detector development, so raw target preparation was not blind.

| Model | Cross-RX packet acc % | TX-equal acc % | Parameters |
|---|---:|---:|---:|
| patch_hstu | 1.41 | 2.06 | 581111 |
| hstem_hstu | 1.42 | 2.00 | 143367 |
| hstem_cnn | 13.31 | 11.88 | 101129 |
| cis_rank16 | 21.00 | 16.28 | 660249 |

HStem-HSTU did not validate cross-RX transfer under this frozen pipeline. Post-hoc same-RX future-block accuracy is 91.24%, versus 1.42% across RX. This is not evidence that CFO compensation would restore transfer. No OSU cross-day data are local; official requests were denied by the configured proxy.

## Decision And Claims

The registered comparison does not establish consistent conditional-RX superiority over both strong fair baselines on both WiSig datasets. OSU independent cross-RX validation failed. A general LoRa/WiFi SOTA claim is not supported. Keep the Shen findings, disclose WiSig/OSU applicability limits, and stop this exploration.

## Evidence

- `registration.json`, `decision_*.json`, `jobs_*.json`: budgets and source lineage.
- `runs/`: complete configs, learning curves, float32 predictions and failure attempts; weights ignored.
- `S07_CODE_AUDIT.md`, `S07_SOURCE_MECHANISMS.md`: code and read-only source diagnostics.
- `S07_WIFI_FEATURE_RESULTS.md`, `S07_WIFI_TRAINING_RESULTS.md`, `S07_WIFI_FINAL_RESULTS.md`: sequential outcomes.
- `S07_OSU_DATA_CARD.md`, `S07_OSU_RESULTS.md`, `S07_OSU_POSTHOC.md`: provenance and negative results.
- `S07_COSTS.md`, `S07_EXECUTION.md`: measured inference costs and execution-time accounting.
- `CONCLUSION_ZH.md`: complete audited Chinese research conclusion.
- `acceptance.json`, `wifi_final_report_evidence.json`: acceptance and frozen evidence hashes.
- `S07_SOURCE_DECISIONS.md`: source-only frontend, recipe and combination decisions.
- `S07_NOVELTY_AND_SOTA_REVIEW.md`: primary-source limits and existing ideas.
