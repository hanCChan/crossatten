# ManyRx Error Diagnosis

S07 frozen frequency/mild recipe: CR 89.8685%, CRD 87.6506%; gap 2.2179 pp.
S09 hard raw-packet source-held score: 90.9665% (different fit-RX pool/recipe; not a causal gap).

| Date Index | RX-Equal Accuracy (%) |
|---|---:|
| 1 | 89.6855 |
| 2 | 90.0662 |
| 3 | 87.2892 |
| 4 | 88.0393 |

| Worst CRD RX | Seed-Mean Accuracy (%) |
|---|---:|
| 24-5 | 68.0071 |
| 8-14 | 74.9167 |
| 8-7 | 76.5000 |
| 13-7 | 77.5370 |
| 1-19 | 77.5833 |
| 18-19 | 78.3000 |

Descriptive RX correlation of low-amplitude bin ratio with CRD accuracy: -0.3498. This is observational, not identification of a causal failure mechanism.
Low amplitude and circular phase dispersion are saved per RX/TX/day; absent TX cells remain absent.

CR and CRD contain DIFFERENT packets and dates. There is no legitimate 'CR-correct packet becomes CRD-wrong' indicator. Only marginal errors for common TX/RX can be compared.
S07-S09 checkpoint files were not distributed/restored. BN running statistics, feature activations and checkpoint interventions are therefore unavailable for those models.
Historical raw/EMA selection curves are analyzed, but are not a substitute for BN measurements.
Hypotheses: threshold discontinuity and low-amplitude phase uncertainty are plausible; the source gate can falsify their utility. RX/date errors alone do not prove either is the main bottleneck.
No fold-specific recipe was changed in response to these diagnostics.

Evidence: `historical_rx_day_errors.csv`, `historical_tx_errors.csv`, `historical_checkpoint_selection.csv`, `spectral_rx_day_tx.csv`, `diagnostic_evidence.json`.
