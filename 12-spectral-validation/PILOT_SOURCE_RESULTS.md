# S12 Pilot Source Results

Decision: **STOP_SOURCE_GATE**. Selected: None.

| Candidate | Dataset | Mean Gain (pp) | Positive Folds | Worst RX Delta (pp) |
|---|---|---:|---:|---:|
| F1 | ManyRx | -1.7400 | 0/4 | -4.7000 |
| F1 | ManySig | 2.1667 | 3/4 | -3.1667 |
| F2 | ManyRx | 0.0750 | 2/4 | -3.5333 |
| F2 | ManySig | 3.4062 | 3/4 | -0.5833 |

Fixed raw_packet checkpoint rule; RX-equal run means, then fold means.
Worst RX gate uses receiver means across matched runs; single-pair minima are also disclosed.
No target predictions, old weights, alternative checkpoint rescue, or new search.
Pilot is exploratory; confirmation inference excludes pilot data.
Full paired accuracy/F1/CE and checks: paired JSON and decision JSON.
