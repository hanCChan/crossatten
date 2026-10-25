# S08 Source Results

Source RX days1-2 only; two held groups x two seeds. Checkpoints from the same fit are correlated, not independent models. Aggregates below are descriptive only; formal candidate selection is fold-local. No target metrics select methods.

| Dataset | Condition | Feature | Rule | Cells | Mean held RX % | Mean worst RX % | Median epoch |
|---|---|---|---|---:|---:|---:|---:|
| ManySig | official | frequency | ema_held | 16 | 89.62 | 85.19 | 8.0 |
| ManySig | official | frequency | ema_packet | 16 | 88.95 | 83.69 | 9.0 |
| ManySig | official | frequency | raw_held | 16 | 89.52 | 85.59 | 7.5 |
| ManySig | official | frequency | raw_packet | 16 | 87.99 | 81.76 | 11.5 |
| ManySig | official_awgn | frequency | ema_held | 16 | 89.57 | 84.81 | 4.0 |
| ManySig | official_awgn | frequency | ema_packet | 16 | 88.20 | 82.11 | 11.0 |
| ManySig | official_awgn | frequency | raw_held | 16 | 89.31 | 84.51 | 7.5 |
| ManySig | official_awgn | frequency | raw_packet | 16 | 85.39 | 79.44 | 17.0 |
| ManySig | official_awgn_full200 | frequency | ema_held | 16 | 89.57 | 84.81 | 4.0 |
| ManySig | official_awgn_full200 | frequency | ema_packet | 16 | 88.20 | 82.11 | 11.0 |
| ManySig | official_awgn_full200 | frequency | raw_held | 16 | 89.31 | 84.51 | 7.5 |
| ManySig | official_awgn_full200 | frequency | raw_packet | 16 | 86.15 | 79.75 | 19.5 |

## Fold-Local Gates

| Dataset | Fold | Supported | Source-selected configuration | Mean delta pp | Worst delta pp |
|---|---:|---|---|---:|---:|
| ManySig | 0 | True | official:frequency:ema_held | +4.76 | +0.63 |
| ManySig | 1 | False | official:frequency:raw_packet | +0.00 | +0.00 |
| ManySig | 2 | True | official:frequency:ema_held | +3.35 | -0.17 |
| ManySig | 3 | False | official:frequency:raw_packet | +0.00 | +0.00 |

Admission requires supported candidates on at least3/4 folds. Unsupported folds fall back to official/raw packet-CE. Held-CE rules refit all source RX for the source-frozen upper median epoch.

Full200 truly ran all200 epochs, but its selected checkpoint may be early. No result licenses target-driven additional features/losses. All exact cells are in `source_results.csv` and the retained `runs/`.
