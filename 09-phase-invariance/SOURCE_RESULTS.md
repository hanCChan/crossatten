# S09 Source Receiver Screening

All arms trained exactly40 epochs, four folds x two source groups x two seeds, both datasets. Checkpoint ranking and evaluation use the same held RX, so source improvements include selection optimism. No target ranking.

| Dataset | Feature | Rule | Mean held RX % | Macro-F1 % | Worst held RX % |
|---|---|---|---:|---:|---:|
| ManyRx | first | raw_held | 91.10 | 89.47 | 72.95 |
| ManyRx | first | ema_held | 91.37 | 89.77 | 72.60 |
| ManyRx | centered | raw_held | 91.33 | 89.75 | 73.35 |
| ManyRx | centered | ema_held | 91.73 | 90.17 | 74.15 |
| ManyRx | curvature | raw_held | 90.40 | 88.69 | 70.95 |
| ManyRx | curvature | ema_held | 90.92 | 89.22 | 73.15 |
| ManySig | first | raw_held | 93.85 | 93.75 | 80.08 |
| ManySig | first | ema_held | 94.24 | 94.07 | 78.17 |
| ManySig | centered | raw_held | 93.17 | 93.04 | 79.83 |
| ManySig | centered | ema_held | 92.96 | 92.69 | 79.92 |
| ManySig | curvature | raw_held | 93.60 | 93.55 | 83.00 |
| ManySig | curvature | ema_held | 94.16 | 94.06 | 84.00 |
