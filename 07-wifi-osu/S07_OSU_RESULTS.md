# S07 OSU Results

Date: 2026-10-25. Completed 24/24 registered fits. Status: complete registered comparison.

Independent 25-class retraining, bidirectional RX1/RX2, seeds5101/5102/5103. Source temporal validation chooses the checkpoint; target time blocks are disjoint. Only one capture per TX/RX; directions are correlated and do not estimate a broad RX population.

| Model | Fits | Packet acc % | TX-equal acc % | Macro-F1 % | Fit SD pp | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| patch_hstu | 6 | 1.41 | 2.06 | 0.69 | 1.47 | 581111 |
| hstem_hstu | 6 | 1.42 | 2.00 | 0.78 | 1.55 | 143367 |
| hstem_cnn | 6 | 13.31 | 11.88 | 8.87 | 6.65 | 101129 |
| cis_rank16 | 6 | 21.00 | 16.28 | 13.14 | 7.46 | 660249 |

All individual directions/seeds are in `osu_metrics.csv`. Detection failure attempts and full class counts are preserved. This is a controlled comparison, not a verified OSU SOTA claim. There is no local cross-day evidence; official download requests were denied by the proxy.

The nominal strict-IB detector pipeline did not validate HStem-HSTU transfer. The post-hoc same-RX future-block control and aliased frequency-offset diagnostics are in `S07_OSU_POSTHOC.md`; they cannot authorize target-based method tuning.
