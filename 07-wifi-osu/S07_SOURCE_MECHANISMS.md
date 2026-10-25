# S07 Source-Only Mechanism Diagnostics

Read-only, after feature fits. No extra training or target reads; never used for selection. Two packets per TX per held-source RX, fixed across frontends. Scores above |6| indicate near-saturated sigmoid entries, not a proven failure cause. Feature masking is out-of-distribution sensitivity, not causal feature importance.

| Dataset | Fold | Frontend | Sample acc % | Block0 init RMS | Trained RMS | Trained saturation |
|---|---:|---|---:|---:|---:|---:|
| ManyRx | 0 | legacy | 67.80 | 0.999 | 0.982 | 0.4394 |
| ManyRx | 0 | time | 53.39 | 0.999 | 0.913 | 0.2069 |
| ManyRx | 0 | diff | 31.36 | 0.999 | 0.964 | 0.1953 |
| ManyRx | 0 | split | 77.12 | 0.999 | 0.995 | 0.2411 |
| ManyRx | 0 | stf | 67.80 | 0.999 | 0.971 | 0.0760 |
| ManyRx | 0 | filter | 61.86 | 0.999 | 0.943 | 0.1277 |
| ManyRx | 0 | frequency | 84.75 | 0.999 | 1.001 | 0.2242 |
| ManyRx | 1 | legacy | 56.78 | 0.999 | 0.889 | 0.3581 |
| ManyRx | 1 | time | 48.31 | 0.999 | 0.835 | 0.1798 |
| ManyRx | 1 | diff | 30.51 | 0.999 | 0.849 | 0.1136 |
| ManyRx | 1 | split | 58.47 | 0.999 | 0.964 | 0.4897 |
| ManyRx | 1 | stf | 41.53 | 0.999 | 0.920 | 0.1816 |
| ManyRx | 1 | filter | 50.00 | 0.999 | 0.827 | 0.1022 |
| ManyRx | 1 | frequency | 80.51 | 0.999 | 0.968 | 0.4372 |
| ManyRx | 2 | legacy | 70.34 | 0.999 | 0.988 | 0.4774 |
| ManyRx | 2 | time | 56.78 | 0.999 | 0.955 | 0.1625 |
| ManyRx | 2 | diff | 50.00 | 0.999 | 0.960 | 0.1658 |
| ManyRx | 2 | split | 71.19 | 0.999 | 1.003 | 0.4251 |
| ManyRx | 2 | stf | 64.41 | 0.999 | 0.955 | 0.0965 |
| ManyRx | 2 | filter | 53.39 | 0.999 | 0.951 | 0.1603 |
| ManyRx | 2 | frequency | 87.29 | 0.999 | 0.998 | 0.4946 |
| ManyRx | 3 | legacy | 49.15 | 0.999 | 0.936 | 0.4467 |
| ManyRx | 3 | time | 61.02 | 0.999 | 0.953 | 0.0847 |
| ManyRx | 3 | diff | 41.53 | 0.999 | 0.967 | 0.1789 |
| ManyRx | 3 | split | 81.36 | 0.999 | 1.001 | 0.4151 |
| ManyRx | 3 | stf | 57.63 | 0.999 | 0.946 | 0.1051 |
| ManyRx | 3 | filter | 65.25 | 0.999 | 0.957 | 0.0820 |
| ManyRx | 3 | frequency | 81.36 | 0.999 | 0.982 | 0.5334 |
| ManySig | 0 | legacy | 62.50 | 0.999 | 0.996 | 0.0577 |
| ManySig | 0 | time | 45.83 | 0.999 | 0.990 | 0.1637 |
| ManySig | 0 | diff | 41.67 | 0.999 | 0.999 | 0.4705 |
| ManySig | 0 | split | 62.50 | 0.999 | 0.999 | 0.0004 |
| ManySig | 0 | stf | 33.33 | 0.999 | 0.997 | 0.0915 |
| ManySig | 0 | filter | 58.33 | 0.999 | 0.993 | 0.1157 |
| ManySig | 0 | frequency | 70.83 | 0.999 | 0.988 | 0.3335 |
| ManySig | 1 | legacy | 54.17 | 0.999 | 0.989 | 0.2039 |
| ManySig | 1 | time | 33.33 | 0.999 | 0.997 | 0.1821 |
| ManySig | 1 | diff | 41.67 | 0.999 | 0.992 | 0.2905 |
| ManySig | 1 | split | 62.50 | 0.999 | 0.993 | 0.0156 |
| ManySig | 1 | stf | 41.67 | 0.999 | 0.988 | 0.0311 |
| ManySig | 1 | filter | 50.00 | 0.999 | 0.996 | 0.1609 |
| ManySig | 1 | frequency | 75.00 | 0.999 | 1.002 | 0.3825 |
| ManySig | 2 | legacy | 50.00 | 0.999 | 0.992 | 0.2906 |
| ManySig | 2 | time | 45.83 | 0.999 | 0.977 | 0.1460 |
| ManySig | 2 | diff | 41.67 | 0.999 | 0.996 | 0.3788 |
| ManySig | 2 | split | 70.83 | 0.999 | 0.999 | 0.0169 |
| ManySig | 2 | stf | 50.00 | 0.999 | 0.981 | 0.1064 |
| ManySig | 2 | filter | 41.67 | 0.999 | 0.984 | 0.1337 |
| ManySig | 2 | frequency | 75.00 | 0.999 | 1.001 | 0.4604 |
| ManySig | 3 | legacy | 58.33 | 0.999 | 0.989 | 0.2163 |
| ManySig | 3 | time | 33.33 | 0.999 | 0.987 | 0.0980 |
| ManySig | 3 | diff | 45.83 | 0.999 | 0.991 | 0.1351 |
| ManySig | 3 | split | 62.50 | 0.999 | 1.002 | 0.2854 |
| ManySig | 3 | stf | 54.17 | 0.999 | 0.998 | 0.0263 |
| ManySig | 3 | filter | 29.17 | 0.999 | 0.988 | 0.1335 |
| ManySig | 3 | frequency | 75.00 | 0.999 | 0.981 | 0.3006 |
