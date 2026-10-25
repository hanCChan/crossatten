# Token scale entering the first HSTU mixer (POST-HOC; fold 0; 512 source_val packets of source RX)

saturated = |q.k/sqrt(d) + bias| > 6.0 (sigmoid slope < 0.0025)

| arm | seed | LayerNorm before mixer 1 | init token RMS | init sat. frac | trained token RMS | trained sat. frac | trained median abs(score) | dev_val acc |
|---|---|---|---|---|---|---|---|---|
| s02_base | 101 | no | 0.08 | 0.000 | 85.03 | 1.000 | 17435.95 | 92.3 |
| s02_base | 102 | no | 0.09 | 0.000 | 134.28 | 1.000 | 41778.48 | 83.2 |
| s02_tfsplit | 101 | no | 0.08 | 0.000 | 33.97 | 0.998 | 1997.32 | 93.9 |
| s02_tfsplit | 102 | no | 0.10 | 0.000 | 56.96 | 1.000 | 7878.08 | 92.8 |
| s02_rawoob_full | 101 | yes (fusion norm_out) | 1.00 | 0.000 | 1.00 | 0.733 | 10.38 | 92.6 |
| s02_rawoob_full | 102 | yes (fusion norm_out) | 1.00 | 0.000 | 1.01 | 0.899 | 16.81 | 85.4 |
| s02_cisoob | 101 | yes (fusion norm_out) | 1.00 | 0.000 | 1.00 | 0.622 | 8.06 | 93.3 |
| s02_cisoob | 102 | yes (fusion norm_out) | 1.00 | 0.000 | 1.00 | 0.837 | 10.80 | 93.5 |
