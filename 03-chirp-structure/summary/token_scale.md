# Post-Hoc Source-Only Token Scale

All eight arms, two folds, exploration seeds 101/102. Each fold uses 500 balanced source-validation packets (ten per TX per source RX), unchanged across arms. CPU evaluation compares the original stable-seed initialization with the source-CE-selected exploration checkpoint. No target packets, checkpoint selection, additional fits, or decision changes.

First-block saturation diagnostic: |q.k/sqrt(d) + bias| > 6.0. RMS is computed across all packet/token/channel elements; score medians and saturation fractions include every packet and token pair. Finite scale does not prove the cause of historical depth4 failures.

| arm | fold | seed | preln | init RMS | trained RMS | trained packet RMS range | init saturated | trained saturated | trained median abs(score) |
|---|---|---|---|---|---|---|---|---|---|
| s03_base | 0 | 101 | False | 0.085 | 114.676 | 100.222-147.590 | 0.0000 | 0.9995 | 31139.027 |
| s03_base | 0 | 102 | False | 0.077 | 141.994 | 122.173-268.263 | 0.0000 | 1.0000 | 175851.719 |
| s03_preln | 0 | 101 | True | 0.999 | 0.951 | 0.943-0.971 | 0.0000 | 0.7298 | 10.171 |
| s03_preln | 0 | 102 | True | 0.999 | 0.982 | 0.969-0.997 | 0.0000 | 0.8425 | 10.768 |
| s03_preln_d4 | 0 | 101 | True | 0.999 | 1.013 | 1.006-1.018 | 0.0000 | 0.9451 | 21.199 |
| s03_preln_d4 | 0 | 102 | True | 0.999 | 1.063 | 1.049-1.069 | 0.0000 | 0.9994 | 129.978 |
| s03_resid | 0 | 101 | True | 0.999 | 1.016 | 1.004-1.026 | 0.0000 | 0.9734 | 16.456 |
| s03_resid | 0 | 102 | True | 0.999 | 0.995 | 0.989-1.001 | 0.0000 | 0.7570 | 7.798 |
| s03_nofft | 0 | 101 | True | 0.999 | 0.996 | 0.986-1.002 | 0.0000 | 0.9116 | 10.285 |
| s03_nofft | 0 | 102 | True | 0.999 | 1.003 | 0.995-1.010 | 0.0000 | 0.9872 | 16.610 |
| s03_symstack | 0 | 101 | True | 0.999 | 1.007 | 0.976-1.011 | 0.0000 | 0.9474 | 17.290 |
| s03_symstack | 0 | 102 | True | 0.999 | 0.974 | 0.961-0.979 | 0.0000 | 0.8861 | 15.367 |
| s03_ifreq | 0 | 101 | True | 0.999 | 0.986 | 0.979-0.995 | 0.0000 | 0.9923 | 22.021 |
| s03_ifreq | 0 | 102 | True | 0.999 | 0.980 | 0.969-0.992 | 0.0000 | 0.8283 | 10.795 |
| s03_hstem | 0 | 101 | True | 0.999 | 0.989 | 0.979-0.995 | 0.0000 | 0.8240 | 10.489 |
| s03_hstem | 0 | 102 | True | 0.999 | 1.016 | 1.007-1.024 | 0.0000 | 0.9677 | 20.901 |
| s03_base | 1 | 101 | False | 0.085 | 106.504 | 94.047-187.245 | 0.0000 | 0.9994 | 9327.567 |
| s03_base | 1 | 102 | False | 0.077 | 176.560 | 133.436-838.603 | 0.0000 | 1.0000 | 1367605.250 |
| s03_preln | 1 | 101 | True | 0.999 | 0.947 | 0.937-0.980 | 0.0000 | 0.8993 | 11.291 |
| s03_preln | 1 | 102 | True | 0.999 | 1.013 | 1.006-1.019 | 0.0000 | 0.9535 | 13.622 |
| s03_preln_d4 | 1 | 101 | True | 0.999 | 1.062 | 1.043-1.068 | 0.0000 | 0.9877 | 56.871 |
| s03_preln_d4 | 1 | 102 | True | 0.999 | 1.021 | 1.012-1.029 | 0.0000 | 0.9987 | 45.231 |
| s03_resid | 1 | 101 | True | 0.999 | 1.023 | 1.018-1.028 | 0.0000 | 0.9686 | 20.406 |
| s03_resid | 1 | 102 | True | 0.999 | 1.020 | 1.015-1.023 | 0.0000 | 0.9806 | 16.626 |
| s03_nofft | 1 | 101 | True | 0.999 | 1.035 | 1.028-1.042 | 0.0000 | 0.9936 | 17.736 |
| s03_nofft | 1 | 102 | True | 0.999 | 0.951 | 0.935-0.970 | 0.0000 | 0.7909 | 10.850 |
| s03_symstack | 1 | 101 | True | 0.999 | 1.031 | 1.012-1.042 | 0.0000 | 0.9956 | 35.757 |
| s03_symstack | 1 | 102 | True | 0.999 | 0.984 | 0.966-0.991 | 0.0000 | 0.9553 | 13.935 |
| s03_ifreq | 1 | 101 | True | 0.999 | 1.009 | 1.003-1.013 | 0.0000 | 0.9978 | 40.904 |
| s03_ifreq | 1 | 102 | True | 0.999 | 0.977 | 0.967-0.984 | 0.0000 | 0.8792 | 9.807 |
| s03_hstem | 1 | 101 | True | 0.999 | 0.992 | 0.987-1.002 | 0.0000 | 0.9240 | 15.984 |
| s03_hstem | 1 | 102 | True | 0.999 | 1.006 | 0.997-1.014 | 0.0000 | 0.8415 | 12.391 |
