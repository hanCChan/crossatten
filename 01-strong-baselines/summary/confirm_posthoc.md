# Stage 01 post-hoc diagnostics (NOT pre-registered; descriptive)

## 1. Source -> unseen-RX generalisation gap (selected checkpoint, mean over 6 fits, %)

| arm | source_val | dev_val | gap dv | confirm | gap cf | final train acc (AWGN-corrupted if AWGN) |
|---|---|---|---|---|---|---|
| xa_trueib_cos_awgn | 94.4 | 94.7 | -0.3 | 92.1 | +2.3 | 76.7 |
| cis_ib_flat | 97.2 | 94.6 | +2.6 | 95.7 | +1.5 | 80.1 |
| cis_flat | 97.3 | 93.1 | +4.2 | 95.2 | +2.1 | 78.0 |
| cis_flat_noawgn | 95.1 | 84.3 | +10.8 | 88.5 | +6.6 | 100.0 |
| xa_trueib_cos | 82.6 | 81.7 | +0.9 | 80.0 | +2.6 | 97.8 |
| cis_gap_noawgn | 83.4 | 69.5 | +13.8 | 71.9 | +11.5 | 94.3 |
| cis_gap | 55.5 | 58.7 | -3.2 | 54.8 | +0.6 | 52.9 |
| xa_lp200_cos_awgn | 82.9 | 52.4 | +30.5 | 57.2 | +25.7 | 79.8 |
| xa_oobnear_cos_awgn | 83.2 | 52.3 | +30.9 | 57.1 | +26.1 | 79.8 |
| tf_cos_awgn | 81.7 | 50.7 | +31.0 | 56.1 | +25.6 | 79.6 |
| xa_cos_awgn | 83.2 | 50.5 | +32.7 | 56.8 | +26.4 | 79.5 |
| xa_nooob_cos_awgn | 57.8 | 49.3 | +8.4 | 49.3 | +8.4 | 48.2 |
| xa_lp200_cos | 82.2 | 45.5 | +36.7 | 54.0 | +28.2 | 100.0 |
| xa_const | 80.0 | 44.5 | +35.5 | 51.8 | +28.1 | 100.0 |
| xa_cos | 81.0 | 44.1 | +36.9 | 53.7 | +27.3 | 100.0 |
| xa_oobnear_cos | 82.1 | 43.4 | +38.7 | 52.6 | +29.4 | 100.0 |
| xa_p1fixed | 59.8 | 40.4 | +19.4 | 41.5 | +18.3 | 61.8 |
| tf_cos | 76.5 | 38.6 | +37.9 | 45.9 | +30.7 | 100.0 |
| xa_nooob_cos | 53.8 | 36.8 | +17.0 | 40.0 | +13.9 | 87.1 |

## 2. Extra paired comparisons (delta = B - A, pp; 6 pairs = 2 folds x 3 seeds)

| B | A | what | role | mean | min | max | +pairs | worst-RX(B) - worst-RX(A) |
|---|---|---|---|---|---|---|---|---|
| xa_trueib_cos_awgn | xa_trueib_cos | AWGN on TrueIB | dev_val | +12.99 | -0.71 | +28.34 | 5/6 | +17.78 |
| xa_trueib_cos_awgn | xa_trueib_cos | AWGN on TrueIB | confirm | +12.06 | -0.20 | +24.88 | 5/6 | +14.17 |
| xa_trueib_cos_awgn | cis_flat | SYSTEM: TrueIB+AWGN (0.58M) vs Shen-CIS official (13.4M) | dev_val | +1.61 | +0.46 | +3.07 | 6/6 | +9.63 |
| xa_trueib_cos_awgn | cis_flat | SYSTEM: TrueIB+AWGN (0.58M) vs Shen-CIS official (13.4M) | confirm | -3.09 | -4.44 | -1.66 | 0/6 | -10.52 |
| xa_trueib_cos_awgn | cis_ib_flat | SYSTEM: TrueIB+AWGN vs strict-IB CIS (13.4M) | dev_val | +0.11 | -3.03 | +2.89 | 3/6 | +19.62 |
| xa_trueib_cos_awgn | cis_ib_flat | SYSTEM: TrueIB+AWGN vs strict-IB CIS (13.4M) | confirm | -3.61 | -4.89 | -2.65 | 0/6 | -7.82 |

## 3. Per-fit accuracy of the strict-IB CrossAttn arms (dev_val / confirm, %)

| arm | fold | seed | best epoch | source_val | dev_val | confirm |
|---|---|---|---|---|---|---|
| xa_trueib_cos | 0 | 201 | 43 | 71.8 | 71.4 | 69.8 |
| xa_trueib_cos | 0 | 202 | 79 | 95.6 | 94.2 | 91.3 |
| xa_trueib_cos | 0 | 203 | 57 | 94.8 | 92.9 | 91.7 |
| xa_trueib_cos | 1 | 201 | 61 | 94.4 | 95.5 | 92.2 |
| xa_trueib_cos | 1 | 202 | 20 | 69.0 | 68.4 | 66.9 |
| xa_trueib_cos | 1 | 203 | 55 | 70.1 | 68.1 | 68.4 |
| xa_trueib_cos_awgn | 0 | 201 | 88 | 95.1 | 93.3 | 91.9 |
| xa_trueib_cos_awgn | 0 | 202 | 94 | 95.2 | 93.5 | 92.3 |
| xa_trueib_cos_awgn | 0 | 203 | 87 | 95.8 | 94.8 | 92.5 |
| xa_trueib_cos_awgn | 1 | 201 | 89 | 93.9 | 96.1 | 92.0 |
| xa_trueib_cos_awgn | 1 | 202 | 97 | 91.8 | 94.2 | 90.7 |
| xa_trueib_cos_awgn | 1 | 203 | 96 | 94.7 | 96.5 | 93.3 |

## 4. Confirmation receivers: seed-mean accuracy per RX (%; both folds evaluate the same 4 RX)

| arm | fold | b200_1 | n210_1 | rtl_4 | rtl_8 | mean |
|---|---|---|---|---|---|---|
| xa_p1fixed | 0 | 51.6 | 49.0 | 28.8 | 60.0 | 47.3 |
| xa_p1fixed | 1 | 44.8 | 46.9 | 14.8 | 35.9 | 35.6 |
| xa_cos | 0 | 48.8 | 63.2 | 30.6 | 87.9 | 57.6 |
| xa_cos | 1 | 61.0 | 67.6 | 25.8 | 44.5 | 49.7 |
| xa_cos_awgn | 0 | 43.4 | 54.2 | 51.9 | 91.0 | 60.1 |
| xa_cos_awgn | 1 | 59.5 | 79.2 | 37.4 | 37.6 | 53.4 |
| xa_lp200_cos_awgn | 0 | 49.1 | 56.0 | 53.4 | 86.4 | 61.2 |
| xa_lp200_cos_awgn | 1 | 63.0 | 79.7 | 37.0 | 32.8 | 53.1 |
| xa_nooob_cos_awgn | 0 | 44.1 | 50.3 | 48.0 | 52.5 | 48.7 |
| xa_nooob_cos_awgn | 1 | 55.7 | 61.8 | 39.4 | 43.0 | 50.0 |
| xa_trueib_cos | 0 | 72.2 | 91.4 | 84.6 | 88.9 | 84.3 |
| xa_trueib_cos | 1 | 65.8 | 82.0 | 74.5 | 81.0 | 75.8 |
| xa_trueib_cos_awgn | 0 | 80.0 | 97.5 | 94.7 | 96.7 | 92.2 |
| xa_trueib_cos_awgn | 1 | 80.2 | 96.5 | 95.2 | 96.0 | 92.0 |
| tf_cos_awgn | 0 | 47.7 | 57.8 | 56.3 | 90.0 | 62.9 |
| tf_cos_awgn | 1 | 62.2 | 71.8 | 36.5 | 26.7 | 49.3 |
| cis_flat_noawgn | 0 | 80.5 | 92.3 | 78.1 | 98.6 | 87.4 |
| cis_flat_noawgn | 1 | 87.9 | 94.5 | 81.6 | 94.6 | 89.6 |
| cis_flat | 0 | 90.5 | 96.5 | 92.1 | 99.4 | 94.6 |
| cis_flat | 1 | 95.2 | 96.8 | 93.5 | 97.8 | 95.8 |
| cis_ib_flat | 0 | 87.8 | 99.1 | 97.9 | 98.2 | 95.7 |
| cis_ib_flat | 1 | 90.0 | 98.2 | 97.0 | 97.7 | 95.7 |
| cis_gap | 0 | 31.9 | 67.8 | 62.5 | 66.0 | 57.1 |
| cis_gap | 1 | 27.8 | 64.3 | 57.6 | 61.0 | 52.6 |
