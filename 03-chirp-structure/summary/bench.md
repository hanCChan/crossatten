# End-to-end cost (NVIDIA A100-SXM4-40GB, torch 2.5.0+cu124)

random weights; raw IQ on device to logits; FFT/scalar ops included in timing, not fully in FLOPs

| arm | params | mflops_per_packet | gpu_b1_ms_p50 | gpu_b1_ms_p95 | gpu_b1_ms_p99 | gpu_b512_pkts_per_s | cpu4t_b1_ms_p50 | cpu4t_b1_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| s03_base | 580008 | 139.2 | 2.321 | 3.469 | 4.253 | 41375 | 6.44 | 8.17 |
| s03_preln | 580136 | 139.2 | 2.276 | 3.533 | 3.742 | 41234 | 7.12 | 9.05 |
| s03_preln_d4 | 622374 | 142.3 | 3.075 | 4.524 | 5.033 | 39055 | 8.34 | 10.61 |
| s03_resid | 580584 | 146.5 | 2.328 | 3.744 | 4.521 | 40470 | 6.53 | 8.52 |
| s03_nofft | 579912 | 135.5 | 1.657 | 1.793 | 1.949 | 46354 | 4.64 | 5.43 |
| s03_symstack | 579912 | 135.5 | 1.726 | 1.794 | 1.816 | 46011 | 3.6 | 4.17 |
| s03_ifreq | 580360 | 142.9 | 2.45 | 3.457 | 3.887 | 40532 | 5.76 | 8.2 |
| s03_hstem | 142392 | 190.6 | 2.53 | 2.99 | 4.013 | 36101 | 6.08 | 7.57 |
