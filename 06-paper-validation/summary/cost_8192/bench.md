# End-to-end cost (NVIDIA A100-SXM4-40GB, torch 2.5.0+cu124)

random weights; raw IQ on device to logits; FFT/scalar ops included in timing, not fully in FLOPs

| arm | params | mflops_per_packet | gpu_b1_ms_p50 | gpu_b1_ms_p95 | gpu_b1_ms_p99 | gpu_b512_pkts_per_s | cpu4t_b1_ms_p50 | cpu4t_b1_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| FAN_full_official | 310481 | 18.5 | 1.711 | 2.842 | 3.374 | 204657 | 2.66 | 3.45 |
| WaveMLP_absolute_full_official | 85258 | 49.9 | 5.897 | 8.055 | 8.629 | 45503 | 14.17 | 16.24 |
| WaveMLP_scaled_full_official | 594442 | 16.6 | 10.789 | 14.273 | 15.076 | 34282 | 16.93 | 20.89 |
| FAN_ib_official | 310481 | 18.5 | 1.868 | 1.955 | 2.574 | 176435 | 2.57 | 2.65 |
| WaveMLP_absolute_ib_official | 85258 | 49.9 | 5.777 | 6.001 | 6.168 | 44375 | 12.74 | 13.75 |
| WaveMLP_scaled_ib_official | 594442 | 16.6 | 10.276 | 10.667 | 11.446 | 35276 | 14.05 | 14.84 |
| S_base | 580136 | 139.2 | 2.164 | 2.237 | 2.32 | 42602 | 4.34 | 4.64 |
| S_hstem | 142392 | 190.6 | 2.533 | 3.806 | 4.393 | 35867 | 5.02 | 6.05 |
| S_cnnonly | 100154 | 187.4 | 1.838 | 1.927 | 2.127 | 37895 | 4.88 | 5.84 |
| S_cnnparam | 138898 | 18.5 | 1.889 | 3.19 | 4.019 | 97639 | 2.6 | 3.8 |
| S_cnnflops | 655358 | 188.1 | 1.55 | 2.058 | 2.309 | 31735 | 6.73 | 8.4 |
