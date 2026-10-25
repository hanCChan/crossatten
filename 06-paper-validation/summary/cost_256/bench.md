# End-to-end cost (NVIDIA A100-SXM4-40GB, torch 2.5.0+cu124)

random weights; raw IQ on device to logits; FFT/scalar ops included in timing, not fully in FLOPs

| arm | params | mflops_per_packet | gpu_b1_ms_p50 | gpu_b1_ms_p95 | gpu_b1_ms_p99 | gpu_b512_pkts_per_s | cpu4t_b1_ms_p50 | cpu4t_b1_ms_p95 |
|---|---|---|---|---|---|---|---|---|
| W0_ManyRx_official | 72232 | 7.7 | 2.015 | 2.836 | 3.27 | 242215 | 1.59 | 2.22 |
| W1_ManyRx_official | 91128 | 10.8 | 2.295 | 3.059 | 3.433 | 200982 | 1.87 | 2.09 |
| W2_ManyRx_official | 91128 | 10.8 | 2.494 | 3.446 | 4.08 | 185488 | 2.02 | 2.67 |
| C0_ManyRx_official | 48890 | 7.7 | 1.567 | 1.9 | 2.062 | 311151 | 1.26 | 1.42 |
| CP_ManyRx_official | 91716 | 23.6 | 1.329 | 1.757 | 1.876 | 247952 | 1.31 | 1.45 |
| CF_ManyRx_official | 52718 | 10.8 | 1.26 | 1.456 | 1.636 | 303505 | 1.02 | 1.3 |
| FAN_ManyRx_official | 151761 | 10.6 | 1.664 | 2.595 | 3.279 | 331868 | 1.83 | 2.98 |
| WaveMLP_default_ManyRx_official | 85258 | 1.4 | 5.521 | 7.296 | 7.746 | 75297 | 3.32 | 3.99 |
| WaveMLP_alternate_ManyRx_official | 84490 | 1.3 | 4.34 | 4.413 | 4.539 | 116966 | 2.7 | 2.78 |
| W0_ManySig_official | 71972 | 7.7 | 1.949 | 1.984 | 2.114 | 245934 | 1.46 | 1.52 |
| W1_ManySig_official | 90868 | 10.8 | 2.209 | 2.262 | 2.328 | 210251 | 1.8 | 1.91 |
| W2_ManySig_official | 90868 | 10.8 | 2.42 | 2.467 | 2.536 | 190376 | 1.93 | 2.01 |
| C0_ManySig_official | 48630 | 7.7 | 1.549 | 1.576 | 1.595 | 316597 | 1.22 | 1.28 |
| CP_ManySig_official | 91456 | 23.6 | 1.313 | 1.34 | 1.372 | 250440 | 1.33 | 1.45 |
| CF_ManySig_official | 52458 | 10.8 | 1.261 | 1.358 | 1.487 | 307799 | 0.99 | 1.07 |
| FAN_ManySig_official | 146605 | 4.7 | 1.496 | 1.969 | 2.293 | 333587 | 1.37 | 1.89 |
| WaveMLP_default_ManySig_official | 83974 | 1.3 | 4.412 | 6.327 | 6.63 | 114524 | 2.69 | 3.97 |
| WaveMLP_alternate_ManySig_official | 84742 | 1.4 | 5.291 | 5.775 | 8.522 | 96140 | 3.4 | 3.51 |
