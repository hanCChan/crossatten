# Ablations And Costs

Common device, batch128; raw IQ to logits in evaluation mode.
| dataset | model | parameter_elements | real_parameter_scalars | counted_mflops | cpu4t_b1_p50_ms | gpu_b1_p50_ms | gpu_b1_p95_ms | gpu_b128_packets_per_second |
|---|---|---|---|---|---|---|---|---|
| ManyRx | hard | 91128 | 91128 | 10.7983 | 1.7879 | 3.3177 | 3.4621 | 38085.4456 |
| ManyRx | soft | 91128 | 91128 | 10.7983 | 1.7981 | 3.3502 | 3.4374 | 37429.8770 |
| ManyRx | unweighted | 91128 | 91128 | 10.7983 | 1.7470 | 3.2475 | 3.3925 | 38725.7305 |
| ManyRx | no_hstu | 48890 | 48890 | 7.6526 | 1.1724 | 1.6768 | 1.7318 | 72503.1725 |
| ManyRx | ask | 100240 | 116369 | 9.3783 | 2.3897 | 2.3491 | 2.4571 | 54126.0999 |
| ManySig | hard | 90868 | 90868 | 10.7978 | 1.7952 | 3.3001 | 3.3942 | 38075.8378 |
| ManySig | soft | 90868 | 90868 | 10.7978 | 1.7820 | 3.3139 | 3.5425 | 38288.3197 |
| ManySig | unweighted | 90868 | 90868 | 10.7978 | 1.7379 | 3.2330 | 3.3669 | 38934.9346 |
| ManySig | no_hstu | 48630 | 48630 | 7.6521 | 1.1687 | 1.7086 | 1.7781 | 73687.2166 |
| ManySig | ask | 99724 | 115853 | 9.3773 | 2.3898 | 2.3566 | 2.4637 | 54200.5438 |

Torch counts supported convolution/linear/matmul only. FFT, QR, complex/scalar and normalization work not fully counted; not a complete total-FLOP comparison
ASKNet has complex parameters. Parameter ELEMENT count and real-scalar degrees are reported separately; they must not be treated as identical across real/complex models.
Soft and hard have exactly equal parameter counts and identical initialized states.
No-HSTU cost is measured but performance benefit cannot be inferred from cost alone.
S09 null-phase capacity control and Shen hierarchical-stem/HSTU evidence remain historical, not fresh WiSig same-config No-HSTU or No-reliability ablation evidence.

Candidate STOP: formal soft/hard and new trained No-HSTU/No-reliability ablations were deliberately NOT run. This honors the preregistered gate and saves36+24 fits.
Therefore neither a new reliability gain nor WiSig HSTU necessity is established by S10.
