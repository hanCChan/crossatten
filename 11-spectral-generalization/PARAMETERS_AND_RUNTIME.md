# Parameters And Runtime

Same A100, Torch2.5.0/CUDA12.4, CPU4threads, batch128, random frozen network weights. IQ already on device to logits; prior estimated offline only from training source packets.
| dataset | model | parameter_elements | real_parameter_scalars | counted_mflops | cpu4t_b1_p50_ms | gpu_b1_p50_ms | gpu_b1_p95_ms | gpu_b128_packets_per_s |
|---|---|---|---|---|---|---|---|---|
| ManyRx | hard | 91128 | 91128 | 10.7983 | 1.6727 | 3.2395 | 3.4008 | 38628.4948 |
| ManyRx | F1 | 91128 | 91128 | 10.7983 | 1.8344 | 3.3591 | 3.5549 | 37393.1366 |
| ManyRx | F2 | 91128 | 91128 | 10.7983 | 1.8027 | 3.2845 | 3.5069 | 37294.4720 |
| ManyRx | ASKNet | 100240 | 116369 | 9.3783 | 2.4263 | 2.3104 | 2.6132 | 55250.1426 |
| ManySig | hard | 90868 | 90868 | 10.7978 | 1.7835 | 3.2234 | 3.4532 | 38335.6576 |
| ManySig | F1 | 90868 | 90868 | 10.7978 | 1.8305 | 3.3519 | 3.5694 | 37690.1590 |
| ManySig | F2 | 90868 | 90868 | 10.7978 | 1.7782 | 3.2726 | 3.6777 | 37541.0603 |
| ManySig | ASKNet | 99724 | 115853 | 9.3773 | 2.4677 | 2.2999 | 2.4612 | 55282.5995 |

No new network accuracy is inferred from these costs. ASKNet complex elements and real scalar freedom are separate. No speed superiority claim based on parameter count.
Torch counter omits/not fully counts FFT, QR, complex and scalar operations; counted FLOPs are partial and not a complete total-operation comparison.
ASKNet uses random weights with tied cutoff logits (eval k8), not trained checkpoints. Its complex operator stays127modes for every cutoff; this is an architectural cost probe, not a checkpoint-specific runtime superiority test.
F2 buffer256floats is not a trainable parameter. Both F1/F2 preserve original token/HSTU dimensions.
