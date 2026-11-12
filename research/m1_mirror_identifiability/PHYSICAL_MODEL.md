# TX → channel/CFO → RX model

The simulator implements, in this order:

\[
s_t[n]=s[n]+\nu_t s^*[n],\quad
v[n]=e^{j(2\pi f_c n/F_s+\phi)}\sum_l h_l s_t[n-l],\quad
y[n]=v[n]+\nu_r v^*[n]+w[n].
\]

The direct gains are fixed to one in this bounded simulation; this is not a claim about calibrated hardware. Complex channel gain and initial phase remain explicit. AWGN has complex variance equal to clean receive power divided by linear SNR. The RX conjugate branch rotates in the opposite CFO direction. Multiplying an entire `a*s+b*conj(s)` by one rotation generally does not reproduce this chain.

For a flat known channel and zero CFO, two identifiable **effective** coefficients are

\[
a=e^{j\phi}h+\nu_r e^{-j\phi}h^*\nu_t^*,\quad
b=e^{j\phi}h\nu_t+\nu_r e^{-j\phi}h^*.
\]

They are not TX-only parameters. RX-only imbalance gives nonzero b even when nu_t=0. With known timing and CFO, a four-column diagnostic design is `[e+ s,e+ s*,e- s*,e- s]`; it becomes rank deficient at CFO=0. The API rejects deficient designs, never hides a singular inverse, and never returns a true-TX estimate.

The source descriptor uses a deliberately simpler two-column projection after a fit-only coarse CFO estimate. It is accepted only if it predicts an untouched repeated segment sufficiently well. The full simulator measures its misspecification; a small projection residual alone is not physical separation.

The L-STF reference uses tones [-24,-20,-16,-12,-8,-4,4,8,12,16,20,24], signs [1,-1,1,-1,-1,1,-1,-1,1,1,1,1], times (1+j), sampled by its continuous trigonometric polynomial at 25 MHz (80 samples per nominal 64-point OFDM interval). This avoids inventing a 20 MHz template for 25 MHz samples or introducing an arbitrary resampling filter. Edge windowing and actual transmitter/channel filters are not modeled as calibrated quantities.

The 0.8 us repetition is 20 samples, the nominal 8 us L-STF is 200 samples. Official energy detection does not certify the absolute field start in a local compact packet. Therefore [20,180) is one **fixed candidate interior**, not an already verified STF boundary. It is never moved after observations. Failure to predict its held-out half is a source validity failure. Nothing uses L-LTF or the final 76 samples to rescue fits.
