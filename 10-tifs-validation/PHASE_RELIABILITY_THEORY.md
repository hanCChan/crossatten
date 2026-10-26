# Phase Reliability Theory

## Model And Assumptions

Let `X_k = A_k exp(j phi_k) + W_k`, with independent proper complex
Gaussian FFT noise satisfying `E|W_k|^2 = sigma_W^2`. At high local SNR,
rotate by the noiseless phase and linearize atan:
`delta phi_k ~= Im(W_k exp(-j phi_k))/A_k`.
Each rotated imaginary component has variance `sigma_W^2/2`, so

`Var(delta phi_k - delta phi_(k-1)) ~= sigma_W^2/2 *
(A_k^-2 + A_(k-1)^-2)`.

This requires small errors and independent adjacent noise. Windowing, colored
receiver noise and correlated FFT bins add a covariance term; low-SNR phase
cannot be analyzed with this linearization. Observed noisy amplitude is not
the latent noiseless amplitude.

## Registered Encoder

Use exactly S09's RMS-normalized FFT Re/Im and standardized log magnitude.
`p_k = z_k conj(z_(k-1)) / max(a_k a_(k-1), 1e-6)`.
The first bin is zero padded, not connected across the FFT boundary.

`r_k = a_k^2/(a_k^2 + 0.05^2)`.
`w_k = r_k r_(k-1)`.
The extra two channels are Re/Im of `w_k p_k`.
No parameters, changed losses or additional network layers are introduced.

The product is a bounded, continuous **surrogate** motivated by uncertainty,
not the exact inverse-variance optimum. The fixed tau is inherited from S09;
it is not a measured noise variance or calibrated probability of correctness.
At two threshold-amplitude bins w=0.25, whereas the hard mask jumps 0 to1.
Strong pairs approach unit weight; either weak bin suppresses the pair.
The existing denominator floor and amplitude estimates make the implementation
continuous but piecewise differentiable. Zero/sparse spectra have finite gradients.

## Identity And Channel Limits

Adjacent products remove a common phase rotation. They do not remove general
RX filters, multipath, CFO, noncircular crop errors or date drift. A circular
time shift induces a common inter-bin rotation, which this encoder retains.
Reliability reduces uncertain observations but may also attenuate useful TX
structure. The source gate tests incremental utility against the hard mask.

Physical feature experiments use identical packets/noise draws, fixed source
normalization and receiver-equal averaging. Gaussian linearization is examined
at high SNR, never asserted across the entire SNR curve.

## Contribution Boundary

Soft reliability weights and phase differences are established operations.
Only a stable, independently reproduced improvement plus mechanism evidence
could justify a new application-level algorithm contribution.
This derivation alone does not establish novelty, optimality or receiver invariance.
