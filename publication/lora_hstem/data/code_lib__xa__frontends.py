"""Signal front-ends in torch (GPU). All inputs are raw I/Q tensors [B, 2, T] (float32).

- rms_normalize : legacy packet RMS normalisation (rfhstu.features.normalize_iq semantics, eps 1e-6)
- awgn          : Shen et al. dataset_preparation.awgn (per-packet SNR ~ U[lo, hi) dB)
- band_limit    : FFT mask |f| <= f_max, IFFT (paper-1 TrueIB pathway when f_max = 62.5 kHz)
- shen_cis      : port of Shen et al. ChannelIndSpectrogram (scipy STFT boxcar 128 / hop 64,
                  two-sided, fftshift, adjacent-frame spectral quotient, log10|.|^2, row crop 0.3);
                  computed in float64 so masked rows stay at the 1e-12 epsilon floor exactly as numpy.
Shen code/data are CC BY-NC-SA 4.0 (github.com/gxhen/receiverAgnosticRFFI); this is a re-implementation.
"""
from __future__ import annotations

import math

import torch

FS = 1_000_000.0
BW = 125_000.0


def freq_shifted(T: int, fs: float = FS, device=None, dtype=torch.float64) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftfreq(T, d=1.0 / fs, dtype=dtype)).to(device)


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sqrt((x[:, 0] ** 2 + x[:, 1] ** 2).mean(-1) + eps)
    return x / p.view(-1, 1, 1)


def awgn(x: torch.Tensor, lo_db: float, hi_db: float, gen: torch.Generator) -> torch.Tensor:
    B = x.shape[0]
    snr = lo_db + (hi_db - lo_db) * torch.rand(B, generator=gen, device=x.device)
    p = (x[:, 0] ** 2 + x[:, 1] ** 2).mean(-1)
    n0 = p / (10.0 ** (snr / 10.0))
    noise = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return x + noise * torch.sqrt(n0 / 2.0).view(B, 1, 1)


def band_limit(x: torch.Tensor, f_max: float, fs: float = FS) -> torch.Tensor:
    T = x.shape[-1]
    z = torch.complex(x[:, 0], x[:, 1])
    Z = torch.fft.fft(z, dim=-1)
    keep = (torch.fft.fftfreq(T, d=1.0 / fs).abs() <= f_max).to(x.device)
    z = torch.fft.ifft(Z * keep, dim=-1)
    return torch.stack([z.real, z.imag], dim=1).to(x.dtype)


def oob_keep_mask(T: int, lo: float, hi: float, fs: float = FS) -> torch.Tensor:
    """fftshift-ordered mask selecting lo < |f| <= hi."""
    f = freq_shifted(T, fs).abs()
    return ((f > lo) & (f <= hi)).float()


def cis_rows(win: int = 128, crop_ratio: float = 0.3) -> tuple[int, int]:
    return math.floor(win * crop_ratio), math.ceil(win * (1 - crop_ratio))


def cis_row_freqs(win: int = 128, crop_ratio: float = 0.3, fs: float = FS) -> torch.Tensor:
    r0, r1 = cis_rows(win, crop_ratio)
    return (torch.arange(r0, r1, dtype=torch.float64) - win // 2) * fs / win


def shen_cis(x: torch.Tensor, f_max: float | None = None, win: int = 128, crop_ratio: float = 0.3,
             fs: float = FS) -> torch.Tensor:
    """[B,2,T] raw I/Q -> [B,1,52,126] CIS (float32). f_max: optional band limit before the STFT (strict CIS)."""
    z = torch.complex(x[:, 0].double(), x[:, 1].double())
    z = z / torch.sqrt((z.abs() ** 2).mean(-1, keepdim=True))
    if f_max is not None:
        keep = (torch.fft.fftfreq(z.shape[-1], d=1.0 / fs, dtype=torch.float64).abs() <= f_max).to(z.device)
        z = torch.fft.ifft(torch.fft.fft(z, dim=-1) * keep, dim=-1)
    hop = win - round(0.5 * win)
    frames = z.unfold(-1, win, hop)                       # [B, nF, win]
    spec = torch.fft.fft(frames, dim=-1) / win            # scipy 'spectrum' scaling for boxcar
    spec = torch.fft.fftshift(spec, dim=-1).transpose(1, 2) + 1e-12
    d = spec[:, :, 1:] / spec[:, :, :-1]
    amp = torch.log10(d.abs() ** 2)
    r0, r1 = cis_rows(win, crop_ratio)
    return amp[:, r0:r1].float().unsqueeze(1)


def shen_cis_numpy_reference(iq_complex, win: int = 128, crop_ratio: float = 0.3):
    """Verbatim logic of Shen et al. ChannelIndSpectrogram.channel_ind_spectrogram (for tests only)."""
    import numpy as np
    from scipy import signal

    data = iq_complex / np.sqrt(np.mean(np.abs(iq_complex) ** 2, axis=1, keepdims=True))
    overlap = round(0.5 * win)
    out = []
    for s in data:
        _, _, spec = signal.stft(s, window="boxcar", nperseg=win, noverlap=overlap, nfft=win,
                                 return_onesided=False, padded=False, boundary=None)
        spec = np.fft.fftshift(spec, axes=0) + 1e-12
        dspec = np.log10(np.abs(spec[:, 1:] / spec[:, :-1]) ** 2)
        n = dspec.shape[0]
        out.append(dspec[math.floor(n * crop_ratio):math.ceil(n * (1 - crop_ratio))])
    return np.stack(out)[:, None]
