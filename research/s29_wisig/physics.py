"""Packet-local CFO estimation and same-support perturbations."""

import math

import torch

FS = 25_000_000
LAG = 20
RAD_TO_HZ = FS / (2 * math.pi * LAG)
BAND_HZ = 7_675_781.25


def complex_iq(x):
    return torch.complex(x[:, 0], x[:, 1])


def cross_product(a, b):
    return (a * b.conj()).sum(-1)


def coherence(a, b):
    power = (a.abs().square().sum(-1) * b.abs().square().sum(-1)).sqrt()
    return (cross_product(a, b).abs() / power.clamp_min(1e-12)).clamp_max(1)


def lag_coherence(z, lag):
    return coherence(z[:, lag:], z[:, :-lag])


def estimate(x):
    segment = complex_iq(x)[:, 20:140]
    blocks = segment.reshape(len(x), 6, LAG)
    later, earlier = blocks[:, 1:], blocks[:, :-1]
    pair = cross_product(later, earlier)
    total = pair.sum(1)
    hz = total.angle() * RAD_TO_HZ
    spread_hz = (pair * total[:, None].conj()).angle().abs().amax(1) * RAD_TO_HZ
    minimum_coherence = coherence(later, earlier).amin(1)
    neighbor = torch.maximum(lag_coherence(segment, 19), lag_coherence(segment, 21))
    margin = lag_coherence(segment, LAG) - neighbor
    reliable = (minimum_coherence >= .7) & (spread_hz <= 5000) & (margin >= .05)
    return hz, reliable, minimum_coherence, spread_hz, margin


def rotate(x, hz):
    time = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype) / FS
    hz = torch.as_tensor(hz, device=x.device, dtype=x.dtype)
    angle = hz[..., None] * (2 * math.pi * time)
    rotated = complex_iq(x) * torch.polar(torch.ones_like(angle), angle)
    return torch.stack((rotated.real, rotated.imag), dim=1)


def correct(x):
    hz, reliable, *_ = estimate(x)
    rotated = rotate(x, -hz)
    return torch.where(reliable[:, None, None], rotated, x)


def matched_time_shift(x, offset):
    """Return aligned/shifted pairs containing identical observed samples."""
    size = abs(offset)
    aligned = x.clone()
    if offset > 0:
        aligned[..., -size:] = 0
    else:
        aligned[..., :size] = 0
    return aligned, aligned.roll(offset, dims=-1)


def characterize(x):
    z = complex_iq(x)
    power = torch.fft.fft(z).abs().square()
    hz = torch.fft.fftfreq(z.shape[-1], 1 / FS, device=z.device)
    inside = power[:, hz.abs() <= BAND_HZ].sum(1) / power.sum(1)
    cfo, reliable, minimum_coherence, spread, margin = estimate(x)
    return dict(cfo_hz=cfo, reliable=reliable, block_min_coherence=minimum_coherence,
                block_spread_hz=spread, lag_margin=margin, inband_fraction=inside,
                early_coherence=lag_coherence(z[:, 20:140], LAG),
                tail_coherence=lag_coherence(z[:, 160:256], LAG))
