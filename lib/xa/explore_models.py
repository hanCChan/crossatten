"""Stage-07 frontends with explicit time/frequency coordinates."""
from __future__ import annotations

import math

import torch
from torch import nn

from legacy_rfhstu.cnn_stem import CNNStemPatchEmbed
from legacy_rfhstu.features import torch_rf_views
from legacy_rfhstu.models import RFHSTUEncoder
from .frontends import rms_normalize
from .paper_models import ValidationRF


def phase_product(x, lag=1):
    if lag <= 0 or lag >= x.shape[-1]:
        raise ValueError("invalid phase-product lag")
    z = torch.complex(x[:, 0], x[:, 1])
    product = z[:, lag:] * z[:, :-lag].conj()
    product = product / (z[:, lag:].abs() * z[:, :-lag].abs()).clamp_min(1e-6)
    out = torch.stack((product.real, product.imag), 1)
    return torch.nn.functional.pad(out, (lag, 0))


def time_views(x, differential=False):
    _, _, _, amplitude_phase = torch_rf_views(x, 25e6, 15.3515625e6, oob_norm="ratio")
    if differential:
        return torch.cat((x, amplitude_phase[:, :1], phase_product(x)), 1)
    return torch.cat((x, amplitude_phase), 1)


def frequency_views(x):
    z = torch.fft.fftshift(torch.fft.fft(torch.complex(x[:, 0], x[:, 1])), dim=-1)
    magnitude = z.abs()
    norm = magnitude.square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
    logmag = torch.log1p(magnitude)
    logmag = (logmag - logmag.mean(-1, keepdim=True)) / logmag.std(-1, keepdim=True).clamp_min(1e-6)
    return torch.stack((z.real / norm, z.imag / norm, logmag), 1)


def periodic_views(x, stop=160, lag=20):
    if stop > x.shape[-1] or stop <= lag:
        raise ValueError("unverified STF region")
    product = phase_product(x, lag)
    mask = torch.zeros_like(product)
    mask[:, :, lag:stop] = 1
    return product * mask


class IdentityFilter(nn.Module):
    def forward(self, x):
        return x


class SmoothComplexFilter(nn.Module):
    """Identity-initialized bounded complex gain on a 16-knot frequency grid."""
    def __init__(self, knots=16):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(1, 1, knots))
        self.phase = nn.Parameter(torch.zeros(1, 1, knots))

    def forward(self, x):
        length = x.shape[-1]
        amplitude = torch.exp(0.5 * torch.tanh(torch.nn.functional.interpolate(
            self.log_gain, size=length, mode="linear", align_corners=True)))[:, 0]
        phase = 0.5 * torch.tanh(torch.nn.functional.interpolate(
            self.phase, size=length, mode="linear", align_corners=True))[:, 0]
        gain = torch.polar(amplitude, phase)
        z = torch.complex(x[:, 0], x[:, 1])
        spectrum = torch.fft.fftshift(torch.fft.fft(z), dim=-1)
        filtered = torch.fft.ifft(torch.fft.ifftshift(spectrum * gain, dim=-1))
        return torch.stack((filtered.real, filtered.imag), 1)


class HierarchicalTokens(CNNStemPatchEmbed):
    def __init__(self, channels):
        super().__init__(channels, 64, 8, 32)
        layers = []
        cin = 32
        for cout in (48, 64, 64):
            layers.extend((nn.Conv1d(cin, cout, 4, 2, 1), nn.BatchNorm1d(cout), nn.GELU()))
            cin = cout
        self.tokenize = nn.Sequential(*layers)


class ExploreRF(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.kind = arm["frontend"]
        self.stf_stop = arm.get("stf_stop", 160)
        differential = self.kind in ("diff", "diff_stf", "diff_split")
        channels = 5 if differential else 4
        if self.kind in ("stf", "diff_stf"):
            channels += 2
        self.filter = SmoothComplexFilter() if self.kind == "filter" else IdentityFilter()
        self.time = HierarchicalTokens(channels) if self.kind != "frequency" else None
        split = self.kind in ("split", "diff_split")
        self.frequency = HierarchicalTokens(3) if self.kind == "frequency" or split else None
        self.frequency_position = nn.Parameter(torch.zeros(1, 32, 64)) if split else None
        self.fusion = nn.Linear(128, 64) if split else None
        self.fusion_kind = arm.get("fusion", "pooled_concat")
        self.gate = nn.Linear(128, 64) if split and self.fusion_kind == "gate" else None
        self.encoder = RFHSTUEncoder(64, num_patches=32, dim=64,
                                    depth=2 if arm.get("mixer", True) else 0)
        self.pre_ln = nn.LayerNorm(64)
        self.classifier = nn.Linear(64, arm["num_classes"])

    def tokens(self, x):
        x = self.filter(rms_normalize(x))
        if self.kind == "frequency":
            return self.frequency(frequency_views(x))
        differential = self.kind in ("diff", "diff_stf", "diff_split")
        features = time_views(x, differential)
        if self.kind in ("stf", "diff_stf"):
            features = torch.cat((features, periodic_views(x, self.stf_stop)), 1)
        h = self.time(features)
        if self.frequency is not None:
            # Frequency bins have their own positions; global pooling makes no
            # one-to-one correspondence with the main time tokens.
            spectral = self.frequency(frequency_views(x)) + self.frequency_position
            spectral = spectral.mean(1, keepdim=True).expand_as(h)
            joined = torch.cat((h, spectral), -1)
            if self.gate is not None:
                h = h + torch.sigmoid(self.gate(joined)) * self.fusion(joined)
            else:
                h = self.fusion(joined)
        return h

    def forward(self, x):
        h = self.pre_ln(self.encoder.prepare_tokens(self.tokens(x)))
        for block in self.encoder.blocks:
            h = block(h)
        return self.classifier(self.encoder.final_norm(h).mean(1))


def build_explore_model(arm):
    if arm["frontend"] == "legacy":
        return ValidationRF(dict(length=256, stem="hierarchical",
                                 num_classes=arm["num_classes"]))
    return ExploreRF(arm)


def physical_augmentation(x, kind, generator, bounds):
    z = torch.complex(x[:, 0], x[:, 1])
    n, length = z.shape
    if kind == "none":
        return x
    if kind == "phase":
        angle = (torch.rand((n, 1), device=x.device, generator=generator) * 2 - 1) * math.pi
        z = z * torch.exp(1j * angle)
    elif kind == "shift":
        shifts = torch.randint(-bounds["shift_samples"], bounds["shift_samples"] + 1,
                               (n, 1), device=x.device, generator=generator)
        indices = torch.arange(length, device=x.device)[None] - shifts
        valid = (indices >= 0) & (indices < length)
        z = z.gather(1, indices.clamp(0, length - 1)) * valid
    elif kind == "cfo":
        freq = (torch.rand((n, 1), device=x.device, generator=generator) * 2 - 1) * bounds["cfo_hz"]
        t = torch.arange(length, device=x.device)[None] / 25e6
        z = z * torch.exp(2j * math.pi * freq * t)
    elif kind == "channel":
        noise = torch.randn((n, 2, 8), device=x.device, generator=generator)
        noise = torch.nn.functional.interpolate(noise, size=length, mode="linear", align_corners=True)
        gain = torch.exp(bounds["channel_log_gain"] * noise[:, 0].tanh()
                         + 1j * bounds["channel_phase"] * noise[:, 1].tanh())
        z = torch.fft.ifft(torch.fft.fft(z) * torch.fft.ifftshift(gain, dim=-1))
    else:
        raise ValueError(kind)
    return torch.stack((z.real, z.imag), 1)
