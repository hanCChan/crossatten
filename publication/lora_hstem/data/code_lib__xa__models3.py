"""Stage-03 single-factor chirp representations on the strict-IB baseline."""
from __future__ import annotations

import math

import torch
from torch import nn

from legacy_rfhstu.cnn_stem import CNNStemPatchEmbed
from legacy_rfhstu.features import torch_rf_views
from legacy_rfhstu.models import DeviceClassifier
from .frontends import BW, FS, band_limit, rms_normalize
from .models import DIM, WINDOW
from .models2 import IB, STEM, CombFFTEmbedder

NS = 1024
SYMBOLS = WINDOW // NS


def nofft_views(x):
    amp = torch.sqrt(x[:, 0].square() + x[:, 1].square() + 1e-6)
    phase = torch.atan2(x[:, 1], x[:, 0]) / math.pi
    return torch.cat([x, torch.stack([torch.log1p(amp), phase], 1)], 1)


def legacy5_views(x):
    iq, fft, _, ap = torch_rf_views(x, FS, BW, fft_norm="log_zscore", oob_norm="ratio")
    return torch.cat([iq, fft, ap], 1)


def symbol_residual(x, normalized=True):
    z = torch.complex(x[:, 0], x[:, 1]).reshape(-1, SYMBOLS, NS)
    # Center differences to avoid amplifying mean-roundoff in exactly repeated packets.
    delta = z - z[:, :1]
    r = (delta - delta.mean(1, keepdim=True)).reshape(-1, WINDOW)
    if normalized:
        r = r / torch.sqrt(r.abs().square().mean(-1, keepdim=True) + 1e-12)
    return torch.stack([r.real, r.imag], 1)


def ifreq_deviation(x):
    z = torch.complex(x[:, 0], x[:, 1])
    dphi = torch.angle(z[:, 1:] * z[:, :-1].conj())
    dphi = torch.cat([dphi, dphi[:, -1:]], -1)
    n = torch.arange(WINDOW, device=x.device, dtype=x.dtype) % NS
    ideal = -BW / 2 + BW * (n + 0.5) / NS
    return ((dphi * (FS / (2 * math.pi)) - ideal).clamp(-2 * BW, 2 * BW) / (2 * BW))[:, None]


class SymbolStackStem(CNNStemPatchEmbed):
    """Shared symbol-local convolutions; each token sees aligned positions in all eight chirps."""

    def __init__(self):
        super().__init__(4, DIM, patch_size=256, stem_dim=32)
        old = self.tokenize
        # Preserve the flattened tokenizer initialization and RNG stream of the nofft control.
        with torch.random.fork_rng(devices=[]):
            self.tokenize = nn.Conv1d(SYMBOLS * 32, DIM, 32, stride=32)
        with torch.no_grad():
            self.tokenize.weight.copy_(old.weight.reshape_as(self.tokenize.weight))
            self.tokenize.bias.copy_(old.bias)
        self.intervention = None

    def forward(self, x):
        b = x.shape[0]
        v = x.reshape(b, 4, SYMBOLS, NS).permute(0, 2, 1, 3)
        if self.intervention == "reverse_symbols":
            v = v.flip(1)
        elif self.intervention == "misalign_symbols":
            v = torch.stack([v[:, s].roll(s * 64, -1) for s in range(SYMBOLS)], 1)
        h = self.stem(v.reshape(b * SYMBOLS, 4, NS))
        h = h.reshape(b, SYMBOLS * 32, NS)
        return self.tokenize(h).transpose(1, 2)


class ChirpEmbedder(CombFFTEmbedder):
    def __init__(self, channels="legacy5", stem="patch"):
        if channels not in ("legacy5", "nofft", "resid", "ifreq"):
            raise ValueError(channels)
        if stem not in ("patch", "symstack", "hierarchical"):
            raise ValueError(stem)
        if stem == "symstack" and channels != "nofft":
            raise ValueError("symstack requires its four-channel nofft control")
        super().__init__(use_oob=False, **STEM)
        self.channels = channels
        self.intervention = None
        if channels != "legacy5":
            nc = {"nofft": 4, "resid": 7, "ifreq": 6}[channels]
            self.cnn_stem = (SymbolStackStem() if stem == "symstack"
                             else CNNStemPatchEmbed(nc, DIM, patch_size=256, stem_dim=32))
        if stem == "hierarchical":
            self.cnn_stem.tokenize = nn.Sequential(
                nn.Conv1d(32, 48, 8, 4, 2), nn.BatchNorm1d(48), nn.GELU(),
                nn.Conv1d(48, 64, 8, 4, 2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 64, 8, 4, 2), nn.BatchNorm1d(64), nn.GELU(),
                nn.Conv1d(64, 64, 4, 4),
            )

    def forward(self, x):
        if self.channels == "legacy5":
            return super().forward(x)
        if self.channels == "nofft":
            v = nofft_views(x)
        elif self.channels == "resid":
            r = symbol_residual(x)
            if self.intervention == "zero_residual":
                r = torch.zeros_like(r)
            elif self.intervention == "shuffle_residual":
                r = r.roll(1, 0)
            v = torch.cat([legacy5_views(x), r], 1)
        else:
            v = torch.cat([legacy5_views(x), ifreq_deviation(x)], 1)
        return self.cnn_stem(v), {}


class ChirpPipeline(nn.Module):
    def __init__(self, channels="legacy5", stem="patch", preln=True, depth=2, num_classes=10):
        super().__init__()
        emb = ChirpEmbedder(channels, stem)
        self.net = DeviceClassifier(emb, num_classes, dim=DIM, depth=depth, use_chirp_embedding=False)
        self.pre_ln = nn.LayerNorm(DIM) if preln else None

    def forward(self, x):
        x = band_limit(rms_normalize(x), IB)
        if self.pre_ln is None:
            return self.net(x)["logits"]
        feats, views = self.net.embedder(x)
        h = self.pre_ln(self.net.encoder.prepare_tokens(feats, oob_features=views.get("oob")))
        for block in self.net.encoder.blocks:
            h = block(h)
        return self.net.classifier(self.net.encoder.final_norm(h).mean(1))


def build_pipeline3(arm):
    if arm["family"] != "chirp":
        raise ValueError(arm["family"])
    return ChirpPipeline(channels=arm.get("channels", "legacy5"), stem=arm.get("stem", "patch"),
                         preln=arm.get("preln", True), depth=arm.get("depth", 2))
