"""Frozen S09 phase features, per-packet STF CFO and isolated P2 controls."""
import importlib.util
import torch
from torch import nn
from core import ROOT
from xa.explore_models import ExploreRF, HierarchicalTokens, frequency_views
from xa.frontends import rms_normalize

spec = importlib.util.spec_from_file_location("s13_frozen_phase", ROOT / "09-phase-invariance/phase_features.py")
phase = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase)
FS, LAG, STOP = 25e6, 20, 160


def cfo(x):
    z = torch.complex(x[:, 0], x[:, 1])
    a, b = z[:, LAG:STOP], z[:, :STOP - LAG]
    product = (a * b.conj()).sum(-1)
    power = (a.abs().square().sum(-1) * b.abs().square().sum(-1)).sqrt().clamp_min(1e-12)
    coherence = product.abs() / power
    return product.angle() * FS / (2 * torch.pi * LAG), coherence.clamp(0, 1)


def rotate(x, hz):
    z = torch.complex(x[:, 0], x[:, 1])
    hz = torch.as_tensor(hz, device=x.device, dtype=x.dtype)
    if hz.ndim == 0:
        hz = hz.expand(len(x))
    angle = 2 * torch.pi * hz[:, None] * torch.arange(x.shape[-1], device=x.device)[None] / FS
    z = z * torch.exp(1j * angle)
    return torch.stack([z.real, z.imag], 1)


def corrected(x):
    estimated, confidence = cfo(x)
    return rotate(x, -torch.where(confidence >= .7, estimated, torch.zeros_like(estimated)))


class WiSigModel(ExploreRF):
    def __init__(self, dataset, variant="B0"):
        super().__init__(dict(frontend="frequency", num_classes=10 if dataset == "ManyRx" else 6))
        self.frequency = HierarchicalTokens(5)
        self.variant = variant
        if variant in ("B2_dual", "B2_dual_raw"):
            self.second = HierarchicalTokens(5)
            self.fusion = nn.Linear(128, 64)
            self.gate = nn.Linear(128, 64)

    def features(self, x):
        return phase.phase_features_from_base(frequency_views(rms_normalize(x)), "first")

    def tokens(self, x):
        raw = rms_normalize(x)
        if self.variant == "B2_corrected":
            return self.frequency(self.features(corrected(raw)))
        first = self.frequency(self.features(raw))
        if self.variant in ("B2_dual", "B2_dual_raw"):
            other = corrected(raw) if self.variant == "B2_dual" else raw
            second = self.second(self.features(other))
            joined = torch.cat([first, second], -1)
            return first + self.gate(joined).sigmoid() * self.fusion(joined)
        return first


def build_wifi(dataset, variant, seed):
    torch.manual_seed(seed)
    return WiSigModel(dataset, variant)
