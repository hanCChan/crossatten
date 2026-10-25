"""Paper-validation adapters; upstream source stays local and unmodified."""
from __future__ import annotations

import importlib
import subprocess
import sys
import types

import torch
from torch import nn

from legacy_rfhstu.models import DeviceClassifier, RFPatchEmbedder
from . import ROOT
from .frontends import band_limit, rms_normalize

UPSTREAM = {
    "IB-RFF": "5a03a04da86c4bbf5cf90f41a55c268bf2152203",
    "WaveMLP-RFF": "1ade72d0082428db2087633a44ae533f1022ca22",
}


def upstream_module(repo, module):
    path = ROOT / "external" / repo
    sha = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if sha != UPSTREAM[repo]:
        raise ValueError(f"unregistered upstream revision: {repo} {sha}")
    namespace = "_xa_ibrff" if repo == "IB-RFF" else "_xa_wavemlp"
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = [str(path / "backbones")]
        sys.modules[namespace] = package
    if repo == "IB-RFF" and "timm" not in sys.modules:
        timm, layers = types.ModuleType("timm"), types.ModuleType("timm.layers")
        layers.trunc_normal_ = torch.nn.init.trunc_normal_
        timm.layers = layers
        sys.modules["timm"], sys.modules["timm.layers"] = timm, layers
    return importlib.import_module(f"{namespace}.{module}")


class UpstreamPipeline(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.fmax = arm.get("input_fmax")
        self.fs = arm.get("sample_rate", 1e6)
        self.method = arm["method"]
        length, classes = arm["length"], arm["num_classes"]
        if self.method in ("FAN", "IB-RFF"):
            self.backbone = upstream_module("IB-RFF", "FAN").FAN(
                seq_len=length, in_chans=2, patch_size=arm["patch_size"],
                embed_dim=128, mlp_ratio=2.0, num_classes=classes,
                bottleneck_type="DSCNN", use_SAF=True, use_FAF=True)
        elif self.method == "WaveMLP":
            self.backbone = upstream_module("WaveMLP-RFF", "MyModel").MyModel(
                seq_len=length, num_classes=classes, patch_size=arm["patch_size"],
                stride=arm["stride"], use_wavelet=True, wavelet_learnable=True,
                wavelet_init="randn", wavelet_levels=arm["wavelet_levels"],
                wavelet_kernel_size=arm["wavelet_kernel_size"])
        else:
            raise ValueError(self.method)

    def preprocess(self, x):
        # WiSig cache already follows the exact official power normalization.
        if x.shape[-1] == 8192:
            x = rms_normalize(x)
        if self.fmax is not None:
            x = band_limit(x, self.fmax, self.fs)
        return x

    def forward_with_hidden(self, x):
        x = self.preprocess(x)
        out = self.backbone(x)
        if self.method in ("FAN", "IB-RFF"):
            hidden, logits = out
        else:
            logits, _, hidden = out
        return logits, hidden.flatten(1), x.flatten(1)

    def forward(self, x):
        return self.forward_with_hidden(x)[0]


class ValidationEmbedder(RFPatchEmbedder):
    def __init__(self, length, stem, width=32):
        super().__init__(
            window_size=length, patch_size=length // 32, dim=64,
            sample_rate=25e6 if length == 256 else 1e6,
            lora_bandwidth=15.3515625e6 if length == 256 else 125e3,
            cnn_stem_dim=width, patch_embed_type="cnn_stem",
            use_oob=False, fft_norm="log_zscore", oob_norm="ratio")
        if stem == "hierarchical":
            if width != 32:
                raise ValueError("hierarchical width is fixed")
            layers = []
            cin = 32
            for cout in (48, 64, 64):
                kernel, stride, padding = (4, 2, 1) if length == 256 else (8, 4, 2)
                layers.extend([nn.Conv1d(cin, cout, kernel, stride, padding),
                               nn.BatchNorm1d(cout), nn.GELU()])
                cin = cout
            if length == 8192:
                layers.append(nn.Conv1d(64, 64, 4, 4))
            self.cnn_stem.tokenize = nn.Sequential(*layers)
        elif stem != "patch":
            raise ValueError(stem)


class ValidationRF(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.length = arm["length"]
        self.fs = 25e6 if self.length == 256 else 1e6
        self.fmax = arm.get("input_fmax")
        embedder = ValidationEmbedder(self.length, arm["stem"], arm.get("width", 32))
        self.net = DeviceClassifier(embedder, arm["num_classes"], dim=64,
                                    depth=2 if arm.get("mixer", True) else 0,
                                    use_chirp_embedding=False)
        self.pre_ln = nn.LayerNorm(64)

    def forward(self, x):
        if x.shape[1:] != (2, self.length):
            raise ValueError("unexpected IQ layout")
        x = rms_normalize(x)
        if self.fmax is not None:
            x = band_limit(x, self.fmax, self.fs)
        feats, _ = self.net.embedder(x)
        h = self.pre_ln(self.net.encoder.prepare_tokens(feats))
        for block in self.net.encoder.blocks:
            h = block(h)
        return self.net.classifier(self.net.encoder.final_norm(h).mean(1))


def hsic_cca(x, y, sigma=5.0):
    """Upstream right-centered Gaussian nHSIC-CCA, with device-safe tensors."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("HSIC needs equal batches with at least two samples")
    n = len(x)
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    center = eye - torch.ones_like(eye) / n

    def kernel(v):
        r = v.square().sum(1, keepdim=True)
        distances = (r - 2 * v @ v.T + r.T).abs()
        return torch.exp(-distances / (2 * sigma * sigma * v.shape[1])) @ center

    kx, ky = kernel(x), kernel(y)
    rx = kx @ torch.linalg.inv(kx + 1e-5 * n * eye)
    ry = ky @ torch.linalg.inv(ky + 1e-5 * n * eye)
    return (rx * ry.T).sum()


def build_paper_model(arm):
    if arm["family"] == "paper_rf":
        return ValidationRF(arm)
    if arm["family"] == "paper_upstream":
        return UpstreamPipeline(arm)
    raise ValueError(arm["family"])
