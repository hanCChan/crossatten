"""Stage-02 pipelines: single-factor variants on the stage-01 base (CrossAttn-TrueIB: strict-IB input, no OOB path).

ib_variant : base with optional  meanchirp (coherent 8-symbol average = projection of the 8192-pt DFT onto the comb
             bins k = 0 mod 8) | combfft (only the FFT view keeps comb bins) | dechirp (multiply by conj ideal
             SF7 upchirp after the IB band limit) | depth (number of mixer blocks)
ib_rawoob  : strict-IB main tokens + paper-1 OOB pathway (ratio view of the FULL-band spectrum, keys/values) restricted
             to a band / side. Clean test of OOB increment: OOB can enter only through the cross-attention path.
ib_cisoob  : strict-IB main tokens + OOB keys/values = channel-independent spectrogram rows outside |f|<=62.5 kHz
             (adjacent-frame spectral quotient; a static receiver/channel response at a given frequency cancels)
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .frontends import BW, FS, band_limit, cis_row_freqs, freq_shifted, rms_normalize, shen_cis
from .models import DIM, PATCH, WINDOW, BandRFPatchEmbedder, TFSplitPipeline
from legacy_rfhstu.features import patchify, torch_rf_views
from legacy_rfhstu.models import DeviceClassifier, RFHSTUBlock, RFPatchEmbedder
from legacy_rfhstu.oob_fusion import OOBCrossAttentionFusion

IB = 62_500.0
STEM = dict(window_size=WINDOW, patch_size=PATCH, dim=DIM, cnn_stem_dim=32, patch_embed_type="cnn_stem",
            fft_norm="log_zscore", oob_norm="ratio")


def upchirp_tiled(T: int = WINDOW, ns: int = 1024, fs: float = FS, bw: float = BW) -> torch.Tensor:
    t = torch.arange(ns, dtype=torch.float64) / fs
    ph = math.pi * bw / (ns / fs) * t ** 2 - math.pi * bw * t
    return torch.polar(torch.ones_like(ph), ph).repeat(T // ns).to(torch.complex64)


def band_side_mask(T: int, lo: float, hi: float, side: str | None = None) -> torch.Tensor:
    f = freq_shifted(T)
    m = (f.abs() > lo) & (f.abs() <= hi)
    if side == "L":
        m &= f < 0
    elif side == "R":
        m &= f > 0
    return m.float()


class CombFFTEmbedder(RFPatchEmbedder):
    def __init__(self, combfft: bool = False, **kw):
        super().__init__(**kw)
        self.combfft = combfft
        k = torch.arange(self.window_size) - self.window_size // 2
        self.register_buffer("comb", ((k % 8) == 0).float().view(1, 1, -1), persistent=False)

    def forward(self, iq):
        if not self.combfft:
            return super().forward(iq)
        iq_v, fft_v, _, ap_v = torch_rf_views(iq, self.sample_rate, self.lora_bandwidth, fft_norm=self.fft_norm,
                                              oob_norm=self.oob_norm)
        return self.cnn_stem(torch.cat([iq_v, fft_v * self.comb, ap_v], dim=1)), {}


class IBVariantPipeline(nn.Module):
    def __init__(self, meanchirp=False, combfft=False, dechirp=False, depth=2, num_classes=10):
        super().__init__()
        emb = CombFFTEmbedder(combfft=combfft, use_oob=False, **STEM)
        self.net = DeviceClassifier(emb, num_classes, dim=DIM, depth=depth, use_chirp_embedding=False)
        self.meanchirp, self.dechirp = meanchirp, dechirp
        if dechirp:
            self.register_buffer("cconj", torch.view_as_real(upchirp_tiled().conj().resolve_conj()), persistent=False)

    def forward(self, x):
        x = rms_normalize(x)
        if self.meanchirp:
            B = x.shape[0]
            x = x.view(B, 2, 8, WINDOW // 8).mean(2, keepdim=True).expand(B, 2, 8, WINDOW // 8).reshape(B, 2, WINDOW)
        x = band_limit(x, IB)
        if self.dechirp:
            z = torch.complex(x[:, 0], x[:, 1]) * torch.view_as_complex(self.cconj)
            x = torch.stack([z.real, z.imag], 1)
        return self.net(x)["logits"]


class IBRawOOBEmbedder(BandRFPatchEmbedder):
    def __init__(self, oob_band=(IB, 1e9), oob_side=None, **kw):
        super().__init__(oob_keep=None, **kw)
        m = band_side_mask(self.window_size, oob_band[0], oob_band[1], oob_side).view(1, 1, -1)
        self.register_buffer("oob_band_mask", patchify(m, self.patch_size), persistent=False)

    def forward(self, iq):
        main, views = super().forward(band_limit(iq, IB))
        _, _, oob_v, _ = torch_rf_views(iq, self.sample_rate, self.lora_bandwidth, fft_norm=self.fft_norm, oob_norm=self.oob_norm)
        views = dict(views)
        views["oob"] = patchify(oob_v, self.patch_size) * self.oob_band_mask
        return main, views


class IBRawOOBPipeline(nn.Module):
    def __init__(self, oob_band=(IB, 1e9), oob_side=None, num_classes=10):
        super().__init__()
        emb = IBRawOOBEmbedder(oob_band=oob_band, oob_side=oob_side, use_oob=True, oob_fusion_type="cross_attn_oob",
                               use_oob_cross_attention=True, **STEM)
        self.net = DeviceClassifier(emb, num_classes, dim=DIM, depth=2, use_chirp_embedding=False)

    def forward(self, x):
        return self.net(rms_normalize(x))["logits"]


class IBCISOOBPipeline(nn.Module):
    def __init__(self, depth=2, num_classes=10):
        super().__init__()
        P = WINDOW // PATCH
        self.embedder = RFPatchEmbedder(use_oob=False, **STEM)
        self.input_proj = nn.Linear(DIM, DIM)
        self.position = nn.Parameter(torch.zeros(1, P, DIM))
        rows = torch.nonzero(cis_row_freqs().abs() > IB).squeeze(1)
        self.register_buffer("oob_rows", rows, persistent=False)
        self.oob_proj = nn.Linear(126, DIM)
        self.oob_pos = nn.Parameter(torch.zeros(1, len(rows), DIM))
        self.fusion = OOBCrossAttentionFusion(DIM, num_heads=4)
        self.blocks = nn.ModuleList([RFHSTUBlock(DIM, P) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(DIM)
        self.classifier = nn.Linear(DIM, num_classes)

    def forward(self, x):
        x = rms_normalize(x)
        main, _ = self.embedder(band_limit(x, IB))
        h = self.input_proj(main) + self.position
        o = self.oob_proj(shen_cis(x)[:, 0][:, self.oob_rows]) + self.oob_pos
        h = self.fusion(h, o)
        for blk in self.blocks:
            h = blk(h)
        return self.classifier(self.final_norm(h).mean(1))


def build_pipeline2(arm: dict) -> nn.Module:
    fam = arm["family"]
    if fam == "ib_variant":
        return IBVariantPipeline(meanchirp=arm.get("meanchirp", False), combfft=arm.get("combfft", False),
                                 dechirp=arm.get("dechirp", False), depth=arm.get("depth", 2))
    if fam == "ib_rawoob":
        return IBRawOOBPipeline(oob_band=tuple(arm.get("oob_band", (IB, 1e9))), oob_side=arm.get("oob_side"))
    if fam == "ib_cisoob":
        return IBCISOOBPipeline()
    if fam == "ib_tfsplit":
        return TFSplitPipeline(input_fmax=IB, use_oob=False)
    raise ValueError(fam)
