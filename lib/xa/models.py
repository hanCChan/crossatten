"""Model pipelines: raw I/Q [B,2,T] -> logits.

crossattn : paper-1 CrossAttn-OOB (legacy rfhstu code, unchanged) + optional input band limit / OOB-band keep mask
tfsplit   : same blocks, but time-domain (I/Q, amp/phase) and frequency-domain (FFT) stems are separate token
            streams with their own position embeddings; OOB tokens share the FFT (frequency) positions
shen_cis  : PyTorch port of Shen et al. TMC2024 CIS ResNet (Keras: glorot init, 'same' padding); head flatten|gap
"""
from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F
from torch import nn

from . import LIB
from .frontends import band_limit, oob_keep_mask, rms_normalize, shen_cis

sys.path.insert(0, str(LIB))
from legacy_rfhstu.cnn_stem import CNNStemPatchEmbed  # noqa: E402
from legacy_rfhstu.features import patchify, torch_rf_views  # noqa: E402
from legacy_rfhstu.models import DeviceClassifier, RFPatchEmbedder  # noqa: E402
from legacy_rfhstu.oob_fusion import OOBCrossAttentionFusion  # noqa: E402

WINDOW, PATCH, DIM = 8192, 256, 64


class BandRFPatchEmbedder(RFPatchEmbedder):
    """Legacy embedder; optionally zero the OOB view outside lo<|f|<=hi (ratio normalisation unchanged: IB RMS)."""

    def __init__(self, oob_keep: tuple[float, float] | None = None, **kw):
        super().__init__(**kw)
        self.oob_keep = tuple(oob_keep) if oob_keep is not None else None
        if self.oob_keep is not None:
            m = oob_keep_mask(self.window_size, *self.oob_keep, fs=self.sample_rate).view(1, 1, -1)
            self.register_buffer("oob_keep_mask", patchify(m, self.patch_size), persistent=False)

    def forward(self, iq):
        main, views = super().forward(iq)
        if self.oob_keep is not None and "oob" in views:
            views = dict(views)
            views["oob"] = views["oob"] * self.oob_keep_mask
        return main, views


class CrossAttnPipeline(nn.Module):
    def __init__(self, fusion: str = "cross_attn_oob", input_fmax: float | None = None,
                 oob_keep: tuple[float, float] | None = None, num_classes: int = 10):
        super().__init__()
        use_oob = fusion != "no_oob"
        emb = BandRFPatchEmbedder(
            oob_keep=oob_keep, window_size=WINDOW, patch_size=PATCH, dim=DIM, cnn_stem_dim=32,
            patch_embed_type="cnn_stem", use_oob=use_oob,
            oob_fusion_type=fusion if use_oob else "concat_oob",
            use_oob_cross_attention=(fusion == "cross_attn_oob"), fft_norm="log_zscore", oob_norm="ratio")
        self.net = DeviceClassifier(emb, num_classes, dim=DIM, depth=2, use_chirp_embedding=False)
        self.input_fmax = input_fmax

    def forward(self, x):
        x = rms_normalize(x)
        if self.input_fmax is not None:
            x = band_limit(x, self.input_fmax)
        return self.net(x)["logits"]


class SegmentHSTUBlock(nn.Module):
    """legacy RFHSTUBlock with a segment-aware relative bias (Toeplitz inside a segment, scalar across)."""

    def __init__(self, dim: int, seg_len: int, n_seg: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dim, self.seg_len, self.n_seg = dim, seg_len, n_seg
        self.qkvu = nn.Linear(dim, dim * 4)
        self.rel = nn.Parameter(torch.zeros(n_seg, 2 * seg_len - 1))
        self.cross = nn.Parameter(torch.zeros(n_seg, n_seg))
        self.av_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self.residual_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        idx = torch.arange(seg_len)
        self.register_buffer("rel_index", idx[:, None] - idx[None, :] + seg_len - 1, persistent=False)
        self.register_buffer("same_seg", torch.block_diag(*[torch.ones(seg_len, seg_len)] * n_seg).bool(), persistent=False)

    def bias(self):
        cross = self.cross.repeat_interleave(self.seg_len, 0).repeat_interleave(self.seg_len, 1)
        within = torch.block_diag(*[self.rel[s][self.rel_index] for s in range(self.n_seg)])
        return torch.where(self.same_seg, within, cross)

    def forward(self, x):
        q, k, v, u = self.qkvu(x).chunk(4, dim=-1)
        score = q @ k.transpose(-2, -1) / math.sqrt(self.dim) + self.bias().unsqueeze(0)
        av = torch.sigmoid(score) @ v
        y = self.dropout(self.out(self.av_norm(av) * self.act(u)))
        return self.residual_norm(x + y)


class TFSplitPipeline(nn.Module):
    """Time stream (I/Q + amp/phase, time positions) || frequency stream (FFT, frequency positions); OOB keys/values
    carry the same frequency positions as the FFT stream. stem_dim 16 per stream keeps parameters ~= legacy (32 shared)."""

    def __init__(self, input_fmax=None, oob_keep=None, use_oob=True, stem_dim=16, depth=2, num_classes=10):
        super().__init__()
        P = WINDOW // PATCH
        self.P, self.input_fmax, self.use_oob = P, input_fmax, use_oob
        self.time_stem = CNNStemPatchEmbed(4, DIM, PATCH, stem_dim, "7,5")
        self.freq_stem = CNNStemPatchEmbed(1, DIM, PATCH, stem_dim, "7,5")
        self.proj_t, self.proj_f = nn.Linear(DIM, DIM), nn.Linear(DIM, DIM)
        self.pos_t = nn.Parameter(torch.zeros(1, P, DIM))
        self.pos_f = nn.Parameter(torch.zeros(1, P, DIM))
        if use_oob:
            self.oob_proj = nn.Linear(PATCH, DIM)
            self.fusion = OOBCrossAttentionFusion(DIM, num_heads=4)
            m = torch.ones(1, 1, WINDOW) if oob_keep is None else oob_keep_mask(WINDOW, *oob_keep).view(1, 1, -1)
            self.register_buffer("oob_mask", patchify(m, PATCH), persistent=False)
        self.blocks = nn.ModuleList([SegmentHSTUBlock(DIM, P, 2) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(DIM)
        self.classifier = nn.Linear(DIM, num_classes)

    def forward(self, x):
        x = rms_normalize(x)
        if self.input_fmax is not None:
            x = band_limit(x, self.input_fmax)
        iq_v, fft_v, oob_v, ap_v = torch_rf_views(x, fft_norm="log_zscore", oob_norm="ratio")
        t = self.proj_t(self.time_stem(torch.cat([iq_v, ap_v], 1))) + self.pos_t
        f = self.proj_f(self.freq_stem(fft_v)) + self.pos_f
        h = torch.cat([t, f], 1)
        if self.use_oob:
            o = self.oob_proj(patchify(oob_v, PATCH) * self.oob_mask) + self.pos_f
            h = self.fusion(h, o)
        for blk in self.blocks:
            h = blk(h)
        return self.classifier(self.final_norm(h).mean(1))


class ShenRes(nn.Module):
    def __init__(self, cin, f, first=False):
        super().__init__()
        self.c1 = nn.Conv2d(cin, f, 3, padding=1)
        self.c2 = nn.Conv2d(f, f, 3, padding=1)
        self.sc = nn.Conv2d(cin, f, 1) if first else None

    def forward(self, x):
        fx = self.c2(F.relu(self.c1(x)))
        return F.relu((self.sc(x) if self.sc is not None else x) + fx)


class ShenCISPipeline(nn.Module):
    """non_adv_net of Shen et al. (deep_learning_models.py). head='flatten': AvgPool2 -> Flatten(13*31*64) -> Dense512;
    head='gap': global average pool -> Dense512; head='rank16': flatten -> Dense16 (no bias) -> Dense512 (the rank-16
    capacity control of gai src/shen_p2a/models.py::FactorizedAuthorCNN, 658,314 params; keeps position information).
    Then L2-normalise -> Dense128 -> ReLU -> Dense(classes)."""

    def __init__(self, head="flatten", input_fmax=None, num_classes=10):
        super().__init__()
        self.head, self.input_fmax = head, input_fmax
        self.stem = nn.Conv2d(1, 32, 7, stride=2)
        self.blocks = nn.Sequential(ShenRes(32, 32), ShenRes(32, 32), ShenRes(32, 64, True), ShenRes(64, 64))
        flat = 13 * 31 * 64
        if head == "flatten":
            self.feat = nn.Linear(flat, 512)
        elif head == "gap":
            self.feat = nn.Linear(64, 512)
        elif head == "rank16":
            self.feat = nn.Sequential(nn.Linear(flat, 16, bias=False), nn.Linear(16, 512))
        else:
            raise ValueError(head)
        self.tx_input = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def features(self, cis):
        h = F.relu(self.stem(F.pad(cis, (2, 3, 2, 3))))   # TF 'same' for k7/s2 on 52x126
        h = self.blocks(h)
        if self.head == "gap":
            h = h.mean((2, 3))
        else:
            h = F.avg_pool2d(h, 2).permute(0, 2, 3, 1).reshape(h.shape[0], -1)   # Keras channels-last flatten
        return F.normalize(self.feat(h), p=2, dim=1, eps=1e-12)

    def forward(self, x):
        return self.classifier(F.relu(self.tx_input(self.features(shen_cis(x, f_max=self.input_fmax)))))


def build_pipeline(arm: dict) -> nn.Module:
    fam = arm["family"]
    if fam in ("paper_rf", "paper_upstream"):
        from .paper_models import build_paper_model
        return build_paper_model(arm)
    if fam == "crossattn":
        return CrossAttnPipeline(fusion=arm.get("fusion", "cross_attn_oob"), input_fmax=arm.get("input_fmax"),
                                 oob_keep=arm.get("oob_keep"))
    if fam == "tfsplit":
        return TFSplitPipeline(input_fmax=arm.get("input_fmax"), oob_keep=arm.get("oob_keep"),
                               use_oob=arm.get("fusion", "cross_attn_oob") != "no_oob")
    if fam == "shen_cis":
        return ShenCISPipeline(head=arm.get("head", "flatten"), input_fmax=arm.get("input_fmax"))
    from .models2 import build_pipeline2   # stage-02 families (strict-IB base variants)
    return build_pipeline2(arm)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
