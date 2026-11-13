"""Only replace token mixers; preserve canonical HStem and classification head."""
import math
import torch
from torch import nn
from core import ROOT
from xa.models3 import ChirpPipeline
from legacy_rfhstu.models import RFHSTUBlock

ARMS = {
    "A0_cnn": dict(kind="none"),
    "A1_mha": dict(kind="mha", inner=64, heads=4),
    "A1_mha_matched": dict(kind="mha", inner=80, heads=4),
    "A2_transformer": dict(kind="transformer", ff=256, heads=4),
    "A2_transformer_matched": dict(kind="transformer", ff=32, heads=4),
    "A3_rfhstu": dict(kind="gated", score="sigmoid"),
    "A4_softmax_gate": dict(kind="gated", score="softmax"),
    "A5_silu_gate": dict(kind="gated", score="silu"),
}


class GatedScore(RFHSTUBlock):
    def __init__(self, score):
        super().__init__(64, 32)
        self.score = score

    def forward(self, x):
        q, k, v, u = self.qkvu(x).chunk(4, dim=-1)
        score = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
        score = score + self.relative_bias[self.relative_index].unsqueeze(0)
        if self.score == "softmax":
            weights = score.softmax(-1)
        elif self.score == "silu":
            weights = torch.nn.functional.silu(score)
        else:
            weights = score.sigmoid()
        y = self.dropout(self.out(self.av_norm(weights @ v) * self.act(u)))
        return self.residual_norm(x + y)


class PlainMHA(nn.Module):
    """Ordinary multihead softmax attention; independent projected width for cost matching."""
    def __init__(self, inner=64, heads=4):
        super().__init__()
        if inner % heads:
            raise ValueError("attention width must be divisible by heads")
        self.inner, self.heads = inner, heads
        self.qkv = nn.Linear(64, 3 * inner)
        self.out = nn.Linear(inner, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.inner // self.heads).unbind(2)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.inner // self.heads)).softmax(-1)
        value = (weights @ v).transpose(1, 2).reshape(b, t, self.inner)
        return self.norm(x + self.out(value))


def build(arm, seed):
    if arm not in ARMS:
        raise ValueError(arm)
    torch.manual_seed(seed)
    canonical = ChirpPipeline(stem="hierarchical", preln=True, depth=2)
    cfg = ARMS[arm]
    if arm == "A3_rfhstu":
        return canonical
    # Isolate replacement RNG so all non-mixer parameters match A3 exactly.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if cfg["kind"] == "none":
            blocks = nn.ModuleList()
        elif cfg["kind"] == "gated":
            blocks = nn.ModuleList([GatedScore(cfg["score"]) for _ in range(2)])
            for new, old in zip(blocks, canonical.net.encoder.blocks):
                new.load_state_dict(old.state_dict())
        elif cfg["kind"] == "mha":
            blocks = nn.ModuleList([PlainMHA(cfg["inner"], cfg["heads"]) for _ in range(2)])
        else:
            blocks = nn.ModuleList([nn.TransformerEncoderLayer(
                64, cfg["heads"], dim_feedforward=cfg["ff"], dropout=0., activation="gelu",
                batch_first=True, norm_first=False) for _ in range(2)])
        canonical.net.encoder.blocks = blocks
    return canonical


def shared_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith("net.encoder.blocks.")}


def nparams(model):
    return sum(p.numel() for p in model.parameters())
