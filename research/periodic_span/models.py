"""Time-coordinate periodic connections with the original 32-token MHA head."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
for relative in ("lib","13-mixer-cfo-mechanisms","17-rx-distortion"):
    p=str(ROOT/relative)
    if p not in sys.path:sys.path.insert(0,p)
from s17_models import build as original_build
from xa.frontends import rms_normalize
from xa.explore_models import phase_product

ARMS=("P0","P1","P2","N19")


def time_features(x):
    """Existing S07 differential time view: IQ, log amplitude, lag1 Re/Im."""
    x=rms_normalize(x)
    amplitude=torch.sqrt(x[:,0].square()+x[:,1].square()+1e-6)
    return torch.cat((x,torch.log1p(amplitude)[:,None],phase_product(x,1)),1)


def valid_counts(length,kernel,dilation,device=None,dtype=torch.float32):
    if length<1 or kernel<1 or kernel%2!=1 or dilation<1:
        raise ValueError("Positive length/dilation and odd kernel required")
    offsets=(torch.arange(kernel,device=device)-kernel//2)*dilation
    index=torch.arange(length,device=device)[:,None]+offsets[None]
    return ((index>=0)&(index<length)).sum(1).to(dtype)[None,None]


def masked_spatial(x,weight,dilation):
    """No wrap; divide only by the number of in-range taps, not padded zeros."""
    kernel=weight.shape[-1]
    result=F.conv1d(x,weight,padding=(kernel//2)*dilation,dilation=dilation)
    return result/valid_counts(x.shape[-1],kernel,dilation,x.device,x.dtype).clamp_min(1)


class ResidualSpan(nn.Module):
    def __init__(self,arm,seed):
        super().__init__()
        if arm not in ARMS[1:]:raise ValueError("Unregistered span arm")
        self.kernel=41 if arm=="P1" else 3
        self.dilation={"P1":1,"P2":20,"N19":19}[arm]
        # Separate constructor RNGs keep common layers identical despite kernel sizes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+501)
            self.reduce=nn.Conv1d(32,4,1,bias=False)
            torch.manual_seed(seed+502)
            self.spatial=nn.Conv1d(4,4,self.kernel,padding=(self.kernel//2)*self.dilation,
                                   dilation=self.dilation,bias=False)
            torch.manual_seed(seed+503)
            self.expand=nn.Conv1d(4,32,1,bias=False)
            self.gate=nn.Conv1d(32,32,1)
        nn.init.zeros_(self.expand.weight)
        nn.init.constant_(self.gate.bias,-2.)
        self.register_buffer("counts",valid_counts(256,self.kernel,self.dilation),persistent=False)

    def forward(self,h):
        z=self.reduce(h)
        counts=self.counts if h.shape[-1]==256 else valid_counts(h.shape[-1],self.kernel,self.dilation,h.device,h.dtype)
        # W_k=kernel*theta_k keeps standard fan-in variance in the interior for
        # both 3 and 41 taps, while retaining the requested valid-count mean.
        periodic=F.gelu(self.spatial(z)*(self.kernel/counts.clamp_min(1)))
        return h+torch.sigmoid(self.gate(h))*self.expand(periodic)


class SpanModel(nn.Module):
    def __init__(self,arm,seed):
        super().__init__()
        if arm not in ARMS:raise ValueError("Unregistered architecture")
        self.arm=arm
        self.backbone=original_build("ManyRx",seed)
        self.span=None if arm=="P0" else ResidualSpan(arm,seed)

    def tokens(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError("Expected full IQ [B,2,256]")
        if self.span is None:return self.backbone.tokens(x)
        # Dilation is in original time samples, BEFORE the three stride-2 layers.
        stem=self.backbone.frequency
        local=stem.stem(time_features(x))
        return stem.tokenize(self.span(local)).transpose(1,2)

    def forward(self,x):
        if self.span is None:return self.backbone(x)
        m=self.backbone
        h=m.pre_ln(m.encoder.prepare_tokens(self.tokens(x)))
        for block in m.encoder.blocks:h=block(h)
        return m.classifier(m.encoder.final_norm(h).mean(1))


def build(arm,seed):return SpanModel(arm,seed)


def common_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith("span.")}
