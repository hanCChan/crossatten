"""Design-only frequency residuals; loading real checkpoints/training is external."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
for relative in ('lib','13-mixer-cfo-mechanisms','17-rx-distortion'):
    path=str(ROOT/relative)
    if path not in sys.path:sys.path.insert(0,path)
from s17_models import build as build_original
from xa.frontends import rms_normalize

ARMS=('F0','F1','F2')


class SpectralResidual(nn.Module):
    """F1: ordinary wide convolution+MLP. F2: parallel fine/wide convolutions."""
    def __init__(self,arm,seed):
        super().__init__()
        if arm not in ('F1','F2'):raise ValueError('Only the fixed F1/F2 design is registered')
        self.arm=arm
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+101)
            self.reduce=nn.Conv1d(32,8,1,bias=False)
            torch.manual_seed(seed+102)
            self.wide=nn.Conv1d(8,8,33,padding=16,bias=False)
            torch.manual_seed(seed+103)
            if arm=='F2':
                self.fine=nn.Conv1d(8,8,3,padding=1,bias=False)
                self.merge=nn.Conv1d(16,8,1,bias=False)
            else:
                self.channel_in=nn.Conv1d(8,20,1,bias=False)
                self.channel_out=nn.Conv1d(20,8,1,bias=False)
            torch.manual_seed(seed+104)
            self.output=nn.Conv1d(8,64,1,bias=False)
        # Only alpha is zero. Zeroing output as well would kill all initial gradients.
        self.alpha=nn.Parameter(torch.zeros(()))

    def forward(self,h):
        x=self.reduce(h)
        coarse=F.gelu(self.wide(x))
        if self.arm=='F2':
            z=F.gelu(self.merge(torch.cat((F.gelu(self.fine(x)),coarse),dim=1)))
        else:
            z=self.channel_out(F.gelu(self.channel_in(coarse)))
        delta=self.output(F.avg_pool1d(z,kernel_size=8,stride=8)).transpose(1,2)
        return self.alpha*delta


class FrozenFrequencyModel(nn.Module):
    def __init__(self,arm,backbone_seed,side_seed=None,backbone_state=None):
        super().__init__()
        if arm not in ARMS:raise ValueError('Unregistered architecture')
        self.arm=arm
        self.backbone=build_original('ManyRx',backbone_seed)
        if backbone_state is not None:
            # Accept the exact old SpanModel-P0 wrapper or the underlying backbone.
            keys=list(backbone_state)
            if keys and all(k.startswith('backbone.') for k in keys):
                backbone_state={k.removeprefix('backbone.'):v for k,v in backbone_state.items()}
            self.backbone.load_state_dict(backbone_state,strict=True)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.residual=None if arm=='F0' else SpectralResidual(arm,backbone_seed if side_seed is None else side_seed)

    def train(self,mode=True):
        super().train(mode)
        # Freeze BN running moments as well as parameters; never adapt to a new RX.
        self.backbone.eval()
        return self

    def tokens(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,256):raise ValueError('Expected full IQ [B,2,256]')
        m=self.backbone
        # Exact P0 normalization/features. No new time view, filter or phase rule.
        v=m.features(rms_normalize(x))
        h=m.frequency.stem(v)
        base=m.frequency.tokenize(h).transpose(1,2)
        return base if self.residual is None else base+self.residual(h)

    def forward(self,x):
        m=self.backbone
        h=m.pre_ln(m.encoder.prepare_tokens(self.tokens(x)))
        for block in m.encoder.blocks:h=block(h)
        return m.classifier(m.encoder.final_norm(h).mean(1))


def build(arm,backbone_seed,side_seed=None,backbone_state=None):
    return FrozenFrequencyModel(arm,backbone_seed,side_seed,backbone_state)
