"""Matched residual channel/token gates on the unchanged frequency backbone."""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch import nn
from quality import proxies

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('quality_frozen_C',ROOT/'research/standard_method_competition_c/engine.py')
C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
A=C.A
ARMS=('B1','B2','N0')


def context(x):
    f,q,_=proxies(x)
    return np.column_stack((f[:,0],f[:,1],f[:,2],q.std(1))).astype(np.float32)


def derangement(n,seed):
    """Sattolo cycle: full role, no self-matched quality, independent of batching."""
    if n<2:raise ValueError('Need at least two packets')
    p=np.arange(n);rng=np.random.default_rng(seed)
    for i in range(n-1,0,-1):
        j=int(rng.integers(i));p[i],p[j]=p[j],p[i]
    assert np.all(p!=np.arange(n))
    return p


def packed(x,arm,seed):
    q=context(x)
    p=derangement(len(x),seed) if arm=='N0' else np.arange(len(x))
    return np.column_stack((x.reshape(len(x),512),q[p])).astype(np.float32),p


class QualityGate(nn.Module):
    def __init__(self,arm,seed):
        super().__init__()
        if arm not in ARMS:raise ValueError(arm)
        self.arm=arm
        self.backbone=A.build_model(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(28100+seed)
            self.gate=nn.Sequential(nn.Linear(68,8),nn.GELU(),nn.Linear(8,64))
        # Identity at initialization, yet all parameters remain in the optimizer.
        nn.init.zeros_(self.gate[-1].weight);nn.init.zeros_(self.gate[-1].bias)

    def forward(self,inputs):
        if inputs.ndim!=2 or inputs.shape[1]!=516:raise ValueError('Need packed full IQ plus four context values')
        x=inputs[:,:512].reshape(-1,2,256)
        h=self.backbone.tokens(x)
        if self.arm=='B1':
            extra=torch.stack((h.mean(-1).sigmoid(),h.square().mean(-1).sqrt().tanh(),
                               h.abs().amax(-1).tanh(),h.std(-1,unbiased=False).tanh()),-1)
        else:
            extra=inputs[:,512:][:,None,:].expand(-1,h.shape[1],-1)
        weights=1+.5*torch.tanh(self.gate(torch.cat((h,extra),-1)))
        h=h*weights
        m=self.backbone
        h=m.pre_ln(m.encoder.prepare_tokens(h))
        for block in m.encoder.blocks:h=block(h)
        return m.classifier(m.encoder.final_norm(h).mean(1))
