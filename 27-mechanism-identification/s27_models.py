"""Reuse S21's unexecuted H3 definition; freeze a separate B readout design."""
import hashlib
import torch
from torch import nn
from s27_common import ROOT
from s26_models import build as prior_build, geometry, shared_hash
from s21_models import build as s21_build
from xa.train import stable_seed


class TokenMLP(nn.Module):
    """Positionwise residual MLP; no cross-token interaction before pooling."""
    def __init__(self):
        super().__init__()
        self.up=nn.Linear(64,128);self.act=nn.GELU();self.down=nn.Linear(128,64);self.norm=nn.LayerNorm(64)
    def forward(self,x):return self.norm(x+self.down(self.act(self.up(x))))


def build(arm,seed):
    if arm=='LA':return s21_build('H3',seed)
    if arm in ['H0','P0','LP']:return prior_build(arm,seed)
    if arm in ['H0_P','LP_P']:
        model=prior_build(arm.split('_')[0],seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(stable_seed('S27B-Readout',seed))
            model.net.encoder.blocks=nn.ModuleList([TokenMLP(),TokenMLP()])
        return model
    raise ValueError(arm)


def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():
        h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
