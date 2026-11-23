"""One reserved factorial cell; no architecture search or training entry."""
import torch
from torch import nn
from s28_core import ROOT
from s27_models import build as prior_build
from s21_models import build as s21_build,geometry
from s27_models import state_hash
from s15_models import shared_hash

ARMS=['H0','LA','H2','H2_LA','LP']

def build(arm,seed):
    if arm=='H2':return s21_build('H2',seed)
    if arm!='H2_LA':return prior_build(arm,seed)
    model=prior_build('H0',seed);layers=model.net.embedder.cnn_stem.tokenize
    assert [i for i,m in enumerate(layers) if isinstance(m,nn.BatchNorm1d)]==[1,4,7]
    assert [i for i,m in enumerate(layers) if isinstance(m,nn.GELU)]==[2,5,8]
    for i in [1,2,4,5,7,8]:layers[i]=nn.Identity()
    return model

def surviving_equal(reference,control):
    a,b=reference.state_dict(),control.state_dict()
    return all(k in a and torch.equal(v,a[k]) for k,v in b.items())
