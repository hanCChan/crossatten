"""Inference-only algebraic transformations; no optimizer or refitting."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from s23_core import ROOT
from s22_models import build

def operator():
    basis=torch.eye(256,dtype=torch.float64)[:,None,:]
    up=F.interpolate(basis,340,mode='linear',align_corners=False)*math.sqrt(256/340)
    # v(column)=T*w(column); row weights obey v=row_w @ T.T.
    return F.interpolate(up,256,mode='linear',align_corners=False).squeeze(1).T.contiguous()

def materialize_s1(model):
    m=copy.deepcopy(model).eval();old=m.net.embedder.cnn_stem.tokenize[0]
    weight=old.smoothed_weight().detach()
    conv=nn.Conv1d(old.in_channels,old.out_channels,256,stride=256,groups=old.groups,bias=True,device=weight.device,dtype=weight.dtype)
    with torch.no_grad():conv.weight.copy_(weight);conv.bias.copy_(old.bias)
    m.net.embedder.cnn_stem.tokenize[0]=conv
    return m.eval()

def fold_h0(model):
    m=copy.deepcopy(model).eval();tokenizer=m.net.embedder.cnn_stem.tokenize;count=0
    for conv_index,bn_index in [(0,1),(3,4),(6,7)]:
        conv,bn=tokenizer[conv_index],tokenizer[bn_index]
        assert isinstance(conv,nn.Conv1d) and isinstance(bn,nn.BatchNorm1d) and not bn.training
        tokenizer[conv_index]=torch.nn.utils.fuse_conv_bn_eval(conv,bn)
        tokenizer[bn_index]=nn.Identity();count+=1
    assert count==3
    return m.eval()

def shifted(x,amount,kind):
    if kind=='circular':return x.roll(amount,-1)
    if kind!='zero':raise ValueError(kind)
    result=torch.zeros_like(x)
    if amount>0:result[...,amount:]=x[...,:-amount]
    elif amount<0:result[...,:amount]=x[...,-amount:]
    else:result.copy_(x)
    return result

def difference(a,b):
    dtype=torch.complex128 if a.is_complex() or b.is_complex() else torch.float64
    d=(a-b).detach().to(dtype);scale=b.detach().to(dtype)
    return dict(max_abs=float(d.abs().max()),rms_abs=float(d.abs().square().mean().sqrt()),relative_l2=float(d.norm()/scale.norm().clamp_min(1e-15)))

def equivalence(a,b,atol=1e-4,rtol=1e-4):
    return dict(**difference(a,b),allclose=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),predictions_equal=int((a.argmax(1)==b.argmax(1)).sum()),packets=len(a),atol=atol,rtol=rtol)
