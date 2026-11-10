"""Fixed small HStem+ordinary MHA; no HSTU/Qwen training."""
import torch
from torch import nn
from s17_core import ROOT
from wifi import build_wifi
from models import PlainMHA

def build(dataset,seed):
    m=build_wifi(dataset,'B0',seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);m.encoder.blocks=nn.ModuleList([PlainMHA(64,4) for _ in range(2)])
    return m
