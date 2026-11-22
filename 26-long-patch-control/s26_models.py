"""One chosen freely learned long-patch arm; no candidate accuracy search."""
import torch
from torch import nn
from s26_common import ROOT
from s21_models import build as archived_build, geometry, shared_hash
from xa.train import stable_seed


def build(arm,seed):
    if arm in ['H0','P0']:
        return archived_build(arm,seed)
    if arm!='LP':
        raise ValueError(arm)
    model=archived_build('P0',seed)
    # Preserve P0's surviving shared initialization and both its CPU and CUDA
    # RNG state. Only the new tokenizer receives its own fixed CPU init stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stable_seed('S26-LongPatch',seed))
        model.net.embedder.cnn_stem.tokenize=nn.Sequential(
            nn.Conv1d(32,64,340,stride=256,padding=42,groups=8),nn.Conv1d(64,64,1))
    return model
