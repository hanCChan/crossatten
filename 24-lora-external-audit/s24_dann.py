"""Audited candidate port, NOT yet cross-framework-validated or approved for fits."""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
from xa.models import ShenCISPipeline
from xa.frontends import band_limit, shen_cis


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -grad


class ShenDANN(ShenCISPipeline):
    def __init__(self, strict_ib=False, num_source_rx=5):
        super().__init__(head='flatten', input_fmax=None, num_classes=10)
        self.strict_ib = strict_ib
        self.rx_input = nn.Linear(512, 128)
        self.rx_classifier = nn.Linear(128, num_source_rx)
        for layer in (self.rx_input, self.rx_classifier):
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

    def forward_heads(self, x):
        if self.strict_ib: x = band_limit(x, 62500)
        feature = self.features(shen_cis(x))
        tx = self.classifier(F.relu(self.tx_input(feature)))
        rx = self.rx_classifier(F.relu(self.rx_input(Reverse.apply(feature))))
        return tx, rx

    def forward(self, x):
        # Inference is always per-packet TX classification without receiver fusion.
        return self.forward_heads(x)[0]
