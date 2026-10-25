#!/usr/bin/env python3
"""Count parameters / FLOPs of third-party SOTA backbones for the literature table (no training, no data).
timm is only used by IB-RFF for trunc_normal_; a shim maps it to torch.nn.init.trunc_normal_ (same defaults)."""
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

EXT = Path(__file__).resolve().parent
shim = Path(tempfile.mkdtemp()) / "timm"
shim.mkdir()
(shim / "__init__.py").write_text("")
(shim / "layers.py").write_text("from torch.nn.init import trunc_normal_\n")
sys.path[:0] = [str(shim.parent), str(EXT / "IB-RFF")]

from backbones.FAN import FAN  # noqa: E402


def count(m, x):
    m.eval()
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        out = m(x)
    return sum(p.numel() for p in m.parameters()), fc.get_total_flops() / 1e6, out


for name, ps, ncls in (("ManyRx", 16, 10), ("ManySig", 32, 6)):
    for var, saf, faf in (("FAN", True, True), ("FAN_C (no SAF/FAF)", False, False)):
        m = FAN(seq_len=256, in_chans=2, patch_size=ps, embed_dim=128, mlp_ratio=2.0, num_classes=ncls,
                bottleneck_type="DSCNN", use_SAF=saf, use_FAF=faf)
        n, mf, (h, y) = count(m, torch.randn(1, 2, 256))
        print(f"IB-RFF {name:7s} patch={ps:2d} {var:19s} params={n:,} MFLOPs/pkt={mf:.2f} tokens={h.shape[1]} logits={tuple(y.shape)}")

# WaveMLP (official configs: ManyRx J=5,K=32 ; ManySig J=4,K=16). Its package is also named 'backbones' -> isolate.
for k in [k for k in sys.modules if k == "backbones" or k.startswith("backbones.")]:
    del sys.modules[k]
sys.path.remove(str(EXT / "IB-RFF"))
sys.path.insert(0, str(EXT / "WaveMLP-RFF"))
from backbones.MyModel import MyModel  # noqa: E402

for name, lv, k, ncls in (("ManyRx", 5, 32, 10), ("ManySig", 4, 16, 6)):
    m = MyModel(num_classes=ncls, use_wavelet=True, wavelet_levels=lv, wavelet_learnable=True, wavelet_init="randn",
                wavelet_kernel_size=k)
    n, mf, (y, _, _) = count(m, torch.randn(1, 2, 256))
    print(f"WaveMLP {name:7s} J={lv} K={k:2d} params={n:,} MFLOPs/pkt(real-valued ops counted by torch)={mf:.2f} logits={tuple(y.shape)}")
