"""Fixed, parameter-preserving overlap intervention and three-BN ablation."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from s21_core import ROOT
from s15_models import build as previous_build, shared_hash

class OverlapConv(nn.Conv1d):
    # State is still the original [80,4,256] learned kernel. No extra RNG draw.
    def __init__(self, original):
        nn.Module.__init__(self)
        for key in ['in_channels','out_channels','groups','dilation','stride','kernel_size','padding','padding_mode']:
            setattr(self, key, getattr(original, key))
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())
        self.effective_kernel_size = 340
        self.effective_padding = 42
    def expanded_weight(self):
        w = F.interpolate(self.weight.reshape(-1,1,256), size=340, mode='linear', align_corners=False)
        return w.reshape(self.out_channels, self.in_channels//self.groups,340)*math.sqrt(256/340)
    def forward(self, x):
        return F.conv1d(x,self.expanded_weight(),self.bias,stride=256,padding=42,groups=self.groups)

def build(arm, seed):
    if arm not in ['H0','P0','H1','H2','H3']: raise ValueError(arm)
    m = previous_build('C2_lite_mha' if arm in ['P0','H1'] else 'A1_mha', seed)
    tokenizer = m.net.embedder.cnn_stem.tokenize
    if arm == 'H1': tokenizer[0] = OverlapConv(tokenizer[0])
    if arm in ['H2','H3']:
        kind = nn.BatchNorm1d if arm == 'H2' else nn.GELU
        count = 0
        for i, layer in enumerate(tokenizer):
            if isinstance(layer, kind): tokenizer[i] = nn.Identity(); count += 1
        assert count == 3
    return m

def geometry(m):
    field=1; jump=1; center=0.; n=8192; layers=[]
    stem=m.net.embedder.cnn_stem
    for name, layer in list(stem.stem.named_modules())+[(f'tokenize.{k}',v) for k,v in stem.tokenize.named_modules()]:
        if not isinstance(layer, nn.Conv1d): continue
        k=getattr(layer,'effective_kernel_size',layer.kernel_size[0]); p=getattr(layer,'effective_padding',layer.padding[0])
        s=layer.stride[0]; d=layer.dilation[0]
        center+=((k-1)*d/2-p)*jump; field+=(k-1)*d*jump; jump*=s
        n=(n+2*p-d*(k-1)-1)//s+1
        layers.append(dict(name=name,kernel=k,padding=p,stride=s,output_length=n))
    return dict(params=sum(p.numel() for p in m.parameters()),cnn_receptive_field=field,token_stride=jump,
                overlap=field-jump,first_token_center=center,tokens=n,bn_layers=sum(isinstance(v,nn.BatchNorm1d) for v in m.modules()),layers=layers,
                scope='Local CNN coordinates only; global FFT and TrueIB prevent an end-to-end physical support interpretation')

def verify_architecture():
    torch.set_num_threads(2); torch.manual_seed(21021)
    x=torch.randn(2,5,8192); audits={}
    for arm in ['H0','P0','H1','H2','H3']:
        m=build(arm,123).eval(); audits[arm]=geometry(m)
        with torch.no_grad(): out=m.net.embedder.cnn_stem(x)
        assert tuple(out.shape)==(2,32,64)
        audits[arm]['token_shape']=list(out.shape)
        assert audits[arm]['first_token_center']==127.5 and audits[arm]['token_stride']==256
    assert audits['H1']['params']==audits['P0']['params']==134266
    assert audits['H0']['params']==133690 and audits['H2']['params']==133338
    assert all(audits[a]['cnn_receptive_field']==350 for a in ['H0','H1','H2','H3'])
    assert audits['P0']['cnn_receptive_field']==266
    for arm, ref in [('H1','P0'),('H2','H0'),('H3','H0')]:
        a,b=build(arm,123),build(ref,123)
        assert shared_hash(a)==shared_hash(b)
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    a=build('H1',123); inp=torch.randn(2,32,8192)
    a.net.embedder.cnn_stem.tokenize(inp).square().mean().backward()
    grad=a.net.embedder.cnn_stem.tokenize[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    # Numerical equality to the prescribed functional operation.
    layer=a.net.embedder.cnn_stem.tokenize[0]
    expected=F.conv1d(inp,F.interpolate(layer.weight.reshape(-1,1,256),340,mode='linear',align_corners=False).reshape(80,4,340)*math.sqrt(256/340),layer.bias,stride=256,padding=42,groups=8)
    assert torch.equal(layer(inp),expected)
    return audits
