"""Data-free design correction: full-rank smoothing and affine-only normalization."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from s22_core import HERE
from s21_models import build as old_build, OverlapConv, geometry, shared_hash

ARMS=['H0','P0','H1','H2','S1','S2']
REFERENCES=['H0','P0','H1','H2']

class SmoothConv(OverlapConv):
    def __init__(self, original):
        super().__init__(original)
        self.effective_kernel_size=256;self.effective_padding=0
    def smoothed_weight(self):
        expanded=self.expanded_weight().reshape(-1,1,340)
        return F.interpolate(expanded,size=256,mode='linear',align_corners=False).reshape_as(self.weight)
    def forward(self,x):
        return F.conv1d(x,self.smoothed_weight(),self.bias,stride=256,groups=self.groups)

class ChannelAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(channels));self.bias=nn.Parameter(torch.zeros(channels))
    def forward(self,x):return x*self.weight[None,:,None]+self.bias[None,:,None]

def build(arm,seed):
    if arm not in ARMS:raise ValueError(arm)
    model=old_build('P0' if arm=='S1' else 'H0' if arm=='S2' else arm,seed)
    tokenizer=model.net.embedder.cnn_stem.tokenize
    if arm=='S1':tokenizer[0]=SmoothConv(tokenizer[0])
    if arm=='S2':
        count=0
        for i,layer in enumerate(tokenizer):
            if isinstance(layer,nn.BatchNorm1d):
                assert torch.equal(layer.weight,torch.ones_like(layer.weight)) and torch.equal(layer.bias,torch.zeros_like(layer.bias))
                tokenizer[i]=ChannelAffine(layer.num_features);count+=1
        assert count==3
    return model

def parameterization_audit():
    basis=torch.eye(256,dtype=torch.float64)[:,None,:]
    expanded=F.interpolate(basis,340,mode='linear',align_corners=False)*math.sqrt(256/340)
    result={}
    for name,matrix in [('proposed_center_crop',expanded[...,42:298].squeeze(1)),('selected_round_trip',F.interpolate(expanded,256,mode='linear',align_corners=False).squeeze(1))]:
        singular=torch.linalg.svdvals(matrix);inactive=(matrix.abs().sum(1)==0).nonzero().flatten().tolist()
        result[name]=dict(rank=int((singular>1e-10).sum()),inactive_base_taps=inactive,inactive_parameter_count=len(inactive)*80*4,singular_max=float(singular.max()),singular_min=float(singular.min()))
    assert result['proposed_center_crop']['rank']==194 and result['selected_round_trip']['rank']==256
    return result

def verify_architecture(cuda=False):
    torch.set_num_threads(2);torch.manual_seed(22022);audits={}
    for arm in ARMS:
        m=build(arm,123).eval();audits[arm]=geometry(m)
        with torch.no_grad():y=m.net.embedder.cnn_stem(torch.randn(2,5,8192))
        assert y.shape==(2,32,64);audits[arm]['token_shape']=list(y.shape)
        assert audits[arm]['first_token_center']==127.5 and audits[arm]['token_stride']==256
    assert audits['S1']['params']==audits['P0']['params']==audits['H1']['params']==134266
    assert audits['S2']['params']==audits['H0']['params']==133690
    assert audits['S1']['cnn_receptive_field']==266 and audits['S2']['cnn_receptive_field']==350
    assert audits['S1']['bn_layers']==audits['S2']['bn_layers']==2
    for arm,ref in [('S1','P0'),('S2','H0')]:
        m,b=build(arm,123),build(ref,123)
        assert shared_hash(m)==shared_hash(b)
        for key,value in m.state_dict().items():assert torch.equal(value,b.state_dict()[key]),key
        assert sum(isinstance(v,ChannelAffine) for v in m.modules())==(3 if arm=='S2' else 0)
    for device in (['cpu','cuda'] if cuda else ['cpu']):
        m=build('S1',123).to(device);layer=m.net.embedder.cnn_stem.tokenize[0];x=torch.randn(2,32,8192,device=device)
        out=layer(x);out.square().mean().backward();grad=layer.weight.grad
        assert torch.isfinite(grad).all() and (grad.abs().sum((0,1))>0).all()
        expected=F.interpolate(F.interpolate(layer.weight.reshape(-1,1,256),340,mode='linear',align_corners=False)*math.sqrt(256/340),256,mode='linear',align_corners=False).reshape_as(layer.weight)
        assert torch.equal(out,F.conv1d(x,expected,layer.bias,stride=256,groups=8))
        m=build('S2',123).to(device);tokenizer=m.net.embedder.cnn_stem.tokenize
        tokenizer(torch.randn(2,32,8192,device=device)).square().mean().backward()
        for layer in tokenizer:
            if isinstance(layer,ChannelAffine):
                assert not list(layer.buffers()) and torch.isfinite(layer.weight.grad).all() and torch.isfinite(layer.bias.grad).all()
    return audits
