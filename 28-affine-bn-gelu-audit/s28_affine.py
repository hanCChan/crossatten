"""Exact finite-length affine composition, including each intermediate crop.

This maps the 32-channel shared-stem output, not raw IQ. Coefficients are
constructed in CPU float64 by local adjoints. No huge dense matrix is built.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class MaterializedTokenizer(nn.Module):
    def __init__(self,tokenizer,length=8192):
        super().__init__()
        self.length=length
        reference=copy.deepcopy(tokenizer).cpu().double().eval()
        folded=[]
        for module in reference:
            if isinstance(module,nn.Conv1d):
                assert module.padding_mode=='zeros' and module.dilation==(1,)
                folded.append(dict(weight=module.weight.detach().clone(),bias=module.bias.detach().clone() if module.bias is not None else torch.zeros(module.out_channels,dtype=torch.float64),
                    stride=module.stride[0],padding=module.padding[0],groups=module.groups))
            elif isinstance(module,nn.BatchNorm1d):
                assert not module.training and module.track_running_stats and folded
                scale=module.weight.detach()/torch.sqrt(module.running_var+module.eps)
                offset=module.bias.detach()-scale*module.running_mean
                folded[-1]['weight']*=scale[:,None,None]
                folded[-1]['bias']=folded[-1]['bias']*scale+offset
            elif not isinstance(module,nn.Identity):raise TypeError('Non-affine tokenizer module: '+type(module).__name__)
        sizes=[length];support=1;jump=1;pad=0
        for layer in folded:
            k=layer['weight'].shape[-1];s=layer['stride'];p=layer['padding']
            sizes.append((sizes[-1]+2*p-k)//s+1)
            support+=(k-1)*jump;pad+=p*jump;jump*=s
        assert sizes==[8192,2048,512,128,32] and (support,jump,pad)==(340,256,42)
        self.support,self.stride,self.padding=support,jump,pad
        boundaries=[]
        for t in range(sizes[-1]):
            lo=hi=t;edge=False
            for layer,n in zip(reversed(folded),reversed(sizes[:-1])):
                lo=lo*layer['stride']-layer['padding'];hi=hi*layer['stride']-layer['padding']+layer['weight'].shape[-1]-1
                edge|=lo<0 or hi>=n;lo=max(lo,0);hi=min(hi,n-1)
            if edge:boundaries.append(t)
        assert boundaries==[0,31]
        self.boundary_positions=boundaries

        def coefficients(t):
            v=torch.eye(64,dtype=torch.float64).unsqueeze(-1);origin=t
            for layer,n in zip(reversed(folded),reversed(sizes[:-1])):
                v=F.conv_transpose1d(v,layer['weight'],stride=layer['stride'],groups=layer['groups'])
                origin=origin*layer['stride']-layer['padding']
                left=max(0,-origin);right=min(v.shape[-1],n-origin)
                v=v[:,:,left:right];origin+=left
            result=torch.zeros(64,32,support,dtype=torch.float64)
            offset=origin-(t*jump-pad)
            result[:,:,offset:offset+v.shape[-1]]=v
            return result
        self.register_buffer('kernel',coefficients(1))
        self.register_buffer('first',coefficients(0))
        self.register_buffer('last',coefficients(31))
        with torch.no_grad():bias=reference(torch.zeros(1,32,length,dtype=torch.float64))[0]
        assert torch.allclose(bias[:,1:31],bias[:,1,None].expand(-1,30),atol=1e-12,rtol=0)
        self.register_buffer('position_bias',bias)

    def naive(self,x,position_bias=False):
        y=F.conv1d(x,self.kernel,stride=self.stride,padding=self.padding)
        return y+(self.position_bias if position_bias else self.position_bias[:,1,None])

    def forward(self,x):
        assert x.shape[1:]==(32,self.length)
        y=F.conv1d(x,self.kernel,stride=self.stride,padding=self.padding)
        # Every nominal support position remains represented. Cropped coordinates
        # are the mathematically zero-padded outside-input values, not data loss.
        y[:,:,0]=x[:,:,:298].flatten(1)@self.first[:,:,42:].reshape(64,-1).T
        y[:,:,-1]=x[:,:,7894:].flatten(1)@self.last[:,:,:298].reshape(64,-1).T
        return y+self.position_bias

    def description(self):
        return dict(input=[32,8192],output=[64,32],kernel=340,stride=256,padding=42,boundary_positions=self.boundary_positions,
            stored_coefficients=sum(v.numel() for v in self.buffers()),trainable_parameters=sum(v.numel() for v in self.parameters()),
            scope='Affine conditional on shared nonlinear stem output; eval-frozen BN only. All 32 positions retained.')

def error_record(a,b,atol,rtol):
    delta=(a-b).abs();relative=delta/a.abs().clamp_min(1e-12)
    return dict(max_abs=float(delta.max()),max_relative=float(relative.max()),rms=float(delta.square().mean().sqrt()),
        allclose=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),atol=atol,rtol=rtol,
        failing_elements=int((delta>atol+rtol*a.abs()).sum()),elements=a.numel())
