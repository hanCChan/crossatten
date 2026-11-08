"""Stem interventions keep the S13 mixer, head and common stem initial weights."""
import copy
import hashlib
import math
import types
import torch
from s14_core import FROZEN
from models import build as s13_build
from xa.models3 import ChirpPipeline
from legacy_rfhstu.models import RFHSTUBlock

BASE={'E1_patch_mha':'A1_mha','E2_patch_hstu':'A3_rfhstu','G1_no_u':'A3_rfhstu','G2_no_rel':'A3_rfhstu'}

def sdpa_forward(self,x):
    b,t,_=x.shape
    q,k,v=self.qkv(x).reshape(b,t,3,self.heads,self.inner//self.heads).unbind(2)
    q,k,v=[a.transpose(1,2) for a in [q,k,v]]
    value=torch.nn.functional.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
    value=value.transpose(1,2).reshape(b,t,self.inner)
    return self.norm(x+self.out(value))

class MechanismBlock(RFHSTUBlock):
    def __init__(self,kind):super().__init__(64,32);self.kind=kind
    def forward(self,x):
        q,k,v,u=self.qkvu(x).chunk(4,-1)
        score=q@k.transpose(-2,-1)/math.sqrt(self.dim)
        if self.kind!='G2_no_rel':score=score+self.relative_bias[self.relative_index].unsqueeze(0)
        av=self.av_norm(score.sigmoid()@v)
        if self.kind!='G1_no_u':av=av*self.act(u)
        return self.residual_norm(x+self.dropout(self.out(av)))

def build(arm,seed):
    if arm.endswith('_sdpa'):
        base=arm[:-5]
        if base not in ['A1_mha','A1_mha_matched']:raise ValueError('Only equivalent MHA inference backends are allowed')
        model=s13_build(base,seed)
        for block in model.net.encoder.blocks:block.forward=types.MethodType(sdpa_forward,block)
        return model
    if arm not in BASE:return s13_build(arm,seed)
    model=s13_build(BASE[arm],seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if arm.startswith('E'):
            patch=ChirpPipeline(stem='patch',preln=True,depth=2)
            model.net.embedder.cnn_stem.tokenize=copy.deepcopy(patch.net.embedder.cnn_stem.tokenize)
        else:
            for i,old in enumerate(model.net.encoder.blocks):
                new=MechanismBlock(arm);new.load_state_dict(old.state_dict());model.net.encoder.blocks[i]=new
    return model

def common_hash(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        if key.startswith('net.embedder.cnn_stem.tokenize.'):continue
        h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def costs(model,arm):
    total=sum(p.numel() for p in model.parameters())
    # Mechanism controls retain projections for exact one-switch comparisons.
    inactive=2*(64*64+64) if arm=='G1_no_u' else 2*63 if arm=='G2_no_rel' else 0
    return dict(params=total,effectively_unused_params=inactive,active_params=total-inactive)

def dense_mixer_flops(model,tokens=32):
    # FMA=2. Consistent dense math even when a profiler misses fused SDPA ops.
    total=0
    for block in model.net.encoder.blocks:
        if hasattr(block,'self_attn'):
            d=block.self_attn.embed_dim;ff=block.linear1.out_features
            total+=2*tokens*(3*d*d+d*d+2*d*ff)+4*tokens*tokens*d
        elif hasattr(block,'qkv'):
            d=64;inner=block.inner
            total+=2*tokens*(3*d*inner+inner*d)+4*tokens*tokens*inner
        else:
            d=block.dim
            total+=2*tokens*(4*d*d+d*d)+4*tokens*tokens*d
    return total
