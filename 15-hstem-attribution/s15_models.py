"""Data-free automatic matching of a single-stride grouped/low-rank Patch stem."""
import hashlib
import copy
import torch
from torch import nn
from s15_core import HERE,read
from models import build as original
from xa.train import stable_seed

BASE={'C2_lite_mha':'A1_mha','C2_lite_hstu':'A3_rfhstu'}
class PatchLite(nn.Sequential):
    def __init__(self,groups,mid):
        super().__init__(nn.Conv1d(32,mid,256,256,groups=groups),nn.Conv1d(mid,64,1))

def search():
    bases={a:original(a,123) for a in ['A1_mha','A3_rfhstu']}
    targets={a:sum(p.numel() for p in m.parameters()) for a,m in bases.items()}
    token_params=sum(p.numel() for p in bases['A1_mha'].net.embedder.cnn_stem.tokenize.parameters())
    candidates=[]
    for g in [1,2,4,8,16,32]:
        for mid in range(g,257,g):
            params=mid*(32//g)*256+mid+64*mid+64
            totals={a:n-token_params+params for a,n in targets.items()}
            error=max(abs(totals[a]/targets[a]-1) for a in targets)
            candidates.append(dict(groups=g,mid=mid,tokenizer_params=params,total_params=totals,max_relative_error=error))
    accepted=[c for c in candidates if c['max_relative_error']<=.01]
    if not accepted:raise ValueError('No candidate meets1% full-model matching tolerance')
    best=min(accepted,key=lambda c:(c['max_relative_error'],c['groups'],c['mid']))
    return dict(target_tokenizer_params=token_params,target_models=targets,candidates=candidates,selected=best,
        rule='Minimize maximum full-model relative parameter error across both mixers; then fewer groups/mid; no data used; full-model tolerance1%')

def build(arm,seed):
    if arm in ['E1_patch_mha','E2_patch_hstu']:
        from xa.models3 import ChirpPipeline
        model=original('A1_mha' if arm=='E1_patch_mha' else 'A3_rfhstu',seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);patch=ChirpPipeline(stem='patch',preln=True,depth=2)
            model.net.embedder.cnn_stem.tokenize=copy.deepcopy(patch.net.embedder.cnn_stem.tokenize)
        return model
    if arm not in BASE:return original(arm,seed)
    model=original(BASE[arm],seed);selected=read(HERE/'architecture_search.json')['selected']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(stable_seed('S15-PatchLite',seed))
        model.net.embedder.cnn_stem.tokenize=PatchLite(selected['groups'],selected['mid'])
    return model

def shared_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith('net.embedder.cnn_stem.tokenize.'):continue
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def receptive_field(model,length=8192):
    field=1;jump=1;center=0.;n=length;path=[]
    stem=model.net.embedder.cnn_stem
    for name,module in list(stem.stem.named_modules())+[(f'tokenize.{k}',v) for k,v in stem.tokenize.named_modules()]:
        if not isinstance(module,nn.Conv1d):continue
        k,s,p,d=[getattr(module,x)[0] for x in ['kernel_size','stride','padding','dilation']]
        center+=((k-1)*d/2-p)*jump;field+=(k-1)*d*jump;jump*=s
        n=(n+2*p-d*(k-1)-1)//s+1
        path.append(dict(layer=name,kernel=k,stride=s,padding=p,groups=module.groups,channels=[module.in_channels,module.out_channels],output_length=n))
    return dict(cnn_receptive_field=field,token_stride=jump,first_token_center=center,tokens=n,path=path,
        scope='CNN coordinate only: global band-limit and FFT views mean this is not the end-to-end physical RF receptive field')
