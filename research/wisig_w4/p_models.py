"""Ordinary CE and prespecified RX class-distance relation penalty."""
from common import *
import copy
import numpy as np
import torch
import torch.nn.functional as F


C=helpers();A=C.A;M=C.M


def build_aux(seed):
    model=A.build_model(24101+seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41501+seed);head=torch.nn.Linear(64,72)
    model.classifier=head.cuda()
    assert sum(p.numel() for p in model.parameters())==86456
    return model


def pooled(model,x):
    h=model.pre_ln(model.encoder.prepare_tokens(model.tokens(x)))
    for block in model.encoder.blocks:h=block(h)
    return model.encoder.final_norm(h).mean(1)


def relation(z):
    if z.ndim!=4:raise ValueError('Need TX,RX,packets,features')
    t,r,s,d=z.shape
    if t<3 or r<2 or s<4:raise ValueError('Insufficient class/RX/packet support')
    centers=z.mean(2).permute(1,0,2)
    i,j=torch.triu_indices(t,t,1,device=z.device)
    distances=((centers[:,i]-centers[:,j])**2).sum(-1)
    norm=distances/(distances.mean(1,keepdim=True)+1e-8)
    u,v=torch.triu_indices(r,r,1,device=z.device)
    loss=((norm[u]-norm[v])**2).mean()
    spread=distances.mean(1)/(centers.square().mean((1,2))+1e-8)
    stats=dict(mean_between_squared_distance=distances.mean().detach(),minimum_relative_spread=spread.min().detach())
    if not torch.isfinite(loss) or not torch.isfinite(spread).all():raise ValueError('Nonfinite relation structure')
    return loss,stats


def balanced_batches(classes,seed,epoch):
    """Canonical train array layout TX,RX,(D1 first24 + D2 first24)."""
    if classes%8:raise ValueError('TX count must be divisible by eight')
    rng=np.random.default_rng(seed+1009*epoch)
    within=np.array([[rng.permutation(48) for _ in range(6)] for _ in range(classes)])
    batches=[]
    for cycle in range(12):
        order=rng.permutation(classes)
        for group in order.reshape(-1,8):
            ids=np.array([((int(tx)*6+rx)*48)+within[tx,rx,cycle*4:cycle*4+4] for tx in group for rx in range(6)]).reshape(-1)
            assert len(ids)==192
            batches.append(ids)
    flat=np.concatenate(batches)
    if not np.array_equal(np.sort(flat),np.arange(classes*6*48)):raise ValueError('Auxiliary packet coverage failed')
    return batches


def initial_gradient_scale(model,train,seed):
    """Initial fit-only autograd probes; zero optimizer updates, no original mutation."""
    original=A.semantic_hash(model.state_dict());rng_before=A.semantic_hash(A.rng_state(torch.Generator(device='cuda').manual_seed(1)))
    stats=[]
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(41601+seed);probe=copy.deepcopy(model).train()
        parameters=[p for n,p in probe.named_parameters() if not n.startswith('classifier.')]
        for ids in balanced_batches(len(np.unique(train['y'])),41301+seed,1)[:4]:
            x=torch.as_tensor(train['x'][ids],device='cuda');y=torch.as_tensor(train['y'][ids],device='cuda')
            h=pooled(probe,x);ce=F.cross_entropy(probe.classifier(h),y);rel,geometry=relation(h.reshape(8,6,4,64))
            gc=torch.autograd.grad(ce,parameters,retain_graph=True);gr=torch.autograd.grad(rel,parameters)
            nc=torch.sqrt(sum(g.square().sum() for g in gc));nr=torch.sqrt(sum(g.square().sum() for g in gr))
            if min(float(nc),float(nr))<=1e-12:raise ValueError('Initial relation/CE gradient is zero')
            if float(geometry['minimum_relative_spread'])<1e-6:raise ValueError('Initial class geometry collapsed')
            stats.append(dict(CE=float(ce),relation=float(rel),CE_gradient_norm=float(nc),relation_gradient_norm=float(nr),
                              minimum_relative_spread=float(geometry['minimum_relative_spread'])))
        del probe
    lam=.1*np.mean([s['CE_gradient_norm'] for s in stats])/np.mean([s['relation_gradient_norm'] for s in stats])
    assert np.isfinite(lam) and lam>0 and original==A.semantic_hash(model.state_dict())
    assert rng_before==A.semantic_hash(A.rng_state(torch.Generator(device='cuda').manual_seed(1)))
    return dict(seed=seed,lambda_value=float(lam),probes=stats,fit_batches=4,optimizer_updates=0,
                initial_state_sha256=original,original_state_and_rng_preserved=True)


def transfer(aux_state,seed):
    downstream=A.build_model(24101+seed);head={k:v.detach().clone() for k,v in downstream.classifier.state_dict().items()}
    shared={k:v for k,v in aux_state.items() if not k.startswith('classifier.')}
    missing,unexpected=downstream.load_state_dict(shared,strict=False)
    assert missing==['classifier.weight','classifier.bias'] and not unexpected
    assert all(torch.equal(v,head[k]) for k,v in downstream.classifier.state_dict().items())
    return downstream
