"""Frozen source-val taps, original model forward, all four trained cells."""
import numpy as np
import torch
from s27b_common import ROOT
from geometry_stats import geometry


@torch.inference_mode()
def extract(model,x):
    captured={};outputs={k:[] for k in ['tokens','pre_backend','final','logits']};handles=[]
    def tap(name):
        def receive(module,args,value):
            assert value.shape[1:]==(32,64)
            captured[name]=value.mean(1).cpu().numpy()
        return receive
    handles.append(model.net.embedder.cnn_stem.register_forward_hook(tap('tokens')))
    handles.append(model.pre_ln.register_forward_hook(tap('pre_backend')))
    def final(module,args):
        assert args[0].shape[1:]==(64,)
        captured['final']=args[0].cpu().numpy()
    handles.append(model.net.classifier.register_forward_pre_hook(final))
    try:
        for start in range(0,len(x),128):
            batch=torch.as_tensor(np.ascontiguousarray(x[start:start+128]),device='cuda')
            logits=model(batch)
            for name in ['tokens','pre_backend','final']:outputs[name].append(captured[name].copy())
            outputs['logits'].append(logits.cpu().numpy())
    finally:
        for handle in handles:handle.remove()
    return {k:np.concatenate(v) for k,v in outputs.items()}


def geometry_record(features,y,rx):
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features.astype(np.float64),axis=1)<=1e-12):
        return dict(status='UNDEFINED_INVALID_OR_ZERO_EMBEDDING',normalization='Fixed perpacket L2; no alternative normalization selected')
    return dict(status='VALID',**geometry(features,y,rx))
