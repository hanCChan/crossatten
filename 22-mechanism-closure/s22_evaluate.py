"""Reproduce baseline inference and collect paired classification diagnostics."""
import argparse
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from s22_core import HERE, read, write, sha, validate, baseline_path, committed, job_id, now
from s22_models import build
from s22_data import shen_train, shen_cell, roles, PROTOCOL
import train as previous

def reference_checks(reg):
    for value in reg['references'].values():
        from pathlib import Path
        for name,digest in value['sha256'].items():
            if sha(Path(value['path'])/name)!=digest: raise ValueError('Historical artifact changed: '+name)

def restore_source():
    reg=validate(); reference_checks(reg); records=[]
    for fold in [0,1]:
        _,va=shen_train(fold)
        x=torch.as_tensor(va['x'],device='cuda'); y=torch.as_tensor(va['y'],device='cuda'); rx=torch.as_tensor(va['rx'],device='cuda')
        for arm,seed in itertools.product(['H0','P0','H1','H2'],[201,202,203]):
            p=baseline_path(arm,fold,seed); cfg=read(p/'config.json'); m=read(p/'metrics.json')
            previous.setup(cfg['data_seed']); model=build(arm,cfg['data_seed']).cuda()
            model.load_state_dict(torch.load(p/'model.pt',map_location='cuda',weights_only=True),strict=True)
            assert cfg['torch']==torch.__version__ and cfg['gpu']==torch.cuda.get_device_name(0)
            logits=previous.predict(model,x)
            ce=float(torch.stack([F.cross_entropy(logits[rx==r],y[rx==r]) for r in torch.unique(rx)]).mean())
            assert abs(ce-m['best_source_ce'])<1e-7,(arm,fold,seed,ce,m['best_source_ce'])
            recomputed=previous.score(logits,y,va['rx'])
            assert recomputed==m['source_validation'],(arm,fold,seed,'source validation mismatch')
            records.append(dict(arm=arm,fold=fold,seed=seed,strict_state_load=True,source_metrics_exact=True,source_ce=ce,source_ce_delta=ce-m['best_source_ce']))
            del model,logits
        del x,y,rx; torch.cuda.empty_cache()
    write(HERE/'BASELINE_SOURCE_RECOVERY.json',dict(time=now(),status='PASS',models=records,target_data_read=False))

def sealed():
    reg=validate(); p=HERE/'decision_p1_source.json'; committed(p); seal=read(p)
    expected={job_id(j) for j in reg['jobs']}
    if set(seal['checkpoints'])!=expected or len(expected)!=12: raise ValueError('Incomplete checkpoint seal')
    for name,digest in seal['checkpoints'].items():
        out=HERE/'runs'/name
        if sha(out/'model.pt')!=digest or read(out/'metrics.json')['model_sha256']!=digest: raise ValueError('Unsealed checkpoint: '+name)
    return reg

@torch.no_grad()
def evaluate(arm,fold,seed):
    reg=sealed()
    if arm in ['H0','P0','H1','H2']:
        source=baseline_path(arm,fold,seed); out=HERE/'runs'/f'reused__{arm}__f{fold}__s{seed}'
        ref=reg['references'][f'{arm}_f{fold}_s{seed}']
        for name,digest in ref['sha256'].items():
            if sha(source/name)!=digest: raise ValueError('Baseline reference changed')
    else: source=out=HERE/'runs'/job_id(dict(arm=arm,fold=fold,seed=seed))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation.json').exists(): return
    cfg=read(source/'config.json'); previous.setup(cfg['data_seed'])
    model=build(arm,cfg['data_seed']).cuda().eval()
    model.load_state_dict(torch.load(source/'model.pt',map_location='cuda',weights_only=True),strict=True)
    embeddings=[]
    handle=model.net.encoder.final_norm.register_forward_hook(lambda module,args,value: embeddings.append(value.mean(1).detach().cpu()))
    result={}; recovery=[]
    try:
        for role,rxs in [('dev_val',roles(fold)['dev_val']),('confirm',PROTOCOL['confirm_receivers'])]:
            saved={}; per={}; old=np.load(source/(f'diagnostics_{role}.npz' if arm in ['H1','H2'] else f'preds_{role}.npz')) if arm in ['H0','P0','H1','H2'] else None
            for rx in rxs:
                data=shen_cell(rx,fold,role); embeddings.clear()
                logits=previous.predict(model,data['x']).cpu().numpy(); features=torch.cat(embeddings).numpy(); y=np.asarray(data['y'])
                if old is not None:
                    if not np.array_equal(old[rx],logits) or not np.array_equal(old['y_'+rx],y):
                        raise ValueError(f'Baseline float32 prediction mismatch: {arm},fold{fold},seed{seed},{rx}; paired report blocked')
                    recovery.append(dict(role=role,rx=rx,float32_predictions_bit_exact=True))
                per[rx]=previous.rx_metrics(logits,y)
                alternatives=logits.copy(); alternatives[np.arange(len(y)),y]=-np.inf
                margins=logits[np.arange(len(y)),y]-alternatives.max(1)
                centroids=np.stack([features[y==tx].mean(0) for tx in range(10)])
                distances=np.linalg.norm(centroids[:,None]-centroids[None,:],axis=-1)
                saved[rx]=logits; saved['y_'+rx]=y; saved['margin_'+rx]=margins
                saved['centroids_'+rx]=centroids; saved['distances_'+rx]=distances
            if old is not None: old.close()
            np.savez_compressed(out/f'diagnostics_{role}.npz',**saved)
            result[role]=dict(summary=previous.summarize(per),per_rx=per)
    finally: handle.remove()
    if arm in ['H0','P0','H1','H2']: assert result==(read(source/'evaluation.json')['metrics'] if arm in ['H1','H2'] else read(source/'target_metrics.json'))
    write(out/'evaluation.json',dict(time=now(),arm=arm,fold=fold,seed=seed,metrics=result,baseline_prediction_recovery=recovery,model_sha256=sha(source/'model.pt')))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--restore-source',action='store_true'); p.add_argument('--arm',choices=['H0','P0','H1','H2','S1','S2']); p.add_argument('--fold',type=int); p.add_argument('--seed',type=int); a=p.parse_args()
    restore_source() if a.restore_source else evaluate(a.arm,a.fold,a.seed)
