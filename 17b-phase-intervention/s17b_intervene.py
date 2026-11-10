"""Fixed source-val D1; require nuisance coverage and TX preservation."""
import numpy as np
import torch
from s17b_core import HERE,ROOT,read,write,sha,validate,now,commit
from s17b_train import configure
from s17b_augment import apply
from s17b_features import phase_descriptor
from s17_jobs import run_id
from s17_models import build
from s17_train import setup,score,predict
from xa.train import stable_seed

def pca_reference(fold):
    p=ROOT/'17-rx-distortion'/f'D0_fold{fold}.json';r=read(p)['families']['phase']['profile_by_rx']
    x=np.asarray(list(r.values()),dtype=float);x-=x.mean(0);scale=x.std(0).clip(1e-4)
    z=x/scale;_,_,v=np.linalg.svd(z,full_matrices=False);basis=v[:3].T
    spread=float(np.mean(np.sum((z@basis)**2,axis=1)))**.5
    if spread<=0:raise ValueError('Degenerate preregistered source phase subspace')
    return scale,basis,spread,sha(p)

def tx_acc(lg,y):return {int(t):float((lg[y==t].argmax(1)==t).mean()) for t in np.unique(y)}

@torch.no_grad()
def probe(model,d,tau,mode,seed,reference,clean_features):
    scale,basis,spread,_=reference;g=torch.Generator(device='cuda').manual_seed(seed);a=torch.Generator(device='cuda').manual_seed(seed+1)
    logits=[];evm=[];ripple=[];projected=0.;total=0.;count=0
    for i in range(0,len(d['y']),128):
        x=torch.as_tensor(np.array(d['x'][i:i+128],copy=True),device='cuda')
        y,info=apply(x,tau,g,a,mode,details=True);logits.append(model(y).cpu().numpy())
        evm.extend(info['evm'].cpu().tolist());ripple.extend(info['amplitude_ripple_db'].cpu().tolist())
        delta=(phase_descriptor(y.cpu().numpy(),support_from=x.cpu().numpy())-clean_features[i:i+len(y)])/scale
        projected+=float(np.sum((delta@basis)**2));total+=float(np.sum(delta**2));count+=len(y)
    return np.concatenate(logits),dict(evm_mean=float(np.mean(evm)),evm_q95=float(np.quantile(evm,.95)),ripple_q95_db=float(np.quantile(ripple,.95)) if mode=='physical' else None,
        phase_subspace_coverage=(projected/count)**.5/spread,phase_subspace_alignment=projected/max(total,1e-12))

def run():
    validate();data,runner=configure();rows=[]
    for j in [v for v in read(HERE/'jobs_pilot.json') if v['arm']=='M0']:
        out=HERE/'runs'/run_id(j);cfg=read(out/'config.json');setup(cfg['data_seed']);m=build(j['dataset'],cfg['data_seed']).cuda()
        m.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True));d,meta=data.load(j);va=d['val'];clean=predict(m,va['x'])
        base=score(clean,va,meta['rx_names']);tx0=tx_acc(clean,va['y']);reference=pca_reference(j['fold']);phi=phase_descriptor(va['x'])
        if abs(base['summary']['acc_mean']-read(out/'metrics.json')['source_validation']['summary']['acc_mean'])>1e-10:raise ValueError('Clean selected source baseline mismatch')
        for tau in [.25,.5,1.]:
            for draw in [0,1]:
                seed=stable_seed('S17B-D1',j['fold'],tau,draw)
                for mode in ['physical','noise','global']:
                    lg,info=probe(m,va,tau,mode,seed,reference,phi);s=score(lg,va,meta['rx_names']);tx=tx_acc(lg,va['y'])
                    rows.append(dict(fold=j['fold'],tau=tau,draw=draw,mode=mode,**info,
                        delta_acc_pp=100*(s['summary']['acc_mean']-base['summary']['acc_mean']),delta_f1_pp=100*(s['summary']['f1_mean']-base['summary']['f1_mean']),
                        worst_tx_delta_pp=100*min(tx[k]-tx0[k] for k in tx),source_phase_reference_sha256=reference[-1]))
        del m;torch.cuda.empty_cache();print('D1 fold',j['fold'],'complete',flush=True)
    levels=[]
    for tau in [.25,.5,1.]:
        folded=[]
        for fold in range(4):
            p=[r for r in rows if r['tau']==tau and r['fold']==fold and r['mode']=='physical'];n=[r for r in rows if r['tau']==tau and r['fold']==fold and r['mode']=='noise']
            v={k:float(np.mean([r[k] for r in p])) for k in ['delta_acc_pp','delta_f1_pp','worst_tx_delta_pp','evm_mean','evm_q95','ripple_q95_db','phase_subspace_coverage','phase_subspace_alignment']}
            v['preserved']=v['delta_acc_pp']>=-1 and v['delta_f1_pp']>=-1 and v['worst_tx_delta_pp']>=-5
            v['physical_bounds']=v['evm_mean']>=.002 and v['evm_q95']<=.10 and v['ripple_q95_db']<=.25
            v['coverage']=v['phase_subspace_coverage']>=.10 and v['phase_subspace_alignment']>=.15
            v['specificity']=v['phase_subspace_alignment']>=float(np.mean([r['phase_subspace_alignment'] for r in n]))+.02
            folded.append(dict(fold=fold,**v))
        allowed=all(v['preserved'] and v['physical_bounds'] for v in folded) and sum(v['coverage'] and v['specificity'] for v in folded)>=3
        levels.append(dict(tau=tau,admitted=bool(allowed),folds=folded))
    allowed=[v['tau'] for v in levels if v['admitted']];tau=max(allowed) if allowed else None
    write(HERE/'D1_interventions.json',dict(time=now(),rows=rows))
    write(HERE/'decision_d1.json',dict(time=now(),status='GO_SOURCE_PILOT' if tau else 'STOP_D1_GATE',selected_tau=tau,levels=levels,
        original_s17_stop_changed=False,real_target_access=False,held_source_read=False))
    commit([HERE/'D1_interventions.json',HERE/'decision_d1.json'],'S17B freeze phase coverage, preservation and negative-control decisions before method training')

if __name__=='__main__':run()
