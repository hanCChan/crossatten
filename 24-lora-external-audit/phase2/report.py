"""Globally sealed Dev reporting only. Never loads confirm/blind."""
import csv,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from execution import HERE,STAGE,ROOT,read,digest,write_json,commit,committed,validate_registry,jobid,setup
from data import load_cell,preprocess
from models import build,forward

def seal():
    r=validate_registry();jobs={}
    for j in r['jobs']:
        out=HERE/'runs'/jobid(j);m=read(out/'metrics.json')
        if m['status']!='COMPLETE' or m['registry_sha256']!=digest(HERE/'pilot_registry.json'):raise RuntimeError('matrix incomplete')
        if m['checkpoint_sha256']!=digest(out/'best.pt'):raise ValueError('model hash mismatch')
        jobs[jobid(j)]=dict(checkpoint_sha256=m['checkpoint_sha256'],metrics_sha256=digest(out/'metrics.json'),config_sha256=digest(out/'config.json'))
    obj=dict(status='ALL_REGISTERED_FITS_SEALED',registry_sha256=digest(HERE/'pilot_registry.json'),jobs=jobs,confirm_blind_evaluation=False)
    p=HERE/'global_seal.json'
    if p.exists():
        if read(p)!=obj:raise ValueError('seal changed')
    else:write_json(p,obj)
    commit([p]+[p for p in (HERE/'runs').glob('*/*.json')], 'S24 seal all registered source-only models before any Dev evaluation')
    committed(p);return obj

def h0_model(fold,seed):
    spec=importlib.util.spec_from_file_location('_s24_h0_models',ROOT/'13-mixer-cfo-mechanisms/models.py')
    sys.path.insert(0,str(ROOT/'13-mixer-cfo-mechanisms'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    record=next(v for v in read(STAGE/'HISTORY_AUDIT.json')['frozen_hstem_checkpoints'] if v['job']['fold']==fold and v['job']['seed']==seed)
    path=Path(record['path'])/'model.pt'
    if digest(path)!=record['artifacts']['model.pt']['sha256']:raise ValueError('frozen H0 changed')
    model=module.build('A1_mha',record['data_seed']).cuda().eval()
    model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));return model

def score(logits,y):
    cm=np.bincount(y*10+logits.argmax(1),minlength=100).reshape(10,10)
    tp=cm.diagonal();f1=2*tp/np.maximum(cm.sum(0)+cm.sum(1),1)
    return dict(acc=float(tp.sum()/cm.sum()),macro_f1=float(f1.mean()),ce=float(F.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y))),confusion=cm.tolist())

@torch.no_grad()
def evaluate():
    sealed=seal();r=validate_registry();records=[]
    # Frozen H0 is evaluated once and reused identically in the two information-budget tables.
    jobs=r['jobs']+[dict(method='h0',view='native',fold=f,seed=s) for f in (0,1) for s in (201,202,203)]
    for j in jobs:
        setup(j['seed']+j['fold']*10000);jid=jobid(j);dest=HERE/'dev'/jid;dest.mkdir(parents=True,exist_ok=True)
        if (dest/'metrics.json').exists():
            saved=read(dest/'metrics.json')
            if saved['seal_sha256']!=digest(HERE/'global_seal.json'):raise ValueError('Dev seal changed')
            records.extend(saved['rows']);continue
        model=h0_model(j['fold'],j['seed']) if j['method']=='h0' else build(j['method']).cuda().eval()
        if j['method']!='h0':model.load_state_dict(torch.load(HERE/'runs'/jid/'best.pt',map_location='cuda',weights_only=True))
        rows=[]
        roles=__import__('s24_common').P['folds'][j['fold']]
        for rx in roles['dev_val']:
            x,y,_=load_cell(rx,j['fold'],'dev',jid,sealed);outputs=[]
            for start in range(0,len(x),128):
                z=torch.from_numpy(np.asarray(x[start:start+128]).copy()).cuda()
                logits=model(z) if j['method']=='h0' else forward(model,j['method'],preprocess(z,j['view'],j['method']))[0]
                outputs.append(logits.cpu().numpy())
            logits=np.concatenate(outputs);np.savez(dest/f'predictions_{rx}.npz',logits=logits,y=y)
            row=dict(method=j['method'],view=j['view'],fold=j['fold'],seed=j['seed'],rx=rx,**score(logits,y));rows.append(row)
        write_json(dest/'metrics.json',dict(rows=rows,seal_sha256=digest(HERE/'global_seal.json'),target_adaptation=False))
        records.extend(rows);del model;torch.cuda.empty_cache()
    scalar=[{k:v for k,v in row.items() if k!='confusion'} for row in records]
    with (HERE/'PER_RX.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(scalar[0]),lineterminator='\n');writer.writeheader();writer.writerows(scalar)
    comparisons=[];summary=[]
    for view in ('native','strict_ib'):
        for method in ('ce','dann','fan','ib','ask','h0'):
            rows=[v for v in scalar if v['method']==method and v['view']==('native' if method=='h0' else view)]
            pairs=[np.mean([v['acc'] for v in rows if v['fold']==fold and v['seed']==seed]) for fold in (0,1) for seed in (201,202,203)]
            rx_means={rx:float(np.mean([v['acc'] for v in rows if v['rx']==rx])) for rx in sorted({v['rx'] for v in rows})}
            summary.append(dict(view=view,method=method,acc=float(np.mean(pairs)),fit_sd=float(np.std(pairs,ddof=1)),
                                macro_f1=float(np.mean([v['macro_f1'] for v in rows])),worst_rx=min(rx_means,key=rx_means.get),worst_rx_acc=min(rx_means.values())))
        for a,b in [('h0','dann'),('h0','ib'),('h0','ask'),('dann','ce'),('ib','fan')]:
            aa={(v['fold'],v['seed'],v['rx']):v['acc'] for v in scalar if v['method']==a and v['view']==('native' if a=='h0' else view)}
            bb={(v['fold'],v['seed'],v['rx']):v['acc'] for v in scalar if v['method']==b and v['view']==view}
            assert aa.keys()==bb.keys()
            rx_deltas={rx:np.mean([aa[k]-bb[k] for k in aa if k[2]==rx]) for rx in sorted({k[2] for k in aa})}
            values=np.array(list(rx_deltas.values()));rng=np.random.default_rng(24016)
            boot=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
            comparisons.append(dict(view=view,a=a,b=b,mean_delta=float(values.mean()),conditional_rx_cluster_ci=np.quantile(boot,[.025,.975]).tolist(),
                                    unique_receivers=len(values),per_rx_delta=rx_deltas,no_multiple_comparison_correction=True))
    write_json(HERE/'RESULTS.json',dict(summary=summary,paired_comparisons=comparisons,seal_sha256=digest(HERE/'global_seal.json'),
                                        statistical_scope='Conditional on these three seeds and ten historically developed RX; RX identities, not packets/seeds, are resampled.'))
    text='# S24 同协议 Dev 结果\n\n所有注册源域模型封存后才统一加载Dev；无confirm/blind评估。\n'
    for view in ('native','strict_ib'):
        text+='\n## '+view+'\n\n| 方法 | RX等权Acc% ± fit SD | Macro-F1% | 最差RX均值% |\n|---|---:|---:|---:|\n'
        for row in summary:
            if row['view']==view:text+=f"|{row['method']}|{row['acc']*100:.3f} ± {row['fit_sd']*100:.3f}|{row['macro_f1']*100:.3f}|{row['worst_rx_acc']*100:.3f}|\n"
    text+='\n逐RX及条件配对区间见PER_RX.csv/RESULTS.json。两视图不是独立重复；confirm不作为新盲测。IB为Shen8192适配。\n'
    (HERE/'RESULTS.zh-CN.md').write_text(text,encoding='utf-8',newline='\n')

if __name__=='__main__':evaluate()
