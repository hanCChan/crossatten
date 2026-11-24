"""Independent audit from saved predictions, histories and permitted Source."""
import csv
import datetime
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from ac_common import HERE,ROOT,CACHE,P,C,read,write,sha,now,validate,identity,committed
from ac_data import check_seal,source
from s28_models import build,state_hash,shared_hash
from s27b_training import setup,learning_rate
from xa.frontends import awgn
from xa.train import stable_seed
from xa.data import source_split
import train as canonical

def score(logits,y):
    v=logits.astype(np.float64);pred=v.argmax(1);cm=np.bincount(y*10+pred,minlength=100).reshape(10,10)
    mx=v.max(1);ce=float(np.mean(mx+np.log(np.exp(v-mx[:,None]).sum(1))-v[np.arange(len(y)),y]))
    return cm,float(np.mean(pred==y)),float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1))),ce

def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def check_record(record,data,mask=None):
    mask=np.ones(len(data['y']),dtype=bool) if mask is None else mask
    cm,acc,f1,ce=score(data['logits'][mask],data['y'][mask])
    np.testing.assert_array_equal(cm,record['confusion'])
    assert abs(acc-record['acc'])<1e-12 and abs(f1-record['macro_f1'])<1e-12 and abs(ce-record['ce'])<1e-10
    np.testing.assert_allclose(cm.diagonal()/cm.sum(1),record['per_tx_recall'],atol=1e-12,rtol=0)

def geom(x,y,rx):
    z=x.astype(np.float64);n=np.linalg.norm(z,axis=1)
    if not np.isfinite(z).all() or np.any(n<=1e-12):return None
    z/=n[:,None];centers=np.stack([z[y==t].mean(0) for t in range(10)]);between=np.mean(((centers-z.mean(0))**2).sum(1))
    d=[];w=[]
    for t in range(10):
        for r in np.unique(rx):
            cell=z[(y==t)&(rx==r)];assert cell.shape==(40,64);center=cell.mean(0)
            d.append(((center-centers[t])**2).sum());w.append(((cell-center)**2).sum(1).mean())
    d,w=np.mean(d),np.mean(w)
    return dict(between_tx=between,conditional_rx=d,within_tx_rx=w,tx_to_within_class_ratio=between/(d+w+1e-12),
        tx_to_rx_centroid_ratio=None if len(np.unique(rx))==1 else between/(d+1e-12))

def main():
    reg=validate();seals={s:check_seal(s) for s in ['A','C']}
    repair=None
    if (HERE/'TECHNICAL_REPAIR_001.json').exists():
        repair=read(HERE/'TECHNICAL_REPAIR_001.json');committed(HERE/'TECHNICAL_REPAIR_001.json');committed(HERE/'repair_io_001.py')
        assert sha(HERE/'repair_io_001.py')==repair['repair_script_sha256'] and sha(HERE/'ac_evaluate.py')==repair['original_evaluator_sha256']
        assert read(HERE/'REPAIR_RESULT_001.json')['scientific_fits']==read(HERE/'REPAIR_RESULT_001.json')['scientific_optimizer_steps']==0
        assert all(sha(HERE/f'seal_{s}.json')==h for s,h in repair['sealed_cohort_sha256'].items())
    a=read(HERE/'EVALUATION_A.json');c=read(HERE/'EVALUATION_C.json');results=read(HERE/'RESULTS.json')
    assert len(a['dev_records'])==150 and len(a['source_records'])==150
    assert len(c['source_records'])==48 and len(c['held_records'])==12 and len(c['geometry'])==72
    assert len(a['baseline_recovery'])==120 and all(q['logits_bit_exact'] for q in a['baseline_recovery'])
    hashes=[];acohort={};record_maps={k:{} for k in ['ADev','ASource','CSource','CHeld']}
    for q in a['dev_records']:
        name=identity(dict(stage='A',arm=q['arm'],fold=q['fold'],seed=q['seed']));path=HERE/'predictions'/name/(q['rx']+'.npz');d=load(path)
        assert d['logits'].shape==(2000,10) and sha(path)==q['predictions_sha256'];check_record(q,d)
        expected_checkpoint=(HERE/'runs'/name/'model.pt') if q['arm']=='H2_LA' else Path(next(t['checkpoint'] for t in reg['references'] if (t['arm'],t['fold'],t['seed'])==(q['arm'],q['fold'],q['seed'])))
        assert sha(expected_checkpoint)==q['checkpoint_sha256']
        labels=np.load(CACHE/q['rx']/'y.npy');rows=np.load(CACHE/q['rx']/'rows.npy')
        np.testing.assert_array_equal(labels,d['y']);np.testing.assert_array_equal(rows,d['rows'])
        assert np.array_equal(np.bincount(d['y'],minlength=10),np.full(10,200))
        if q['arm']!='H2_LA':
            ref=next(t for t in reg['references'] if t['arm']==q['arm'] and t['fold']==q['fold'] and t['seed']==q['seed'])['predictions'][q['rx']]
            old=load(ref['path']);np.testing.assert_array_equal(old[ref['logits_key']],d['logits'])
        record_maps['ADev'][q['arm'],q['fold'],q['seed'],q['rx']]=float(np.mean(d['logits'].argmax(1)==d['y']))
    for f in (0,1):
        for arm in ['H0','LA','H2','H2_LA','LP']:
            for s in (201,202,203):
                d=load(HERE/'predictions'/identity(dict(stage='A',arm=arm,fold=f,seed=s))/'SOURCE.npz')
                assert d['logits'].shape==(2000,10)
                if f in acohort:
                    for k in ['y','rx','rows']:np.testing.assert_array_equal(d[k],acohort[f][k])
                else:acohort[f]={k:d[k] for k in ['y','rx','rows']}
                for ri,rx in enumerate(P['folds'][f]['source']):
                    q=next(q for q in a['source_records'] if (q['arm'],q['fold'],q['seed'],q['rx'])==(arm,f,s,rx));mask=d['rx']==ri
                    assert mask.sum()==400;check_record(q,d,mask);record_maps['ASource'][arm,f,s,rx]=q['acc']
    csource={};cheld={}
    for j in [j for j in reg['jobs'] if j['stage']=='C']:
        name=identity(j);base=HERE/'predictions'/name;f,s,arm=j['fold'],j['seed'],j['arm']
        for role in ['SOURCE','HELD']:
            d=load(base/(role+'.npz'));features=load(base/(role+'_features.npz'));count=1600 if role=='SOURCE' else 2000
            assert d['logits'].shape==(count,10)
            if role=='SOURCE':
                csource[arm,f,s]=d
                for ri,rx in enumerate(C['splits'][f]['training_receivers']):
                    q=next(q for q in c['source_records'] if (q['arm'],q['fold'],q['seed'],q['rx'])==(arm,f,s,rx));check_record(q,d,d['rx']==ri)
                    record_maps['CSource'][arm,f,s,rx]=q['acc']
                indices=np.arange(1600)
            else:
                cheld[arm,f,s]=d;q=next(q for q in c['held_records'] if (q['arm'],q['fold'],q['seed'])==(arm,f,s));check_record(q,d)
                record_maps['CHeld'][arm,f,s,q['rx']]=q['acc'];_,indices=source_split(d['y'],d['rows'])
                np.testing.assert_array_equal(d['rows'],np.load(CACHE/q['rx']/'rows.npy'));np.testing.assert_array_equal(d['y'],np.load(CACHE/q['rx']/'y.npy'))
            for k in ['logits','y','rx','rows']:np.testing.assert_array_equal(features[k],d[k][indices])
            for layer in ['tokens','pre_backend','final']:
                assert features[layer].shape==(len(indices),64)
                q=next(q for q in c['geometry'] if (q['arm'],q['fold'],q['seed'],q['role'],q['layer'])==(arm,f,s,role,layer))
                independent=geom(features[layer],features['y'],features['rx'])
                if independent is None:assert q['status'].startswith('UNDEFINED')
                else:
                    assert q['status']=='VALID'
                    for key,value in independent.items():
                        if value is None:assert q[key] is None
                        else:np.testing.assert_allclose(value,q[key],rtol=1e-9,atol=1e-10)
    for label,records,section in [('ADev',a['dev_records'],results['A']['dev_summary']),('ASource',a['source_records'],results['A']['source_summary']),
        ('CSource',c['source_records'],results['C']['source_summary']),('CHeld',c['held_records'],results['C']['held_summary'])]:
        for q in section:
            selected=[r for r in records if r['arm']==q['arm']];fits=[np.mean([r['acc'] for r in selected if r['fold']==f and r['seed']==s])*100 for f in (0,1) for s in (201,202,203)]
            np.testing.assert_allclose(fits,q['fit_accuracies_pct'],atol=1e-10,rtol=0)
            assert abs(np.mean(fits)-q['accuracy_pct'])<1e-10 and abs(np.std(fits,ddof=1)-q['fit_sd_pp'])<1e-10
    contrasts=[results['A']['primary_interaction']]+results['A']['secondary']
    for q in contrasts:
        values=[];fits=[]
        for rx in sorted({r['rx'] for r in a['dev_records']}):
            v=[sum(w*r['acc'] for arm,w in q['weights'].items() for r in a['dev_records'] if r['arm']==arm and r['rx']==rx and r['seed']==s)*100 for s in (201,202,203)]
            values.append(np.mean(v))
        for f in (0,1):
            for s in (201,202,203):
                fits.append(np.mean([sum(w*record_maps['ADev'][arm,f,s,rx] for arm,w in q['weights'].items())*100 for rx in P['folds'][f]['dev_val']]))
        rng=np.random.default_rng(28028);values=np.array(values);ci=np.quantile(values[rng.integers(0,10,(10000,10))].mean(1),[.025,.975])
        np.testing.assert_allclose(ci,q['conditional_rx_ci95_pp'],atol=1e-10,rtol=0);np.testing.assert_allclose(fits,q['fit_deltas_pp'],atol=1e-10,rtol=0)
        mean=values.mean();sign=np.sign(mean);folds=np.array(fits).reshape(2,3).mean(1)
        rule=abs(mean)>=.5 and (ci[0]>0 or ci[1]<0) and sum(np.array(fits)*sign>0)>=5 and all(folds*sign>0)
        assert bool(rule)==q['two_sided_practical_rule_met'] and abs(mean-q['mean_delta_pp'])<1e-10
    pilot=results['C']['pilot'];favorable=0
    for q in pilot['pairs']:
        f,s=q['fold'],q['seed'];dif={}
        for arm in ['H0','LP']:
            d=csource[arm,f,s];h=cheld[arm,f,s]
            source_acc=np.mean(d['logits'].argmax(1)==d['y'])*100;held_acc=np.mean(h['logits'].argmax(1)==h['y'])*100
            assert abs(source_acc-q[arm]['source_accuracy_pct'])<1e-10 and abs(held_acc-q[arm]['held_accuracy_pct'])<1e-10
            dif[arm]=(source_acc,held_acc)
        dh=dif['H0'][1]-dif['LP'][1];dg=(dif['H0'][0]-dif['H0'][1])-(dif['LP'][0]-dif['LP'][1])
        assert abs(dh-q['held_delta_pp'])<1e-10 and abs(dg-q['delta_gap_pp'])<1e-10;favorable+=dh>0 and dg<0
    folds=[dict(held=np.mean([q['held_delta_pp'] for q in pilot['pairs'] if q['fold']==f]),gap=np.mean([q['delta_gap_pp'] for q in pilot['pairs'] if q['fold']==f])) for f in (0,1)]
    rule=all(q['held']>=.5 and q['gap']<=-.5 for q in folds) and favorable>=5
    assert favorable==pilot['favorable_fit_pairs'] and bool(rule)==pilot['consistent_pilot_signal']
    verified=[]
    for stage in ['A','C']:
        for f in (0,1):
            tr,va=source(stage,f);steps=125 if stage=='A' else 100
            for j in [j for j in reg['jobs'] if j['stage']==stage and j['fold']==f]:
                name=identity(j);p=HERE/'runs'/name;cfg=read(p/'config.json');m=read(p/'metrics.json')
                with (p/'history.csv').open(encoding='utf-8',newline='') as file:history=list(csv.DictReader(file))
                assert len(history)==m['epochs']==200 and m['optimizer_steps']==200*steps
                best=min(history,key=lambda r:float(r['source_val_ce']));assert int(best['epoch'])==m['best_epoch'] and abs(float(best['source_val_ce'])-m['best_source_ce'])<1e-12
                for i,row in enumerate(history,1):assert abs(float(row['lr'])-learning_rate(i*steps-1,steps))<1e-14
                for file,h in seals[stage]['models'][name].items():assert sha(p/file)==h
                seed=stable_seed('P-H5x160' if stage=='A' else 'S27C-H4x160',f,j['seed']);setup(seed);model=build(j['arm'],seed)
                assert state_hash(model)==cfg['initial_full_sha256'] and shared_hash(model)==cfg['initial_shared_sha256'] and cfg['pretrained_checkpoint_loaded'] is False
                model=model.cuda().eval();model.load_state_dict(torch.load(p/'model.pt',map_location='cuda',weights_only=True));logits=canonical.predict(model,va['x'])
                saved=load(HERE/'predictions'/name/'SOURCE.npz')
                np.testing.assert_array_equal(logits.cpu().numpy(),saved['logits'])
                for key in ['y','rx','rows']:np.testing.assert_array_equal(saved[key],va[key])
                assert canonical.score(logits,va['y'],va['rx'])==m['source_validation']
                y=torch.as_tensor(va['y'],device='cuda');rx=torch.as_tensor(va['rx'],device='cuda')
                ce=float(torch.stack([F.cross_entropy(logits[rx==r],y[rx==r]) for r in torch.unique(rx)]).mean());assert abs(ce-m['best_source_ce'])<1e-7
                og=torch.Generator(device='cuda').manual_seed(seed);ng=torch.Generator(device='cuda').manual_seed(seed+1)
                order=torch.randperm(len(tr['y']),generator=og,device='cuda');aug=awgn(torch.as_tensor(tr['x'],device='cuda')[order[:64]],0,50,ng)
                streams=read(p/'streams.json')
                assert hashlib.sha256(order.cpu().numpy().tobytes()).hexdigest()==streams['permutation_sha256']
                assert hashlib.sha256(aug.cpu().numpy().tobytes()).hexdigest()==streams['first_augmented_batch_sha256']
                if stage=='C':
                    other=next(t for t in reg['jobs'] if t['stage']=='C' and t['fold']==f and t['seed']==j['seed'] and t['arm']!=j['arm'])
                    assert streams==read(HERE/'runs'/identity(other)/'streams.json')
                verified.append(dict(id=name,epochs=200,updates=steps*200,source_best_exact=True,initialization_and_RF_streams_exact=True))
                del model,logits,y,rx,aug;torch.cuda.empty_cache()
            del tr,va
    for ref in reg['references']:
        for file,h in ref['sha256'].items():assert sha(Path(ref['checkpoint']).parent/file)==h
    import json
    events=[json.loads(s) for s in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines()]
    for q in events:
        assert q['rx'] not in P['confirm_receivers']+P['blind_receivers_never_open']
        if q['role']=='source':assert q['rx'] in (P['folds'][q['fold']]['source'] if q['stage']=='A' else C['splits'][q['fold']]['training_receivers'])
        else:
            assert q['role'] in ['dev_report','held_report']
            for seal in seals.values():assert datetime.datetime.fromisoformat(q['time'])>=datetime.datetime.fromisoformat(seal['time'])
    assert sum(q['role']=='dev_report' for q in events)==10 and sum(q['role']=='held_report' for q in events)==2
    for p in sorted((HERE/'predictions').rglob('*.npz')):hashes.append(dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p)))
    assert len(hashes)==228
    write(HERE/'EXPERIMENT_INTEGRITY.json',dict(time=now(),status='PASS',scientific_fits=18,retained_optimizer_steps=390000,
        synthetic_engineering_updates=read(HERE/'GATE.json')['synthetic_optimizer_updates'],reused_models_unchanged=24,
        target_predictions=324000,source_predictions=79200,C_geometry_vectors=72000,A_RX_units=10,C_held_RX_units=2,
        all_model_selection_source_only=True,all_targets_after_both_seals=True,C_no_pretrained_weights=True,
        A_interaction_independently_recomputed=True,C_gap_rule_independently_recomputed=True,no_new_Confirm_blind=True,
        original_S28_numeric_failure_preserved=True,IO_only_repair_preserved=repair,IO_repair_scientific_fits=0,
        fits=verified,prediction_hashes=hashes,results_sha256=sha(HERE/'RESULTS.json'),verifier_sha256=sha(__file__)))
    print('PASS:18fits,390000updates,324000target/79200Source predictions,72geometryrecords,streams,statistics andoldhashes.',flush=True)

if __name__=='__main__':main()
