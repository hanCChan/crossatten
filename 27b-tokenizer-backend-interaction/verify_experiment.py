"""Independent saved-prediction, geometry, source-stream and historical audit."""
import csv
import datetime
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from s27b_common import HERE,ROOT,P,read,write,sha,validate,identity,now
from s27b_data import source,check_seal
from s27b_models import build,surviving_hash,state_hash
from s27b_training import setup
from xa.train import stable_seed
from xa.frontends import awgn
import train as canonical

ARMS=['H0','LP','H0_P','LP_P']


def recalculate(logits,y):
    z=logits.astype(np.float64);pred=z.argmax(1)
    cm=np.bincount(y*10+pred,minlength=100).reshape(10,10)
    accuracy=float(np.mean(pred==y));f1=float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)))
    maximum=z.max(1);ce=float(np.mean(maximum+np.log(np.exp(z-maximum[:,None]).sum(1))-z[np.arange(len(y)),y]))
    return cm,accuracy,f1,ce


def geometry_independent(x,y,rx):
    z=x.astype(np.float64);norm=np.sqrt((z*z).sum(1))
    if not np.isfinite(z).all() or np.any(norm<=1e-12):return None
    z=z/norm[:,None];overall=z.mean(0);centers=np.stack([z[y==t].mean(0) for t in range(10)])
    between=float(np.mean(np.sum((centers-overall)**2,axis=1)));dr=[];within=[]
    for t in range(10):
        for r in range(5):
            cell=z[(y==t)&(rx==r)];assert cell.shape==(40,64);center=cell.mean(0)
            dr.append(float(np.sum((center-centers[t])**2)));within.append(float(np.mean(np.sum((cell-center)**2,axis=1))))
    d=float(np.mean(dr));w=float(np.mean(within))
    return dict(between_tx=between,conditional_rx=d,within_tx_rx=w,
        tx_to_rx_centroid_ratio=between/(d+1e-12),tx_to_within_class_ratio=between/(d+w+1e-12))


def main():
    registration=validate();seal=check_seal();dev=read(HERE/'EVALUATION.json');src=read(HERE/'SOURCE_EVALUATION.json')
    resume_checks=[]
    for receipt in sorted(HERE.glob('RESUME_AUDIT_*.json')):
        audit=read(receipt);assert audit['frozen_scientific_hashes']=='PASS' and audit['best_checkpoint_source_consistency']=='PASS'
        for record in audit['records']:
            if record['status']!='RESUMABLE':continue
            recovery=read(HERE/'runs'/record['id']/f"recovery_epoch{record['epoch']+1}.json")
            assert recovery['resume_sha256']==record['resume_sha256'] and recovery['from_completed_epoch']==record['epoch']
            resume_checks.append(dict(id=record['id'],from_completed_epoch=record['epoch'],snapshot_sha256=record['resume_sha256']))
    assert len(dev['records'])==120 and dev['packet_predictions']==240000
    assert len(src['records'])==120 and len(src['geometry'])==72 and src['packet_predictions']==48000
    assert dev['seal_sha256']==sha(HERE/'seal.json')
    assert all(q['metrics_recovered_exactly'] for q in src['models'])
    cohort={};hashes=[];ce_rows=[]
    for q in dev['records']:
        path=HERE/'predictions'/identity(q['fold'],q['seed'],q['arm'])/(q['rx']+'.npz')
        with np.load(path,allow_pickle=False) as data:logits=data['logits'];y=data['y'];rows=data['rows']
        assert logits.shape==(2000,10) and y.shape==rows.shape==(2000,) and np.isfinite(logits).all()
        np.testing.assert_array_equal(np.bincount(y,minlength=10),np.full(10,200))
        key=(q['fold'],q['rx'])
        if key in cohort:
            np.testing.assert_array_equal(y,cohort[key][0]);np.testing.assert_array_equal(rows,cohort[key][1])
        else:cohort[key]=(y.copy(),rows.copy())
        cm,acc,f1,ce=recalculate(logits,y);np.testing.assert_array_equal(cm,q['confusion'])
        assert abs(acc-q['acc'])<1e-12 and abs(f1-q['macro_f1'])<1e-12
        ce_rows.append(dict(arm=q['arm'],fold=q['fold'],seed=q['seed'],rx=q['rx'],ce=ce))
        hashes.append(dict(path=path.relative_to(HERE).as_posix(),sha256=sha(path)))
    results=read(HERE/'RESULTS.json')
    for summary in results['summary']:
        q=[r for r in dev['records'] if r['arm']==summary['arm']]
        fits=[np.mean([r['acc'] for r in q if r['fold']==f and r['seed']==s])*100 for f in (0,1) for s in (201,202,203)]
        np.testing.assert_allclose(fits,summary['fit_accuracies_pct'],atol=1e-12,rtol=0)
        assert abs(np.mean(fits)-summary['accuracy_pct'])<1e-12 and abs(np.std(fits,ddof=1)-summary['fit_sd_pp'])<1e-12
        assert abs(np.mean([r['macro_f1'] for r in q])*100-summary['macro_f1_pct'])<1e-12
        for rx,v in summary['per_rx_accuracy_pct'].items():assert abs(np.mean([r['acc'] for r in q if r['rx']==rx])*100-v)<1e-12
    groups={a:{(q['fold'],q['seed'],q['rx']):q['acc'] for q in dev['records'] if q['arm']==a} for a in ARMS}
    keys=groups['H0'].keys();assert all(g.keys()==keys for g in groups.values())
    interactions={k:((groups['H0'][k]-groups['LP'][k])-(groups['H0_P'][k]-groups['LP_P'][k]))*100 for k in keys}
    rxnames=sorted({k[2] for k in keys});rxvalues=np.array([np.mean([v for k,v in interactions.items() if k[2]==rx]) for rx in rxnames])
    rng=np.random.default_rng(27028);ci=np.quantile(rxvalues[rng.integers(0,10,(10000,10))].mean(1),[.025,.975])
    primary=results['primary_interaction'];np.testing.assert_allclose(ci,primary['conditional_rx_ci95_pp'],atol=1e-10,rtol=0)
    assert abs(rxvalues.mean()-primary['mean_delta_pp'])<1e-10
    fits=[np.mean([v for k,v in interactions.items() if k[:2]==(f,s)]) for f in (0,1) for s in (201,202,203)]
    np.testing.assert_allclose(fits,primary['fit_deltas_pp'],atol=1e-10,rtol=0)
    for item in results['secondary_contrasts']:
        perkey={k:sum(w*groups[arm][k] for arm,w in item['weights'].items())*100 for k in keys}
        perrx=np.array([np.mean([v for k,v in perkey.items() if k[2]==rx]) for rx in rxnames])
        generator=np.random.default_rng(27028);bounds=np.quantile(perrx[generator.integers(0,10,(10000,10))].mean(1),[.025,.975])
        assert abs(perrx.mean()-item['mean_delta_pp'])<1e-10
        np.testing.assert_allclose(bounds,item['conditional_rx_ci95_pp'],atol=1e-10,rtol=0)
    source_cohorts={};geometry_checks=[]
    for record in src['models']:
        name=record['id'];path=HERE/'predictions/source'/(name+'.npz');assert sha(path)==record['features_sha256']
        with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        fold=record['fold'];y=data['y'];rx=data['rx'];rows=data['rows'];assert data['logits'].shape==(2000,10)
        if fold in source_cohorts:
            for key in ['y','rx','rows']:np.testing.assert_array_equal(data[key],source_cohorts[fold][key])
        else:source_cohorts[fold]={k:data[k].copy() for k in ['y','rx','rows']}
        for ri,receiver in enumerate(P['folds'][fold]['source']):
            mask=rx==ri;q=next(q for q in src['records'] if q['arm']==record['arm'] and q['fold']==fold and q['seed']==record['seed'] and q['rx']==receiver)
            cm,acc,f1,ce=recalculate(data['logits'][mask],y[mask]);np.testing.assert_array_equal(cm,q['confusion'])
            assert abs(acc-q['acc'])<1e-12 and abs(f1-q['macro_f1'])<1e-12 and abs(ce-q['ce'])<3e-6
        for layer in ['tokens','pre_backend','final']:
            assert data[layer].shape==(2000,64)
            q=next(q for q in src['geometry'] if q['arm']==record['arm'] and q['fold']==fold and q['seed']==record['seed'] and q['layer']==layer)
            independent=geometry_independent(data[layer],y,rx)
            if independent is None:assert q['status'].startswith('UNDEFINED')
            else:
                assert q['status']=='VALID'
                for metric,value in independent.items():np.testing.assert_allclose(value,q[metric],atol=1e-9,rtol=1e-9)
            geometry_checks.append(dict(id=name,layer=layer,status=q['status']))
        hashes.append(dict(path=path.relative_to(HERE).as_posix(),sha256=sha(path)))
    source_groups={arm:{(q['fold'],q['seed'],q['rx']):q['acc'] for q in src['records'] if q['arm']==arm} for arm in ARMS}
    for item in results['source_summary']:
        selected=[q for q in src['records'] if q['arm']==item['arm']]
        sf=[np.mean([q['acc'] for q in selected if q['fold']==f and q['seed']==s])*100 for f in (0,1) for s in (201,202,203)]
        np.testing.assert_allclose(sf,item['fit_accuracies_pct'],atol=1e-12,rtol=0)
        assert abs(np.mean(sf)-item['accuracy_pct'])<1e-12
        assert abs(np.mean([q['macro_f1'] for q in selected])*100-item['macro_f1_pct'])<1e-12
    source_interactions={k:((source_groups['H0'][k]-source_groups['LP'][k])-(source_groups['H0_P'][k]-source_groups['LP_P'][k]))*100 for k in source_groups['H0']}
    source_fits=[np.mean([v for k,v in source_interactions.items() if k[:2]==(f,s)]) for f in (0,1) for s in (201,202,203)]
    np.testing.assert_allclose(source_fits,results['source_interaction']['fit_deltas_pp'],atol=1e-10,rtol=0)
    np.testing.assert_allclose(np.array(source_fits)-np.array(fits),results['source_minus_dev_interaction']['fit_deltas_pp'],atol=1e-10,rtol=0)
    fit_records=[]
    for fold in (0,1):
        tr,va=source(fold,with_rows=True)
        for key in ['y','rx','rows']:np.testing.assert_array_equal(va[key],source_cohorts[fold][key])
        for seed in (201,202,203):
            for arm in ['H0_P','LP_P']:
                name=identity(fold,seed,arm);folder=HERE/'runs'/name;cfg=read(folder/'config.json');metric=read(folder/'metrics.json')
                with (folder/'history.csv').open(encoding='utf-8',newline='') as stream:history=list(csv.DictReader(stream))
                assert len(history)==metric['epochs']==200 and metric['optimizer_steps']==25000
                best=min(history,key=lambda q:float(q['source_val_ce']));assert int(best['epoch'])==metric['best_epoch']
                assert abs(float(best['source_val_ce'])-metric['best_source_ce'])<1e-12
                for filename,expected in seal['models'][name].items():assert sha(folder/filename)==expected
                setup(cfg['data_seed']);model=build(arm,cfg['data_seed']);base=build(arm.split('_')[0],cfg['data_seed'])
                assert surviving_hash(model)==surviving_hash(base)==cfg['initial_surviving_sha256']
                assert state_hash(model)==read(folder/'pairing.json')['initial_full_sha256']
                del base;model=model.cuda().eval();model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda',weights_only=True))
                logits=canonical.predict(model,va['x']);score=canonical.score(logits,va['y'],va['rx']);assert score==metric['source_validation']
                ce=float(torch.stack([F.cross_entropy(logits[torch.as_tensor(va['rx']==i,device='cuda')],torch.as_tensor(va['y'][va['rx']==i],device='cuda')) for i in range(5)]).mean())
                assert abs(ce-metric['best_source_ce'])<1e-7
                order=torch.Generator(device='cuda').manual_seed(cfg['data_seed']);noise=torch.Generator(device='cuda').manual_seed(cfg['data_seed']+1)
                permutation=torch.randperm(8000,generator=order,device='cuda');batch=awgn(torch.as_tensor(tr['x'],device='cuda')[permutation[:64]],0,50,noise)
                stream=read(folder/'streams.json');reference=next(q for q in registration['reference_streams'] if q['fold']==fold and q['seed']==seed)
                for key,array in [('permutation_sha256',permutation.cpu().numpy()),('first_augmented_batch_sha256',batch.cpu().numpy())]:
                    assert hashlib.sha256(array.tobytes()).hexdigest()==stream[key]==reference[key]
                fit_records.append(dict(id=name,epochs=200,updates=25000,source_metric_exact=True,paired_streams_exact=True,model_sha256=sha(folder/'model.pt')))
                del model,logits,batch;torch.cuda.empty_cache()
        del tr,va
    import json
    events=[json.loads(line) for line in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(e['role']=='dev_report_only' for e in events)==10
    for e in events:
        assert e['role'] in ['source','dev_report_only'] and e['rx'] not in P['confirm_receivers']+P['blind_receivers_never_open']
        assert e['rx'] in P['folds'][e['fold']]['source' if e['role']=='source' else 'dev_val']
        if e['role']=='dev_report_only':assert datetime.datetime.fromisoformat(e['time'])>=datetime.datetime.fromisoformat(seal['time'])
    historical=set()
    for q in read(ROOT/'23-hstem-mechanism-evidence/registration.json')['references'].values():
        for filename,h in q['sha256'].items():assert sha(Path(q['path'])/filename)==h
        historical.add(q['path'])
    for root,folder in [(ROOT/'26-long-patch-control',Path('D:/crossatten/worktrees/s26-long-patch-control/26-long-patch-control/runs')),
        (ROOT/'27-mechanism-identification',Path('D:/crossatten/worktrees/s27-mechanism-identification/27-mechanism-identification/runs'))]:
        for name,files in read(root/'seal.json')['models'].items():
            for filename,h in files.items():assert sha(folder/name/filename)==h
            historical.add(str(folder/name))
    changed=subprocess.check_output(['git','diff',registration['base_git'],'--name-only'],cwd=ROOT,text=True).splitlines();assert all(p.startswith(HERE.name+'/') for p in changed)
    costs=read(HERE/'COSTS.json')['rows'];gate=read(HERE/'GATE.json')['architecture']
    assert len(costs)==4 and {q['arm'] for q in costs}==set(ARMS)
    for q in costs:
        assert q['params']==gate[q['arm']]['params'] and set(q['batches'])=={'1','128'}
        for values in q['batches'].values():
            samples=np.array(values['samples']);assert len(samples)==15 and np.isfinite(samples).all() and samples.min()>0
            assert abs(np.median(samples)-values['p50_ms'])<1e-10
    write(HERE/'EXPERIMENT_INTEGRITY.json',dict(status='PASS',time=now(),new_fits=12,scientific_optimizer_steps=300000,
        engineering_synthetic_optimizer_steps=8,models_evaluated=24,target_packet_predictions=240000,source_packet_predictions=48000,
        geometry_vectors=144000,geometry_records_checked=geometry_checks,source_recovery=read(HERE/'SOURCE_RECOVERY.json')['all_metrics_exact'],
        baseline_logit_recovery_all=all(q['logits_bit_exact'] for q in dev['baseline_recovery']),
        all_model_selection_source_only=True,all_twelve_sealed_before_dev=True,all_fits_rx_retained=True,
        interaction_conditional_interval_independently_recomputed=True,target_CE_from_saved_logits=ce_rows,
        original_scientific_code_unchanged=True,historical_reference_models_unchanged=len(historical),new_confirm_blind_reads=0,
        fits=fit_records,resume_snapshots_checked=resume_checks,prediction_hashes=hashes,results_sha256=sha(HERE/'RESULTS.json'),seal_sha256=sha(HERE/'seal.json'),
        verifier_sha256=sha(__file__),C_fits=0,IQ_only_fits=0,ASK_strict_fits=0))
    print('PASS: all12fits,240000Dev/48000Sourcepredictions,72geometryrecords,interaction,sourceRNG andhistoricalhashes.',flush=True)


if __name__=='__main__':main()
