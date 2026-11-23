"""Independently recompute saved target metrics and audit source/fit integrity."""
import csv
import datetime
import hashlib
from pathlib import Path
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from s27_common import HERE,ROOT,P,read,write,sha,validate,identity,now
from s27_data import check_seal,source
from s27_models import build
from s27_training import setup
from xa.frontends import awgn
from xa.train import stable_seed
import train as canonical


def main():
    registration=validate();seal=check_seal();evaluation=read(HERE/'EVALUATION.json')
    assert evaluation['status']=='COMPLETE_REPORT_ONLY' and len(evaluation['records'])==90
    assert evaluation['seal_sha256']==sha(HERE/'seal.json')
    assert evaluation['packet_predictions']==180000
    cohort={};all_rows=[];prediction_hashes=[];recomputed_cells=[]
    for row in evaluation['records']:
        path=HERE/'predictions'/f"{row['arm']}_f{row['fold']}_s{row['seed']}"/(row['rx']+'.npz')
        assert row['rx'] in P['folds'][row['fold']]['dev_val']
        with np.load(path,allow_pickle=False) as data:
            logits=data['logits'].astype(np.float64);y=data['y'];rawrows=data['rows']
        assert logits.shape==(2000,10) and y.shape==rawrows.shape==(2000,) and np.isfinite(logits).all()
        np.testing.assert_array_equal(np.bincount(y,minlength=10),np.full(10,200))
        key=(row['fold'],row['rx'])
        if key in cohort:
            np.testing.assert_array_equal(y,cohort[key][0]);np.testing.assert_array_equal(rawrows,cohort[key][1])
        else:cohort[key]=(y.copy(),rawrows.copy())
        predictions=logits.argmax(1)
        cm=np.bincount(y*10+predictions,minlength=100).reshape(10,10)
        np.testing.assert_array_equal(cm,row['confusion'])
        accuracy=float(np.mean(predictions==y));f1=float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1)))
        maxima=logits.max(1);ce=float(np.mean(maxima+np.log(np.exp(logits-maxima[:,None]).sum(1))-logits[np.arange(2000),y]))
        assert abs(accuracy-row['acc'])<1e-12 and abs(f1-row['macro_f1'])<1e-12
        # The frozen evaluator archives CM/accuracy/F1/recall, not target CE.
        # Compute CE from saved logits here and cross-check a separate Torch
        # implementation; never invent an original evaluator CE field.
        torch_ce=float(F.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y).long()))
        assert abs(ce-torch_ce)<1e-12
        recomputed_cells.append(dict(arm=row['arm'],fold=row['fold'],seed=row['seed'],rx=row['rx'],
            accuracy=accuracy,macro_f1=f1,ce_from_saved_logits=ce))
        all_rows.append(dict(arm=row['arm'],fold=row['fold'],seed=row['seed'],rx=row['rx'],acc=accuracy,macro_f1=f1))
        prediction_hashes.append(dict(path=path.relative_to(HERE).as_posix(),sha256=sha(path)))
    assert len(cohort)==10
    results=read(HERE/'RESULTS.json')
    for summary in results['summary']:
        rr=[r for r in all_rows if r['arm']==summary['arm']]
        fits=np.array([np.mean([r['acc'] for r in rr if r['fold']==f and r['seed']==s])*100 for f in (0,1) for s in (201,202,203)])
        np.testing.assert_allclose(fits,summary['fit_accuracies_pct'],atol=1e-12,rtol=0)
        assert abs(fits.mean()-summary['accuracy_pct'])<1e-12
        assert abs(fits.std(ddof=1)-summary['fit_sd_pp'])<1e-12
        assert abs(np.mean([r['macro_f1'] for r in rr])*100-summary['macro_f1_pct'])<1e-12
        for rx,value in summary['per_rx_accuracy_pct'].items():
            assert abs(np.mean([r['acc'] for r in rr if r['rx']==rx])*100-value)<1e-12
        assert min(summary['per_rx_accuracy_pct'].values())==summary['worst_rx_accuracy_pct']
    for comparison in results['paired_comparisons']:
        a={(r['fold'],r['seed'],r['rx']):r['acc'] for r in all_rows if r['arm']==comparison['a']}
        b={(r['fold'],r['seed'],r['rx']):r['acc'] for r in all_rows if r['arm']==comparison['b']}
        d=np.array([np.mean([a[k]-b[k] for k in a if k[2]==rx])*100 for rx in sorted({k[2] for k in a})])
        rng=np.random.default_rng(27027);ci=np.quantile(d[rng.integers(0,10,(10000,10))].mean(1),[.025,.975])
        np.testing.assert_allclose(ci,comparison['conditional_rx_ci95_pp'],atol=1e-12,rtol=0)
        assert abs(d.mean()-comparison['mean_delta_pp'])<1e-12
    fits=[];source_checks=[]
    for fold in (0,1):
        train,val=source(fold);xx=torch.from_numpy(val['x']).cuda();yy=torch.from_numpy(val['y']).cuda();rr=torch.from_numpy(val['rx']).cuda()
        for seed in (201,202,203):
            name=identity(fold,seed);folder=HERE/'runs'/name;cfg=read(folder/'config.json');metric=read(folder/'metrics.json')
            with (folder/'history.csv').open(newline='',encoding='utf-8') as stream:history=list(csv.DictReader(stream))
            assert len(history)==metric['epochs']==200 and metric['optimizer_steps']==25000
            best=min(history,key=lambda r:float(r['source_val_ce']))
            assert int(best['epoch'])==metric['best_epoch'] and abs(float(best['source_val_ce'])-metric['best_source_ce'])<1e-12
            assert cfg['params']==metric['params']==133690 and cfg['data_seed']==stable_seed('P-H5x160',fold,seed)
            assert cfg['registration_sha256']==sha(HERE/'registration.json')
            for filename,expected in seal['models'][name].items():assert sha(folder/filename)==expected
            setup(cfg['data_seed']);model=build('LA',cfg['data_seed']).cuda().eval()
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cuda',weights_only=True),strict=True)
            logits=canonical.predict(model,xx)
            current=canonical.score(logits,yy,val['rx'])
            ce=float(torch.stack([F.cross_entropy(logits[rr==r],yy[rr==r]) for r in torch.unique(rr)]).mean())
            assert current==metric['source_validation'] and abs(ce-metric['best_source_ce'])<1e-7
            order=torch.Generator(device='cuda').manual_seed(cfg['data_seed']);noise=torch.Generator(device='cuda').manual_seed(cfg['data_seed']+1)
            perm=torch.randperm(8000,generator=order,device='cuda')
            batch=awgn(torch.from_numpy(train['x']).cuda()[perm[:64]],0,50,noise)
            actual=read(folder/'streams.json')
            assert hashlib.sha256(perm.cpu().numpy().tobytes()).hexdigest()==actual['permutation_sha256']
            assert hashlib.sha256(batch.cpu().numpy().tobytes()).hexdigest()==actual['first_augmented_batch_sha256']
            fits.append(dict(id=name,epochs=200,best_epoch=metric['best_epoch'],model_sha256=sha(folder/'model.pt'),
                source_accuracy=metric['source_validation']['summary']['acc_mean'],fit_wall_seconds=metric['wall_seconds']))
            source_checks.append(dict(id=name,source_ce_delta=ce-metric['best_source_ce'],source_metrics_exact=True,first_training_stream_recovered=True))
            del model,logits,batch;torch.cuda.empty_cache()
        del train,val,xx,yy,rr;torch.cuda.empty_cache()
    import json
    events=[json.loads(line) for line in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(e['role'] in ['source','dev_report_only'] for e in events)
    for e in events:
        assert e['rx'] not in P['confirm_receivers']+P['blind_receivers_never_open']
        assert e['rx'] in P['folds'][e['fold']]['source' if e['role']=='source' else 'dev_val']
        if e['role']=='dev_report_only':
            assert datetime.datetime.fromisoformat(e['time'])>=datetime.datetime.fromisoformat(seal['time'])
    assert sum(e['role']=='dev_report_only' for e in events)==10
    # All historical reference files are verified, including arms not newly run.
    references=read(ROOT/'23-hstem-mechanism-evidence/registration.json')['references']
    historical=0
    for r in references.values():
        for filename,expected in r['sha256'].items():assert sha(Path(r['path'])/filename)==expected
        historical+=1
    for record in registration['references']:
        for filename,expected in record['artifacts_sha256'].items():assert sha(Path(record['checkpoint']).parent/filename)==expected
    previous=read(ROOT/'26-long-patch-control/seal.json')
    for name, files in previous['models'].items():
        folder=Path('D:/crossatten/worktrees/s26-long-patch-control/26-long-patch-control/runs')/name
        for filename,expected in files.items():assert sha(folder/filename)==expected
        historical+=1
    changed=subprocess.check_output(['git','diff',registration['base_git'],'--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith(HERE.name+'/') for p in changed)
    costs=read(HERE/'COSTS.json');assert len(costs['rows'])==3
    for row in costs['rows']:
        for batch in row['batches'].values():assert len(batch['samples'])==15 and min(batch['samples'])>0
    write(HERE/'EXPERIMENT_INTEGRITY.json',dict(status='PASS',time=now(),new_fits=6,scientific_optimizer_steps=150000,
        engineering_synthetic_optimizer_steps=8,
        models_evaluated=18,target_packet_predictions=180000,target_cells=90,unique_held_rx=10,
        all_model_selection_source_only=True,all_six_sealed_before_dev=True,all_seeds_rx_retained=True,
        checkpoint_source_metrics_recovered=source_checks,confusion_accuracy_f1_ce_recomputed=True,
        target_ce_scope='Derived from saved logits during this audit; not a metric field in the original frozen evaluator.',
        recomputed_cells=recomputed_cells,
        paired_conditional_intervals_recomputed=True,source_roles_checked=True,new_confirm_blind_reads=0,
        historical_reference_models_unchanged=historical,fits=fits,prediction_hashes=prediction_hashes,
        source_recovery=read(HERE/'SOURCE_RECOVERY.json')['all_metrics_exact'],
        baseline_logit_recovery_all=all(r['logits_bit_exact'] for r in evaluation['baseline_recovery']),
        results_sha256=sha(HERE/'RESULTS.json'),seal_sha256=sha(HERE/'seal.json'),verifier_sha256=sha(__file__),
        original_scientific_code_unchanged=True,
        cost_scope='Common separate S27 session. Per-fit wall sums overlap under2 jobs; not exclusive GPU active time.'))
    print('PASS: all6 fits, 180000 target predictions, source selection/streams, conditional statistics and historical hashes.',flush=True)


if __name__=='__main__':main()
