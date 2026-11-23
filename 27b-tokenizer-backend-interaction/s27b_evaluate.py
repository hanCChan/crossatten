"""Common four-cell inference and fixed source geometry after all12 are sealed."""
from pathlib import Path
import time
import numpy as np
import torch
from s27b_common import HERE,P,read,write,sha,validate,identity,now
from s27b_data import cell,source,check_seal
from s27b_models import build
from s27b_training import setup
from s27b_stats import ARMS
from s27b_features import extract,geometry_record
from xa.train import stable_seed
import train as canonical


def load(arm,fold,seed,registration):
    data_seed=stable_seed('P-H5x160',fold,seed);setup(data_seed)
    checkpoint=(HERE/'runs'/identity(fold,seed,arm)/'model.pt' if arm.endswith('_P') else
        Path(next(r['checkpoint'] for r in registration['references'] if r['arm']==arm and r['fold']==fold and r['seed']==seed)))
    model=build(arm,data_seed).cuda().eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=True)
    return model,checkpoint


def evaluate():
    registration=validate();check_seal();records=[];recovery=[]
    for fold in (0,1):
        for rx in P['folds'][fold]['dev_val']:
            data=cell(rx,fold,'dev_report_only');x=np.array(data['x'],copy=True)
            for arm in ARMS:
                for seed in (201,202,203):
                    model,checkpoint=load(arm,fold,seed,registration)
                    logits=canonical.predict(model,x).cpu().numpy();y=data['y']
                    out=HERE/'predictions'/identity(fold,seed,arm);out.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(out/(rx+'.npz'),logits=logits,y=y,rows=data['rows'])
                    records.append(dict(arm=arm,fold=fold,seed=seed,rx=rx,checkpoint_sha256=sha(checkpoint),**canonical.rx_metrics(logits,y)))
                    if not arm.endswith('_P'):
                        ref=next(r for r in registration['references'] if r['arm']==arm and r['fold']==fold and r['seed']==seed)
                        if ref['prediction_format']=='legacy_bundle':
                            path=checkpoint.parent/'preds_dev_val.npz';assert sha(path)==ref['predictions_sha256'][rx]
                            with np.load(path,allow_pickle=False) as old:old_y=old['y_'+rx];old_logits=old[rx]
                        else:
                            path=Path(ref['prediction_directory'])/(rx+'.npz');assert sha(path)==ref['predictions_sha256'][rx]
                            with np.load(path,allow_pickle=False) as old:old_y=old['y'];old_logits=old['logits']
                        np.testing.assert_array_equal(y,old_y)
                        recovery.append(dict(arm=arm,fold=fold,seed=seed,rx=rx,logits_bit_exact=bool(np.array_equal(logits,old_logits)),
                            max_absolute_logit_difference=float(np.abs(logits-old_logits).max()),changed_argmax_packets=int(np.sum(logits.argmax(1)!=old_logits.argmax(1)))))
                    del model;torch.cuda.empty_cache()
            print('Report-only Dev',fold,rx,'all12models',flush=True)
    assert len(records)==120
    write(HERE/'EVALUATION.json',dict(status='COMPLETE_REPORT_ONLY',time=now(),records=records,packet_predictions=240000,
        fitted_new_models=12,reused_models=12,baseline_recovery=recovery,seal_sha256=sha(HERE/'seal.json'),
        scope='CommonGPU originalFP32/TF32matmuloff/cuDNN on; batch128, all4cells/allseeds/all10RX.'))


def source_features():
    registration=validate();check_seal();records=[];geometries=[];model_rows=[]
    for fold in (0,1):
        training,data=source(fold,with_rows=True);del training
        for arm in ARMS:
            for seed in (201,202,203):
                model,checkpoint=load(arm,fold,seed,registration)
                features=extract(model,data['x']);name=identity(fold,seed,arm)
                out=HERE/'predictions/source';out.mkdir(parents=True,exist_ok=True)
                path=out/(name+'.npz');np.savez_compressed(path,**features,y=data['y'],rx=data['rx'],rows=data['rows'])
                for layer in ['tokens','pre_backend','final']:
                    geometries.append(dict(arm=arm,fold=fold,seed=seed,layer=layer,**geometry_record(features[layer],data['y'],data['rx'])))
                for ri,rx in enumerate(P['folds'][fold]['source']):
                    mask=data['rx']==ri;logits=features['logits'][mask];y=data['y'][mask]
                    ce=float(torch.nn.functional.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y).long()))
                    records.append(dict(arm=arm,fold=fold,seed=seed,rx=rx,ce=ce,**canonical.rx_metrics(logits,y)))
                old=read((HERE/'runs'/name/'metrics.json') if arm.endswith('_P') else checkpoint.parent/'metrics.json')
                score=canonical.score(features['logits'],data['y'],data['rx'])
                model_rows.append(dict(id=name,arm=arm,fold=fold,seed=seed,metrics_recovered_exactly=score==old['source_validation'],
                    features_sha256=sha(path),checkpoint_sha256=sha(checkpoint)))
                print('Source frozen taps',name,'all3layers',flush=True)
                del model;torch.cuda.empty_cache()
    assert len(geometries)==72 and len(records)==120
    write(HERE/'SOURCE_EVALUATION.json',dict(status='COMPLETE_DESCRIPTIVE_SOURCE_ONLY',time=now(),records=records,geometry=geometries,
        models=model_rows,packet_predictions=48000,pooled_feature_vectors=144000,
        scope='Original sourceval already used forselection. All24models,3predeclaredmean64Dtaps,perpacketL2,balanced40packets/TX/RX. No causal informationflow,MI or physicalinvariance claim.'))


def costs():
    validate();setup(27028);rows=[]
    for arm in ARMS:
        model=build(arm,27028).cuda().eval();batches={}
        for batch in (1,128):
            x=torch.randn(batch,2,8192,device='cuda')
            with torch.no_grad():
                for _ in range(3):model(x)
                samples=[]
                for _ in range(15):
                    torch.cuda.synchronize();start=time.perf_counter();model(x);torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1000)
            batches[str(batch)]=dict(p50_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),samples=samples)
        rows.append(dict(arm=arm,params=sum(p.numel() for p in model.parameters()),batches=batches))
        del model;torch.cuda.empty_cache()
    write(HERE/'COSTS.json',dict(status='COMPLETE',time=now(),rows=rows,gpu=torch.cuda.get_device_name(0),
        scope='Separate commonB GPUresidentIQ-to-logits, allfrontend included, hosttransfer excluded;3warmups15reps, batch1/128; notpooledwitholdtimings.'))
