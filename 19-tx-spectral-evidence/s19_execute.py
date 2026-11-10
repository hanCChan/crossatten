"""Fixed source pilot: estimate train evidence, fit16, seal20, evaluate once."""
import argparse
from contextlib import nullcontext
import hashlib
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from s19_core import HERE, ROOT, HISTORY, S18, read, write, sha, now, validate, committed, access, job_id
from s19_data import load_fit, load_held, teacher, evidence_ids, locations
from s19_models import build_model
from s19_evidence import collect_evidence
from s19_train import setup, predict, save_predictions, fit
from s19_metrics import score, decide


def verify_s18():
    output=HERE/'S18_VERIFIED.json'
    if output.exists(): return
    head=subprocess.check_output(['git','-C',str(S18),'rev-parse','HEAD'],text=True).strip()
    if not head.startswith('933d4a6'): raise ValueError('Expected available S18 commit933d4a6')
    manifest=read(S18/'18-selective-rx/MANIFEST.json')
    for name,digest in manifest['files_sha256'].items():
        if sha(S18/'18-selective-rx'/name)!=digest: raise ValueError('S18 artifact mismatch: '+name)
    write(output,dict(time=now(),status='PASS',commit=head,verified_files=len(manifest['files_sha256']),
                     manifest_sha256=sha(S18/'18-selective-rx/MANIFEST.json'),
                     decision=read(S18/'18-selective-rx/d0_support_matched/decision_d0.json')['status'],weights_or_iq_read=False))


def evidence(fold):
    validate()
    record=HERE/f'evidence_f{fold}.json'
    if record.exists(): return
    attempt=HERE/f'evidence_attempt_f{fold}.json'
    if attempt.exists(): raise RuntimeError('Interrupted evidence attempt retained; no automatic retry')
    write(attempt,dict(time=now(),status='RUNNING',fold=fold))
    data,meta,_,indices=load_fit(fold);tr=data['train'];selected=evidence_ids(tr,fold)
    run,cfg,digest=teacher(fold);setup(cfg['data_seed'])
    model=build_model('B0','ManyRx',cfg['data_seed']).cuda()
    model.load_state_dict(torch.load(run/'model.pt',map_location='cpu',weights_only=True));model.eval()
    started=time.perf_counter()
    arrays,summary=collect_evidence(model,**{k:tr[k][selected] for k in ['x','y','rx','day']},device='cuda',batch=128,role='train')
    if summary['tx_labels']!=list(range(10)):raise ValueError('Teacher TX class mapping mismatch')
    arrays['bar_q']=np.array(summary['bar_q'],dtype=np.float32)
    arrays['q']=np.array(summary['q'],dtype=np.float64)
    arrays['selected_train_indices']=selected
    arrays['selected_global_indices']=indices['train'][selected]
    arrays['val_global_indices']=indices['val']
    if np.intersect1d(arrays['selected_global_indices'],arrays['val_global_indices']).size: raise ValueError('Evidence/validation overlap')
    cache=HERE/'cache';cache.mkdir(exist_ok=True)
    path=cache/f'evidence_f{fold}.npz';np.savez_compressed(path,**arrays)
    write(record,dict(time=now(),fold=fold,teacher_sha256=digest,teacher_source='S17B frozen baseline; teacher selected using historical source validation',
          evidence_role='train_only',packets=len(selected),global_indices_sha256=hashlib.sha256(arrays['selected_global_indices'].tobytes()).hexdigest(),
          cache_sha256=sha(path),seconds=time.perf_counter()-started,summary=summary))
    # Strict B0 reuse: reproduce its source-val metrics and retain logits for QA.
    out=HERE/'runs'/job_id(fold,'B0');out.mkdir(parents=True,exist_ok=True)
    logits=predict(model,data['val']['x']);s=score(logits,data['val'],meta['rx_names'])
    previous=read(run/'metrics.json')
    if abs(s['summary']['acc']-previous['source_validation']['summary']['acc_mean'])>1e-10: raise ValueError('B0 reuse parity failure')
    save_predictions(out/'source_val_predictions.npz',logits,data['val'])
    write(out/'config.json',dict(cfg,arm='B0',fold=fold,reused=True,original_config_sha256=sha(run/'config.json'),original_checkpoint_path=str(run/'model.pt')))
    write(out/'metrics.json',dict(fold=fold,arm='B0',status='COMPLETE',reused=True,epochs=previous['epochs'],best_epoch=previous['best_epoch'],
          best_source_ce=previous['best_source_ce'],source_validation=s,model_sha256=digest,prediction_sha256=sha(out/'source_val_predictions.npz'),
          wall_seconds=0.,historical_teacher_fit_seconds=previous['wall_seconds']))
    write(attempt,dict(time=now(),status='COMPLETE',fold=fold))
    print('evidence fold',fold,'complete; train packets',len(selected),flush=True)


def seal():
    validate();path=HERE/'seal_pilot.json'
    if path.exists(): committed(path);return
    checkpoints={};metadata={}
    for fold in range(4):
        for arm in ['B0','B1','B2','B3','B4']:
            jid=job_id(fold,arm);out=HERE/'runs'/jid;r=read(out/'metrics.json')
            if r['status']!='COMPLETE':raise ValueError('Incomplete formal job')
            source=locations(fold)[1]/'model.pt' if arm=='B0' else out/'model.pt'
            digest=sha(source)
            if digest!=r['model_sha256']:raise ValueError('Checkpoint changed before seal')
            checkpoints[jid]=dict(path=str(source),sha256=digest,fold=fold,arm=arm)
            metadata[jid]={name:sha(out/name) for name in ['config.json','metrics.json']}
    write(path,dict(time=now(),all_checkpoints_complete=True,formal_new_fits=16,reused_B0=4,
                   checkpoints=checkpoints,metadata_sha256=metadata,evidence_sha256={str(f):sha(HERE/f'evidence_f{f}.json') for f in range(4)},
                   real_target_access=False,held_source_read_before_seal=False))
    # A local commit makes sealing concrete before any held-source data opens.
    subprocess.run(['git','-C',str(ROOT),'add','--',HERE.relative_to(ROOT).as_posix()],check=True)
    subprocess.run(['git','-C',str(ROOT),'-c','user.name=Codex','-c','user.email=codex@local.invalid','commit','-m',
                    'S19 seal all20 source checkpoints before held-source evaluation','--',HERE.relative_to(ROOT).as_posix()],check=True)


def held_evaluate():
    validate();committed(HERE/'seal_pilot.json');seal_record=read(HERE/'seal_pilot.json')
    for jid,metadata in seal_record['metadata_sha256'].items():
        for name,digest in metadata.items():
            if sha(HERE/'runs'/jid/name)!=digest:raise ValueError('Sealed metadata changed: '+jid+'/'+name)
    marker=HERE/'held_attempt.json'
    if marker.exists():
        if read(marker)['status']=='COMPLETE':return
        raise RuntimeError('Interrupted held-source evaluation retained; no automatic retry')
    write(marker,dict(time=now(),status='RUNNING',seal_sha256=sha(HERE/'seal_pilot.json')))
    for fold in range(4):
        data,names,sources=load_held(fold)
        for arm in ['B0','B1','B2','B3','B4']:
            out=HERE/'runs'/job_id(fold,arm);cfg=read(out/'config.json');ref=seal_record['checkpoints'][job_id(fold,arm)]
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Unsealed checkpoint')
            setup(cfg['data_seed']);model=build_model(arm,'ManyRx',cfg['data_seed']).cuda()
            model.load_state_dict(torch.load(ref['path'],map_location='cpu',weights_only=True));model.eval()
            access(fold=fold,arm=arm,kind='sealed_checkpoint',path=ref['path'],sha256=ref['sha256'],seal_sha256=sha(HERE/'seal_pilot.json'))
            logits=predict(model,data['x']);save_predictions(out/'held_predictions.npz',logits,data)
            write(out/'held_metrics.json',dict(time=now(),fold=fold,arm=arm,model_sha256=ref['sha256'],
                  prediction_sha256=sha(out/'held_predictions.npz'),held_source=score(logits,data,names),held_sources=sources,rx_names=names))
        print('held source fold',fold,'all5 arms evaluated',flush=True)
    write(marker,dict(time=now(),status='COMPLETE',seal_sha256=sha(HERE/'seal_pilot.json')))


def decision():
    validate();path=HERE/'decision_pilot.json'
    if path.exists():return
    rows=[read(HERE/'runs'/job_id(f,a)/'held_metrics.json') for f in range(4) for a in ['B0','B1','B2','B3','B4']]
    write(path,dict(time=now(),**decide(rows),new_fits=16,reused_B0=4,real_target_access=False,ManySig_read=False,Shen_read=False))
    print(read(path)['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['verify-s18','evidence','fit','seal','held','decide','run']);a=p.parse_args()
    if a.phase=='verify-s18':verify_s18()
    else:
        validate()
        if a.phase in ['evidence','run']:
            for f in range(4):evidence(f)
        if a.phase in ['fit','run']:
            for f in range(4):
                for arm in ['B1','B2','B3','B4']:fit(f,arm)
        if a.phase in ['seal','run']:seal()
        if a.phase in ['held','run']:held_evaluate()
        if a.phase in ['decide','run']:decision()
