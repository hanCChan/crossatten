"""Freeze source-only metadata plan. Never decode IQ or model/prediction files."""
from __future__ import annotations
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''): h.update(block)
    return h.hexdigest()


def save(path, value):
    if path.exists(): raise FileExistsError('Frozen registration artifact exists: '+str(path))
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def vector_record(ids, y, rx, day, names):
    return dict(rows=len(ids), dtype=str(ids.dtype), indices_array_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                rx=sorted({names[int(r)] for r in rx[ids]}), days=sorted(map(int,np.unique(day[ids]))),
                tx_day_support=[dict(tx=t,day=d,rows=int(((y[ids]==t)&(day[ids]==d)).sum())) for t in range(10) for d in (1,2)],
                tx_rx_day_support=[dict(rx=names[int(r)],tx=t,day=d,rows=int(((rx[ids]==r)&(y[ids]==t)&(day[ids]==d)).sum())) for r in sorted(np.unique(rx[ids])) for t in range(10) for d in (1,2)])


def source_views(y, rx, day, train, val, names, legal_fit):
    if any(a.ndim!=1 or a.dtype.kind not in 'iu' for a in (y,rx,day,train,val)):
        raise ValueError('Only integer metadata/index vectors are allowed')
    if not (len(y)==len(rx)==len(day)) or set(day)!={1,2}:
        raise ValueError('Source metadata length/date differs')
    if np.any(y<0) or np.any(y>=10) or np.any(rx<0) or np.any(rx>=len(names)):
        raise ValueError('Unregistered TX/RX metadata code')
    actual = {names[int(r)] for r in np.unique(rx)}
    if actual != set(legal_fit) or len(actual)!=18:
        raise ValueError('Only the legal fold2 fit18 pool may be partitioned')
    if not np.array_equal(np.sort(np.r_[train,val]),np.arange(len(y))):
        raise ValueError('Original train/val partition differs')
    ordered = sorted(legal_fit,key=lambda r:hashlib.sha256(('C-source-CV-v1|'+r).encode()).hexdigest())
    views, arrays = [], {}
    for v in range(2):
        held = ordered[6*v:6*(v+1)]
        active = [r for r in legal_fit if r not in held]
        codes = [names.index(r) for r in held]
        tr = train[~np.isin(rx[train],codes)]
        va = val[~np.isin(rx[val],codes)]
        selection = np.flatnonzero(np.isin(rx,codes)).astype(np.int64)
        if any(set(a)&set(b) for a,b in ((tr,va),(tr,selection),(va,selection))):
            raise ValueError('Role rows overlap')
        if set(np.r_[tr,va,selection]) != set(range(len(y))):
            raise ValueError('View does not account for the original pool')
        rec = dict(id=f'V{v}',fit_rx=active,source_selection_rx=held,
                   fit_train=vector_record(tr,y,rx,day,names),checkpoint_val=vector_record(va,y,rx,day,names),source_selection=vector_record(selection,y,rx,day,names))
        if any(not np.any(rx[tr]==names.index(r)) for r in active):
            raise ValueError('A registered training RX is globally missing')
        rec['updates_per_epoch'] = (len(tr)+127)//128
        rec['gradient_updates_per_fit'] = 40*rec['updates_per_epoch']
        views.append(rec)
        arrays[rec['id']] = dict(fit_train=tr,checkpoint_val=va,source_selection=selection)
    if set(views[0]['source_selection_rx']) & set(views[1]['source_selection_rx']):
        raise ValueError('Source selection RX repeat between views')
    return views, arrays


def main():
    if (HERE/'PRE_REGISTRATION.json').exists() or (HERE/'ACCESS_SCOPE.json').exists():
        raise FileExistsError('Refuse to rerun metadata registration')
    base=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if not base.startswith('140d2bc'): raise ValueError('Register against reviewed B completion')
    previous=read(ROOT/'research/controlled_training_a/PRE_REGISTRATION.json')
    names=('fit_y','fit_rx','fit_day','train_ids','val_ids')
    scope={n:previous['inputs'][n] for n in names}
    save(HERE/'ACCESS_SCOPE.json',dict(scope='REGISTRATION_METADATA_ONLY',base_commit=base,metadata_array_whitelist=scope,
         iq_payload_decode_admitted=False,checkpoint_or_prediction_decode_admitted=False,new_training_admitted=False))
    values, log = {}, []
    for n in names:
        item=scope[n];path=Path(item['path'])
        if sha(path)!=item['sha256']: raise ValueError('Metadata hash differs: '+n)
        values[n]=np.load(path,allow_pickle=False)
        log.append(dict(name=n,path=str(path),sha256=item['sha256'],action='registered_metadata_load',rows=len(values[n])))
    views,_=source_views(values['fit_y'],values['fit_rx'],values['fit_day'],values['train_ids'],values['val_ids'],previous['rx_names'],previous['fit_rx'])
    save(HERE/'SOURCE_PLAN.json',dict(status='SOURCE_ONLY_DEVELOPMENT_PLAN',partition_rule='SHA256(C-source-CV-v1|RX) first6/next6 held within legal fit18',
         views=views,source_selection_rx_count=12,original_packet_split_preserved=True,
         original_pool_rows=len(values['fit_y']),labels_only_planning=True,iq_decodes=0))
    receiver_csv=ROOT/'10-tifs-validation/final_receiver_metrics.csv'
    with receiver_csv.open(encoding='utf-8-sig',newline='') as f:
        historical=sorted({r['rx'] for r in csv.DictReader(f) if r['dataset']=='ManyRx'})
    save(HERE/'CONFIRMATION_GATE.json',dict(status='CLOSED_NO_REGISTERED_UNSEEN_RECEIVERS',historically_reported_ManyRx_rx=historical,
         historically_reported_count=len(historical),evidence_path=receiver_csv.relative_to(ROOT).as_posix(),evidence_sha256=sha(receiver_csv),
         fresh_confirmation_rx=[],new_receiver_payload_access_admitted=False,final_full_source_refits_admitted=False,
         historical_held6_scoring_admitted=False,target8_and_dates3_4_admitted=False,
         requirements=['New or provably unused compatible RX cohort and TX mapping','No selection, BN calibration or hyperparameter tuning with its data or statistics','Frozen methods/configurations/checkpoints and separate final evaluation registration']))
    jobs=[]
    for view in views:
        for stream in range(3):
            for label,wd in (('wd0',0.),('wd1e3',.001)):
                for family in ('ERM_EMA','RX_GDRO'):
                    jobs.append(dict(id=f"{view['id']}_S{stream}_{label}_{family}",view=view['id'],stream=f'S{stream}',family=family,
                         readouts=['C0','C1'] if family=='ERM_EMA' else ['C2'],weight_decay=wd,
                         initialization_seed=24101+stream,order_seed=24201+stream,stochastic_seed=24301,
                         epochs=40,batch=128,updates_per_epoch=view['updates_per_epoch'],max_gradient_updates=view['gradient_updates_per_fit']))
    save(HERE/'JOBS.json',jobs)
    save(HERE/'ACCESS_LOG.json',dict(records=log,iq_array_decodes=0,checkpoint_loads=0,prediction_loads=0,scientific_model_forwards=0,new_scientific_fits=0,
         note='Synthetic method-kernel QA is separate; no scientific model was constructed or evaluated. Script actions only, not OS monitoring.'))
    # Full IQ is hashed as bytes for the future input registry, never np.load'ed.
    hashed=[]
    for key,item in previous['inputs'].items():
        path=Path(item['path'])
        if sha(path)!=item['sha256']:raise ValueError('Future source input hash differs')
        hashed.append(str(path))
    import torch
    import torch.optim.swa_utils as swa
    runtime=Path(swa.__file__).resolve()
    save(HERE/'REFERENCES.json',dict(group_dro=dict(paper='https://arxiv.org/abs/1911.08731v2',
         repository='https://github.com/kohpangwei/group_DRO',commit='cbbc1c5b06844e46b87e264326b56056d2a437d1',
         loss='https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/loss.py',
         default_step=.01,full_upstream_trainer_used=False),
         ema=dict(documentation='https://docs.pytorch.org/docs/stable/optim.html#weight-averaging-swa-and-ema',torch_version=str(torch.__version__),
         actual_local_source=str(runtime),actual_local_source_sha256=sha(runtime),decay=.999,
         actual_version_pinned_instead_of_online_stable=True)))
    code=dict(previous['code_sha256'])
    for relative,digest in code.items():
        if sha(ROOT/relative)!=digest:raise ValueError('Historical registered code changed: '+relative)
    for p in list(HERE.glob('*.py'))+[HERE/'PREREGISTRATION.zh-CN.md',ROOT/'tests/test_standard_method_competition_c.py']:
        code[p.relative_to(ROOT).as_posix()]=sha(p)
    dependencies=[ROOT/'research/controlled_training_b/DECISION.json',ROOT/'10-tifs-validation/PROTOCOL_AND_LEAKAGE_AUDIT.md',receiver_csv]
    registration=dict(schema_version=1,stage='C',status='C_SOURCE_METHOD_COMPETITION_PREREGISTERED_CONFIRMATION_GATE_CLOSED',
         scope_this_turn='PREREGISTRATION_ONLY_NO_TRAINING',base_commit=base,registered_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
         maximum_development_gradient_fits=24,gradient_fits_executed=0,ema_shadow_readouts=12,total_arm_candidates=36,
         max_gradient_epochs=960,max_BN_calibration_passes=480,streams=3,source_views=2,
         weight_decay_candidates=[0.,.001],ema_decay=.999,group_dro_step_size=.01,group_dro_groups='RX only',
         checkpoint_selection='RX-equal CE on checkpoint_val; strict less, earliest tie',
         candidate_selection='Per-method minimum source_selection RX-equal CE averaged over12RX and3fixed streams; smaller wd tie',
         early_stopping=False,epochs=40,common_gradient_update_budget=True,
         determinism=dict(algorithms=True,warn_only=False,cublas_workspace_config=':4096:8',cudnn_deterministic=True,
                          cudnn_benchmark=False,matmul_allow_tf32=False,cudnn_allow_tf32=False,cpu_threads=2),
         primary_endpoint='selected',auxiliary_endpoints=['epoch10','final40'],bootstrap_replicates=10000,bootstrap_seed=24501,
         source_input_whitelist=previous['inputs'],fit_rx=previous['fit_rx'],rx_names=previous['rx_names'],source_cell_sha256=previous['source_cell_sha256'],
         forbidden_historical_held_rx=previous['held_rx'],forbidden_target_rx=previous['target_rx'],allowed_days=[1,2],
         code_sha256=code,evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in dependencies},
         artifact_sha256={p.name:sha(p) for p in HERE.glob('*.json')},
         integrated_training_runner_registered=False,execution_engineering_addendum_required=True,
         final_confirmation_access_admitted=False,new_generalization_claim_admitted=False,AB_attribution_line_closed=True,
         pre_registration_raw_file_hash_reads=hashed,pre_registration_iq_decodes=0)
    save(HERE/'PRE_REGISTRATION.json',registration)
    print(json.dumps(dict(status=registration['status'],gradient_fits=0,registered_gradient_budget=len(jobs),source_views=[dict(id=v['id'],train=v['fit_train']['rows'],checkpoint_val=v['checkpoint_val']['rows'],selection=v['source_selection']['rows']) for v in views],iq_decodes=0)))


if __name__=='__main__':main()
