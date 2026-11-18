"""Explicit budget amendment; original training code and registry remain immutable."""
from pathlib import Path
import datetime,json,sys
HERE=Path(__file__).resolve().parent
PHASE=HERE.parent
STAGE=PHASE.parent
ROOT=STAGE.parent
sys.path.insert(0,str(PHASE))
from execution import read,digest,atomic_json,committed,validate_registry,jobid,commit,setup
from s24_common import P

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def policy():
    p=HERE/'AMENDMENT.json';committed(p);a=read(p)
    r=validate_registry()
    if a['original_registry_sha256']!=digest(PHASE/'pilot_registry.json'):raise ValueError('original registry changed')
    for rel,h in a['code_sha256'].items():
        if digest(ROOT/rel)!=h:raise ValueError('amendment code changed: '+rel)
    active={jobid(j) for j in r['jobs'] if not(j['method']=='ask' and j['view']=='strict_ib')}
    deferred={jobid(j) for j in r['jobs'] if j['method']=='ask' and j['view']=='strict_ib'}
    if set(a['active_job_ids'])!=active or set(a['deferred_job_ids'])!=deferred:raise ValueError('budget membership changed')
    return a,r

def authorized_training(job):
    a,r=policy();jid=jobid(job)
    if jid not in a['active_job_ids']:raise PermissionError('Deferred by user budget amendment: '+jid)
    if job not in r['jobs']:raise PermissionError('unregistered training configuration')
    return a

def cohort_jobs(cohort,r):
    if cohort=='baseline48':return [j for j in r['jobs'] if j['method'] in ('ce','dann','fan','ib')]
    if cohort=='native_ask6':return [j for j in r['jobs'] if j['method']=='ask' and j['view']=='native']
    if cohort=='retained54':return [j for j in r['jobs'] if not(j['method']=='ask' and j['view']=='strict_ib')]
    raise PermissionError('Unknown cohort')

def h0_records():
    return {f"h0__native__f{x['job']['fold']}__s{x['job']['seed']}":x for x in read(STAGE/'HISTORY_AUDIT.json')['frozen_hstem_checkpoints']}

def create_seal(cohort):
    a,r=policy();jobs={}
    for j in cohort_jobs(cohort,r):
        jid=jobid(j);out=PHASE/'runs'/jid;m=read(out/'metrics.json');c=read(out/'config.json')
        if m['status']!='COMPLETE' or c['job']!=j or m['job']!=j:raise ValueError('cohort incomplete or config changed: '+jid)
        if m['registry_sha256']!=digest(PHASE/'pilot_registry.json'):raise ValueError('wrong registry')
        if digest(out/'best.pt')!=m['checkpoint_sha256'] or digest(out/'config.json')!=m['config_sha256']:raise ValueError('checkpoint/config changed: '+jid)
        jobs[jid]=dict(checkpoint_sha256=digest(out/'best.pt'),metrics_sha256=digest(out/'metrics.json'),config_sha256=digest(out/'config.json'))
    h0={}
    for jid,record in h0_records().items():
        p=Path(record['path'])/'model.pt';expected=record['artifacts']['model.pt']['sha256']
        if digest(p)!=expected:raise ValueError('frozen H0 changed')
        h0[jid]=dict(checkpoint_sha256=expected,path=str(p))
    obj=dict(status='COHORT_SEALED_BEFORE_DEV',cohort=cohort,amendment_sha256=digest(HERE/'AMENDMENT.json'),
             original_registry_sha256=digest(PHASE/'pilot_registry.json'),jobs=jobs,h0=h0,
             authorized_prediction_jobs=sorted(list(jobs)+(list(h0) if cohort in ('baseline48','retained54') else [])),
             no_target_selection=True,confirm_blind_access=False)
    path=HERE/f'seal_{cohort}.json'
    if path.exists():
        if read(path)!=obj:raise ValueError('immutable cohort seal changed')
    else:atomic_json(path,obj)
    commit([path],f'S24 budget amendment: seal {cohort} before report-only Dev evaluation')
    committed(path);return obj

_verified={}
def verify_seal(cohort):
    a,r=policy();path=HERE/f'seal_{cohort}.json';committed(path);token=digest(path)
    if _verified.get(cohort,{}).get('token')==token:return _verified[cohort]['seal']
    s=read(path)
    if s['status']!='COHORT_SEALED_BEFORE_DEV' or s['amendment_sha256']!=digest(HERE/'AMENDMENT.json'):raise PermissionError('invalid seal')
    expected={jobid(j) for j in cohort_jobs(cohort,r)}
    if set(s['jobs'])!=expected:raise PermissionError('incomplete cohort')
    for jid,entry in s['jobs'].items():
        out=PHASE/'runs'/jid
        if digest(out/'best.pt')!=entry['checkpoint_sha256'] or digest(out/'config.json')!=entry['config_sha256'] or digest(out/'metrics.json')!=entry['metrics_sha256']:raise PermissionError('sealed artifact changed')
    for jid,entry in s['h0'].items():
        if digest(Path(entry['path']))!=entry['checkpoint_sha256']:raise PermissionError('H0 changed')
    _verified[cohort]=dict(token=token,seal=s)
    return s

def dev_identity(jid,rx,cohort):
    # Role validation occurs before any waveform/cache IO.
    a,r=policy()
    expected={jobid(j):j for j in cohort_jobs(cohort,r)}
    if cohort in ('baseline48','retained54'):
        expected.update({jid:dict(method='h0',view='native',fold=record['job']['fold'],seed=record['job']['seed']) for jid,record in h0_records().items()})
    if jid not in expected:raise PermissionError('job not in sealed report cohort')
    j=expected[jid]
    if rx in P['blind_receivers_never_open'] or rx in P['confirm_receivers'] or rx not in P['folds'][j['fold']]['dev_val']:raise PermissionError('RX not authorized for this fold report')
    return j

def access_events():
    p=HERE/'access.jsonl'
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []

_cells={}
def dev_cell(jid,rx,cohort):
    j=dev_identity(jid,rx,cohort);s=verify_seal(cohort)
    import numpy as np
    if rx not in _cells:
        d=Path('D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/shen')/rx
        man=next(x for x in read(STAGE/'DATA_AUDIT.json')['files'] if x['rx']==rx)
        x=np.load(d/'x.npy',mmap_mode='r');y=np.load(d/'y.npy');rows=np.load(d/'rows.npy')
        import hashlib
        for arr,key in [(x,'selected_iq_float32_sha256'),(y,'selected_labels_sha256'),(rows,'selected_rows_sha256')]:
            if hashlib.sha256(arr.tobytes()).hexdigest()!=man[key]:raise ValueError('cache/identity changed')
        if x.shape!=(2000,2,8192) or not np.array_equal(np.bincount(y,minlength=10),np.full(10,200)):raise ValueError('packet budget changed')
        _cells[rx]=(x,y,rows)
    event=dict(time=now(),role='dev_report_only',cohort=cohort,job=jid,fold=j['fold'],rx=rx,packets=2000,
               amendment_sha256=digest(HERE/'AMENDMENT.json'),seal_sha256=digest(HERE/f'seal_{cohort}.json'))
    with (HERE/'access.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(event)+'\n')
    return _cells[rx]
