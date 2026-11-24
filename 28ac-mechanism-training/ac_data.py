"""Four-source C never opens its held file before the full cohort seal."""
import hashlib
import numpy as np
from ac_common import HERE,CACHE,P,C,validate,read,sha,event,committed,identity
from xa.data import source_split

def check_seal(stage):
    path=HERE/f'seal_{stage}.json'
    if not path.exists():raise PermissionError('Incomplete cohort: '+stage)
    r=validate()
    committed(path);s=read(path);assert s['registration_sha256']==sha(HERE/'registration.json')
    jobs=[j for j in r['jobs'] if j['stage']==stage]
    assert set(s['models'])=={identity(j) for j in jobs}
    for name,record in s['models'].items():
        for file,h in record.items():assert sha(HERE/'runs'/name/file)==h
    return s

def authorize(stage,rx,fold,role):
    if fold not in (0,1) or rx in P['confirm_receivers']+P['blind_receivers_never_open']:raise PermissionError('Forbidden RX/fold')
    if stage=='A' and role=='source' and rx in P['folds'][fold]['source']:return
    if stage=='C' and role=='source' and rx in C['splits'][fold]['training_receivers']:return
    if (stage=='A' and role=='dev_report' and rx in P['folds'][fold]['dev_val']) or (stage=='C' and role=='held_report' and rx==C['splits'][fold]['held_source_receiver']):
        # Delaying A Dev until C also seals prevents A observations influencing C.
        check_seal('A');check_seal('C');return
    raise PermissionError('Forbidden stage/RX/data role before waveform IO')

def cell(stage,rx,fold,role):
    authorize(stage,rx,fold,role);reg=validate();path=CACHE/rx;expected=reg['source_caches'][rx]
    x=np.load(path/'x.npy',mmap_mode='r');y=np.load(path/'y.npy');rows=np.load(path/'rows.npy')
    assert x.shape==(2000,2,8192) and y.shape==rows.shape==(2000,)
    for a,k in [(x,'selected_iq_float32_sha256'),(y,'selected_labels_sha256'),(rows,'selected_rows_sha256')]:
        assert hashlib.sha256(a.tobytes()).hexdigest()==expected[k]
    event(stage=stage,role=role,fold=fold,rx=rx,packets=2000,rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
    return dict(x=x,y=y,rows=rows)

def source(stage,fold):
    receivers=P['folds'][fold]['source'] if stage=='A' else C['splits'][fold]['training_receivers'];parts=[[],[]]
    for i,rx in enumerate(receivers):
        d=cell(stage,rx,fold,'source')
        for part,indices in zip(parts,source_split(d['y'],d['rows'])):
            part.append(dict(x=np.array(d['x'][indices]),y=d['y'][indices],rows=d['rows'][indices],rx=np.full(len(indices),i)))
    result=[{k:np.concatenate([p[k] for p in part]) for k in ['x','y','rx','rows']} for part in parts]
    expected=(8000,2000) if stage=='A' else (6400,1600)
    assert tuple(len(p['y']) for p in result)==expected
    return result
