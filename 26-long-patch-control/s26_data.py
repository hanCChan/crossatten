"""Original cached packets; forbid Confirm/blind and pre-seal Dev IO."""
import hashlib
import numpy as np
from s26_common import HERE,CACHE,P,read,sha,committed,validate,event,identity
from xa.data import source_split


def authorize(rx,fold,role):
    if fold not in (0,1) or rx in P['confirm_receivers'] or rx in P['blind_receivers_never_open']:
        raise PermissionError('Forbidden fold/RX')
    if role=='source' and rx in P['folds'][fold]['source']:
        return
    if role=='dev_report_only' and rx in P['folds'][fold]['dev_val']:
        check_seal()
        return
    raise PermissionError('Forbidden data role before waveform IO')


def check_seal():
    p=HERE/'seal.json'
    if not p.exists():raise PermissionError('All six checkpoints must be sealed before Dev IO')
    r=validate();committed(p);s=read(p)
    assert s['registration_sha256']==sha(HERE/'registration.json')
    assert set(s['models'])=={identity(j['fold'],j['seed']) for j in r['jobs']}
    for name, record in s['models'].items():
        for filename,expected in record.items():
            assert sha(HERE/'runs'/name/filename)==expected,(name,filename)
    return s


def cell(rx,fold,role):
    authorize(rx,fold,role)
    r=validate();expected=r['source_caches'][rx];path=CACHE/rx
    x=np.load(path/'x.npy',mmap_mode='r');y=np.load(path/'y.npy');rows=np.load(path/'rows.npy')
    assert x.shape==(2000,2,8192) and y.shape==rows.shape==(2000,)
    for value,key in [(x,'selected_iq_float32_sha256'),(y,'selected_labels_sha256'),(rows,'selected_rows_sha256')]:
        assert hashlib.sha256(value.tobytes()).hexdigest()==expected[key],(rx,key)
    event(role=role,fold=fold,rx=rx,packets=2000,mode='read_only',rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest())
    return dict(x=x,y=y,rows=rows)


def source(fold):
    output=[[],[]]
    for ri,rx in enumerate(P['folds'][fold]['source']):
        d=cell(rx,fold,'source')
        for part,indices in zip(output,source_split(d['y'],d['rows'])):
            part.append(dict(x=np.asarray(d['x'][indices]),y=np.asarray(d['y'][indices]),rx=np.full(len(indices),ri)))
    result=[{k:np.concatenate([d[k] for d in part]) for k in ['x','y','rx']} for part in output]
    assert len(result[0]['y'])==8000 and len(result[1]['y'])==2000
    return result
