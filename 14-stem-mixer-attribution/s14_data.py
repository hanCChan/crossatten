"""Only read S13's completed, role-authorized caches; never mutate S13 logs/cache."""
import numpy as np
from s14_core import HERE,ROOT,S13,read,sha,access,committed
from xa.data import source_split
from xa.wisig import receivers

PROTOCOL=read(ROOT/'configs/protocol.json')
def roles(fold):return next(f for f in PROTOCOL['folds'] if f['fold']==fold)
def shen_cell(rx,fold,role):
    allowed=roles(fold)['source'] if role=='source' else roles(fold)['dev_val'] if role=='dev_val' else PROTOCOL['confirm_receivers'] if role=='confirm' else []
    if rx not in allowed or rx in PROTOCOL['blind_receivers_never_open']:raise ValueError('Forbidden Shen RX/role')
    if role!='source':committed(HERE/'decision_p1_source.json')
    path=S13/'cache/shen'/rx
    manifest=read(path/'manifest.json')
    import hashlib
    x=np.load(path/'x.npy',mmap_mode='r');y=np.load(path/'y.npy',mmap_mode='r');rows=np.load(path/'rows.npy',mmap_mode='r')
    if x.shape!=(2000,2,8192) or y.shape!=(2000,) or set(np.unique(y))!=set(range(10)):raise ValueError('Shen cache shape/classes mismatch')
    if hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()!=manifest['x_sha256']:raise ValueError('Shen cache hash mismatch')
    if hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()!=manifest['rows_sha256']:raise ValueError('Shen row selection mismatch')
    access(dataset='Shen',rx=rx,fold=fold,role=role,source_manifest_sha256=sha(path/'manifest.json'),mode='read_only')
    return dict(x=x,y=y,rows=rows)
def shen_train(fold):
    output=[[],[]]
    for ri,rx in enumerate(roles(fold)['source']):
        d=shen_cell(rx,fold,'source')
        for out,ids in zip(output,source_split(d['y'],d['rows'])):
            out.append(dict(x=np.asarray(d['x'][ids]),y=np.asarray(d['y'][ids]),rx=np.full(len(ids),ri)))
    return [dict(x=np.concatenate([d['x'] for d in v]),y=np.concatenate([d['y'] for d in v]),rx=np.concatenate([d['rx'] for d in v])) for v in output]
def wifi_cell(dataset,fold,rx,day):
    source,_=receivers(dataset,fold)
    if rx not in source or day not in [1,2]:raise ValueError('Forbidden WiSig target RX/date')
    path=S13/'cache/wisig'/dataset/f'rx_{rx}_day{day}.npz'
    access(dataset=dataset,fold=fold,rx=rx,day=day,role='source_diagnostic',sha256=sha(path),mode='read_only')
    with np.load(path) as z:return dict(x=z['x'],y=z['y'])
