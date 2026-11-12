"""Only the original source-validation rows; no training or target loader."""
import hashlib
import numpy as np
from s23_core import HERE,ROOT,S13,read,sha,write,now
from xa.data import source_split

PROTOCOL=read(ROOT/'configs/protocol.json')
def roles(fold):return next(f for f in PROTOCOL['folds'] if f['fold']==fold)
def source_val(fold):
    output=[]
    for ri,rx in enumerate(roles(fold)['source']):
        if rx in PROTOCOL['blind_receivers_never_open']:raise ValueError('Forbidden receiver')
        p=S13/'cache/shen'/rx;manifest=read(p/'manifest.json')
        x=np.load(p/'x.npy',mmap_mode='r');y=np.load(p/'y.npy',mmap_mode='r');rows=np.load(p/'rows.npy',mmap_mode='r')
        assert x.shape==(2000,2,8192)
        assert hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()==manifest['rows_sha256']
        _,ids=source_split(y,rows)
        assert len(ids)==400 and np.array_equal(np.bincount(y[ids],minlength=10),np.full(10,40))
        selected=np.ascontiguousarray(x[ids]);labels=np.asarray(y[ids]);original_rows=np.asarray(rows[ids])
        output.append(dict(x=selected,y=labels,rx=np.full(400,ri),rows=original_rows))
        with (HERE/'access.jsonl').open('a',encoding='utf-8') as f:
            import json
            f.write(json.dumps(dict(time=now(),dataset='Shen',role='source_val',fold=fold,rx=rx,selected_packets=400,selected_x_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),manifest_sha256=sha(p/'manifest.json'),mode='read_only'))+'\n')
    return {k:np.concatenate([d[k] for d in output]) for k in ['x','y','rx','rows']}

def probe_ids(data):
    return np.asarray([np.flatnonzero((data['rx']==r)&(data['y']==y))[0] for r in range(5) for y in range(10)])
