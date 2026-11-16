"""Guarded cache: exact PACKETS identities, source-only until a committed global seal."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent;ROOT=STAGE.parent
sys.path.insert(0,str(STAGE))
from s24_common import P,sha,source_path,split_indices,write_json
CACHE=Path('D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/shen')

def log(**fields):
    import datetime
    record=dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),**fields)
    with (HERE/'access.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(record)+'\n')

def load_cell(rx,fold,role,who,seal=None):
    f=next(f for f in P['folds'] if f['fold']==fold)
    if rx in P['blind_receivers_never_open'] or rx in P['confirm_receivers']:raise PermissionError(rx)
    if role=='source':
        source_path(Path('D:/crossatten/stage_1/data/raw'),rx,fold)
    elif role=='dev':
        if rx not in f['dev_val'] or seal is None:raise PermissionError('Dev needs global seal')
        if seal['status']!='ALL_REGISTERED_FITS_SEALED':raise PermissionError('unsealed')
        from execution import committed,read,validate_registry,digest,jobid
        canonical=HERE/'global_seal.json'
        committed(canonical)
        if read(canonical)!=seal:raise PermissionError('fabricated seal')
        registry=validate_registry()
        if seal['registry_sha256']!=digest(HERE/'pilot_registry.json') or set(seal['jobs'])!={jobid(j) for j in registry['jobs']}:raise PermissionError('incomplete seal')
        for jid,record in seal['jobs'].items():
            if read(HERE/'runs'/jid/'metrics.json')['status']!='COMPLETE':raise PermissionError('fit incomplete')
    else:raise PermissionError(role)
    # Check exact source hashes recorded at phase one. Cached data for held RX was source in the other fold.
    man=json.loads((STAGE/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    expected=next(v for v in man['files'] if v['rx']==rx)
    d=CACHE/rx
    x=np.load(d/'x.npy',mmap_mode='r');y=np.load(d/'y.npy');rows=np.load(d/'rows.npy')
    if sha(rows.tobytes())!=expected['selected_rows_sha256'] or sha(x.tobytes())!=expected['selected_iq_float32_sha256']:raise ValueError('cache changed: '+rx)
    if sha(y.tobytes())!=expected['selected_labels_sha256']:raise ValueError('labels changed')
    log(rx=rx,fold=fold,role=role,who=who,packets=len(x))
    return x,y,rows

def source(fold,who):
    out={'train':[],'val':[]}
    with (STAGE/'PACKETS.csv').open(encoding='utf-8') as f:table=list(csv.DictReader(f))
    for domain,rx in enumerate(P['folds'][fold]['source']):
        x,y,rows=load_cell(rx,fold,'source',who)
        tr,va=split_indices(y,rows)
        for key,ids,split in [('train',tr,'train'),('val',va,'source_val')]:
            allowed={int(r['raw_row']) for r in table if int(r['fold'])==fold and r['rx']==rx and r['split']==split}
            assert set(rows[ids].tolist())==allowed
            out[key].append((np.asarray(x[ids]),y[ids],np.full(len(ids),domain,dtype=np.int64)))
    return {key:tuple(torch.from_numpy(np.concatenate([v[i] for v in cells])) for i in range(3)) for key,cells in out.items()}

def preprocess(x,view,method):
    if view=='strict_ib':
        z=torch.complex(x[:,0].double(),x[:,1].double())
        mask=torch.fft.fftfreq(8192,d=1e-6,device=x.device,dtype=torch.float64).abs()<=62500
        z=torch.fft.ifft(torch.fft.fft(z)*mask)
        x=torch.stack((z.real,z.imag),1).float()
    if method in ('fan','ib','ask'):
        x=x/torch.sqrt((x[:,0]**2+x[:,1]**2).mean(-1).clamp_min(1e-12))[:,None,None]
    return x
