"""Legal fold-specific source cells only, immutable S13 IQ caches."""
import numpy as np
from sklearn.model_selection import train_test_split
from s17_core import HERE,S13,read,write,sha,access
from xa.wisig import receivers
from s17_jobs import split_seed,cache_key

def groups(dataset,fold,group):
    source,_=receivers(dataset,fold);k=6 if dataset=='ManyRx' else 2
    if group not in ['first','last']:raise ValueError('Unknown source group')
    held=source[:k] if group=='first' else source[-k:]
    return [rx for rx in source if rx not in held],held

def cells(dataset,fold,names,days=(1,2),role='fit'):
    source,_=receivers(dataset,fold)
    if not set(names)<=set(source) or tuple(days)!=(1,2) or role not in ['fit','held']:
        raise ValueError('Real target RX/date3-4/unknown role forbidden')
    out={}
    for rx in names:
        for day in days:
            p=S13/'cache/wisig'/dataset/f'rx_{rx}_day{day}.npz'
            access(dataset=dataset,fold=fold,rx=rx,day=day,role=role,sha256=sha(p),mode='read_only')
            with np.load(p) as z:out[rx,day]=dict(x=z['x'],y=z['y'])
    return out

def cache(job,held=False):
    fit,test=groups(job['dataset'],job['fold'],job['group']);names=sorted(fit+test)
    dest=HERE/'cache'/cache_key(job);dest.mkdir(parents=True,exist_ok=True)
    for part,rxs in [('fit',fit)]+([('held',test)] if held else []):
        manifest=dest/f'{part}_manifest.json'
        if manifest.exists():
            for name,digest in read(manifest)['sha256'].items():
                if sha(dest/name)!=digest:raise ValueError('Local source cache changed')
            continue
        raw=cells(job['dataset'],job['fold'],rxs,role=part);values={k:[] for k in ['x','y','rx','day']}
        for (rx,day),d in raw.items():
            values['x'].append(d['x']);values['y'].append(d['y'])
            values['rx'].append(np.full(len(d['y']),names.index(rx)));values['day'].append(np.full(len(d['y']),day))
        for key,rows in values.items():np.save(dest/f'{part}_{key}.npy',np.concatenate(rows))
        if part=='fit':
            tr,va=train_test_split(np.arange(sum(len(v) for v in values['y'])),test_size=.3,random_state=split_seed(job))
            np.save(dest/'train_ids.npy',tr);np.save(dest/'val_ids.npy',va)
        files=list(dest.glob(f'{part}_*.npy'))+([dest/'train_ids.npy',dest/'val_ids.npy'] if part=='fit' else [])
        write(manifest,dict(rx_names=names,fit_rx=fit,held_rx=test,sha256={p.name:sha(p) for p in files},source_only=True))
    return dest,read(dest/'fit_manifest.json')

def load(job,held=False):
    dest,manifest=cache(job,held);pool={k:np.load(dest/f'fit_{k}.npy',mmap_mode='r') for k in ['x','y','rx','day']}
    out={}
    for part in ['train','val']:
        ids=np.load(dest/f'{part}_ids.npy');out[part]={k:np.array(v[ids],copy=True) for k,v in pool.items()}
    if held:out['held']={k:np.load(dest/f'held_{k}.npy',mmap_mode='r') for k in ['x','y','rx','day']}
    return out,manifest
