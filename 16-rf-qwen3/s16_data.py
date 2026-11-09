"""Common source-pretrained frozen RF tokens; no real target receivers/dates."""
import copy
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from s16_core import HERE,S13,read,write,sha,access
from xa.wisig import receivers
from wifi import build_wifi,phase
from xa.explore_models import frequency_views
from xa.frontends import rms_normalize

def groups(dataset,fold,group='first'):
    source,_=receivers(dataset,fold);count=6 if dataset=='ManyRx' else 2
    if group not in ['first','last']:raise ValueError('Unknown source holdout group')
    held=source[:count] if group=='first' else source[-count:]
    return [r for r in source if r not in held],held

def cells(dataset,fold,names,days=(1,2),role='fit'):
    source,_=receivers(dataset,fold)
    if role not in ['fit','held','diagnostic'] or not set(names)<=set(source) or tuple(days)!=(1,2):raise ValueError('Forbidden WiSig RX/date')
    result={}
    for rx in names:
        for day in days:
            p=S13/'cache/wisig'/dataset/f'rx_{rx}_day{day}.npz'
            access(dataset=dataset,fold=fold,rx=rx,day=day,role=role,sha256=sha(p),mode='read_only')
            with np.load(p) as z:result[rx,day]=dict(x=z['x'],y=z['y'])
    return result

def flatten(raw):
    names=sorted({rx for rx,day in raw});xs=[];ys=[];rs=[]
    for (rx,day),d in raw.items():xs.append(d['x']);ys.append(d['y']);rs.append(np.full(len(d['y']),names.index(rx)))
    return dict(x=np.concatenate(xs),y=np.concatenate(ys),rx=np.concatenate(rs),rx_names=names)

class RFTokenizer(torch.nn.Module):
    def __init__(self,base):super().__init__();self.frequency=copy.deepcopy(base.frequency);self.requires_grad_(False);self.eval()
    def forward(self,x):
        # Exactly S13 B0 tokens(): normalize, then features() normalizes again.
        base=frequency_views(rms_normalize(rms_normalize(x)))
        return self.frequency(phase.phase_features_from_base(base,'first'))

def front_ref(job):
    if job['group']=='first' and job['split_seed']==16001:
        name=f"p2__{job['dataset']}__B0__f{job['fold']}__first__s16001";root=S13
    else:
        name=f"{job['front_phase']}__{job['dataset']}__B0__f{job['fold']}__{job['group']}__s{job['split_seed']}";root=HERE
    p=root/'runs'/name;m=read(p/'metrics.json');cfg=read(p/'config.json')
    if sha(p/'model.pt')!=m['model_sha256']:raise ValueError('RF front checkpoint changed')
    if cfg['job']['seed']!=job['split_seed'] or cfg['job']['group']!=job['group']:raise ValueError('RF validation partition mismatch')
    if cfg['job']['arm']!='B0' or cfg['job']['dataset']!=job['dataset'] or cfg['job']['fold']!=job['fold']:
        raise ValueError('RF front arm/dataset/fold mismatch')
    if root==S13 and (HERE/'design.json').exists():
        reference=next(r for r in read(HERE/'design.json')['front_references'] if r['fold']==job['fold'])
        if m['model_sha256']!=reference['checkpoint_sha256'] or sha(p/'config.json')!=reference['config_sha256']:
            raise ValueError('Pilot RF front differs from frozen design reference')
    return p,m,cfg

@torch.no_grad()
def prepare(job):
    p,metric,cfg=front_ref(job);key=f"{job['dataset']}_f{job['fold']}_{job['group']}_s{job['split_seed']}"
    dest=HERE/'cache'/key;manifest=dest/'manifest.json'
    if manifest.exists():
        m=read(manifest)
        if m['front_sha256']!=metric['model_sha256']:raise ValueError('RF token cache uses another front')
        for name,digest in m['files'].items():
            if sha(dest/name)!=digest:raise ValueError('RF token cache changed')
        return dest,m
    fit,held=groups(job['dataset'],job['fold'],job['group']);all_names=sorted(fit+held)
    base=build_wifi(job['dataset'],'B0',cfg['data_seed'])
    base.load_state_dict(torch.load(p/'model.pt',map_location='cpu',weights_only=True))
    tokenizer=RFTokenizer(base).cuda();del base
    dest.mkdir(parents=True,exist_ok=True);parts={}
    for part,names in [('fit',fit),('held',held)]:
        raw=cells(job['dataset'],job['fold'],names,role='fit' if part=='fit' else 'held');xs=[];ys=[];rs=[];ds=[]
        for (rx,day),d in raw.items():
            values=[]
            for i in range(0,len(d['x']),128):values.append(tokenizer(torch.from_numpy(np.ascontiguousarray(d['x'][i:i+128])).cuda()).cpu().numpy())
            xs.append(np.concatenate(values));ys.append(d['y']);rs.append(np.full(len(d['y']),all_names.index(rx)));ds.append(np.full(len(d['y']),day))
        parts[part]=dict(x=np.concatenate(xs),y=np.concatenate(ys),rx=np.concatenate(rs),day=np.concatenate(ds))
    ids=train_test_split(np.arange(len(parts['fit']['y'])),test_size=.3,random_state=job['split_seed'])
    for name,index in [('train',ids[0]),('val',ids[1])]:
        for k,v in parts['fit'].items():np.save(dest/f'{name}_{k}.npy',np.ascontiguousarray(v[index]))
    for k,v in parts['held'].items():np.save(dest/f'held_{k}.npy',np.ascontiguousarray(v))
    if len(ids[0])!=cfg['train_packets'] or len(ids[1])!=cfg['val_packets']:raise ValueError('RF source split does not match pretraining')
    files={f.name:sha(f) for f in dest.glob('*.npy')}
    m=dict(front_sha256=metric['model_sha256'],front_config_sha256=sha(p/'config.json'),rf_active_params=sum(v.numel() for v in tokenizer.parameters()),
        frozen_front=True,rx_names=all_names,fit_rx=fit,held_rx=held,split_seed=job['split_seed'],counts={n:len(parts['held']['y']) if n=='held' else len(ids[0 if n=='train' else 1]) for n in ['train','val','held']},files=files)
    write(manifest,m);del tokenizer;torch.cuda.empty_cache();return dest,m

def load(job):
    path,manifest=prepare(job)
    return {part:{k:np.load(path/f'{part}_{k}.npy',mmap_mode='r') for k in ['x','y','rx','day']} for part in ['train','val','held']},manifest
