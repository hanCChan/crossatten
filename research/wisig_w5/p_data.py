"""Selective auxiliary IQ and original C roles, with no historical writers."""
import sys
import numpy as np
from common import *
from p_models import C,A
sys.path.insert(0,str(ROOT/'research/wisig_w3'))
from selective_pickle import read_rows


def auxiliary(plan,diagnostic=False):
    if diagnostic:
        committed([HERE/'P_MODEL_SEAL.json']);assert read(HERE/'P_MODEL_SEAL.json')['new_fits']==12
        txs=plan['aux_diagnostic_TX'];role='aux24_after_model_seal'
    else:txs=plan['aux_train_TX'];role='aux72_fit_and_checkpoint_validation'
    source=plan['raw_pickle'];p=Path(source['path'])
    assert p.stat().st_size==source['bytes']
    # Full-file SHA reads are opaque identity checks, not ndarray decoding.
    checked(source,'opaque_source_pickle_hash',role)
    lookup={(c['tx'],c['rx'],c['day']):c for c in plan['cells']}
    allowed={c['key'] for c in plan['cells'] if c['tx'] in txs}
    buckets={'diagnostic':[]} if diagnostic else {'train':[],'val':[]};access=[]
    for t,tx in enumerate(txs):
        for r,rx in enumerate(plan['aux_RX']):
            for day in (1,2):
                cell=lookup[tx,rx,day];assert cell['eq']==0 and cell['rows']==[0,32]
                raw=read_rows(p,cell,allowed);x=raw.transpose(0,2,1)
                power=(x[:,0]**2+x[:,1]**2).mean(1)
                if not np.isfinite(x).all() or not np.isfinite(power).all() or (power<=0).any():raise ValueError('Invalid source IQ')
                x=(x/np.sqrt(power)[:,None,None]).astype(np.float32)
                for part,lo,hi in ([('diagnostic',0,32)] if diagnostic else [('train',0,24),('val',24,32)]):
                    keys=np.array([f'{tx}|{rx}|D{day}|raw|{i}' for i in range(lo,hi)])
                    ids=(((t*6+r)*2+(day-1))*32)+np.arange(lo,hi)
                    buckets[part].append(dict(x=x[lo:hi],y=np.full(hi-lo,t,np.int64),rx=np.full(hi-lo,r,np.int64),
                        day=np.full(hi-lo,day,np.int64),keys=keys,pool_ids=ids.astype(np.int64)))
                access.append(dict(key=cell['key'],offset=cell['offset'],rows=[0,32],bytes=raw.nbytes,
                    raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),role=role))
    data={part:{k:np.concatenate([b[k] for b in cells]) for k in ('x','y','rx','day','keys','pool_ids')} for part,cells in buckets.items()}
    write('P_AUX24_READS.json' if diagnostic else 'P_AUX72_READS.json',dict(status='PASS',cells=access,
        packets=sum(len(v['y']) for v in data.values()),role=role,forbidden_RX_day_TX_reads=0))
    return data


def source_metadata(plan):
    source=plan['source'];pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids','val_ids'):
        pool[name]=np.load(checked(source['source_input_whitelist'][name],'read_C_source_metadata',name),allow_pickle=False)
    assert set(pool['fit_day'])=={1,2} and pool['fit_y'].shape==(34668,)
    assert {source['rx_names'][int(r)] for r in pool['fit_rx']}==set(source['fit_rx'])
    pool['keys']=A.make_keys(pool['fit_y'],pool['fit_rx'],pool['fit_day'],source['rx_names'],source['source_cell_sha256'])
    ids=C.role_ids(source,plan['view'],pool)
    return pool,ids


def downstream(plan,held=False):
    if held:committed([HERE/'P_MODEL_SEAL.json'])
    pool,ids=source_metadata(plan)
    roles=('source_selection',) if held else ('fit_train','checkpoint_val')
    p=checked(plan['source']['source_input_whitelist']['fit_x'],'mmap_C_V1_admitted_IQ_rows',','.join(roles))
    x=np.load(p,mmap_mode='r',allow_pickle=False);assert x.shape==(34668,2,256) and x.dtype==np.float32
    result={};audit=[]
    for role in roles:
        i=ids[role];rx={plan['source']['rx_names'][int(r)] for r in pool['fit_rx'][i]}
        expected=set(plan['view']['source_selection_rx'] if held else plan['view']['fit_rx']);assert rx==expected
        assert not rx&set(plan['source']['forbidden_target_rx']+plan['source']['forbidden_historical_held_rx'])
        result[role]=dict(x=np.array(x[i],copy=True),y=pool['fit_y'][i].astype(np.int64),rx=pool['fit_rx'][i].astype(np.int64),
            day=pool['fit_day'][i].astype(np.int64),keys=pool['keys'][i],pool_ids=np.asarray(i))
        audit.append(dict(role=role,rows=len(i),RX=sorted(rx),keys_sha256=A.semantic_hash(pool['keys'][i]),
            indices_sha256=hashlib.sha256(i.tobytes()).hexdigest()))
    write('P_HELD_READS.json' if held else 'P_DOWNSTREAM_FIT_READS.json',dict(status='PASS',roles=audit,IQ_rows_read=sum(r['rows'] for r in audit),
        forbidden_target_D3_D4_historical_held6_reads=0))
    return result
