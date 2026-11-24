"""All targets are report-only after A6 and C12 complete-cohort seals."""
import gc
from pathlib import Path
import numpy as np
import torch
from ac_common import HERE,P,C,validate,read,write,sha,identity,now
from ac_data import check_seal,cell,source
from ac_stats import summary,contrast,c_pilot
from s28_models import build
from s27b_training import setup
from s27b_features import extract
from geometry_stats import geometry
from xa.data import source_split
from xa.train import stable_seed
import train as canonical

def metrics(logits,y):
    result=canonical.rx_metrics(logits,y);v=logits.astype(np.float64);mx=v.max(1)
    result['ce']=float(np.mean(mx+np.log(np.exp(v-mx[:,None]).sum(1))-v[np.arange(len(y)),y]))
    other=logits.copy();other[np.arange(len(y)),y]=-np.inf
    result['true_margin_mean']=float(np.mean(logits[np.arange(len(y)),y]-other.max(1)))
    result['per_tx_recall']=(np.array(result['confusion']).diagonal()/np.maximum(np.array(result['confusion']).sum(1),1)).tolist()
    return result

def load_model(stage,arm,fold,seed,reg):
    data_seed=stable_seed('P-H5x160' if stage=='A' else 'S27C-H4x160',fold,seed);setup(data_seed)
    if stage=='A' and arm!='H2_LA':path=Path(next(q['checkpoint'] for q in reg['references'] if q['arm']==arm and q['fold']==fold and q['seed']==seed))
    else:path=HERE/'runs'/identity(dict(stage=stage,arm=arm,fold=fold,seed=seed))/'model.pt'
    model=build(arm,data_seed).cuda().eval();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True),strict=True)
    return model,path

def save_predictions(stage,arm,fold,seed,rx,**data):
    name=identity(dict(stage=stage,arm=arm,fold=fold,seed=seed));out=HERE/'predictions'/name;out.mkdir(parents=True,exist_ok=True)
    path=out/(rx+'.npz');np.savez_compressed(path,**data);return path

def a_evaluate(reg):
    dev=[];src=[];recovery=[]
    for f in (0,1):
        _,va=source('A',f)
        for arm in ['H0','LA','H2','H2_LA','LP']:
            for s in (201,202,203):
                model,path=load_model('A',arm,f,s,reg);lg=canonical.predict(model,va['x']).cpu().numpy();old=read(path.parent/'metrics.json')
                assert canonical.score(lg,va['y'],va['rx'])==old['source_validation']
                save_predictions('A',arm,f,s,'SOURCE',logits=lg,y=va['y'],rx=va['rx'],rows=va['rows'])
                for ri,rx in enumerate(P['folds'][f]['source']):
                    mask=va['rx']==ri;src.append(dict(arm=arm,fold=f,seed=s,rx=rx,**metrics(lg[mask],va['y'][mask])))
                del model;torch.cuda.empty_cache()
        for rx in P['folds'][f]['dev_val']:
            data=cell('A',rx,f,'dev_report')
            for arm in ['H0','LA','H2','H2_LA','LP']:
                for s in (201,202,203):
                    model,path=load_model('A',arm,f,s,reg);lg=canonical.predict(model,np.array(data['x'])).cpu().numpy()
                    prediction=save_predictions('A',arm,f,s,rx,logits=lg,y=data['y'],rows=data['rows'])
                    dev.append(dict(arm=arm,fold=f,seed=s,rx=rx,checkpoint_sha256=sha(path),predictions_sha256=sha(prediction),**metrics(lg,data['y'])))
                    if arm!='H2_LA':
                        ref=next(q for q in reg['references'] if q['arm']==arm and q['fold']==f and q['seed']==s);record=ref['predictions'][rx]
                        assert sha(record['path'])==record['sha256']
                        with np.load(record['path'],allow_pickle=False) as z:
                            assert np.array_equal(z[record['y_key']],data['y'])
                            same=np.array_equal(z[record['logits_key']],lg)
                        assert same,(arm,f,s,rx,'archived prediction mismatch')
                        recovery.append(dict(arm=arm,fold=f,seed=s,rx=rx,logits_bit_exact=True))
                    del model;torch.cuda.empty_cache()
            print('A report-only Dev',f,rx,'all15models',flush=True)
        del va;gc.collect()
    write(HERE/'EVALUATION_A.json',dict(time=now(),dev_records=dev,source_records=src,baseline_recovery=recovery,
        dev_predictions=300000,source_predictions=60000,models=30,old_models=24,new_models=6))

def geometry_record(x,y,rx):
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x.astype(np.float64),axis=1)<=1e-12):return dict(status='UNDEFINED_ZERO_OR_NONFINITE')
    q=geometry(x,y,rx)
    if len(np.unique(rx))==1:
        q['algebraic_regularized_Jc']=q['tx_to_rx_centroid_ratio'];q['tx_to_rx_centroid_ratio']=None
        q['across_RX_interpretation']='NOT_IDENTIFIABLE_SINGLE_RX'
    return dict(status='VALID',**q)

def c_evaluate(reg):
    src=[];held=[];geom=[]
    for f in (0,1):
        _,va=source('C',f);rx=C['splits'][f]['held_source_receiver'];d=cell('C',rx,f,'held_report');_,probe=source_split(d['y'],d['rows'])
        for arm in ['H0','LP']:
            for s in (201,202,203):
                model,path=load_model('C',arm,f,s,reg)
                for role,data,indices in [('SOURCE',va,np.arange(len(va['y']))),('HELD',d,probe)]:
                    features=extract(model,data['x']);lg=features['logits']
                    ri=data['rx'] if role=='SOURCE' else np.zeros(len(data['y']),dtype=np.int64)
                    if role=='SOURCE':
                        assert canonical.score(lg,data['y'],ri)==read(path.parent/'metrics.json')['source_validation']
                        for j,name in enumerate(C['splits'][f]['training_receivers']):
                            mask=ri==j;src.append(dict(arm=arm,fold=f,seed=s,rx=name,**metrics(lg[mask],data['y'][mask])))
                    else:held.append(dict(arm=arm,fold=f,seed=s,rx=rx,checkpoint_sha256=sha(path),**metrics(lg,data['y'])))
                    save_predictions('C',arm,f,s,role,logits=lg,y=data['y'],rx=ri,rows=data['rows'])
                    target=HERE/'predictions'/identity(dict(stage='C',arm=arm,fold=f,seed=s))/(role+'_features.npz')
                    np.savez_compressed(target,**{k:features[k][indices] for k in ['tokens','pre_backend','final']},
                        y=data['y'][indices],rx=ri[indices],rows=data['rows'][indices],logits=lg[indices])
                    for layer in ['tokens','pre_backend','final']:
                        geom.append(dict(arm=arm,fold=f,seed=s,role=role,layer=layer,**geometry_record(features[layer][indices],data['y'][indices],ri[indices])))
                del model;gc.collect();torch.cuda.empty_cache()
                print('C source/held frozen evaluation',arm,f,s,'all3fixedtaps',flush=True)
        del va,d;gc.collect()
    write(HERE/'EVALUATION_C.json',dict(time=now(),source_records=src,held_records=held,geometry=geom,
        source_predictions=19200,held_predictions=24000,geometry_vectors=72000,receiver_units=2,models=12))

def report_results():
    a=read(HERE/'EVALUATION_A.json');c=read(HERE/'EVALUATION_C.json')
    pairs=[('GELU_with_BN',{'H0':1,'LA':-1}),('GELU_without_BN',{'H2':1,'H2_LA':-1}),
        ('BN_with_GELU',{'H0':1,'H2':-1}),('BN_without_GELU',{'LA':1,'H2_LA':-1}),('H2_LA_minus_LP',{'H2_LA':1,'LP':-1})]
    weights={'H0':1,'LA':-1,'H2':-1,'H2_LA':1}
    write(HERE/'RESULTS.json',dict(time=now(),status='COMPLETE_REGISTERED_A_AND_C',new_fits=18,reused_models=24,
        A=dict(dev_summary=[summary(a['dev_records'],arm) for arm in ['H0','LA','H2','H2_LA','LP']],
            source_summary=[summary(a['source_records'],arm) for arm in ['H0','LA','H2','H2_LA','LP']],
            primary_interaction=contrast(a['dev_records'],weights),source_interaction=contrast(a['source_records'],weights),
            secondary=[dict(name=name,**contrast(a['dev_records'],w)) for name,w in pairs]),
        C=dict(source_summary=[summary(c['source_records'],arm) for arm in ['H0','LP']],held_summary=[summary(c['held_records'],arm) for arm in ['H0','LP']],
            pilot=c_pilot(c['source_records'],c['held_records'])),scope='Exploratory A; C has two historically exposed heldRX only. No population inference or unique physical/optimization cause.'))

def main():
    reg=validate();check_seal('A');check_seal('C')
    if not (HERE/'EVALUATION_A.json').exists():a_evaluate(reg)
    if not (HERE/'EVALUATION_C.json').exists():c_evaluate(reg)
    report_results()

if __name__=='__main__':main()
