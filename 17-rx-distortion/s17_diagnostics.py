"""D0: source-training TX/day-conditioned nuisance statistics; fixed exit."""
import numpy as np
from s17_core import HERE,ROOT,read,write,sha,validate,now,commit
from s17_data import load
from s17_features import features,analyze,cell_profiles
from s17_jobs import run_id,profiles_path,cache_key

def profile_only(job):
    data,meta=load(job);tr=data['train'];f=features(tr['x']);profiles={}
    for family in ['amp','iq']:
        c=cell_profiles(f[family],tr,meta['rx_names']);ids=np.unique(c['rx'])
        profiles[family]=np.stack([c['residual'][c['rx']==r].mean(0) for r in ids]).tolist()
    record=dict(job=job,profiles=profiles,scope='Source train only; TX/day-conditioned RX-associated proxies, not pure hardware estimates',
        caps=dict(amp_db=3,iq_beta=.05),fit_rx=meta['fit_rx'],source_manifest_sha256=sha(HERE/'cache'/cache_key(job)/'fit_manifest.json'))
    write(profiles_path(job),record);return record

def run():
    reg=validate();jobs=read(HERE/'jobs_pilot.json');baselines=[j for j in jobs if j['arm']=='M0'];rows=[]
    for j in baselines:
        data,meta=load(j);ref=next(r for r in reg['d0_frozen_references'] if r['fold']==j['fold'])
        if sha(ROOT/ref['metrics'])!=ref['sha256']:raise ValueError('Frozen S16 source reference changed')
        per=read(ROOT/ref['metrics'])['source_validation']['per_rx'];errors={}
        for rx,v in per.items():
            cm=np.asarray(v['confusion']);support=cm.sum(1);valid=support>0
            errors[meta['rx_names'].index(rx)]=1-float(np.mean(np.diag(cm)[valid]/support[valid]))
        f=features(data['train']['x']);record=dict(job=j,families={},error_reference=ref,
            error_definition='S16 frozen MHA source-validation TX-macro error pooled over days1-2; per-day errors unavailable')
        for family in ['amp','phase','iq']:
            record['families'][family]=analyze(f[family],data['train'],meta['rx_names'],errors)
            print('D0',j['fold'],family,record['families'][family]['stable'],flush=True)
        write(HERE/f"D0_fold{j['fold']}.json",record);profile_only(j);rows.append(record)
    decisions={}
    for family in ['amp','phase','iq']:
        values=[r['families'][family] for r in rows];rhos=[v['source_val_error_spearman'] for v in values]
        stable=sum(v['stable'] for v in values)>=3;associated=float(np.median(rhos))>=.2 and sum(v>0 for v in rhos)>=3
        decisions[family]=dict(eligible=bool(stable and associated and family!='phase'),stable_folds=int(sum(v['stable'] for v in values)),
            error_rho_median=float(np.median(rhos)),positive_error_association_folds=int(sum(v>0 for v in rhos)),
            reason='Phase audited/intervened, not converted to an estimated compensator from unpaired 256-sample packets' if family=='phase' else 'RX stability plus model error association required')
    result=dict(time=now(),status='GO_D1_SCREEN' if any(v['eligible'] for v in decisions.values()) else 'STOP_NO_STABLE_ERROR_LINK',
        decisions=decisions,eligible=[k for k,v in decisions.items() if v['eligible']],held_source_read=False,real_target_access=False)
    write(HERE/'decision_d0.json',result)
    commit(list(HERE.glob('D0_fold*.json'))+[HERE/'decision_d0.json']+list((HERE/'profiles').glob('*.json')),'S17 freeze D0 source mechanism evidence and nuisance bounds before interventions')

if __name__=='__main__':run()
