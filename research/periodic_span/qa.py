"""Independent arithmetic/role/trace verification; reads metadata and sealed outputs only."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while c:=f.read(16<<20):h.update(c)
    return h.hexdigest()


def main():
    reg=read(HERE/'REGISTRATION.json');jobs=read(HERE/'JOBS.json');seal=read(HERE/'TRAINING_SEAL.json')
    manifests=read(HERE/'EVALUATION_MANIFEST.json');summary=read(HERE/'SUMMARY.json');checks={}
    with gzip.open(HERE/'ANALYSIS.json.gz','rt',encoding='utf-8') as f:analysis=json.load(f)
    def require(name,test):
        checks[name]=bool(test)
        if not test:raise ValueError('Independent QA failed: '+name)
    require('all24_unique_jobs',len(jobs)==24 and len({j['id'] for j in jobs})==24)
    require('unchanged_fixed_window_STOP',sha(ROOT/'research/periodic_hstem/AUDIT_RESULT.json')==reg['prior_fixed_window_result_sha256'])
    for relative,digest in reg['code_sha256'].items():require('code:'+relative,sha(ROOT/relative)==digest)
    for relative,digest in reg['artifacts_sha256'].items():require('registered_artifact:'+relative,sha(HERE/relative)==digest)
    for name in ('REGISTRATION.json','TRAINING_SEAL.json','RUN_EVIDENCE.json'):
        blob=subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:research/periodic_span/'+name])
        require('committed:'+name,blob==(HERE/name).read_bytes())
    require('seal_registration',seal['registration_sha256']==sha(HERE/'REGISTRATION.json'))
    require('all24_sealed',set(seal['runs'])=={j['id'] for j in jobs} and all(seal['checks'].values()))
    require('evaluation_matches_seal',manifests['seal_sha256']==sha(HERE/'TRAINING_SEAL.json'))
    require('summary_decision_matches_full_analysis',summary['decision']==analysis['decision'])
    for endpoint,block in analysis['endpoints'].items():
        require('summary_aggregates:'+endpoint,summary['endpoints'][endpoint]['aggregates']=={a:{k:v for k,v in d.items() if k!='tx_rx_stream_mean'} for a,d in block['aggregates'].items()})
        require('summary_comparisons:'+endpoint,summary['endpoints'][endpoint]['comparisons']=={a:{k:v for k,v in d.items() if k not in ('cells','tx_rx_stream_mean')} for a,d in block['comparisons'].items()})
    # Only the five metadata files are ever opened here; never fit_x.
    pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids','val_ids'):
        path=Path(reg['source_input_whitelist'][name]['path'])
        require('metadata_hash:'+name,sha(path)==reg['source_input_whitelist'][name]['sha256'])
        pool[name]=np.load(path,allow_pickle=False)
    plan=read(HERE/'SOURCE_PLAN.json');views={v['id']:v for v in plan['views']}
    roles={}
    for name,view in views.items():
        held=[reg['rx_names'].index(r) for r in view['source_selection_rx']]
        r=pool['fit_rx']
        roles[name]=dict(fit_train=pool['train_ids'][~np.isin(r[pool['train_ids']],held)],
            checkpoint_val=pool['val_ids'][~np.isin(r[pool['val_ids']],held)],
            source_selection=np.flatnonzero(np.isin(r,held)).astype(np.int64))
        require('roles_disjoint:'+name,not any(np.intersect1d(roles[name][a],roles[name][b]).size for a,b in
            (('fit_train','checkpoint_val'),('fit_train','source_selection'),('checkpoint_val','source_selection'))))
    initial={};histories={};costs={}
    for job in jobs:
        key=job['id'];out=HERE/'artifacts'/key;c=read(out/'COMPLETE.json');history=read(out/'HISTORY.json')
        require('completion:'+key,sha(out/'COMPLETE.json')==seal['runs'][key]['complete_sha256'])
        require('budget:'+key,c['epochs']==40 and c['updates']==5080 and len(history)==40 and history[-1]['updates']==5080)
        require('completion_registration:'+key,c['registration_sha256']==sha(HERE/'REGISTRATION.json'))
        for name,digest in c['files_sha256'].items():require('fit_bytes:'+key+'/'+name,sha(out/name)==digest)
        best=min(range(40),key=lambda k:history[k]['val_rx_ce'])+1
        require('earliest_CE_minimum:'+key,c['best_epoch']==best)
        config=read(out/'CONFIG.json')
        require('no_selection_training:'+key,not config['source_selection_IQ_read'] and set(config['allowed_training_RX'])==set(views[job['view']]['fit_rx']))
        require('active_params:'+key,c['active_parameters']==reg['parameters'][job['arm']])
        with np.load(out/'SAMPLE_KEYS.npz',allow_pickle=False) as f:
            require('fit_row_identity:'+key,np.array_equal(f['train_ids'],roles[job['view']]['fit_train']) and np.array_equal(f['val_ids'],roles[job['view']]['checkpoint_val']))
            for epoch in range(1,41):
                order=np.load(out/'orders'/f'epoch_{epoch:02d}.npy',allow_pickle=False)
                require('permutation:'+key+':'+str(epoch),np.array_equal(np.sort(order),np.arange(len(f['train_ids']))))
        initial[key]=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)['model']
        histories[key]=history;costs[key]=dict(costs=c['costs'],total_wall_seconds=c['total_wall_seconds'],peak_cuda_allocated_gib=c['peak_cuda_allocated_gib'])
        for endpoint in ('selected','epoch10','final40'):
            ck=torch.load(out/(endpoint+'.pt'),map_location='cpu',weights_only=False)
            expected_epoch={'selected':best,'epoch10':10,'final40':40}[endpoint]
            require('endpoint_epoch:'+key+'/'+endpoint,ck['epoch']==expected_epoch)
            require('endpoint_hash:'+key+'/'+endpoint,sha(out/(endpoint+'.pt'))==seal['runs'][key]['checkpoints'][endpoint])
            with np.load(out/f'{endpoint}_val.npz',allow_pickle=False) as f:
                require('checkpoint_val_roles:'+key+'/'+endpoint,np.array_equal(f['pool_ids'],roles[job['view']]['checkpoint_val']))
                logits=f['logits'].astype(np.float64);y=f['y'];rx=f['rx']
                ce=np.log(np.exp(logits-logits.max(1,keepdims=True)).sum(1))+logits.max(1)-logits[np.arange(len(y)),y]
                equal_ce=np.mean([ce[rx==r].mean() for r in np.unique(rx)])
                require('checkpoint_CE:'+key+'/'+endpoint,abs(equal_ce-ck['checkpoint_val_rx_ce'])<2e-6)
    for view in views:
        for stream in ('S0','S1','S2'):
            ids=[j['id'] for j in jobs if j['view']==view and j['stream']==stream]
            for name,tensor in initial[ids[0]].items():
                if name.startswith('backbone.'):
                    require('common_initial:'+view+'/'+stream+'/'+name,all(torch.equal(tensor,initial[k][name]) for k in ids))
            for k in range(40):
                for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
                    require('paired_order:'+view+'/'+stream+'/'+str(k)+'/'+field,len({histories[i][k][field] for i in ids})==1)
            p,n=[f'{view}_{stream}_{a}' for a in ('P2','N19')]
            require('P2_N19_all_weights:'+view+'/'+stream,all(torch.equal(t,initial[n][name]) for name,t in initial[p].items()))
    expected={f'{j["id"]}/{e}_selection.npz' for j in jobs for e in ('selected','epoch10','final40')}
    require('complete72_predictions',set(manifests['files'])==expected)
    predictions={}
    for endpoint in ('selected','epoch10','final40'):
        preds={}
        for job in jobs:
            name=job['id']+'/'+endpoint+'_selection.npz';path=HERE/'artifacts'/name
            require('prediction_bytes:'+name,sha(path)==manifests['files'][name])
            with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
            rows=roles[job['view']]['source_selection']
            require('selection_roles:'+name,np.array_equal(data['pool_ids'],rows) and np.array_equal(data['y'],pool['fit_y'][rows]) and np.array_equal(data['day'],pool['fit_day'][rows]))
            require('selection_RX_names:'+name,np.array_equal(data['rx'],np.asarray(reg['rx_names'])[pool['fit_rx'][rows]]))
            require('finite_logits:'+name,data['logits'].shape==(len(rows),10) and np.isfinite(data['logits']).all())
            preds[(job['arm'],job['view'],job['stream'])]=data
        predictions[endpoint]=preds
    # Independent accuracy/CE and class-condition risk calculations, without C metric helpers.
    comparisons={};draws=np.random.default_rng(24501).integers(0,12,size=(10000,12))
    all_rx=sorted(r for v in views.values() for r in v['source_selection_rx'])
    for endpoint,preds in predictions.items():
        accuracy={};cross_entropy={}
        for (arm,view,stream),data in preds.items():
            logits=data['logits'].astype(np.float64);y=data['y'];prob=logits.argmax(1)
            losses=logits.max(1)+np.log(np.exp(logits-logits.max(1,keepdims=True)).sum(1))-logits[np.arange(len(y)),y]
            for rx in np.unique(data['rx']):
                mask=data['rx']==rx
                accuracy[(arm,stream,rx)]=float((prob[mask]==y[mask]).mean())
                cross_entropy[(arm,stream,rx)]=float(losses[mask].mean())
        for arm in ('P0','P1','P2','N19'):
            a=np.mean([accuracy[(arm,s,r)] for s in ('S0','S1','S2') for r in all_rx])
            ce=np.mean([cross_entropy[(arm,s,r)] for s in ('S0','S1','S2') for r in all_rx])
            result=analysis['endpoints'][endpoint]['aggregates'][arm]['rx_equal']
            require('independent_accuracy:'+endpoint+'/'+arm,abs(a-result['accuracy'])<1e-12)
            require('independent_CE:'+endpoint+'/'+arm,abs(ce-result['ce'])<1e-12)
        for first,second in (('P1','P2'),('P0','P2'),('N19','P2'),('P0','P1'),('P0','N19')):
            name=second+'-'+first;record=analysis['endpoints'][endpoint]['comparisons'][name]
            rx_delta=np.asarray([np.mean([accuracy[(second,s,r)]-accuracy[(first,s,r)] for s in ('S0','S1','S2')]) for r in all_rx])
            intervals=np.quantile(rx_delta[draws].mean(1),(.025,.975))
            require('independent_delta:'+endpoint+'/'+name,abs(rx_delta.mean()-record['rx_equal']['accuracy_delta'])<1e-12)
            require('independent_bootstrap:'+endpoint+'/'+name,np.allclose(intervals,record['paired_rx_bootstrap_95_intervals']['accuracy_delta'],rtol=0,atol=1e-12))
            risk=[]
            for view in views:
                base=preds[(first,view,'S0')]
                for rx in views[view]['source_selection_rx']:
                    for tx in range(10):
                        mask=(base['rx']==rx)&(base['y']==tx)
                        if not mask.any():continue
                        changes=[]
                        for stream in ('S0','S1','S2'):
                            left,right=preds[(first,view,stream)],preds[(second,view,stream)]
                            require('paired_inputs:'+endpoint+'/'+name+'/'+view+'/'+stream,np.array_equal(left['keys'],right['keys']))
                            changes.append(float((left['logits'][mask].argmax(1)==left['y'][mask]).mean()-(right['logits'][mask].argmax(1)==right['y'][mask]).mean()))
                        if mask.sum()>=50:risk.append(dict(rx=rx,tx=tx,support=int(mask.sum()),risk_increase=float(np.mean(changes))))
            worst=max(r['risk_increase'] for r in risk)
            require('independent_local_risk:'+endpoint+'/'+name,abs(worst-record['worst_supported_tx_rx_risk_increase']['risk_increase'])<1e-12)
            if endpoint=='selected':
                gate=dict(mean_accuracy_at_least_1pp=float(rx_delta.mean())>=.01,
                    paired_rx_interval_lower_above_zero=float(intervals[0])>0,
                    both_views_positive=all(np.mean([accuracy[(second,s,r)]-accuracy[(first,s,r)] for s in ('S0','S1','S2') for r in v['source_selection_rx']])>0 for v in views.values()),
                    at_least_two_streams_positive=sum(np.mean([accuracy[(second,s,r)]-accuracy[(first,s,r)] for r in all_rx])>0 for s in ('S0','S1','S2'))>=2,
                    worst_paired_rx_at_least_minus_2pp=float(rx_delta.min())>=-.02,
                    supported_tx_rx_risk_increase_at_most_5pp=all(r['risk_increase']<=.05 for r in risk))
                comparisons[name]=dict(checks=gate,accuracy_delta=float(rx_delta.mean()),max_supported_risk_increase=worst,supported_groups=len(risk))
                if name in ('P2-P1','P2-P0'):require('promotion_gate:'+name,gate==summary['decision']['promotion_gates'][name]['checks'])
    log=[json.loads(line) for line in (HERE/'ACCESS_LOG.jsonl').read_text(encoding='utf-8').splitlines()]
    specificity=dict(accuracy_positive=comparisons['P2-N19']['accuracy_delta']>0,
        CE_improves=analysis['endpoints']['selected']['aggregates']['P2']['rx_equal']['ce']<analysis['endpoints']['selected']['aggregates']['N19']['rx_equal']['ce'],
        both_views_positive=comparisons['P2-N19']['checks']['both_views_positive'])
    require('independent_physical_specificity',specificity==summary['decision']['specificity_P2_minus_N19'])
    promoted=all(all(comparisons[name]['checks'].values()) for name in ('P2-P1','P2-P0')) and all(specificity.values())
    expected_status='PERIODIC_SPAN_WORTH_SEPARATE_CONFIRMATION' if promoted else 'STOP_NO_PERIODIC_SPAN_PROMOTION'
    require('independent_final_decision',summary['decision']['status']==expected_status)
    whitelist={a['path'] for a in reg['source_input_whitelist'].values()}
    require('only_source_whitelist_access',all(row['path'] in whitelist for row in log))
    selection=[r for r in log if r['action']=='copy_source_selection_rows']
    require('exactly_two_sealed_selection_copies',len(selection)==2 and all(r['job']=='sealed_selection_evaluation' for r in selection))
    require('exact_training_role_copies',all(sum(row['job']==j['id'] and row['action']==action for row in log)==1
        for j in jobs for action in ('copy_fit_train_rows','copy_checkpoint_val_rows')))
    from datetime import datetime
    seal_commit=subprocess.check_output(['git','-C',str(ROOT),'log','-1','--format=%cI','--','research/periodic_span/TRAINING_SEAL.json'],text=True).strip()
    require('selection_after_committed_seal',all(datetime.fromisoformat(r['time'])>=datetime.fromisoformat(seal_commit) for r in selection))
    require('exact_scientific_totals',sum(read(HERE/'artifacts'/j['id']/'COMPLETE.json')['updates'] for j in jobs)==121920)
    result=dict(status='PASS',checks=checks,check_count=len(checks),primary_comparisons=comparisons,
        independent_IQ_reads=0,scientific_fits=24,epochs=960,gradient_updates=121920,
        source_selection_predictions=72,decision=summary['decision'],costs=costs,
        verified_artifacts=dict(registration_sha256=sha(HERE/'REGISTRATION.json'),training_seal_sha256=sha(HERE/'TRAINING_SEAL.json'),
            evaluation_manifest_sha256=sha(HERE/'EVALUATION_MANIFEST.json'),analysis_sha256=sha(HERE/'ANALYSIS.json.gz')))
    (HERE/'INDEPENDENT_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),decision=result['decision']['status'])))


if __name__=='__main__':main()
