"""Second implementation of sealed statistics and execution audit; never reads IQ."""
from __future__ import annotations

from datetime import datetime
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

import common as H
import executor as E

HERE,ROOT=H.HERE,H.ROOT
STREAMS=('S0','S1','S2')
ARMS=('F0','F1','F2','E0')
checks={}


def require(name,value):
    checks[name]=bool(value)
    if not value:raise ValueError('Independent recomputation failed: '+name)


def close(name,value,expected,tolerance=1e-12):
    require(name,np.allclose(value,expected,rtol=0,atol=tolerance))


def committed(name):
    blob=subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:research/spectral_scale_execution/'+name])
    require('committed:'+name,blob.replace(b'\r\n',b'\n')==(HERE/name).read_bytes().replace(b'\r\n',b'\n'))


def metadata(source,views):
    pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids','val_ids'):
        record=source['source_input_whitelist'][name];path=Path(record['path'])
        require('metadata_SHA:'+name,H.sha(path)==record['sha256'])
        pool[name]=np.load(path,allow_pickle=False)
    counters={};keys=[]
    for rx,day in zip(pool['fit_rx'],pool['fit_day']):
        name=source['rx_names'][int(rx)];cell=name+'|'+str(int(day));index=counters.get(cell,0);counters[cell]=index+1
        keys.append(cell+'|'+source['source_cell_sha256'][cell]+'|'+str(index))
    pool['keys']=np.asarray(keys)
    roles={}
    for view,record in views.items():
        codes=[source['rx_names'].index(r) for r in record['source_selection_rx']];rx=pool['fit_rx']
        roles[view]=dict(fit_train=pool['train_ids'][~np.isin(rx[pool['train_ids']],codes)],
            checkpoint_val=pool['val_ids'][~np.isin(rx[pool['val_ids']],codes)],
            source_selection=np.flatnonzero(np.isin(rx,codes)).astype(np.int64))
        for role,rows in roles[view].items():
            import hashlib
            require('role:'+view+'/'+role,len(rows)==record[role]['rows'] and hashlib.sha256(rows.tobytes()).hexdigest()==record[role]['indices_array_sha256'])
    return pool,roles


def metrics(data,mask):
    y=data['y'][mask];z=data['logits'][mask].astype(np.float64);p=z.argmax(1)
    f=[]
    for label in range(10):
        tp=int(((y==label)&(p==label)).sum());den=int((y==label).sum()+(p==label).sum())
        f.append(2*tp/den if den else 0.)
    mx=z.max(1);ce=mx+np.log(np.exp(z-mx[:,None]).sum(1))-z[np.arange(len(y)),y]
    return dict(accuracy=float((p==y).mean()),macro_f1=float(np.mean(f)),ce=float(ce.mean()))


def training(jobs,source,views,pool,roles,seal):
    initials={};histories={};completions={};trace_totals=0
    for job in jobs:
        name=job['id'];out=HERE/'artifacts/scientific'/name
        _,complete=E.verify_complete(job);completions[name]=complete
        require('sealed_completion:'+name,H.sha(out/'COMPLETE.json')==seal['runs'][name]['complete_sha256'])
        history=H.read(out/'HISTORY.json');histories[name]=history;initials[name]=H.read(out/'INITIAL.json');initial=initials[name]
        require('exact_budget:'+name,len(history)==40 and complete['epochs']==40 and complete['updates']==5080)
        reference=H.read(HERE/'REFERENCE_RECOVERY.json')['references'][job['view']+'/'+job['stream']+'/F0']['semantic_state_sha256']
        require('registered_backbone:'+name,initial['frozen_backbone_sha256']==reference)
        require('only3201_active:'+name,initial['optimizer_parameter_elements']==sum(r['elements'] for r in initial['trainable_parameters'])==3201 and all(r['name'].startswith('residual.') for r in initial['trainable_parameters']))
        config=H.read(out/'CONFIG.json')
        require('only_fit_RX:'+name,not config['source_selection_IQ_read'] and config['allowed_training_RX']==views[job['view']]['fit_rx'])
        with np.load(out/'SAMPLE_KEYS.npz',allow_pickle=False) as f:sample={k:f[k] for k in f.files}
        for kind,role in (('train','fit_train'),('val','checkpoint_val')):
            require('sample_identity:'+name+'/'+kind,np.array_equal(sample[kind+'_ids'],roles[job['view']][role]) and np.array_equal(sample[kind+'_keys'],pool['keys'][roles[job['view']][role]]))
        with np.load(out/'GRADIENT_TRACE.npz',allow_pickle=False) as f:columns=list(f['columns']);trace=f['values']
        require('gradient_rows:'+name,trace.shape==(5080,13));trace_totals+=len(trace)
        require('all_gradients_parameters_finite:'+name,np.isfinite(trace).all() and (trace[:,10:13]==1).all())
        require('sequential_updates:'+name,np.array_equal(trace[:,0],np.arange(1,5081)))
        require('initial_alpha0_task_identity:'+name,trace[0,5]==trace[0,8]==0)
        journal=[json.loads(line) for line in (out/'GRADIENT_TRACE.jsonl').read_text(encoding='utf-8').splitlines()]
        require('gradient_journal_exact:'+name,np.array_equal(trace,np.asarray([[r[k] for k in columns] for r in journal])))
        state=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        backbone={k.removeprefix('backbone.'):v for k,v in state['model'].items() if k.startswith('backbone.')}
        require('initial_state_identity:'+name,H.semantic_hash(backbone)==reference and float(state['model']['residual.alpha'])==0)
        for index,row in enumerate(history,1):
            prefix=name+'/epoch'+str(index)
            state=torch.load(out/'epoch_states'/f'epoch_{index:02d}.pt',map_location='cpu',weights_only=False)
            require(prefix+'/full_state_hash',H.semantic_hash(state)==row['state_sha256'])
            backbone={k.removeprefix('backbone.'):v for k,v in state['model'].items() if k.startswith('backbone.')}
            require(prefix+'/all_frozen_weights_BN',H.semantic_hash(backbone)==reference)
            require(prefix+'/registered_updates',state['updates']==row['updates']==127*index)
            require(prefix+'/alpha_trace',float(state['model']['residual.alpha'])==row['alpha']==trace[127*index-1,6])
            require(prefix+'/rollback_evidence',row['alpha0_rollback_bitwise'] and row['backbone_and_BN_unchanged'])
            order=np.load(out/'orders'/f'epoch_{index:02d}.npy',allow_pickle=False)
            require(prefix+'/actual_permutation',np.array_equal(np.sort(order),np.arange(len(sample['train_ids']))))
            require(prefix+'/actual_order_hash',H.semantic_hash(order)==row['relative_order_sha256'] and H.semantic_hash(sample['train_keys'][order])==row['canonical_order_sha256'])
        scores=[initial['epoch0_val_rx_ce']]+[r['val_rx_ce'] for r in history]
        best=min(range(41),key=lambda i:scores[i]);require('earliest_CE_min0to40:'+name,complete['best_epoch']==best)
        for endpoint,epoch in (('epoch0',0),('selected',best),('epoch10',10),('final40',40)):
            ck=torch.load(out/(endpoint+'.pt'),map_location='cpu',weights_only=False)
            require('checkpoint_epoch:'+name+'/'+endpoint,ck['epoch']==epoch)
            with np.load(out/(endpoint+'_val.npz'),allow_pickle=False) as f:data={k:f[k] for k in f.files}
            require('validation_role:'+name+'/'+endpoint,np.array_equal(data['pool_ids'],roles[job['view']]['checkpoint_val']) and np.array_equal(data['keys'],sample['val_keys']))
            per_rx=[metrics(data,data['rx']==rx)['ce'] for rx in np.unique(data['rx'])]
            close('independent_validation_CE:'+name+'/'+endpoint,float(np.mean(per_rx)),ck['checkpoint_val_rx_ce'],2e-6)
        print('QA verified full fit states and gradients:',name,flush=True)
    require('all60960_gradient_updates',trace_totals==60960)
    for view in views:
        for stream in STREAMS:
            a,b=[f'{view}_{stream}_{arm}' for arm in ('F1','F2')]
            for field in ('frozen_backbone_sha256','common_residual_sha256','rng_sha256','scheduler_sha256'):
                require('paired_initial:'+view+'/'+stream+'/'+field,initials[a][field]==initials[b][field])
            for epoch in range(40):
                for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
                    require('paired_order:'+view+'/'+stream+'/'+str(epoch+1)+'/'+field,histories[a][epoch][field]==histories[b][epoch][field])
    return completions


def statistics(analysis,manifest,source,views,pool,roles):
    all_rx=sorted(r for v in views.values() for r in v['source_selection_rx'])
    require('12_observed_RX',len(all_rx)==len(set(all_rx))==12)
    draws=np.random.default_rng(24501).integers(0,12,size=(10000,12));independent={}
    expected={f'evaluation/{v}_{s}_{a}/reference.npz' for v in views for s in STREAMS for a in ('F0','E0')}
    expected|={f'evaluation/{v}_{s}_{a}/{e}.npz' for v in views for s in STREAMS for a in ('F1','F2') for e in H.ENDPOINTS}
    require('exact48_predictions',set(manifest['files'])==expected)
    for name,digest in manifest['files'].items():require('selection_prediction_SHA:'+name,H.sha(HERE/'artifacts'/name)==digest)
    for endpoint in H.ENDPOINTS:
        data={};scores={};rx_views={r:v for v,x in views.items() for r in x['source_selection_rx']}
        for arm in ARMS:
            for view in views:
                for stream in STREAMS:
                    suffix='reference' if arm in ('F0','E0') else endpoint
                    path=HERE/'artifacts/evaluation'/f'{view}_{stream}_{arm}'/(suffix+'.npz')
                    with np.load(path,allow_pickle=False) as f:d={k:f[k] for k in f.files}
                    data[arm,view,stream]=d;rows=roles[view]['source_selection'];prefix=endpoint+'/'+arm+'/'+view+'/'+stream
                    require('source_selection_identity:'+prefix,np.array_equal(d['pool_ids'],rows) and np.array_equal(d['keys'],pool['keys'][rows]) and np.array_equal(d['y'],pool['fit_y'][rows]) and np.array_equal(d['day'],pool['fit_day'][rows]) and np.array_equal(d['rx'],np.asarray(source['rx_names'])[pool['fit_rx'][rows]]))
                    require('source_selection_finite:'+prefix,d['logits'].shape==(len(rows),10) and np.isfinite(d['logits']).all())
                    for rx in views[view]['source_selection_rx']:scores[arm,stream,rx]=metrics(d,d['rx']==rx)
        block=analysis['endpoints'][endpoint]
        for arm in ARMS:
            aggregate=block['aggregates'][arm]
            for metric in ('accuracy','macro_f1','ce'):
                means={r:float(np.mean([scores[arm,s,r][metric] for s in STREAMS])) for r in all_rx}
                close('independent_'+metric+':'+endpoint+'/'+arm,float(np.mean(list(means.values()))),aggregate['rx_equal'][metric])
                for rx in all_rx:close('independent_RX_'+metric+':'+endpoint+'/'+arm+'/'+rx,means[rx],aggregate['per_rx'][rx][metric])
                for view,x in views.items():close('independent_view_'+metric+':'+endpoint+'/'+arm+'/'+view,float(np.mean([means[r] for r in x['source_selection_rx']])),aggregate['per_view'][view][metric])
                for stream in STREAMS:close('independent_stream_'+metric+':'+endpoint+'/'+arm+'/'+stream,float(np.mean([scores[arm,stream,r][metric] for r in all_rx])),aggregate['per_stream'][stream][metric])
        for baseline in ('F1','F0','E0'):
            name='F2-'+baseline;record=block['comparisons'][name];paired={}
            for rx in all_rx:
                view=rx_views[rx];rows=[]
                for stream in STREAMS:
                    left,right=data[baseline,view,stream],data['F2',view,stream];mask=left['rx']==rx
                    require('paired_keys:'+endpoint+'/'+name+'/'+view+'/'+stream,np.array_equal(left['keys'],right['keys']))
                    lp,rp=left['logits'][mask].argmax(1),right['logits'][mask].argmax(1);y=left['y'][mask]
                    l,r=lp==y,rp==y
                    rows.append(dict(accuracy_delta=float(r.mean()-l.mean()),
                        macro_f1_delta=scores['F2',stream,rx]['macro_f1']-scores[baseline,stream,rx]['macro_f1'],
                        ce_delta=scores['F2',stream,rx]['ce']-scores[baseline,stream,rx]['ce'],
                        corrected_fraction=float((~l&r).mean()),new_error_fraction=float((l&~r).mean()),disagreement=float((lp!=rp).mean())))
                paired[rx]={k:float(np.mean([row[k] for row in rows])) for k in rows[0]}
            intervals={}
            for metric in paired[all_rx[0]]:
                values=np.asarray([paired[r][metric] for r in all_rx]);intervals[metric]=np.quantile(values[draws].mean(1),(.025,.975))
                close('independent_pair_mean:'+endpoint+'/'+name+'/'+metric,float(values.mean()),record['rx_equal'][metric])
                close('independent_pair_interval:'+endpoint+'/'+name+'/'+metric,intervals[metric],record['paired_rx_bootstrap_95_intervals'][metric])
                for rx in all_rx:close('independent_pair_RX:'+endpoint+'/'+name+'/'+metric+'/'+rx,paired[rx][metric],record['per_rx'][rx][metric])
            risk=[]
            for view,x in views.items():
                d=data[baseline,view,'S0']
                for rx in x['source_selection_rx']:
                    for tx in range(10):
                        mask=(d['rx']==rx)&(d['y']==tx)
                        if not mask.any():continue
                        values=[]
                        for stream in STREAMS:
                            l,r=data[baseline,view,stream],data['F2',view,stream]
                            values.append(float((l['logits'][mask].argmax(1)==l['y'][mask]).mean()-(r['logits'][mask].argmax(1)==r['y'][mask]).mean()))
                        risk.append(dict(rx=rx,tx=tx,support=int(mask.sum()),risk_increase=float(np.mean(values))))
            expected_risk={(r['rx'],r['tx_class_index']):r for r in record['tx_rx_stream_mean']}
            require('all_observed_TX_RX:'+endpoint+'/'+name,len(risk)==len(expected_risk))
            for row in risk:
                expected=expected_risk[row['rx'],row['tx']]
                require('TX_RX_support:'+endpoint+'/'+name+'/'+row['rx']+'/'+str(row['tx']),row['support']==expected['packets_per_stream'])
                close('TX_RX_risk:'+endpoint+'/'+name+'/'+row['rx']+'/'+str(row['tx']),row['risk_increase'],expected['risk_increase'])
            supported=[r for r in risk if r['support']>=50]
            worst=max(r['risk_increase'] for r in supported)
            close('independent_supported_max_risk:'+endpoint+'/'+name,worst,record['worst_supported_tx_rx_risk_increase']['risk_increase'])
            deltas=np.asarray([paired[r]['accuracy_delta'] for r in all_rx])
            close('independent_worst_paired_RX:'+endpoint+'/'+name,float(deltas.min()),record['worst_paired_rx']['accuracy_delta'])
            gates=dict(mean_accuracy_at_least_1pp=float(deltas.mean())>=.01,
                paired_rx_interval_lower_above_zero=float(intervals['accuracy_delta'][0])>0,
                both_views_positive=all(np.mean([paired[r]['accuracy_delta'] for r in x['source_selection_rx']])>0 for x in views.values()),
                at_least_two_streams_positive=sum(np.mean([scores['F2',s,r]['accuracy']-scores[baseline,s,r]['accuracy'] for r in all_rx])>0 for s in STREAMS)>=2,
                worst_paired_rx_at_least_minus_2pp=float(deltas.min())>=-.02,
                supported_tx_rx_risk_increase_at_most_5pp=all(r['risk_increase']<=.05 for r in supported))
            if endpoint=='selected':
                gates={k:bool(v) for k,v in gates.items()}
                require('independent_joint_gates:'+name,gates==analysis['decision']['gates'][name]['checks'])
                independent[name]=dict(checks=gates,accuracy_delta=float(deltas.mean()),max_supported_risk_increase=worst,supported_groups=len(supported))
    promoted=all(all(v['checks'].values()) for v in independent.values())
    status='F2_WORTH_SEPARATE_CONFIRMATION' if promoted else 'STOP_F2_FROZEN_BACKBONE_RESIDUAL'
    require('independent_final_decision',status==analysis['decision']['status'])
    return independent


def main():
    if (HERE/'INDEPENDENT_QA.json').exists():raise FileExistsError('Keep original independent audit')
    _,ex,_,jobs,source=E.scientific_context();seal=E.require_full_seal(jobs)
    for name in ('EXECUTION_AUTHORIZATION.json','SOURCE_PREFLIGHT.json','TRAINING_SEAL.json','RUN_EVIDENCE.json'):committed(name)
    for stage in ('periodic_hstem','periodic_span','spectral_scale_review'):
        old=ROOT/'research'/stage
        for name,item in H.read(old/'ARTIFACT_MANIFEST.json')['files'].items():require('old_STOP_design:'+stage+'/'+name,H.sha(old/name)==item['sha256'])
    for name,item in H.read(HERE/'ARTIFACT_MANIFEST.json')['files'].items():require('original_engineering:'+name,H.sha(HERE/name)==item['sha256'])
    plan=H.read(H.DESIGN/'FUTURE_SOURCE_PLAN.json');views={v['id']:v for v in plan['views']};pool,roles=metadata(source,views)
    complete=training(jobs,source,views,pool,roles,seal)
    with gzip.open(HERE/'ANALYSIS.json.gz','rt',encoding='utf-8') as f:analysis=json.load(f)
    summary=H.read(HERE/'SUMMARY.json');manifest=H.read(HERE/'EVALUATION_MANIFEST.json')
    require('evaluation_matches_seal',manifest['seal_sha256']==H.sha(HERE/'TRAINING_SEAL.json'))
    require('summary_decision',summary['decision']==analysis['decision'])
    for endpoint,block in analysis['endpoints'].items():
        require('full_summary_aggregates:'+endpoint,summary['endpoints'][endpoint]['aggregates']=={a:{k:v for k,v in x.items() if k!='tx_rx_stream_mean'} for a,x in block['aggregates'].items()})
        require('full_summary_comparisons:'+endpoint,summary['endpoints'][endpoint]['comparisons']=={a:{k:v for k,v in x.items() if k not in ('cells','tx_rx_stream_mean')} for a,x in block['comparisons'].items()})
    independent=statistics(analysis,manifest,source,views,pool,roles)
    references=H.read(HERE/'REFERENCE_INPUTS.json')['references'];allowed={str(Path(v['path']).resolve()) for v in source['source_input_whitelist'].values()}
    for group in references.values():
        for item in group.values():
            allowed.add(str((HERE/item['copy_relative']).resolve()))
            allowed.update(str(Path(item[k]['path']).resolve()) for k in ('checkpoint_val_prediction','source_selection_prediction'))
    accesses=[json.loads(line) for line in (HERE/'SCIENTIFIC_ACCESS_LOG.jsonl').read_text(encoding='utf-8').splitlines()]
    require('only_source_whitelist_accesses',all(str(Path(row['path']).resolve()) in allowed for row in accesses))
    for job in jobs:
        for action in ('copy_fit_train_rows','copy_checkpoint_val_rows'):
            require('single_training_role_access:'+job['id']+'/'+action,sum(r['job']==job['id'] and r['action']==action for r in accesses)==1)
    selection=[r for r in accesses if r['action']=='copy_source_selection_rows']
    require('only_two_sealed_selection_copies',len(selection)==2 and all(r['job']=='sealed_selection_evaluation' for r in selection))
    timestamp=subprocess.check_output(['git','-C',str(ROOT),'log','-1','--format=%cI','--','research/spectral_scale_execution/TRAINING_SEAL.json'],text=True).strip()
    require('selection_after_committed_all12_seal',all(datetime.fromisoformat(r['time'])>=datetime.fromisoformat(timestamp) for r in selection))
    attempts=list((HERE/'artifacts/scientific').glob('*/ATTEMPT.json'))
    require('only12_attempts_no_failures',len(attempts)==12 and all(H.read(p)['status']=='COMPLETE' for p in attempts) and not list((HERE/'artifacts/scientific').glob('*/FAILURE.json')))
    value=dict(status='PASS',check_count=len(checks),checks=checks,independent_comparisons=independent,
        scientific_fits=12,epochs=480,updates=60960,source_selection_predictions=48,independent_IQ_reads=0,
        independently_decoded_full_states=492,decision=analysis['decision'],
        verified_artifacts=dict(execution_sha256=H.sha(HERE/'EXECUTION_REGISTRATION.json'),authorization_sha256=H.sha(HERE/'EXECUTION_AUTHORIZATION.json'),
            preflight_sha256=H.sha(HERE/'SOURCE_PREFLIGHT.json'),seal_sha256=H.sha(HERE/'TRAINING_SEAL.json'),evaluation_sha256=H.sha(HERE/'EVALUATION_MANIFEST.json'),analysis_sha256=H.sha(HERE/'ANALYSIS.json.gz')),
        method='Independent numpy arithmetic and classwise F1; no Stage C metric/comparison/promotion helpers used for recomputation.',time=H.now())
    H.write(HERE/'INDEPENDENT_QA.json',value)
    print(dict(status='PASS',checks=len(checks),decision=analysis['decision']['status']))


if __name__=='__main__':main()
