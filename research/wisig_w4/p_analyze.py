"""Fixed W4-P final classification endpoint, all groups and the registered gates."""
import csv
import numpy as np
from common import *
from metrics import metrics,rx_interval


def table(name,rows):
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze():
    validate('P_EVALUATION_REGISTRATION.json');receipt=read(HERE/'P_PREDICTION_RECEIPT.json')
    if (HERE/'P_RESULT.json').exists():raise RuntimeError('Keep first scientific W4-P analysis')
    plan=read(HERE/'DATA_PLAN.json');rx=plan['view']['source_selection_rx'];rows=[];cells=[];overall=[];arrays={}
    for key,item in receipt['predictions'].items():
        path=checked(item,'analyze_sealed_source_selection_predictions',key)
        arm,seed=key.split('/');seed=int(seed)
        with np.load(path,allow_pickle=False) as f:p={k:f[k] for k in f.files}
        assert set(p['rx'])==set(rx) and set(p['day'])=={1,2} and p['logits'].shape==(11468,10)
        arrays[arm,seed]=p
        overall.append(dict(arm=arm,seed=seed,**metrics(p['logits'],p['y'])))
        for r in rx:
            mask=p['rx']==r
            rows.append(dict(arm=arm,seed=seed,RX=r,**metrics(p['logits'][mask],p['y'][mask])))
            for tx in range(10):
                cm=mask&(p['y']==tx);n=int(cm.sum())
                cells.append(dict(arm=arm,seed=seed,RX=r,TX=tx,packets=n,
                    errors=int(np.sum(p['logits'][cm].argmax(1)!=p['y'][cm])) if n else 0,
                    error_pct=float(100*np.mean(p['logits'][cm].argmax(1)!=p['y'][cm])) if n else None,
                    support_status='SUPPORTED' if n else 'NO_OBSERVATIONS'))
    ref=arrays['A0',0]
    for p in arrays.values():
        assert all(np.array_equal(p[k],ref[k]) for k in ('y','rx','day','keys','pool_ids'))
    lookup={(r['arm'],r['seed'],r['RX']):r for r in rows};summary={}
    for arm in ('A0','A1','A2'):
        by_seed=[float(np.mean([lookup[arm,s,r]['accuracy_pct'] for r in rx])) for s in range(3)]
        by_rx={r:float(np.mean([lookup[arm,s,r]['accuracy_pct'] for s in range(3)])) for r in rx}
        summary[arm]=dict(RX_equal_accuracy_pct=float(np.mean(by_seed)),per_seed_RX_equal_accuracy_pct=by_seed,
            per_RX_seed_mean_accuracy_pct=by_rx,worst_RX_seed_mean_accuracy_pct=min(by_rx.values()),
            RX_equal_macro_F1_pct=float(np.mean([r['macro_F1_pct'] for r in rows if r['arm']==arm])),
            packet_accuracy_pct=float(np.mean([r['accuracy_pct'] for r in overall if r['arm']==arm])),
            packet_macro_F1_pct=float(np.mean([r['macro_F1_pct'] for r in overall if r['arm']==arm])))
    comparisons={}
    for arm,control in (('A2','A1'),('A2','A0'),('A1','A0')):
        dr={r:summary[arm]['per_RX_seed_mean_accuracy_pct'][r]-summary[control]['per_RX_seed_mean_accuracy_pct'][r] for r in rx}
        ds=np.array(summary[arm]['per_seed_RX_equal_accuracy_pct'])-summary[control]['per_seed_RX_equal_accuracy_pct']
        comparisons[arm+'_minus_'+control]=dict(mean_gain_pp=float(ds.mean()),per_seed_gain_pp=ds.tolist(),positive_seeds=int((ds>0).sum()),
            per_RX_seed_mean_gain_pp=dr,worst_RX_seed_mean_gain_pp=min(dr.values()),
            worst_seed_RX_gain_pp=min(lookup[arm,s,r]['accuracy_pct']-lookup[control,s,r]['accuracy_pct'] for s in range(3) for r in rx),
            conditional_RX_interval=rx_interval(list(dr.values())))
    c1=comparisons['A2_minus_A1'];c0=comparisons['A2_minus_A0']
    gates=dict(A2_vs_A1_gain_ge_1pp=c1['mean_gain_pp']>=1,all_three_seed_gains_positive=c1['positive_seeds']==3,
        worst_seed_averaged_RX_vs_A1_ge_minus2pp=c1['worst_RX_seed_mean_gain_pp']>=-2,
        A2_overall_not_worse_than_A0=c0['mean_gain_pp']>=0,worst_seed_averaged_RX_vs_A0_ge_minus2pp=c0['worst_RX_seed_mean_gain_pp']>=-2)
    result=dict(status='W4P_PASS_FIRST_SOURCE_VIEW_ONLY' if all(gates.values()) else 'STOP_W4P_RELATIONAL_PRETRAINING_NOT_PROMOTED',
        time=now(),summary=summary,comparisons=comparisons,gates=gates,new_neural_fits=12,A0_reused=3,new_gradient_updates=56400,
        source_view='V0',unique_internal_source_RX=6,actual_target_RX_or_D3_D4_reads=0,second_source_view_started=False,
        macro_F1_definition='All ten labels, zero division=0, same for every arm; absent TX/RX cells are separately disclosed.',
        evidence_boundary='Auxiliary data plus extra optimization is confounded in A1-A0; A2-A1 isolates the registered relation penalty under matching auxiliary budget. No hardware mechanism or SOTA established.')
    table('P_RX_METRICS.csv',rows);table('P_TX_RX_RISKS.csv',cells);table('P_PACKET_METRICS.csv',overall)
    write('P_RESULT.json',result);print(json.dumps(dict(status=result['status'],summary=summary,gates=gates),indent=2),flush=True)


def auxiliary_analysis():
    receipt=read(HERE/'P_PREDICTION_RECEIPT.json');path=checked(receipt['auxiliary_features'],'analyze_aux24_sealed_features')
    plan=read(HERE/'DATA_PLAN.json');rx=plan['aux_RX'];rows=[];summaries={}
    with np.load(path,allow_pickle=False) as f:
        assert f['TX'].tolist()==plan['aux_diagnostic_TX'] and f['RX'].tolist()==rx
        for seed in range(3):
            for arm in ('A1','A2'):
                key=f'{arm}_S{seed}';z=f[key].astype(np.float64);assert z.shape==(24,6,2,32,64)
                mu=z.mean(3);ii,jj=np.triu_indices(24,1);relations=[];spreads=[]
                for day in (0,1):
                    centers=mu[:,:,day].transpose(1,0,2)
                    distances=((centers[:,ii]-centers[:,jj])**2).sum(-1);norm=distances/(distances.mean(1,keepdims=True)+1e-8)
                    rr,ss=np.triu_indices(6,1);relations.append(float(((norm[rr]-norm[ss])**2).mean()))
                    spreads.extend((distances.mean(1)/(np.mean(centers**2,axis=(1,2))+1e-8)).tolist())
                for source in range(6):
                    for dest in range(6):
                        if source==dest:continue
                        for sd,dd in ((0,1),(1,0)):
                            prototypes=mu[:,source,sd];query=z[:,dest,dd].reshape(-1,64);truth=np.repeat(np.arange(24),32)
                            distance=np.maximum((query*query).sum(1)[:,None]+(prototypes*prototypes).sum(1)[None,:]-2*query@prototypes.T,0)
                            true=distance[np.arange(len(query)),truth];other=distance.copy();other[np.arange(len(query)),truth]=np.inf
                            margin=(other.min(1)-true)/(distance.mean(1)+1e-12)
                            rows.append(dict(arm=arm,seed=seed,prototype_RX=rx[source],query_RX=rx[dest],prototype_day=sd+1,query_day=dd+1,
                                queries=768,prototype_diagnostic_accuracy_pct=float(100*(distance.argmin(1)==truth).mean()),normalized_margin=float(margin.mean())))
                mine=[r for r in rows if r['arm']==arm and r['seed']==seed]
                summaries[key]=dict(prototype_diagnostic_accuracy_pct=float(np.mean([r['prototype_diagnostic_accuracy_pct'] for r in mine])),
                    normalized_margin=float(np.mean([r['normalized_margin'] for r in mine])),normalized_RX_relation_MSE=float(np.mean(relations)),
                    minimum_relative_class_spread=min(spreads))
    table('P_AUX24_PAIR_METRICS.csv',rows)
    delta=[summaries[f'A2_S{s}']['prototype_diagnostic_accuracy_pct']-summaries[f'A1_S{s}']['prototype_diagnostic_accuracy_pct'] for s in range(3)]
    write('P_AUX24_RESULT.json',dict(status='POST_SEAL_AUXILIARY_DIAGNOSTIC_ONLY',summary=summaries,
        A2_minus_A1_prototype_accuracy_pp=dict(mean=float(np.mean(delta)),per_seed=delta),
        uses_labeled_support_prototypes=True,RX_seen_during_auxiliary_pretraining=True,
        not_downstream10TX_accuracy=True,not_zero_shot_unknown_RX=True,not_used_for_checkpoint_or_gate=True,
        data_qualification='24 TX excluded from W4 fitting; these observations were part of prior W3 exploration, not historically untouched.'))


if __name__=='__main__':analyze();auxiliary_analysis()
