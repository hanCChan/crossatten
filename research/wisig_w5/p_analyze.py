"""Fixed W5-A final classification endpoint, all groups and the registered gates."""
import csv
import numpy as np
from common import *
from metrics import metrics,rx_interval


def table(name,rows):
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze():
    validate('P_EVALUATION_REGISTRATION.json');receipt=read(HERE/'P_PREDICTION_RECEIPT.json')
    if (HERE/'P_RESULT.json').exists():raise RuntimeError('Keep first scientific W5-A analysis')
    plan=read(HERE/'DATA_PLAN.json');rx=plan['view']['source_selection_rx'];rows=[];cells=[];overall=[];arrays={}
    for key,item in receipt['predictions'].items():
        path=checked(item,'analyze_sealed_source_selection_predictions',key)
        arm,seed=key.split('/');seed=int(seed)
        with np.load(path,allow_pickle=False) as f:p={k:f[k] for k in f.files}
        assert set(p['rx'])==set(rx) and set(p['day'])=={1,2} and p['logits'].shape==(11500,10)
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
    for arm in ('A0','A1'):
        by_seed=[float(np.mean([lookup[arm,s,r]['accuracy_pct'] for r in rx])) for s in range(3)]
        by_rx={r:float(np.mean([lookup[arm,s,r]['accuracy_pct'] for s in range(3)])) for r in rx}
        summary[arm]=dict(RX_equal_accuracy_pct=float(np.mean(by_seed)),per_seed_RX_equal_accuracy_pct=by_seed,
            per_RX_seed_mean_accuracy_pct=by_rx,worst_RX_seed_mean_accuracy_pct=min(by_rx.values()),
            RX_equal_macro_F1_pct=float(np.mean([r['macro_F1_pct'] for r in rows if r['arm']==arm])),
            packet_accuracy_pct=float(np.mean([r['accuracy_pct'] for r in overall if r['arm']==arm])),
            packet_macro_F1_pct=float(np.mean([r['macro_F1_pct'] for r in overall if r['arm']==arm])))
    comparisons={}
    for arm,control in (('A1','A0'),):
        dr={r:summary[arm]['per_RX_seed_mean_accuracy_pct'][r]-summary[control]['per_RX_seed_mean_accuracy_pct'][r] for r in rx}
        ds=np.array(summary[arm]['per_seed_RX_equal_accuracy_pct'])-summary[control]['per_seed_RX_equal_accuracy_pct']
        comparisons[arm+'_minus_'+control]=dict(mean_gain_pp=float(ds.mean()),per_seed_gain_pp=ds.tolist(),positive_seeds=int((ds>0).sum()),
            per_RX_seed_mean_gain_pp=dr,worst_RX_seed_mean_gain_pp=min(dr.values()),
            worst_seed_RX_gain_pp=min(lookup[arm,s,r]['accuracy_pct']-lookup[control,s,r]['accuracy_pct'] for s in range(3) for r in rx),
            conditional_RX_interval=rx_interval(list(dr.values())))
    c1=comparisons['A1_minus_A0']
    gates=dict(A1_vs_A0_gain_ge_1pp=c1['mean_gain_pp']>=1,all_three_seed_gains_positive=c1['positive_seeds']==3,
        worst_seed_averaged_RX_vs_A0_ge_minus2pp=c1['worst_RX_seed_mean_gain_pp']>=-2)
    result=dict(status='W5A_PASS_CROSS_VIEW_SOURCE_REPLICATION' if all(gates.values()) else 'STOP_W5A_A1_NOT_REPLICATED',
        time=now(),summary=summary,comparisons=comparisons,gates=gates,new_neural_fits=6,A0_reused=3,new_gradient_updates=28200,
        source_view='V1',unique_internal_source_RX=6,actual_target_RX_or_D3_D4_reads=0,W5_B_started=False,
        macro_F1_definition='All ten labels, zero division=0; absent TX/RX cells separately disclosed.',
        evidence_boundary='Previously developed source RX, not historically untouched. A1 adds both auxiliary data and computation; no causal mechanism or real-target/SOTA claim.')
    table('P_RX_METRICS.csv',rows);table('P_TX_RX_RISKS.csv',cells);table('P_PACKET_METRICS.csv',overall)
    write('P_RESULT.json',result);print(json.dumps(dict(status=result['status'],summary=summary,gates=gates),indent=2),flush=True)



if __name__=='__main__':analyze()
