"""Recompute conditional W1 and W2 decisions from saved packet readouts, no IQ."""
from pathlib import Path
import csv
import json
import numpy as np

HERE=Path(__file__).resolve().parent


def read(name):return json.loads((HERE/name).read_text(encoding='utf8'))


def w1_checks():
    checks={};effects={'F0':[],'E0':[]};all_ids=[]
    for view in ('V0','V1'):
        with np.load(HERE/'artifacts'/(view+'_packets.npz'),allow_pickle=False) as z:
            p={k:z[k] for k in z.files}
        all_ids.extend(p['ids'].tolist())
        checks[view+'_quality_definition']=bool(np.allclose(p['features'][:,0],p['pairs'].mean(1),atol=1e-15,rtol=0))
        checks[view+'_source_days_only']=set(p['day'])=={1,2}
        checks[view+'_all_packets_finite']=bool(p['valid'].all())
        for model in ('F0','E0'):
            per_rx=[]
            for rx in sorted(set(p['rx'])):
                per_tx=[]
                for tx in sorted(set(p['y'][p['rx']==rx])):
                    per_day=[]
                    for day in sorted(set(p['day'][(p['rx']==rx)&(p['y']==tx)])):
                        pos=np.flatnonzero((p['rx']==rx)&(p['y']==tx)&(p['day']==day))
                        rank=sorted(pos,key=lambda i:(p['features'][i,0],p['ids'][i]))
                        cut=len(rank)//4;low=rank[:cut];other=rank[cut:]
                        per_day.append(100*(p[model][:,low].mean()-p[model][:,other].mean()))
                    per_tx.append(float(np.mean(per_day)))
                per_rx.append(float(np.mean(per_tx)))
            effects[model].extend(per_rx)
    checks['W1_unique_packet_ids']=len(all_ids)==len(set(all_ids))==22968
    result=read('W1_RESULT.json')
    for model,v in effects.items():
        checks[model+'_independent_conditional_effect']=abs(np.mean(v)-result['primary'][model]['difference_pp'])<1e-10
        samples=np.random.default_rng(27113).integers(0,len(v),(20000,len(v)))
        ci=np.percentile(np.asarray(v)[samples].mean(1),[2.5,97.5])
        checks[model+'_independent_RX_interval']=bool(np.allclose(ci,result['primary'][model]['conditional_rx_cluster_ci95_pp'],atol=1e-10,rtol=0))
    return checks


def w2_checks():
    if not (HERE/'W2_RESULT.json').exists():return {}
    checks={};result=read('W2_RESULT.json');reg=read('W2_REGISTRATION.json')
    for view in ('V0','V1'):
        for stream in ('S0','S1','S2'):
            orders=[];common=[]
            for arm in ('B1','B2','N0'):
                folder=HERE/'artifacts'/f'{view}_{stream}_{arm}'
                hist=json.loads((folder/'HISTORY.json').read_text(encoding='utf8'))
                checks[f'{view}_{stream}_{arm}_full_budget']=len(hist)==40 and hist[-1]['updates']==5080
                checks[f'{view}_{stream}_{arm}_BN_state_preserved']=all(h['calibration_preserved_training_state'] for h in hist)
                orders.append([h['canonical_order_sha256'] for h in hist])
                with np.load(folder/'CONTEXT_PAIRING.npz',allow_pickle=False) as p:
                    checks[f'{view}_{stream}_{arm}_context_pairing']=all(
                        np.all(p[k]!=np.arange(len(p[k]))) if arm=='N0' else np.array_equal(p[k],np.arange(len(p[k]))) for k in ('train','val'))
            checks[f'{view}_{stream}_identical_epoch_orders']=orders[0]==orders[1]==orders[2]
    accuracy={};rx_sets={};low_groups={}
    for view in ('V0','V1'):
        for stream in ('S0','S1','S2'):
            for arm in ('B0','B1','B2','N0'):
                for cond in ('clean','noise20','noise10','noise0'):
                    with np.load(HERE/'artifacts'/f'{view}_{stream}_{arm}_{cond}_selection.npz',allow_pickle=False) as p:
                        y,rx=p['y'],p['rx'];err=p['logits'].argmax(1)!=y
                        checks[f'{view}_{stream}_{arm}_{cond}_full_packet_count']=len(y)==(11468 if view=='V0' else 11500)
                        for r in np.unique(rx):accuracy[(view,stream,arm,cond,str(r))]=100*(1-err[rx==r].mean())
    for control in ('B0','B1','N0'):
        rx_delta=[]
        for view in ('V0','V1'):
            names=sorted({k[-1] for k in accuracy if k[0]==view})
            for r in names:
                rx_delta.append(np.mean([accuracy[(view,s,'B2','clean',r)]-accuracy[(view,s,control,'clean',r)] for s in ('S0','S1','S2')]))
        actual=next(c for c in result['comparisons'] if c['control']==control)
        checks[control+'_independent_effect']=abs(np.mean(rx_delta)-actual['mean_delta_pp'])<1e-10
        rng=np.random.default_rng(27113);samples=rng.integers(0,12,(20000,12))
        ci=np.percentile(np.asarray(rx_delta)[samples].mean(1),[2.5,97.5])
        checks[control+'_independent_interval']=bool(np.allclose(ci,actual['conditional_rx_ci95_pp'],atol=1e-10,rtol=0))
    checks['W2_no_promotion_unless_all_gates']=((result['status'].startswith('PASS'))==all(result['gates'].values()))
    checks['W2_exact_total_updates']=sum(json.loads((HERE/'artifacts'/j['id']/'COMPLETE.json').read_text())['updates'] for j in reg['jobs'])==91440
    return checks


def main():
    checks={k:bool(v) for k,v in {**w1_checks(),**w2_checks()}.items()}
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,count=len(checks),additional_IQ_reads=0,additional_fits=0)
    name='RESULTS_QA.json' if (HERE/'W2_RESULT.json').exists() else 'W1_QA.json'
    (HERE/name).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['status'],len(checks),'checks',flush=True)
    if not all(checks.values()):raise ValueError('Result QA failed')


if __name__=='__main__':main()
