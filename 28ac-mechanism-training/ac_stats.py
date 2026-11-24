"""Frozen A factorial interaction and two-held-RX C pilot decision."""
import numpy as np

def summary(rows,arm):
    q=[r for r in rows if r['arm']==arm]
    fits=[float(np.mean([r['acc'] for r in q if r['fold']==f and r['seed']==s])*100) for f in (0,1) for s in (201,202,203)]
    rx={name:float(np.mean([r['acc'] for r in q if r['rx']==name])*100) for name in sorted({r['rx'] for r in q})}
    return dict(arm=arm,accuracy_pct=float(np.mean(fits)),fit_sd_pp=float(np.std(fits,ddof=1)),fit_accuracies_pct=fits,
        macro_f1_pct=float(np.mean([r['macro_f1'] for r in q])*100),CE=float(np.mean([r['ce'] for r in q])),
        per_rx_accuracy_pct=rx,worst_rx=min(rx,key=rx.get),worst_rx_accuracy_pct=min(rx.values()))

def contrast(rows,weights):
    groups={a:{(r['fold'],r['seed'],r['rx']):r['acc'] for r in rows if r['arm']==a} for a in weights}
    keys=next(iter(groups.values())).keys();assert len(keys)==30 and all(q.keys()==keys for q in groups.values())
    values={k:sum(w*groups[a][k] for a,w in weights.items())*100 for k in keys}
    rx={r:float(np.mean([v for k,v in values.items() if k[2]==r])) for r in sorted({k[2] for k in keys})}
    rng=np.random.default_rng(28028);x=np.array(list(rx.values()));assert len(x)==10
    ci=np.quantile(x[rng.integers(0,10,(10000,10))].mean(1),[.025,.975])
    fits=[float(np.mean([v for k,v in values.items() if k[:2]==(f,s)])) for f in (0,1) for s in (201,202,203)]
    folds=[float(np.mean([v for k,v in values.items() if k[0]==f])) for f in (0,1)]
    mean=float(x.mean());sign=1 if mean>0 else -1 if mean<0 else 0
    consistent=sum(v*sign>0 for v in fits)
    return dict(weights=weights,mean_delta_pp=mean,conditional_rx_ci95_pp=ci.tolist(),per_rx_delta_pp=rx,fit_deltas_pp=fits,fold_deltas_pp=folds,
        same_direction_fit_pairs=consistent,two_sided_practical_rule_met=bool(abs(mean)>=.5 and (ci[0]>0 or ci[1]<0) and consistent>=5 and all(v*sign>0 for v in folds)))

def c_pilot(source,held):
    records=[]
    for f in (0,1):
        for s in (201,202,203):
            metrics={}
            for arm in ['H0','LP']:
                src=[r for r in source if r['arm']==arm and r['fold']==f and r['seed']==s];assert len(src)==4
                h=next(r for r in held if r['arm']==arm and r['fold']==f and r['seed']==s)
                metrics[arm]=dict(source_accuracy_pct=float(np.mean([q['acc'] for q in src])*100),held_accuracy_pct=h['acc']*100,held_rx=h['rx'])
                metrics[arm]['gap_pp']=metrics[arm]['source_accuracy_pct']-metrics[arm]['held_accuracy_pct']
            records.append(dict(fold=f,seed=s,**metrics,held_delta_pp=metrics['H0']['held_accuracy_pct']-metrics['LP']['held_accuracy_pct'],
                source_delta_pp=metrics['H0']['source_accuracy_pct']-metrics['LP']['source_accuracy_pct'],delta_gap_pp=metrics['H0']['gap_pp']-metrics['LP']['gap_pp']))
    folds=[dict(fold=f,held_rx=next(q['H0']['held_rx'] for q in records if q['fold']==f),
        **{key:float(np.mean([q[key] for q in records if q['fold']==f])) for key in ['held_delta_pp','source_delta_pp','delta_gap_pp']}) for f in (0,1)]
    favorable=sum(q['held_delta_pp']>0 and q['delta_gap_pp']<0 for q in records)
    return dict(pairs=records,fold_means=folds,receiver_units=2,paired_repeats=6,population_inference=False,
        favorable_fit_pairs=favorable,consistent_pilot_signal=bool(all(q['held_delta_pp']>=.5 and q['delta_gap_pp']<=-.5 for q in folds) and favorable>=5),
        means={key:float(np.mean([q[key] for q in records])) for key in ['held_delta_pp','source_delta_pp','delta_gap_pp']},
        fit_sd={key:float(np.std([q[key] for q in records],ddof=1)) for key in ['held_delta_pp','source_delta_pp','delta_gap_pp']})
