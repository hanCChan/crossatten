"""Fixed four-cell interaction and conditional RX summaries."""
import numpy as np

ARMS=['H0','LP','H0_P','LP_P']


def interaction(hm,lm,hp,lp):return (hm-lm)-(hp-lp)


def contrast(rows,weights,seed=27028):
    groups={a:{(r['fold'],r['seed'],r['rx']):r['acc'] for r in rows if r['arm']==a} for a in ARMS}
    keys=groups['H0'].keys();assert len(keys)==30 and all(g.keys()==keys for g in groups.values())
    values={k:sum(w*groups[a][k] for a,w in weights.items())*100 for k in keys}
    perrx={rx:float(np.mean([v for k,v in values.items() if k[2]==rx])) for rx in sorted({k[2] for k in keys})}
    x=np.array(list(perrx.values()));rng=np.random.default_rng(seed)
    ci=np.quantile(x[rng.integers(0,10,(10000,10))].mean(1),[.025,.975])
    fits=[float(np.mean([v for k,v in values.items() if k[:2]==(f,s)])) for f in (0,1) for s in (201,202,203)]
    folds=[float(np.mean([v for k,v in values.items() if k[0]==f])) for f in (0,1)]
    return dict(weights=weights,mean_delta_pp=float(x.mean()),conditional_rx_ci95_pp=ci.tolist(),
        per_rx_delta_pp=perrx,fit_deltas_pp=fits,fold_deltas_pp=folds,positive_fit_pairs=sum(v>0 for v in fits),
        practical_rule_met=bool(x.mean()>=.5 and ci[0]>0 and sum(v>0 for v in fits)>=5 and min(folds)>0))


def summary(rows,arm):
    selected=[r for r in rows if r['arm']==arm]
    fits=[np.mean([r['acc'] for r in selected if r['fold']==f and r['seed']==s])*100 for f in (0,1) for s in (201,202,203)]
    rx={name:float(np.mean([r['acc'] for r in selected if r['rx']==name])*100) for name in sorted({r['rx'] for r in selected})}
    return dict(arm=arm,accuracy_pct=float(np.mean(fits)),fit_sd_pp=float(np.std(fits,ddof=1)),
        fit_accuracies_pct=[float(v) for v in fits],macro_f1_pct=float(np.mean([r['macro_f1'] for r in selected])*100),
        per_rx_accuracy_pct=rx,worst_rx=min(rx,key=rx.get),worst_rx_accuracy_pct=min(rx.values()))
