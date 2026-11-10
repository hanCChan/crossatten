import numpy as np

def delta_stats(rows):
    if not rows:raise ValueError('Empty contrast')
    if set(r['fold'] for r in rows)!=set(range(4)):raise ValueError('All four paired folds are required')
    folds={f:np.mean([r['delta_pp'] for r in rows if r['fold']==f]) for f in range(4)}
    rx={}
    for r in rows:rx.setdefault(r['rx'],[]).append(r['delta_pp'])
    # Resample whole RX clusters including their fold occurrences. RXs occur in
    # unequal numbers of folds; averaging cluster means would change the estimand.
    if len({sum(r['fold']==f for r in rows) for f in range(4)})!=1:raise ValueError('Fold receiver counts must match')
    values=np.asarray([np.mean(v) for v in rx.values()]);rng=np.random.default_rng(16016)
    sums=np.asarray([sum(v) for v in rx.values()]);counts=np.asarray([len(v) for v in rx.values()])
    index=rng.integers(len(rx),size=(5000,len(rx)))
    draws=sums[index].sum(1)/counts[index].sum(1)
    return dict(mean_pp=float(np.mean(list(folds.values()))),positive_folds=int(sum(v>0 for v in folds.values())),folds={str(k):float(v) for k,v in folds.items()},
        worst_rx_pp=float(values.min()),ci95_pp=np.quantile(draws,[.025,.975]).tolist(),receiver_count=len(values),uncertainty='conditional RX bootstrap, no multiplicity correction')

def gate(stats,confirm=False):
    a=stats['ManyRx'];checks=dict(manyrx_mean=a['mean_pp']>=1,manyrx_positive=a['positive_folds']>=3,worst_rx=a['worst_rx_pp']>=-2,manyrx_ci=a['ci95_pp'][0]>0)
    if confirm:
        b=stats['ManySig'];checks.update(worst_rx=checks['worst_rx'] and b['worst_rx_pp']>=-2,manysig_mean=b['mean_pp']>=0,
            manysig_folds=sum(v>=-1 for v in b['folds'].values())>=3)
    return dict(admitted=all(checks.values()),checks=checks,stats=stats)

