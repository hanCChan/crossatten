"""Fixed complete-arm paired summaries; no model selection or refitting."""
import csv
import numpy as np
from s27_common import HERE,read,write,validate,now


def main():
    registration=validate();evaluation=read(HERE/'EVALUATION.json');rows=evaluation['records']
    summaries=[];pairs=[];per_rx=[];tx_rows=[]
    for arm in ['H0','LA','LP']:
        selected=[r for r in rows if r['arm']==arm]
        fits=[np.mean([r['acc'] for r in selected if r['fold']==f and r['seed']==s]) for f in (0,1) for s in (201,202,203)]
        rx_means={rx:float(np.mean([r['acc'] for r in selected if r['rx']==rx])) for rx in sorted({r['rx'] for r in selected})}
        summaries.append(dict(arm=arm,accuracy_pct=float(np.mean(fits)*100),macro_f1_pct=float(np.mean([r['macro_f1'] for r in selected])*100),
            fit_sd_pp=float(np.std(fits,ddof=1)*100),fit_accuracies_pct=[float(v*100) for v in fits],
            worst_rx=min(rx_means,key=rx_means.get),worst_rx_accuracy_pct=min(rx_means.values())*100,per_rx_accuracy_pct={r:v*100 for r,v in rx_means.items()}))
        per_rx.extend(dict(arm=arm,rx=r,accuracy_pct=v*100) for r,v in rx_means.items())
        for row in selected:
            cm=np.asarray(row['confusion'])
            for tx in range(10):tx_rows.append(dict(arm=arm,fold=row['fold'],seed=row['seed'],rx=row['rx'],tx_raw_id=tx+31,
                packets=int(cm[tx].sum()),correct=int(cm[tx,tx]),accuracy_pct=float(cm[tx,tx]/cm[tx].sum()*100)))
    for a,b in [('H0','LA'),('LA','LP'),('H0','LP')]:
        aa={(r['fold'],r['seed'],r['rx']):r['acc'] for r in rows if r['arm']==a}
        bb={(r['fold'],r['seed'],r['rx']):r['acc'] for r in rows if r['arm']==b}
        assert aa.keys()==bb.keys() and len(aa)==30
        differences={rx:float(np.mean([aa[k]-bb[k] for k in aa if k[2]==rx])*100) for rx in sorted({k[2] for k in aa})}
        values=np.array(list(differences.values()));rng=np.random.default_rng(27027)
        ci=np.quantile(values[rng.integers(0,10,(10000,10))].mean(1),[.025,.975])
        fit_deltas=[float(np.mean([aa[k]-bb[k] for k in aa if k[0]==f and k[1]==s])*100) for f in (0,1) for s in (201,202,203)]
        fold_deltas=[float(np.mean([aa[k]-bb[k] for k in aa if k[0]==f])*100) for f in (0,1)]
        criterion=bool(values.mean()>=.5 and ci[0]>0 and sum(v>0 for v in fit_deltas)>=5 and min(fold_deltas)>0)
        pairs.append(dict(a=a,b=b,mean_delta_pp=float(values.mean()),conditional_rx_ci95_pp=ci.tolist(),
            positive_fit_pairs=sum(v>0 for v in fit_deltas),fit_deltas_pp=fit_deltas,fold_deltas_pp=fold_deltas,
            per_rx_delta_pp=differences,primary_practical_rule_met=criterion if a=='H0' and b=='LA' else None))
    for name,rr in [('per_rx.csv',per_rx),('tx_rx.csv',tx_rows)]:
        with (HERE/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rr[0]),lineterminator='\n');writer.writeheader();writer.writerows(rr)
    write(HERE/'RESULTS.json',dict(status='COMPLETE',time=now(),summary=summaries,paired_comparisons=pairs,
        new_fits=6,scope='Developmental post-S26 GELU intervention; source-CE selected; all six seeds and all ten RX retained. Conditional intervals are not independent acquisition evidence.',
        interpretation_rule=registration['statistics']['interpretation']))
    print('Complete fixed report:',summaries,pairs,flush=True)


if __name__=='__main__':main()
