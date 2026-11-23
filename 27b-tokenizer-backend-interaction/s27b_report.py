"""All four cells and preregistered difference of differences; no selection."""
import csv
import numpy as np
from s27b_common import HERE,read,write,validate,now
from s27b_stats import ARMS,contrast,summary


def main():
    validate();rows=read(HERE/'EVALUATION.json')['records'];source=read(HERE/'SOURCE_EVALUATION.json')
    primary=contrast(rows,dict(H0=1,LP=-1,H0_P=-1,LP_P=1))
    contrasts=[dict(name=n,**contrast(rows,w)) for n,w in [
        ('tokenizer_MHA',dict(H0=1,LP=-1)),('tokenizer_MLP',dict(H0_P=1,LP_P=-1)),
        ('backend_HStem',dict(H0=1,H0_P=-1)),('backend_LongPatch',dict(LP=1,LP_P=-1))]]
    summaries=[summary(rows,a) for a in ARMS];source_summaries=[summary(source['records'],a) for a in ARMS]
    src=contrast(source['records'],dict(H0=1,LP=-1,H0_P=-1,LP_P=1))
    source_interaction={k:v for k,v in src.items() if k not in ['conditional_rx_ci95_pp','practical_rule_met']}
    gaps=[x-y for x,y in zip(src['fit_deltas_pp'],primary['fit_deltas_pp'])]
    rx=[dict(rx=rx,interaction_pp=value,**{a:next(s for s in summaries if s['arm']==a)['per_rx_accuracy_pct'][rx] for a in ARMS}) for rx,value in primary['per_rx_delta_pp'].items()]
    tx=[]
    for row in rows:
        cm=np.array(row['confusion'])
        for t in range(10):tx.append(dict(arm=row['arm'],fold=row['fold'],seed=row['seed'],rx=row['rx'],tx_raw_id=t+31,
            packets=int(cm[t].sum()),correct=int(cm[t,t]),accuracy_pct=float(cm[t,t]/cm[t].sum()*100)))
    for name,rr in [('per_rx.csv',rx),('tx_rx.csv',tx),('source_geometry.csv',source['geometry'])]:
        fields=list(dict.fromkeys(k for r in rr for k in r))
        with (HERE/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rr)
    write(HERE/'RESULTS.json',dict(status='COMPLETE',time=now(),new_fits=12,reused_models=12,summary=summaries,
        primary_interaction=primary,secondary_contrasts=contrasts,source_summary=source_summaries,
        source_interaction=source_interaction,source_minus_dev_interaction=dict(mean_pp=float(np.mean(gaps)),fit_deltas_pp=gaps),
        scope='Developmental fourcellbackend interaction. Sourcegeometry/sourceDevdifferences descriptive; B is not C. Allfits/RX retained; no universal attention necessity orphysicalRF invariance claim.'))
    print('Complete fourcell report. Primary interaction:',primary,flush=True)


if __name__=='__main__':main()
