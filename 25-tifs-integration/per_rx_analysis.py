"""Post-hoc full RX/TX decomposition using existing S24 confusion matrices only."""
import csv
import numpy as np
from s25_common import HERE, ROOT, read, validate, write


def save(name, rows):
    with (HERE/name).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    validate()
    folder=ROOT/'24-lora-external-audit/phase2/budget_update'
    rows=[r for path in sorted((folder/'evaluations').glob('*/metrics.json')) for r in read(path)['rows']]
    receivers=[]; tx_rows=[]; confusions=[]; contrasts=[]
    for method,view in sorted({(r['method'],r['view']) for r in rows}):
        selected=[r for r in rows if r['method']==method and r['view']==view]
        for rx in sorted({r['rx'] for r in selected}):
            cells=[r for r in selected if r['rx']==rx]
            assert len(cells)==3
            cm=np.sum([np.asarray(r['confusion'],dtype=np.int64) for r in cells],axis=0)
            receivers.append(dict(method=method,view=view,rx=rx,accuracy_pct=float(np.trace(cm)/cm.sum()*100),
                                  min_seed_accuracy_pct=min(r['acc'] for r in cells)*100,max_seed_accuracy_pct=max(r['acc'] for r in cells)*100))
            for tx in range(10):
                tx_rows.append(dict(method=method,view=view,rx=rx,tx_raw_id=tx+31,correct=int(cm[tx,tx]),
                                    packets=int(cm[tx].sum()),accuracy_pct=float(cm[tx,tx]/cm[tx].sum()*100)))
                for prediction in range(10):
                    if prediction!=tx and cm[tx,prediction]:
                        confusions.append(dict(method=method,view=view,rx=rx,true_tx_raw_id=tx+31,
                                               predicted_tx_raw_id=prediction+31,count=int(cm[tx,prediction])))
    for view in ['native','strict_ib']:
        for baseline in ['ce','dann','fan','ib']+(['ask'] if view=='native' else []):
            a={r['rx']:r['accuracy_pct'] for r in receivers if r['method']=='h0'}
            b={r['rx']:r['accuracy_pct'] for r in receivers if r['method']==baseline and r['view']==view}
            deltas={rx:a[rx]-b[rx] for rx in a}
            total=float(np.mean(list(deltas.values())))
            pluto=deltas['pluto_1']
            contrasts.append(dict(view=view,baseline=baseline,mean_h0_minus_baseline_pp=total,
                pluto_delta_pp=pluto,pluto_weighted_contribution_pp=pluto/10,
                pluto_share_of_mean_delta=pluto/10/total if abs(total)>1e-12 else None,
                other_nine_mean_delta_pp=float(np.mean([v for k,v in deltas.items() if k!='pluto_1'])),
                rx_better=sum(v>0 for v in deltas.values()),rx_worse=sum(v<0 for v in deltas.values()),per_rx_delta_pp=deltas))
    changes=[]
    for method in ['ce','dann','fan','ib']:
        for rx in sorted({r['rx'] for r in receivers}):
            native=next(r['accuracy_pct'] for r in receivers if r['method']==method and r['view']=='native' and r['rx']==rx)
            strict=next(r['accuracy_pct'] for r in receivers if r['method']==method and r['view']=='strict_ib' and r['rx']==rx)
            changes.append(dict(method=method,rx=rx,native_pct=native,strict_ib_pct=strict,strict_minus_native_pp=strict-native))
    save('per_rx.csv',receivers);save('per_tx_rx.csv',tx_rows);save('confusion_pairs.csv',confusions);save('information_budget_effect.csv',changes)
    write(HERE/'PER_RX_ANALYSIS.json',dict(status='COMPLETE',receivers=receivers,paired_rx_contributions=contrasts,
          information_budget_effect=changes,new_waveform_reads=0,new_fits=0,
          scope='Post-hoc descriptive decomposition of all fixed RX; no excluded RX, new hardware cause, checkpoint choice or new independent test.'))
    print('Complete: full RX/TX/error decomposition, using existing confusion matrices only.')


if __name__=='__main__':
    main()
