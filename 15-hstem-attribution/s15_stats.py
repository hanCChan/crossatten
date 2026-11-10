"""Paired receiver contrasts; seeds/folds are repeats, never extra receivers."""
import numpy as np

def paired(rows,key='delta_pp'):
    if not rows:raise ValueError('No paired receiver observations')
    rx={};fits={}
    for r in rows:
        rx.setdefault(r['rx'],[]).append(r[key])
        fits.setdefault((r['fold'],r['seed']),[]).append(r[key])
    values=np.asarray([np.mean(v) for v in rx.values()]);means=np.asarray([np.mean(v) for v in fits.values()])
    rng=np.random.default_rng(14014);boot=rng.choice(values,(5000,len(values)),replace=True).mean(1)
    return dict(mean_delta_pp=float(means.mean()),positive_pairs=int((means>0).sum()),pairs=len(fits),receivers=len(rx),
        worst_receiver_delta_pp=float(values.min()),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
        uncertainty='RX-cluster bootstrap conditional on these fitted seeds; no multiplicity correction')

def factorial(pm,ph,hm,hh):
    dm=hm-pm;dh=hh-ph
    return dict(hstem_effect_mha=dm,hstem_effect_hstu=dh,interaction=dh-dm,
        stem_main=(dm+dh)/2,mixer_effect_patch=ph-pm,mixer_effect_hstem=hh-hm,
        mixer_main=((ph-pm)+(hh-hm))/2)

def mechanism_gate(comparisons):
    needed=['A1_mha','A1_mha_matched','A2_transformer','A2_transformer_matched']
    details={}
    for arm in needed:
        s=comparisons[arm]
        details[arm]=dict(mean=s['mean_delta_pp']>=1,pairs=s['positive_pairs']>=5,
            worst=s['worst_receiver_delta_pp']>=-2,ci=s['ci95_pp'][0]>0)
    return dict(admitted=all(all(v.values()) for v in details.values()),checks=details,
        rule='dev_val only; >=1pp, >=5/6 positive, worstRX>=-2pp, RX CI lower>0 against all4 attention controls; descriptive budget gate')
