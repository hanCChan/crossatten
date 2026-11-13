"""W4-E: pair historical predictions only; no IQ, torch, model fitting or routing."""
import argparse,csv
import numpy as np
from common import *


def save_csv(name,rows):
    with (HERE/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def register():
    validate('ADMISSION_REGISTRATION.json')
    files=[HERE/n for n in ('e_audit.py','common.py','E_INPUTS.json','PREREGISTRATION.zh-CN.md')]
    files+=[ROOT/n for n in ('19-tx-spectral-evidence/s19_data.py','20-asknet-source-comparison/s20_run.py','20-asknet-source-comparison/s20_finalize.py')]
    write('E_REGISTRATION.json',dict(status='FROZEN_BEFORE_PAIRED_PREDICTION_VALUES',time=now(),
        primary_oracle_headroom_pp=3,aggregation='Mean of four fold RX-equal oracle minus best single in that fold',
        fusion='At most one RX-equal fit-validation CE-minimizing scalar alpha per fold; T=1; 64 derivative bisections',
        files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},new_neural_fits=0))


def load_prediction(item,role):
    path=checked(item,'load_saved_S20_prediction',role)
    with np.load(path,allow_pickle=False) as p:
        if set(p.files)!={'logits','y','rx','day'}:raise ValueError('Historical prediction schema differs')
        out={k:p[k] for k in p.files}
    n=len(out['y'])
    assert out['logits'].shape==(n,10) and np.isfinite(out['logits']).all()
    assert set(out['day'])=={1,2} and set(out['y'])<=set(range(10))
    return out


def quadrant(a,b,y,mask):
    n=int(mask.sum())
    if not n:return dict(packets=0,hstem_accuracy=None,asknet_accuracy=None,hstem_wrong_asknet_right=None,
        hstem_right_asknet_wrong=None,both_wrong=None,both_right=None,oracle_accuracy=None,oracle_minus_best_single_pp=None)
    ca=(a.argmax(1)==y)[mask];cb=(b.argmax(1)==y)[mask]
    pa,pb=ca.mean(),cb.mean();oracle=(ca|cb).mean()
    return dict(packets=n,hstem_accuracy=float(pa),asknet_accuracy=float(pb),
        hstem_wrong_asknet_right=float((~ca&cb).mean()),hstem_right_asknet_wrong=float((ca&~cb).mean()),
        both_wrong=float((~ca&~cb).mean()),both_right=float((ca&cb).mean()),oracle_accuracy=float(oracle),
        oracle_minus_best_single_pp=float(100*(oracle-max(pa,pb))))


def scalar_alpha(a,b,y,rx):
    # Convex CE of alpha*a+(1-alpha)*b; no temperatures or held-label input.
    a=a.astype(np.float64);b=b.astype(np.float64);delta=a-b
    groups=np.unique(rx);weight=np.zeros(len(y))
    for r in groups:weight[rx==r]=1/(len(groups)*np.sum(rx==r))
    def derivative(alpha):
        l=b+alpha*delta;l-=l.max(1,keepdims=True);p=np.exp(l);p/=p.sum(1,keepdims=True)
        return float(np.sum(weight*((p*delta).sum(1)-delta[np.arange(len(y)),y])))
    lo,hi=0.,1.
    if derivative(lo)>=0:return 0.
    if derivative(hi)<=0:return 1.
    for _ in range(64):
        mid=(lo+hi)/2
        if derivative(mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2


def run():
    validate('E_REGISTRATION.json')
    if (HERE/'E_ATTEMPT.json').exists():raise RuntimeError('No silent repetition of W4-E')
    write('E_ATTEMPT.json',dict(time=now(),registration_sha256=sha(HERE/'E_REGISTRATION.json'),commit=git('rev-parse','HEAD'),new_neural_fits=0))
    inputs=read(HERE/'E_INPUTS.json');folds=[];groups=[];identity=[];arrays={};loaded={}
    for pair in inputs['pairs']:
        f=pair['fold'];pred={}
        for model,item in pair['models'].items():
            checked(item['config'],'prediction_fit_role_metadata');checked(item['held_metadata'],'prediction_source_metadata')
            pred[model]=load_prediction(item['held'],f'fold{f}:held-source-saved-only')
        a,b=pred['HStem'],pred['ASK']
        assert all(np.array_equal(a[k],b[k]) for k in ('y','rx','day'))
        rx=np.array(pair['rx_names'])[a['rx']];assert set(rx)==set(pair['held_RX'])
        assert not set(rx)&set(pair['fit_RX'])
        keys=np.empty(len(rx),dtype='U100');positions=[]
        for cell in pair['held_source_cells']:
            ids=np.flatnonzero((rx==cell['rx'])&(a['day']==cell['day']));positions.extend(ids.tolist())
            keys[ids]=[cell['sha256']+'|'+str(i) for i in range(len(ids))]
        assert np.array_equal(positions,np.arange(len(rx))) and len(set(keys))==len(rx)
        for model,p in pred.items():
            for key in ('logits','y','rx','day'):arrays[f'f{f}_{model}_{key}']=p[key]
        arrays[f'f{f}_keys']=keys;arrays[f'f{f}_RX_names']=np.array(pair['rx_names'])
        loaded[f]=(a,b,rx)
        fr=[]
        for r in pair['held_RX']:
            stat=quadrant(a['logits'],b['logits'],a['y'],rx==r)
            row=dict(fold=f,level='RX',RX=r,TX=-1,**stat);fr.append(row);groups.append(row)
            for tx in range(10):groups.append(dict(fold=f,level='TX_RX',RX=r,TX=tx,**quadrant(a['logits'],b['logits'],a['y'],(rx==r)&(a['y']==tx))))
        for tx in range(10):groups.append(dict(fold=f,level='TX',RX='ALL',TX=tx,**quadrant(a['logits'],b['logits'],a['y'],a['y']==tx)))
        averages={k:float(np.mean([v[k] for v in fr])) for k in ('hstem_accuracy','asknet_accuracy','oracle_accuracy')}
        gap=100*(averages['oracle_accuracy']-max(averages['hstem_accuracy'],averages['asknet_accuracy']))
        folds.append(dict(fold=f,**averages,oracle_headroom_vs_fold_best_pp=gap,packets=len(rx)))
        identity.append(dict(fold=f,training_RX_disjoint=True,source_cell_SHA_equal=True,labels_RX_day_order_equal=True,
            derived_packet_keys_unique=len(keys),keys_sha256=hashlib.sha256(keys.tobytes()).hexdigest(),
            provenance='Same sealed RX/day source-cell arrays, common deterministic loader and sequential inference; cache-row-derived keys, not independently recorded RF emission IDs'))
    headroom=float(np.mean([v['oracle_headroom_vs_fold_best_pp'] for v in folds]));promote=headroom>=3
    fusion=[];weights=[]
    if promote:
        for pair in inputs['pairs']:
            f=pair['fold'];a,b,rx=loaded[f]
            va=load_prediction(pair['models']['HStem']['fit_validation'],f'fold{f}:fit-RX-validation-only')
            vb=load_prediction(pair['models']['ASK']['fit_validation'],f'fold{f}:fit-RX-validation-only')
            assert all(np.array_equal(va[k],vb[k]) for k in ('y','rx','day'))
            vrx=np.array(pair['rx_names'])[va['rx']];assert set(vrx)==set(pair['fit_RX']) and not set(vrx)&set(pair['held_RX'])
            fm=read(checked(pair['source_manifest'],'fit_cache_manifest_for_prediction_identity'));cache=Path(pair['source_manifest']['path']).parent
            ids=np.load(checked(pair['validation_indices'],'fit_validation_indices'),allow_pickle=False)
            for k in ('y','rx','day'):
                file=cache/f'fit_{k}.npy';assert sha(file)==fm['sha256'][file.name]
                expected=np.load(file,allow_pickle=False)[ids];assert np.array_equal(expected,va[k])
            alpha=scalar_alpha(va['logits'],vb['logits'],va['y'],vrx)
            weights.append(dict(fold=f,alpha_HStem=alpha,alpha_ASKNet=1-alpha,temperature=1,
                parameter_fit_RX=pair['fit_RX'],held_RX_used_to_fit_parameters=False,
                fit_HStem_prediction_sha256=pair['models']['HStem']['fit_validation']['sha256'],fit_ASKNet_prediction_sha256=pair['models']['ASK']['fit_validation']['sha256']))
            for name,p in (('HStem',va),('ASK',vb)):
                for k in ('logits','y','rx','day'):arrays[f'f{f}_fit_{name}_{k}']=p[k]
            for r in pair['held_RX']:
                mask=rx==r
                for method,logit in (('HStem',a['logits']),('ASKNet',b['logits']),('equal_logit',(a['logits']+b['logits'])/2),('fit_CE_scalar',alpha*a['logits']+(1-alpha)*b['logits'])):
                    fusion.append(dict(fold=f,RX=r,method=method,accuracy=float((logit.argmax(1)[mask]==a['y'][mask]).mean()),packets=int(mask.sum())))
    rx_rows=[g for g in groups if g['level']=='RX'];unique=sorted({v['RX'] for v in rx_rows})
    concentrations={}
    for direction in ('hstem_wrong_asknet_right','hstem_right_asknet_wrong'):
        masses={r:float(np.mean([v[direction] for v in rx_rows if v['RX']==r])) for r in unique}
        total=sum(masses.values());shares={r:v/total if total else 0 for r,v in masses.items()}
        tx_mass={str(t):float(np.mean([v[direction] for v in groups if v['level']=='TX' and v['TX']==t and v['packets']])) for t in range(10)}
        ts=sum(tx_mass.values());txshares={t:v/ts if ts else 0 for t,v in tx_mass.items()}
        concentrations[direction]=dict(RX_equal_error_mass=masses,RX_mass_share=shares,top2_RX_mass_share=sum(sorted(shares.values(),reverse=True)[:2]),
            RX_HHI=sum(s*s for s in shares.values()),TX_equal_error_mass=tx_mass,TX_mass_share=txshares,top2_TX_mass_share=sum(sorted(txshares.values(),reverse=True)[:2]))
    save_csv('E_GROUP_METRICS.csv',groups)
    if fusion:save_csv('E_FUSION_RX.csv',fusion)
    write('E_IDENTITY_AUDIT.json',dict(status='PASS_CANONICAL_SOURCE_CACHE_LINEAGE_PAIRING',folds=identity,raw_IQ_reads=0,
        caveat='No independent RF emission ID field existed in saved NPZ. Pairing is supported by the original common loader and exact cache-cell provenance.'))
    if weights:write('E_FIT_ONLY_FUSION_WEIGHTS.json',dict(status='FIXED_RULE_FIT_RX_ONLY',weights=weights))
    ART.mkdir(exist_ok=True);np.savez_compressed(ART/'E_PAIRED.npz',**arrays)
    write('E_RESULT.json',dict(status='E_ORACLE_GATE_PASS_FIXED_FUSION_EVALUATED' if promote else 'STOP_E_ORACLE_HEADROOM_BELOW_3PP',
        time=now(),folds=folds,primary_oracle_headroom_pp=headroom,threshold_pp=3,unique_RX=len(unique),
        concentrations=concentrations,fusion_candidate_count=int(promote),fusion_neural_fits=0,
        fusion_mean_accuracy={m:float(np.mean([r['accuracy'] for r in fusion if r['method']==m])) for m in sorted({r['method'] for r in fusion})},
        oracle_uses_true_labels=True,Oracle_is_not_deployment_accuracy=True,
        raw_IQ_reads=0,new_neural_fits=0,paired_prediction_cache_sha256=sha(ART/'E_PAIRED.npz')))
    print('W4-E:',headroom,'pp oracle headroom;',read(HERE/'E_RESULT.json')['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('register','run'));args=p.parse_args();globals()[args.command]()
