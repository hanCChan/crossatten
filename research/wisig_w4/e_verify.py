"""Independent W4-E arithmetic/KKT/provenance audit; no IQ or neural imports."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b):assert np.allclose(a,b,rtol=1e-11,atol=1e-11),(a,b)


def main():
    result=read(HERE/'E_RESULT.json');inputs=read(HERE/'E_INPUTS.json');reg=read(HERE/'E_REGISTRATION.json')
    for path,digest in reg['files_sha256'].items():assert sha(ROOT/path)==digest,path
    path=HERE/'artifacts/E_PAIRED.npz';assert sha(path)==result['paired_prediction_cache_sha256']
    cache=np.load(path,allow_pickle=False)
    with (HERE/'E_GROUP_METRICS.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    fold_values=[];rx_values=[];checks=0;fold_group={}
    for pair in inputs['pairs']:
        f=pair['fold'];a=cache[f'f{f}_HStem_logits'];b=cache[f'f{f}_ASK_logits'];y=cache[f'f{f}_HStem_y']
        for k in ('y','rx','day'):assert np.array_equal(cache[f'f{f}_HStem_{k}'],cache[f'f{f}_ASK_{k}'])
        assert not set(pair['fit_RX'])&set(pair['held_RX'])
        r=cache[f'f{f}_RX_names'][cache[f'f{f}_HStem_rx']]
        ca=a.argmax(1)==y;cb=b.argmax(1)==y
        quads=dict(hstem_accuracy=ca,asknet_accuracy=cb,hstem_wrong_asknet_right=(1-ca)*cb,
            hstem_right_asknet_wrong=ca*(1-cb),both_wrong=(1-ca)*(1-cb),both_right=ca*cb,oracle_accuracy=np.maximum(ca,cb))
        assert np.all(quads['hstem_wrong_asknet_right']+quads['hstem_right_asknet_wrong']+quads['both_wrong']+quads['both_right']==1)
        frows=[v for v in rows if int(v['fold'])==f]
        assert len(frows)==76
        for row in frows:
            mask=np.ones(len(y),dtype=bool)
            if row['level']!='TX':mask&=r==row['RX']
            if row['level']!='RX':mask&=y==int(row['TX'])
            assert int(row['packets'])==int(mask.sum());checks+=1
            if not mask.any():
                assert all(row[k]=='' for k in quads);continue
            for k,value in quads.items():close(float(row[k]),value[mask].mean());checks+=1
        rx_summary={k:np.array([v[r==rx].mean() for rx in pair['held_RX']]) for k,v in quads.items()}
        point=100*(rx_summary['oracle_accuracy'].mean()-max(rx_summary['hstem_accuracy'].mean(),rx_summary['asknet_accuracy'].mean()))
        close(point,result['folds'][f]['oracle_headroom_vs_fold_best_pp']);fold_values.append(point)
        best='hstem_accuracy' if rx_summary['hstem_accuracy'].mean()>rx_summary['asknet_accuracy'].mean() else 'asknet_accuracy'
        for i,rx in enumerate(pair['held_RX']):rx_values.append((rx,100*(rx_summary['oracle_accuracy'][i]-rx_summary[best][i])))
        fold_group[f]=(a,b,y,r)
    close(np.mean(fold_values),result['primary_oracle_headroom_pp']);assert result['fusion_candidate_count']==int(np.mean(fold_values)>=3)
    if result['fusion_candidate_count']:
        weights=read(HERE/'E_FIT_ONLY_FUSION_WEIGHTS.json')['weights']
        with (HERE/'E_FUSION_RX.csv').open(encoding='utf-8',newline='') as f:fr=list(csv.DictReader(f))
        for pair in inputs['pairs']:
            f=pair['fold'];alpha=weights[f]['alpha_HStem'];assert 0<=alpha<=1
            a=cache[f'f{f}_fit_HStem_logits'].astype(np.float64);b=cache[f'f{f}_fit_ASK_logits'].astype(np.float64)
            y=cache[f'f{f}_fit_HStem_y'];rx=cache[f'f{f}_RX_names'][cache[f'f{f}_fit_HStem_rx']]
            assert set(rx)==set(pair['fit_RX']) and not set(rx)&set(pair['held_RX'])
            for k in ('y','rx','day'):assert np.array_equal(cache[f'f{f}_fit_HStem_{k}'],cache[f'f{f}_fit_ASK_{k}'])
            logits=alpha*a+(1-alpha)*b;logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
            residual=(p*(a-b)).sum(1)-(a-b)[np.arange(len(y)),y]
            derivative=np.mean([residual[rx==r].mean() for r in pair['fit_RX']])
            assert (alpha<1e-10 and derivative>=-1e-9) or (alpha>1-1e-10 and derivative<=1e-9) or abs(derivative)<1e-9
            a,b,y,rx=fold_group[f]
            expected=dict(HStem=a,ASKNet=b,equal_logit=(a+b)/2,fit_CE_scalar=alpha*a+(1-alpha)*b)
            for row in [v for v in fr if int(v['fold'])==f]:
                mask=rx==row['RX'];close(float(row['accuracy']),(expected[row['method']].argmax(1)[mask]==y[mask]).mean());checks+=1
        for method,value in result['fusion_mean_accuracy'].items():close(value,np.mean([float(v['accuracy']) for v in fr if v['method']==method]))
    # Check historical metadata still agrees with its original archive manifests.
    for pair in inputs['pairs']:
        for model,item in pair['models'].items():
            directory=Path(item['config']['path']).parents[2];original=read(directory/'MANIFEST.json')['files_sha256']
            for kind in ('config','held_metadata'):
                p=Path(item[kind]['path']);assert sha(p)==item[kind]['sha256']==original[p.relative_to(directory).as_posix()]
        fm=read(pair['source_manifest']['path']);cache_dir=Path(pair['source_manifest']['path']).parent
        for name,kind in (('train_ids.npy','train_indices'),('val_ids.npy','validation_indices')):
            assert sha(cache_dir/name)==fm['sha256'][name]==pair[kind]['sha256']
    unique=sorted({r for r,v in rx_values});delta=np.array([np.mean([v for r,v in rx_values if r==rx]) for rx in unique])
    rng=np.random.default_rng(42001);draw=rng.integers(len(unique),size=(10000,len(unique)));interval=np.quantile(delta[draw].mean(1),[.025,.975])
    fusion_risk={}
    if result['fusion_candidate_count']:
        for method in ('equal_logit','fit_CE_scalar'):
            paired=[]
            for f in range(4):
                for r in inputs['pairs'][f]['held_RX']:
                    vals={v['method']:float(v['accuracy']) for v in fr if int(v['fold'])==f and v['RX']==r}
                    paired.append(dict(fold=f,RX=r,gain_vs_ASK_pp=100*(vals[method]-vals['ASKNet']),gain_vs_HStem_pp=100*(vals[method]-vals['HStem'])))
            fusion_risk[method]=dict(worst_fold_RX_gain_vs_ASK_pp=min(v['gain_vs_ASK_pp'] for v in paired),
                worst_unique_RX_gain_vs_ASK_pp=min(np.mean([v['gain_vs_ASK_pp'] for v in paired if v['RX']==r]) for r in unique),paired=paired)
    report=dict(status='PASS',numerical_checks=checks,own_code_sha256=sha(__file__),all_historical_metadata_seals_match=True,
        all_quadrant_rows_recomputed=True,fit_only_scalar_KKT_verified=True,raw_IQ_reads=0,neural_fits=0,
        oracle_headroom_primary_pp=float(np.mean(fold_values)),unique_RX_oracle_gain_pp=float(delta.mean()),
        unique_RX_conditional_oracle_gain_CI95_pp=interval.tolist(),fusion_risk=fusion_risk,
        note='Oracle and per-fold best single are labeled diagnostics; S20 source protocol differs from W4-P V0.')
    (HERE/'E_INDEPENDENT_QA.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS W4-E independent quadrant, provenance and fit-only fusion audit',checks,'checks')


if __name__=='__main__':main()
