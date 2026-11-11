"""Prespecified selected-endpoint decision; no endpoint/alpha/model search."""
from __future__ import annotations

import gzip
import numpy as np

import common as H
import executor as E

ARMS=('F0','F1','F2','E0')
KEYS={(v,s) for v in ('V0','V1') for s in ('S0','S1','S2')}


def analyze_arrays(predictions,replicates=10000):
    if set(predictions)!=set(H.ENDPOINTS):raise ValueError('All three predeclared endpoints required')
    for endpoint,arms in predictions.items():
        if set(arms)!=set(ARMS) or any(set(arms[a])!=KEYS for a in ARMS):raise ValueError('All four arms, both views and three streams required')
    draws=np.random.default_rng(24501).integers(0,12,size=(replicates,12))
    endpoints={}
    for endpoint,arms in predictions.items():
        for view,stream in KEYS:
            reference=arms['F0'][(view,stream)]
            for arm in ARMS[1:]:H.S._align(reference,arms[arm][(view,stream)])
        aggregates={a:H.S.aggregate_candidate({k:H.S._summary(v) for k,v in arms[a].items()}) for a in ARMS}
        comparisons={f'F2-{base}':H.S.paired_comparison(arms[base],arms['F2'],draws) for base in ('F1','F0','E0')}
        endpoints[endpoint]=dict(aggregates=aggregates,comparisons=comparisons)
    gates={name:H.S.promotion_check(value) for name,value in endpoints['selected']['comparisons'].items()}
    promoted=all(g['passes'] for g in gates.values())
    return dict(primary_endpoint='selected',auxiliary_endpoints=['epoch10','final40'],endpoints=endpoints,
        decision=dict(status='F2_WORTH_SEPARATE_CONFIRMATION' if promoted else 'STOP_F2_FROZEN_BACKBONE_RESIDUAL',
            candidate='F2 only',primary_contrast='F2-F1',gates=gates,alpha_or_endpoint_search=False,
            independent_CRD_confirmation=False,interpretation='Fixed source-CV development on reused source roles; not a new RX population or seed-population result.'),
        bootstrap=dict(unit='12 observed RX clusters after averaging3 fixed streams',replicates=replicates,seed=24501))


def main():
    _,_,design,jobs,source=E.scientific_context();E.require_full_seal(jobs)
    manifest=H.read(H.HERE/'EVALUATION_MANIFEST.json')
    expected={f'evaluation/{j["id"]}/{e}.npz' for j in jobs for e in H.ENDPOINTS}
    expected|={f'evaluation/{v}_{s}_{a}/reference.npz' for v,s in KEYS for a in ('F0','E0')}
    if set(manifest['files'])!=expected or manifest['seal_sha256']!=H.sha(H.HERE/'TRAINING_SEAL.json'):raise ValueError('Complete48-file evaluation block tied to the committed seal required')
    for name,digest in manifest['files'].items():
        if H.sha(H.HERE/'artifacts'/name)!=digest:raise ValueError('Prediction artifact changed')
    views={v['id']:v for v in H.read(H.DESIGN/'FUTURE_SOURCE_PLAN.json')['views']}
    predictions={}
    for endpoint in H.ENDPOINTS:
        arms={a:{} for a in ARMS}
        for arm in ARMS:
            for view,stream in KEYS:
                name='reference' if arm in ('F0','E0') else endpoint
                path=H.HERE/'artifacts/evaluation'/f'{view}_{stream}_{arm}'/(name+'.npz')
                with np.load(path,allow_pickle=False) as f:data={k:f[k] for k in f.files}
                arms[arm][(view,stream)]=H.S.validate_source_predictions(data,views[view],source)
        predictions[endpoint]=arms
    result=analyze_arrays(predictions,design['bootstrap']['replicates'])
    result['evidence']=dict(design_sha256=H.sha(H.DESIGN/'REGISTRATION.json'),execution_sha256=H.sha(H.HERE/'EXECUTION_REGISTRATION.json'),
        seal_sha256=H.sha(H.HERE/'TRAINING_SEAL.json'),evaluation_sha256=H.sha(H.HERE/'EVALUATION_MANIFEST.json'))
    with gzip.open(H.HERE/'ANALYSIS.json.gz','wt',encoding='utf-8') as f:H.json.dump(result,f,ensure_ascii=False,allow_nan=False)
    H.write(H.HERE/'DECISION.json',result['decision'])
    summary=dict(primary_endpoint='selected',decision=result['decision'],endpoints={})
    for endpoint,block in result['endpoints'].items():
        summary['endpoints'][endpoint]=dict(aggregates={a:{k:v for k,v in data.items() if k!='tx_rx_stream_mean'} for a,data in block['aggregates'].items()},
            comparisons={a:{k:v for k,v in data.items() if k not in ('cells','tx_rx_stream_mean')} for a,data in block['comparisons'].items()})
    H.write(H.HERE/'SUMMARY.json',summary)
    print(result['decision']['status'])


if __name__=='__main__':main()
