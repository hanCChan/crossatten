"""Retrospective decision from twelve sealed prediction members; no IQ/weights."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PREVIOUS=ROOT/'research/periodic_span'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def write_new(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def pair_metrics(left,right):
    """Mean within RX; caller averages paired streams within each RX."""
    for data in (left,right):
        n=len(data['y'])
        if data['logits'].shape!=(n,10) or not np.isfinite(data['logits']).all():raise ValueError('Invalid logits')
        if len(np.unique(data['keys']))!=n:raise ValueError('Duplicate canonical key')
    lookup={key:i for i,key in enumerate(right['keys'])}
    if set(left['keys'])!=set(lookup):raise ValueError('Different input keys')
    take=np.asarray([lookup[k] for k in left['keys']],dtype=np.int64)
    right={k:v[take] for k,v in right.items()}
    for key in ('keys','y','rx','day','pool_ids'):
        if not np.array_equal(left[key],right[key]):raise ValueError('Conflicting paired '+key)
    a=left['logits'].argmax(1);b=right['logits'].argmax(1);y=left['y']
    correct_a=a==y;correct_b=b==y
    events=dict(baseline_accuracy=correct_a,candidate_accuracy=correct_b,
        corrected=(~correct_a)&correct_b,new_errors=correct_a&(~correct_b),
        disagreement=a!=b,oracle=correct_a|correct_b,both_wrong=(~correct_a)&(~correct_b))
    rows={str(rx):{k:float(v[left['rx']==rx].mean()) for k,v in events.items()} for rx in np.unique(left['rx'])}
    return rows


def register():
    manifest=read(PREVIOUS/'EVALUATION_MANIFEST.json')
    members={k:v for k,v in manifest['files'].items() if k.endswith('/selected_selection.npz') and k.split('/')[0].rsplit('_',1)[-1] in ('P0','P2')}
    if len(members)!=12:raise ValueError('Exactly twelve primary P0/P2 members required')
    documents=[
        'research/periodic_span/SUMMARY.json','research/periodic_span/REGISTRATION.json',
        'research/periodic_span/SOURCE_PLAN.json','research/periodic_span/TRAINING_SEAL.json',
        'research/periodic_span/EVALUATION_MANIFEST.json','research/periodic_span/ARTIFACT_ARCHIVE.json',
        'research/periodic_span/DECISION.json','research/periodic_hstem/AUDIT_RESULT.json',
        '07-wifi-osu/RESULTS.md','07-wifi-osu/registration.json','07-wifi-osu/S07_WIFI_FEATURE_RESULTS.md',
        '09-phase-invariance/RESULTS.md','09-phase-invariance/SOURCE_RESULTS.md','09-phase-invariance/phase_features.py',
        '19-tx-spectral-evidence/RESULTS.zh-CN.md','19-tx-spectral-evidence/s19_models.py',
        '20-asknet-source-comparison/RESULTS.zh-CN.md','20-asknet-source-comparison/UPSTREAM.json',
        '20-asknet-source-comparison/COSTS_AND_STRUCTURE.json','20-asknet-source-comparison/s20_run.py',
        'research/classification_failure_diagnostics/SUMMARY.json',
        'research/classification_failure_diagnostics/D4_FRONTEND_AUDIT.zh-CN.md',
        'lib/xa/explore_models.py','lib/legacy_rfhstu/multiscale.py',
        'research/standard_method_competition_c/SUMMARY.json',
        'research/standard_method_competition_c/EVALUATION_MANIFEST.json']
    upstream=read(ROOT/'20-asknet-source-comparison/UPSTREAM.json')
    file=Path(upstream['source_directory'])/'WiSig/backbones/PatchNet.py'
    expected=upstream['files']['WiSig/backbones/PatchNet.py']['sha256']
    if sha(file)!=expected:raise ValueError('Official cached source differs from archived upstream SHA')
    value=dict(status='RETROSPECTIVE_DECISION_REVIEW_ONLY',registered_at=now(),
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        known_information='c4b2efd results and user-provided complementarity statistics were already seen; not a blind hypothesis test.',
        maximum_scientific_fits=0,IQ_reads_allowed=0,checkpoint_reads_allowed=0,feature_cache_reads_allowed=0,
        historical_prediction_decodes_allowed=0,current_prediction_members=members,
        prediction_archive=dict(path=str(PREVIOUS/'SOURCE_PREDICTIONS.zip'),sha256=sha(PREVIOUS/'SOURCE_PREDICTIONS.zip')),
        documents={name:sha(ROOT/name) for name in documents},
        official_source=dict(path=str(file),sha256=expected,commit=upstream['commit'],executed=False),
        target_or_new_RX_evaluation_allowed=False,automatic_training_after_registration=False)
    write_new(HERE/'REVIEW_SCOPE.json',value)
    print(json.dumps(dict(status=value['status'],primary_prediction_members=len(members),scientific_fits=0)))


def audit():
    scope=read(HERE/'REVIEW_SCOPE.json');access=[]
    def logged(p,action):
        access.append(dict(time=now(),path=str(p),action=action))
        (HERE/'ACCESS_LOG.json').write_text(json.dumps(access,indent=2)+'\n',encoding='utf-8',newline='\n')
    def document(name):
        p=ROOT/name;logged(p,'read_frozen_derived_or_static_document')
        if sha(p)!=scope['documents'][name]:raise ValueError('Review document changed: '+name)
        return read(p)
    for name,digest in scope['documents'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen review input changed')
    archive=Path(scope['prediction_archive']['path']);logged(archive,'hash_sealed_archive_bytes')
    if sha(archive)!=scope['prediction_archive']['sha256']:raise ValueError('Prediction archive changed')
    summary=document('research/periodic_span/SUMMARY.json')
    source=document('research/periodic_span/REGISTRATION.json')
    plan=document('research/periodic_span/SOURCE_PLAN.json')
    views={v['id']:v for v in plan['views']};predictions={}
    with zipfile.ZipFile(archive) as z:
        for name,digest in scope['current_prediction_members'].items():
            logged(archive,'decode_only_registered_member:'+name)
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Sealed prediction changed')
            with np.load(io.BytesIO(raw),allow_pickle=False) as f:data={k:f[k] for k in f.files}
            view,stream,arm=name.split('/')[0].split('_')
            if not set(data['day'])<={1,2} or set(data['rx'])!=set(views[view]['source_selection_rx']):raise ValueError('Wrong prediction roles')
            if set(data['rx']) & (set(source['forbidden_historical_held_rx'])|set(source['forbidden_target_rx'])):raise ValueError('Forbidden current-stage RX')
            predictions[(view,stream,arm)]=data
    cells={};per_rx={}
    for view in ('V0','V1'):
        for stream in ('S0','S1','S2'):
            rows=pair_metrics(predictions[(view,stream,'P0')],predictions[(view,stream,'P2')])
            cells[view+'/'+stream]=rows
            for rx,values in rows.items():per_rx.setdefault(rx,[]).append(values)
    if len(per_rx)!=12 or any(len(rows)!=3 for rows in per_rx.values()):raise ValueError('Expected twelve distinct RX and three paired streams')
    means={rx:{k:float(np.mean([r[k] for r in rows])) for k in rows[0]} for rx,rows in per_rx.items()}
    metrics={k:float(np.mean([v[k] for v in means.values()])) for k in next(iter(means.values()))}
    metrics['fraction_of_P0_errors_recovered']=metrics['corrected']/(1-metrics['baseline_accuracy'])
    ref=summary['endpoints']['selected']['comparisons']['P2-P0']['rx_equal']
    checks=dict(corrected_matches_frozen_summary=abs(metrics['corrected']-ref['corrected_fraction'])<1e-12,
        new_errors_match_frozen_summary=abs(metrics['new_errors']-ref['new_error_fraction'])<1e-12,
        disagreement_matches_frozen_summary=abs(metrics['disagreement']-ref['disagreement'])<1e-12,
        oracle_identity=abs(metrics['oracle']-metrics['baseline_accuracy']-metrics['corrected'])<1e-12,
        accuracy_difference_identity=abs(metrics['candidate_accuracy']-metrics['baseline_accuracy']-metrics['corrected']+metrics['new_errors'])<1e-12)
    if not all(checks.values()):raise ValueError('Complementarity arithmetic failed')
    history=document('research/classification_failure_diagnostics/SUMMARY.json')
    ask_history=[r for r in history['paired_models'] if r['role']=='held_source']
    structure=document('20-asknet-source-comparison/COSTS_AND_STRUCTURE.json')
    upstream=Path(scope['official_source']['path']);logged(upstream,'hash_and_read_official_code_without_execution')
    if sha(upstream)!=scope['official_source']['sha256']:raise ValueError('Official source changed')
    _=upstream.read_text(encoding='utf-8')
    value=dict(status='REVIEW_COMPLETE_NO_TRAINING',weighting='RX equal after averaging three paired fixed streams within RX; packet equal only within each RX/stream',
        P0_P2=metrics,per_rx=means,by_view_stream=cells,checks=checks,
        historical_ASK_HStem=dict(source_folds=ask_history,unique_RX_uncertainty=history['receiver_uncertainty'],
            scope='Existing derived source summaries only; fold1-3 reuse held RX and inputs. Not a direct numerical comparison to current P0/P2 roles.'),
        static_ASK_scope=structure['structure_scope'],
        scientific_fits=0,IQ_reads=0,checkpoint_reads=0,feature_cache_reads=0,
        newly_decoded_prediction_members=12,review_scope_sha256=sha(HERE/'REVIEW_SCOPE.json'),
        oracle_interpretation='Only the two fixed label-dependent predictor choices; not a bound on jointly learned representations or a deployable selector.')
    write_new(HERE/'EVIDENCE.json',value)
    print(json.dumps(dict(status=value['status'],P0_P2=metrics,checks=checks),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','audit']);a=p.parse_args()
    {'register':register,'audit':audit}[a.command]()
