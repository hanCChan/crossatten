"""Verify the design and derived arithmetic only; never decodes prediction/IQ arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    r=read(HERE/'REGISTRATION.json');scope=read(HERE/'REVIEW_SCOPE.json');e=read(HERE/'EVIDENCE.json');checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('Design QA failed: '+name)
    check('execution_closed',r['status']=='DESIGN_PREREGISTERED_EXECUTION_CLOSED' and r['current_turn']['scientific_fits_authorized']==0)
    check('no_scientific_fits_or_new_inputs',all(v==0 for v in r['current_turn'].values()))
    for name,digest in r['code_sha256'].items():check('pinned_code:'+name,sha(ROOT/name)==digest)
    for name,digest in r['artifact_sha256'].items():check('pinned_artifact:'+name,sha(HERE/name)==digest)
    check('review_scope',e['review_scope_sha256']==sha(HERE/'REVIEW_SCOPE.json'))
    for name,digest in scope['documents'].items():check('historical_document:'+name,sha(ROOT/name)==digest)
    check('official_cached_source_SHA',sha(Path(scope['official_source']['path']))==scope['official_source']['sha256'])
    # Re-average only the already-derived means, using Python statistics rather than numpy.
    rows={}
    for group,receivers in e['by_view_stream'].items():
        for rx,value in receivers.items():rows.setdefault(rx,[]).append(value)
    check('12_distinct_RX_3_streams',len(rows)==12 and all(len(v)==3 for v in rows.values()))
    for name in ('baseline_accuracy','candidate_accuracy','corrected','new_errors','disagreement','oracle','both_wrong'):
        value=mean(mean(row[name] for row in receiver) for receiver in rows.values())
        check('independent_mean:'+name,abs(value-e['P0_P2'][name])<1e-12)
    p=e['P0_P2']
    check('oracle_identity',abs(p['oracle']-(p['baseline_accuracy']+p['corrected']))<1e-12)
    check('error_recovery_fraction',abs(p['fraction_of_P0_errors_recovered']-p['corrected']/(1-p['baseline_accuracy']))<1e-12)
    access=read(HERE/'ACCESS_LOG.json');allowed={str(ROOT/name) for name in scope['documents']}
    allowed|={scope['prediction_archive']['path'],scope['official_source']['path']}
    check('read_paths_whitelisted',all(v['path'] in allowed for v in access))
    members=[v['action'].split(':',1)[1] for v in access if v['action'].startswith('decode_only_registered_member:')]
    check('exact12_primary_members',len(members)==12 and set(members)==set(scope['current_prediction_members']))
    check('no_IQ_or_checkpoint_file_paths',not any(Path(v['path']).suffix in ('.npy','.pt','.pth','.pkl') for v in access))
    jobs=read(HERE/'PROPOSED_JOBS.json')
    expected={(v,f'S{s}',a) for v in ('V0','V1') for s in range(3) for a in ('F1','F2')}
    check('exact12_future_jobs',len(jobs)==12 and {(j['view'],j['stream'],j['arm']) for j in jobs}==expected)
    check('fixed_future_budget',sum(j['epochs'] for j in jobs)==480 and sum(j['max_gradient_updates'] for j in jobs)==60960)
    check('checkpoint_rule',all(j['checkpoint_candidate_epochs']==list(range(41)) and j['auxiliary_epochs']==[10,40] for j in jobs))
    check('same_active_and_total_parameters',r['parameters']['F1']==r['parameters']['F2']==dict(total=85627,trainable=3201))
    # Independent coefficient counting for the two different modules.
    multiscale=32*8+8*8*3+8*8*33+16*8+8*64+1
    ordinary=32*8+8*8*33+8*20+20*8+8*64+1
    check('all_parameters_accounted_for',multiscale==ordinary==3201 and 82426+multiscale==85627)
    refs=read(HERE/'FROZEN_REFERENCES.json');seal=read(ROOT/'research/periodic_span/TRAINING_SEAL.json')
    check('six_original_P0_references',len(refs)==6 and all(v['sha256']==seal['runs'][v['job']]['checkpoints']['selected'] for v in refs.values()))
    ema=read(HERE/'EMA_REFERENCE.json');manifest=read(ROOT/ema['original_manifest_path'])
    check('EMA_reference_is_unchanged',sha(ROOT/ema['original_manifest_path'])==ema['original_manifest_sha256'] and len(ema['members'])==6 and all(manifest['files'][k]==h for k,h in ema['members'].items()))
    for stage in ('periodic_hstem','periodic_span'):
        old=ROOT/'research'/stage
        for name,value in read(old/'ARTIFACT_MANIFEST.json')['files'].items():check('original_STOP_artifact:'+stage+'/'+name,sha(old/name)==value['sha256'])
    code=read(HERE/'DESIGN_BYTES.json')
    check('code_archive_SHA',sha(HERE/'DESIGN_CODE_BYTES.zip')==code['archive_sha256'])
    with zipfile.ZipFile(HERE/'DESIGN_CODE_BYTES.zip') as archive:
        check('code_archive_members',set(archive.namelist())==set(code['files']))
        for name,digest in code['files'].items():check('exact_design_bytes:'+name,hashlib.sha256(archive.read(name)).hexdigest()==digest)
    value=dict(status='PASS',check_count=len(checks),checks=checks,IQ_reads=0,checkpoint_reads=0,
        prediction_decodes_by_verifier=0,optimizer_steps=0,scientific_fits=0,
        status_scope='Design and previously derived source evidence only; no F1/F2 performance evaluation')
    (HERE/'DESIGN_QA.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),scientific_fits=0)))


if __name__=='__main__':main()
