"""Freeze public science delivery and verify archives; no IQ, model load or fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(16<<20):h.update(data)
    return h.hexdigest()
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def freeze():
    if read(HERE/'INDEPENDENT_QA.json')['status']!='PASS':raise ValueError('Independent audit must pass')
    if read(HERE/'TRAINING_INVOCATION.json')['status']!='COMPLETED':raise ValueError('Original training invocation did not finish')
    if (HERE/'SCIENTIFIC_CODE_BYTES.zip').exists():raise FileExistsError('Preserve original science delivery')
    code={}
    with zipfile.ZipFile(HERE/'SCIENTIFIC_CODE_BYTES.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('scientific_qa.py','finish_science.py','close_science.py'):
            path=HERE/name;relative=path.relative_to(ROOT).as_posix();code[relative]=sha(path);z.write(path,relative)
    write_new(HERE/'SCIENTIFIC_CODE_BYTES.json',dict(archive_sha256=sha(HERE/'SCIENTIFIC_CODE_BYTES.zip'),files=code,
        scope='Postprocessing only; scientific fitting code remains frozen in EXECUTION_CODE_BYTES.zip'))
    for original,target in (('training.log','TRAINING_CONSOLE.txt'),('preflight.log','PREFLIGHT_CONSOLE.txt'),
        ('evaluation.log','EVALUATION_CONSOLE.txt'),('analysis.log','ANALYSIS_CONSOLE.txt'),
        ('scientific_qa.log','INDEPENDENT_QA_CONSOLE.txt'),('benchmark.log','BENCHMARK_CONSOLE.txt'),
        ('archive.log','ARCHIVE_CONSOLE.txt'),('report.log','REPORT_CONSOLE.txt')):
        path=HERE/target
        if path.exists():raise FileExistsError('Preserve original console evidence')
        # PowerShell native redirection can use UTF16; decode only its console file.
        data=(HERE/original).read_bytes();encoding='utf-16' if data.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8'
        path.write_text(data.decode(encoding).replace('\r\n','\n'),encoding='utf-8',newline='\n')
    decision=read(HERE/'DECISION.json')
    write_new(HERE/'SCIENCE_STATUS.json',dict(status='COMPLETED_'+decision['status'],scientific_fits=12,epochs=480,updates=60960,
        scientific_failures=0,scientific_retries=0,bookkeeping_launch_rejections=1,source_selection_predictions=48,
        source_scope='Only registered fit18 / Day1-2; selection only after committed all12 seal',
        next_candidate_authorized=False,merged_main=False,decision=decision))
    files={}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name not in ('SCIENCE_ARTIFACT_MANIFEST.json','SCIENCE_CLOSURE_QA.json') and path.suffix!='.log':
            files[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    write_new(HERE/'SCIENCE_ARTIFACT_MANIFEST.json',dict(status='COMPLETE_SCIENTIFIC_DELIVERY',files=files,
        original_engineering_manifest_unchanged=True,source_IQ_uploaded=False,model_weights_uploaded=False))


def verify():
    checks={}
    def require(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('Science delivery verification failed: '+name)
    ex=read(HERE/'EXECUTION_REGISTRATION.json')
    for name,digest in ex['code_sha256'].items():require('frozen_execution:'+name,sha(ROOT/name)==digest)
    for name,digest in ex['artifact_sha256'].items():require('original_engineering:'+name,sha(HERE/name)==digest)
    for name,item in read(HERE/'SCIENCE_ARTIFACT_MANIFEST.json')['files'].items():require('public_artifact:'+name,sha(HERE/name)==item['sha256'])
    for archive,manifest,field in (('EXECUTION_CODE_BYTES.zip',ex,'code_sha256'),('SCIENTIFIC_CODE_BYTES.zip',read(HERE/'SCIENTIFIC_CODE_BYTES.json'),'files')):
        with zipfile.ZipFile(HERE/archive) as z:
            require('code_members:'+archive,set(z.namelist())==set(manifest[field]))
            for name,digest in manifest[field].items():require('exact_code_bytes:'+name,hashlib.sha256(z.read(name)).hexdigest()==digest)
    records=read(HERE/'ARTIFACT_ARCHIVE.json')
    for field in ('predictions','public_traces'):
        item=records[field];path=HERE/item['path'];require('archive_SHA:'+field,sha(path)==item['sha256'])
        with zipfile.ZipFile(path) as z:
            require('complete_archive_members:'+field,set(z.namelist())==set(item['files']))
            for name,record in item['files'].items():require('archive_member_SHA:'+name,hashlib.sha256(z.read(name)).hexdigest()==record['sha256'])
    item=records['local_full_execution'];require('local_full_execution_SHA',sha(HERE/item['path'])==item['sha256'])
    for name,item in records['local_files'].items():require('local_full_state_file:'+name,sha(HERE/'artifacts'/name)==item['sha256'])
    pf=read(HERE/'SOURCE_PREFLIGHT.json')
    require('all_real_reference_parity_zero',all(r['max_abs_logit_difference']==0 and r['identical_argmax'] for rows in pf['checks'].values() for r in rows.values()))
    require('all_initial_branch_bitwise',all(all(r['alpha0_branch_bitwise'].values()) for rows in pf['checks'].values() for name,r in rows.items() if name.endswith('/F0')))
    require('same_authorization_identity',pf['authorization_sha256']==sha(HERE/'EXECUTION_AUTHORIZATION.json'))
    note=read(HERE/'AUTHORIZATION_TRANSCRIPTION_NOTE.json')
    require('exact_user_approval_preserved',sha(HERE/'USER_AUTHORIZATION.zh-CN.md')==note['corrected_transcript_sha256'] and sha(HERE/'EXECUTION_AUTHORIZATION.json')==note['original_authorization_sha256'])
    require('independent_QA_pass',read(HERE/'INDEPENDENT_QA.json')['status']=='PASS')
    require('same_final_decision',read(HERE/'SCIENCE_STATUS.json')['decision']==read(HERE/'DECISION.json')==read(HERE/'SUMMARY.json')['decision'])
    for name in ('EXECUTION_AUTHORIZATION.json','SOURCE_PREFLIGHT.json','TRAINING_SEAL.json','RUN_EVIDENCE.json'):
        blob=subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:research/spectral_scale_execution/'+name])
        require('committed_boundary:'+name,blob.replace(b'\r\n',b'\n')==(HERE/name).read_bytes().replace(b'\r\n',b'\n'))
    value=dict(status='PASS',checks=checks,check_count=len(checks),IQ_reads=0,model_loads=0,optimizer_updates=0)
    if (HERE/'SCIENCE_CLOSURE_QA.json').exists():
        if read(HERE/'SCIENCE_CLOSURE_QA.json')!=value:raise ValueError('Recorded closure QA differs from re-verification')
    else:write_new(HERE/'SCIENCE_CLOSURE_QA.json',value)
    print(dict(status='PASS',checks=value['check_count'],scientific_fits=12))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','verify']);args=parser.parse_args()
    {'freeze':freeze,'verify':verify}[args.command]()
