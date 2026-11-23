"""Seal coverage/provenance while retaining the failed native numeric contract."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from pypdf import PdfReader
from s28_core import HERE,ROOT,P,read,write,sha,now,validate,committed

def main():
    reg=validate();summary=read(HERE/'SUMMARY.json');affine=read(HERE/'AFFINE_AUDIT.json');precision=read(HERE/'PRECISION_DIAGNOSTIC.json')
    assert summary['scientific_fits']==0 and not list((HERE/'runs').glob('*/metrics.json'))
    assert summary['FP64_all_pass'] and not summary['native_contract_all_pass'] and summary['strict_probe_all_pass']
    assert len(affine['models'])==len(precision['models'])==6 and summary['native_changed_predictions']==1
    assert all(not q['source_FP32']['errors']['tokens']['allclose'] for q in affine['models'])
    assert all(len(q['source_FP32']['corrected_per_position_max_abs'])==32 and q['source_FP32']['packets']==2000 for q in affine['models'])
    assert all(q['source_FP64']['packets']==50 and q['source_FP64']['errors']['naive_interior']['allclose'] and not q['source_FP64']['errors']['naive']['allclose'] for q in affine['models'])
    assert read(HERE/'FACTORIAL_READINESS.json')['interaction'] is None
    assert read(HERE/'PLAN_A.json')['automatic_fit_budget']==read(HERE/'PLAN_C.original.json')['automatic_fit_budget']==0
    assert sha(HERE/'PLAN_C.original.json')==sha(ROOT/'27-mechanism-identification/PLAN_C.json')
    assert read(HERE/'EVIDENCE_MATRIX.json')['rows'][:-4]==read(ROOT/'27b-tokenizer-backend-interaction/EVIDENCE_MATRIX.json')['rows']
    ledger=read(HERE/'ENGINEERING_UPDATES.json');assert ledger['updates']==sum(q.get('updates',1) for q in ledger['events'])==7
    assert reg['synthetic_gate_optimizer_updates']==7
    historical=[]
    for q in reg['references']:
        for file,h in q['sha256'].items():assert sha(Path(q['checkpoint']).parent/file)==h
        historical.append(q['id'])
    events=[json.loads(s) for s in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(events)==20
    for e in events:
        assert e['role']=='audit_source_val' and e['rx'] in P['folds'][e['fold']]['source'] and e['optimizer_access'] is False
    plan=read(HERE/'PRECISION_PLAN.json');committed(HERE/'PRECISION_PLAN.json')
    assert sha(HERE/'AFFINE_AUDIT.json')==plan['original_audit_sha256'] and sha(HERE/'s28_precision.py')==plan['script_sha256']
    documents=[];visual=read(HERE/'VISUAL_QA.json');assert visual['status']=='PASS'
    for d in read(HERE/'DOCUMENT_BUILD.json')['documents']:
        p=Path(d['path']);assert sha(p)==d['sha256'];pages=PdfReader(p).pages
        v=next(t for t in visual['documents'] if t['sha256']==sha(p));assert v['reviewed_pages']==list(range(1,len(pages)+1))
        log=Path(d['log']).read_text(encoding='utf-8',errors='replace')
        assert not re.search(r'Overfull|LaTeX Warning|undefined|Missing character|Fatal error',log)
        assert all('\ufffd' not in (page.extract_text() or '') for page in pages)
        documents.append(dict(path=p.relative_to(HERE).as_posix(),sha256=sha(p),pages=len(pages),
            renders=[dict(page=n,sha256=sha(Path(d['render'])/f'page-{n}.png')) for n in range(1,len(pages)+1)]))
    protected=[HERE/'registration.json',HERE/'AFFINE_AUDIT.json',HERE/'SOURCE_RECOVERY.json',HERE/'access.jsonl']
    before={p.name:sha(p) for p in protected}
    guard=subprocess.run([sys.executable,str(HERE/'launch.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert guard.returncode==1 and 'NO TRAINING STARTED' in guard.stdout.decode('utf-8')
    assert before=={p.name:sha(p) for p in protected}
    write(HERE/'LAUNCH_GUARD.json',dict(time=now(),status='PASS',output=guard.stdout.decode('utf-8').strip(),protected_artifact_sha256=before,
        no_new_fit_or_RF_access=True))
    logs=[]
    for p in sorted((HERE/'local').glob('*.log')):
        raw=p.read_bytes()
        for encoding in ['utf-8','utf-16','gb18030']:
            try:text=raw.decode(encoding);break
            except UnicodeError:continue
        else:raise UnicodeError(str(p))
        assert 'Traceback (most recent call last):' not in text
        out=HERE/(p.stem.upper().replace('.','_')+'.txt');out.write_text(text.replace('\r\n','\n'),encoding='utf-8',newline='\n')
        logs.append(dict(raw_file=p.name,raw_sha256=sha(p),raw_encoding=encoding,published=out.name,sha256=sha(out)))
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    changed=subprocess.check_output(['git','diff',reg['base_git'],'--name-only'],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith(HERE.name+'/') for p in changed)
    write(HERE/'STATUS.json',dict(time=now(),status='COMPLETE_READ_ONLY_AUDIT_WITH_NATIVE_PRECISION_FAILURE',scientific_fits=0,
        source_models_recovered=24,source_predictions=48000,affine_models=6,native_FP32_contract='FAILED_PRESERVED_ALL_SIX',
        FP64_contract='PASS_ALL_SIX',TF32_off_probe='PASS_ALL_SIX_50_PACKETS_EACH',A_completed=0,A_planned=6,C_completed=0,C_planned=12,
        S28B_fits=0,synthetic_engineering_updates=7,Dev_Confirm_blind_reads=0,remaining_automatic_fit_budget=0,
        user_training_authorization='REQUIRED_BY_USER_ATTACHMENT_NOT_YET_GIVEN',visual='PASS',documents=documents))
    artifacts=[]
    for p in sorted(HERE.rglob('*')):
        rel=p.relative_to(HERE)
        if not p.is_file() or any(x in ['local','converted','tmp','__pycache__'] for x in rel.parts) or p.name=='FINAL_AUDIT.json':continue
        artifacts.append(dict(path=rel.as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    write(HERE/'FINAL_AUDIT.json',dict(time=now(),status='PASS_COVERAGE_AND_PROVENANCE_WITH_DISCLOSED_NATIVE_NUMERIC_FAILURE',
        mathematical_numeric_outcomes=summary,original_scientific_code_unchanged=True,original_models_unchanged=historical,
        original_C_plan_byte_exact=True,original_native_failures_not_overwritten=True,no_new_scientific_fits=True,
        synthetic_optimizer_updates=7,Source_cache_read_events=len(events),new_Dev_Confirm_blind_reads=0,all_six_models_and_all32positions_retained=True,
        initial_preregistration_commit=subprocess.check_output(['git','rev-parse','e807396'],cwd=ROOT,text=True).strip(),
        precision_plan_sha256=sha(HERE/'PRECISION_PLAN.json'),scientific_registration_sha256=sha(HERE/'registration.json'),
        logs=logs,documents=documents,artifacts=artifacts))
    print('Audit coverage/provenance PASS; native precision failures retained;',len(artifacts),'public artifacts; scientific fits0.',flush=True)

if __name__=='__main__':main()
