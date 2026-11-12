"""Publication byte archive and closure audit only; never loads scientific arrays."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    r=read(HERE/'REGISTRATION.json');a=read(HERE/'ENGINEERING_AMENDMENT.json');checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('Closure failed:'+name)
    codes=dict(r['code_sha256']);codes.update(a['updated_code_sha256'])
    for name,digest in codes.items():check('actual_code:'+name,sha(ROOT/name)==digest)
    for name,digest in r['document_sha256'].items():check('frozen_design:'+name,sha(HERE/name)==digest)
    for name,digest in r['history_sha256'].items():check('historical_STOP:'+name,sha(ROOT/name)==digest)
    for name,digest in a['preserved_failure_files'].items():check('first_engineering_attempt_preserved:'+name,sha(HERE/name)==digest)
    check('user_request_preserved',sha(HERE/'USER_REQUEST.txt')==r['user_request_sha256'])
    check('independent_QR_pass',read(HERE/'INDEPENDENT_QA.json')['status']=='PASS')
    check('stage_closed_no_future_training',read(HERE/'DECISION.json')['closed'] and not read(HERE/'DECISION.json')['M1_B_authorized'] and not read(HERE/'DECISION.json')['M1_C_authorized'])
    log=[json.loads(x) for x in (HERE/'ACCESS_LOG.jsonl').read_text(encoding='utf-8').splitlines()]
    actions=[x for x in log if x['action'].startswith(('copy_only','independent_QR_replay'))]
    check('same_source_rows_twice_only',len(actions)==2 and all(x['action'].endswith(':1872') for x in actions))
    # Exact bytes survive Git's CRLF conversion by an explicit archive.
    archive=HERE/'EXECUTION_BYTES.zip';files={}
    if archive.exists():raise FileExistsError('Keep original execution byte archive')
    paths=set(codes)|set(r['parent_protocol_sha256'])|set(r['official_snapshots'])
    paths|={p.relative_to(ROOT).as_posix() for p in HERE.iterdir() if p.is_file() and p.name!='ARTIFACT_MANIFEST.json' and p.suffix in ('.py','.md','.txt')}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(paths):z.write(ROOT/name,name);files[name]=sha(ROOT/name)
    with zipfile.ZipFile(archive) as z:
        check('archive_member_set',set(z.namelist())==set(files))
        for name,digest in files.items():check('archived_bytes:'+name,hashlib.sha256(z.read(name)).hexdigest()==digest)
    write(HERE/'EXECUTION_BYTES.json',dict(archive_sha256=sha(archive),files=files,raw_IQ_included=False))
    write(HERE/'CLOSURE_QA.json',dict(status='PASS',checks=checks,check_count=len(checks),IQ_decodes=0,classifier_fits=0,neural_updates=0))
    # Replaces only this stage's initial report manifest, never historical evidence.
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='ARTIFACT_MANIFEST.json' and p.suffix!='.log'}
    write(HERE/'ARTIFACT_MANIFEST.json',dict(status=read(HERE/'DECISION.json')['status'],files=files,raw_IQ_included=False,classifier_fits=0,neural_updates=0))
    print(dict(status='PASS',closure_checks=len(checks),files=len(files)))


if __name__=='__main__':main()
