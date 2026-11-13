"""Package and test a NumPy-only, feature-level W3 review without raw IQ."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OUT=ROOT/'output/review'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    assert read(HERE/'B_INDEPENDENT_AUDIT_QA.json')['status']=='PASS'
    OUT.mkdir(parents=True,exist_ok=True);archive=OUT/'wisig_w3_ab_review.zip'
    files={ROOT/rel for rel in read(HERE/'B_EXECUTION_REGISTRATION.json')['files_sha256']}
    files.update(p for p in HERE.iterdir() if p.is_file() and p.name!='B_REVIEW_PACKAGE.json')
    files.update((HERE/'artifacts'/name) for name in ('B_FEATURES.npz','B_OPERATORS.npz',
        '002_FullWiSig_create_compact_datasets.py','data_utilities.py'))
    files.add(HERE/'B_EXECUTION_REGISTRATION.json')
    def build():
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    build()
    folder=OUT/('w3_portable_verification_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    folder.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (folder/name).resolve().is_relative_to(folder.resolve()):raise ValueError('Unexpected archive path')
        z.extractall(folder)
    command=['D:/crossatten/.venv/Scripts/python.exe',str(folder/'research/wisig_w3/audit_b.py'),'--portable']
    completed=subprocess.run(command,cwd=folder,capture_output=True,text=True,encoding='utf-8')
    if completed.returncode:raise RuntimeError(completed.stdout+completed.stderr)
    portable=folder/'research/wisig_w3/B_PORTABLE_AUDIT_QA.json'
    assert read(portable)['status']=='PASS'
    shutil.copyfile(portable,HERE/'B_PORTABLE_AUDIT_QA.json');files.add(HERE/'B_PORTABLE_AUDIT_QA.json')
    build()
    # Adding the QA receipt must not alter any file tested in the fresh directory.
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:
            assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
    report=dict(status='PASS_PORTABLE_FEATURE_LEVEL_REVIEW',archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha(archive),files=len(files),portable_audit_checks=read(portable)['checks_passed'],
        verified_in_fresh_directory=str(folder),verification_command='python research/wisig_w3/audit_b.py --portable',
        dependencies=['Python 3.10+','NumPy'],raw_IQ_included=False,neural_weights_included=False,
        included=['Frozen 64D source-only EMA/ERM features','All shallow operator parameters','Registered code and metadata','Full metric tables and results','Report and figures','NumPy-only verifier'],
        reproduction_boundary='Feature-to-result arithmetic and fit-TX normal equations; not raw-IQ-to-feature extraction',
        all_final_archive_entry_hashes_checked=True)
    write(HERE/'B_REVIEW_PACKAGE.json',report);print(completed.stdout);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
