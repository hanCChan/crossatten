"""Assemble exact replay inputs and selected weights; exclude raw IQ/optimizer caches."""
import hashlib,json,shutil,subprocess,sys,zipfile
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;ART=HERE/'artifacts'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def collect():
    records={}
    def add(item,relative=None):
        p=Path(item['path']);digest=sha(p);assert digest==item['sha256'],str(p)
        if relative is None:
            relative=p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else f'research/wisig_w5/artifacts/review_metadata/{digest}{p.suffix}'
        dest=ROOT/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if p.resolve()!=dest.resolve():shutil.copyfile(p,dest)
        assert sha(dest)==digest
        records[str(p).replace('\\','/')]=dict(source=str(p),relative=relative,sha256=digest,bytes=p.stat().st_size)
    seal=read(HERE/'P_MODEL_SEAL.json');receipt=read(HERE/'P_PREDICTION_RECEIPT.json');tq=read(HERE/'P_TRAINING_QA.json')
    for run in seal['runs']:
        for key in ('complete','selected_predictions'):add(run[key])
        add(run['selected_checkpoint'],f'research/wisig_w5/artifacts/review_checkpoints/{run["job"]}.pt')
        for name in ('HISTORY.json','INITIAL.json'):
            p=Path(run['complete']['path']).parent/name;add(dict(path=str(p),sha256=sha(p)))
    for seed,ref in read(HERE/'DATA_PLAN.json')['A0'].items():
        add(ref['checkpoint'],f'research/wisig_w5/artifacts/review_checkpoints/A0_S{seed}.pt')
        for key in ('validation','initialization','sample_keys'):add(ref[key])
        # The copied source-selection file is byte-identical to its historical reference.
        add(ref['selection'],f'research/wisig_w5/artifacts/P_A0_S{seed}_selection.npz')
    for item in receipt['predictions'].values():add(item)
    for item in (tq['history_evidence'],tq['role_metadata']):add(item)
    # Historical W4 plan is metadata required by the unchanged-TX audit.
    q=ROOT/'research/wisig_w4/DATA_PLAN.json';add(dict(path=str(q),sha256=sha(q)))
    index=dict(status='EXACT_REPLAY_INPUTS_AND_SELECTED_WEIGHTS',raw_IQ_included=False,optimizer_states_included=False,
        selected_weight_files=9,files=list(records.values()),
        scope='Predictions/features/training histories/role metadata plus 6 selected new checkpoints and 3 reused A0 checkpoints; no raw training replay.')
    (HERE/'REPLAY_FILES.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Collected',len(records),'exact input aliases and 9 selected checkpoints')


def package():
    assert read(HERE/'INDEPENDENT_QA.json')['status']=='PASS'
    out=ROOT/'output/review';out.mkdir(parents=True,exist_ok=True);archive=out/'wisig_w5_review.zip'
    files={p for p in HERE.iterdir() if p.is_file() and p.name not in ('REVIEW_PACKAGE.json','PACKAGE_STDOUT.log')}
    for v in read(HERE/'REPLAY_FILES.json')['files']:files.add(ROOT/v['relative'])
    for name in ('ADMISSION_REGISTRATION.json','EXECUTION_REGISTRATION.json','P_EVALUATION_REGISTRATION.json'):
        files.update(ROOT/p for p in read(HERE/name)['files_sha256'])
    def build():
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    build();folder=out/('w5_portable_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));folder.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        assert all((folder/n).resolve().is_relative_to(folder.resolve()) for n in z.namelist());z.extractall(folder)
    result=subprocess.run([sys.executable,str(folder/'research/wisig_w5/audit_w5.py'),'--portable'],cwd=folder,capture_output=True,text=True,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    p=folder/'research/wisig_w5/PORTABLE_QA.json';assert read(p)['status']=='PASS'
    shutil.copyfile(p,HERE/'PORTABLE_QA.json');files.add(HERE/'PORTABLE_QA.json');build()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in files:assert hashlib.sha256(z.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
    report=dict(status='PASS_FRESH_DIRECTORY_PORTABLE_REPLAY',archive=str(archive),bytes=archive.stat().st_size,
        sha256=sha(archive),entries=len(files),portable_checks=read(HERE/'PORTABLE_QA.json')['checks_passed'],
        selected_checkpoints=9,raw_IQ_included=False,optimizer_states_included=False,
        reproduction_boundary='Prediction/feature-to-result, role/history/hash audit; not retraining raw IQ',
        command='python research/wisig_w5/audit_w5.py --portable')
    (HERE/'REVIEW_PACKAGE.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(result.stdout);print(json.dumps(report,indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('command',choices=('collect','package'));args=p.parse_args();globals()[args.command]()
