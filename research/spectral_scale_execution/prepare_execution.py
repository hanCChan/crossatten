"""Freeze accepted engineering bytes/evidence. Never grants science authorization."""
from pathlib import Path
import hashlib
import json
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
DESIGN=ROOT/'research/spectral_scale_review'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    if (HERE/'EXECUTION_REGISTRATION.json').exists():raise FileExistsError('Keep frozen engineering registration')
    if (HERE/'EXECUTION_AUTHORIZATION.json').exists():raise ValueError('Engineering registration never supplies scientific authorization')
    qa=read(HERE/'ENGINEERING_QA.json');cuda=read(HERE/'CUDA_ACCEPTANCE.json');cpu=read(HERE/'CPU_ACCEPTANCE.json')
    if qa['status']!='PASS' or not all(qa['checks'].values()) or cpu['status']!='PASS' or not all(cuda['checks'].values()):raise ValueError('Engineering acceptance is incomplete')
    new(HERE/'AUTHORIZATION_STATUS.json',dict(status='NO_SCIENTIFIC_EXECUTION_AUTHORIZATION',scientific_fits_authorized=0,
        scientific_fits_started=0,source_IQ_reads=0,source_prediction_decodes=0,
        user_instruction='下一步先补齐执行器工程证据，得到明确执行授权后完成原定 12 次侧路拟合。',
        pending='Explicit user authorization for source checkpoint_val parity preflight and the unchanged12-fit protocol',
        source_checkpoint_val_parity='NOT_STARTED',source_selection_scoring='NOT_STARTED'))
    code=dict(read(DESIGN/'REGISTRATION.json')['code_sha256'])
    dependencies=[
        'research/periodic_span/engine.py','research/periodic_span/models.py','research/periodic_span/REGISTRATION.json',
        'research/periodic_span/TRAINING_SEAL.json','research/periodic_span/DECISION.json',
        'research/periodic_hstem/AUDIT_RESULT.json','research/spectral_scale_review/REGISTRATION.json',
        'research/standard_method_competition_c/engine.py','research/standard_method_competition_c/methods.py',
        'research/standard_method_competition_c/analyze.py','research/controlled_training_a/runner.py',
        'research/controlled_training_a/analyze.py','tests/test_spectral_scale_execution.py']
    dependencies += [p.relative_to(ROOT).as_posix() for p in sorted(HERE.glob('*.py'))]
    for name in dependencies:code[name]=sha(ROOT/name)
    for name,digest in code.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen design bytes changed before execution registration')
    with zipfile.ZipFile(HERE/'EXECUTION_CODE_BYTES.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(code):z.write(ROOT/name,name)
    synthetic=HERE/cuda['synthetic_artifact_directory'];members={}
    with zipfile.ZipFile(HERE/'SYNTHETIC_TRACES.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(synthetic.rglob('*')):
            if not path.is_file() or path.suffix=='.pt':continue
            name=path.relative_to(synthetic).as_posix();z.write(path,name)
            members[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    new(HERE/'SYNTHETIC_ARCHIVE.json',dict(archive_sha256=sha(HERE/'SYNTHETIC_TRACES.zip'),files=members,
        synthetic_updates=240,synthetic_epochs=80,source_checkpoints_included=False,source_IQ_included=False,
        full_synthetic_states='Local ignored artifacts only; see LOCAL_SYNTHETIC_MANIFEST.json'))
    names=['ENGINEERING_SCOPE.json','REFERENCE_INPUTS.json','REFERENCE_RECOVERY.json','ENGINEERING_ACCESS_LOG.jsonl',
        'CPU_ACCEPTANCE.json','CUDA_ACCEPTANCE.json','ENGINEERING_QA.json','LOCAL_SYNTHETIC_MANIFEST.json',
        'AUTHORIZATION_STATUS.json','EXECUTION_ADDENDUM.zh-CN.md','README.zh-CN.md',
        'EXECUTION_CODE_BYTES.zip','SYNTHETIC_TRACES.zip','SYNTHETIC_ARCHIVE.json']
    value=dict(status='ENGINEERING_ACCEPTED_SCIENTIFIC_EXECUTION_CLOSED',design_commit='21212557782a96382446be0181c648bf7b5678b8',
        design_sha256=sha(DESIGN/'REGISTRATION.json'),code_sha256=code,artifact_sha256={name:sha(HERE/name) for name in names},
        environment=cuda['environment'],engineering=dict(source_checkpoints_restored=12,CPU_tests=8,CUDA_synthetic_updates=240,
            CUDA_synthetic_epochs=80,CPU_synthetic_updates=12,QA_checks=qa['check_count']),
        scientific_execution=dict(current_fits_authorized=0,current_fits_started=0,current_source_IQ_reads=0,
            future_fits=12,future_epochs=480,future_updates=60960,actual_source_checkpoint_val_parity='PENDING_EXPLICIT_AUTHORIZATION',
            all12_committed_seal_before_source_selection=True,selected_only_decision=True),
        authorization_file_not_created=True)
    new(HERE/'EXECUTION_REGISTRATION.json',value)
    public={}
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.name!='ARTIFACT_MANIFEST.json' and path.suffix!='.log':
            public[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path))
    new(HERE/'ARTIFACT_MANIFEST.json',dict(status=value['status'],scientific_fits=0,source_IQ_reads=0,files=public))
    print(json.dumps(dict(status=value['status'],code_files=len(code),published_files=len(public),science_authorized=0)))


if __name__=='__main__':main()
