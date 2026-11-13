"""Seal identities, numerical tests and code before admitting any official IQ."""
import subprocess
from pathlib import Path
import run_b as b


def main():
    if (b.HERE/'B_FEATURE_ATTEMPT.json').exists():raise RuntimeError('Cannot preregister after official feature readout')
    for test,count,name in (('test_selective_pickle.py',4,'index'),('test_transport.py',5,'transport')):
        p=subprocess.run(['D:/crossatten/.venv/Scripts/python.exe',str(b.HERE/test)],capture_output=True,text=True)
        if p.returncode or f'Ran {count} tests' not in p.stderr:raise RuntimeError(p.stdout+p.stderr)
    b.write(b.HERE/'B_SYNTHETIC_ACCEPTANCE.json',dict(status='PASS',index_tests=4,transport_tests=5,
        official_IQ_rows_decoded=0,feature_preflight=b.read(b.HERE/'B_FEATURE_PREFLIGHT.json')['status']))
    manifest=b.read(b.HERE/'B_SOURCE_MANIFEST.json');b.write(b.HERE/'B_ROLES.json',b.roles(manifest))
    files=list(b.HERE.glob('*.py'))
    for folder in ('lib','13-mixer-cfo-mechanisms','17-rx-distortion','09-phase-invariance'):
        files+=list((b.ROOT/folder).rglob('*.py'))
    files+=[b.ROOT/'research/controlled_training_a/runner.py',b.ROOT/'research/standard_method_competition_c/engine.py',
        b.ROOT/'research/standard_method_competition_c/methods.py']
    for name in ('ACCESS_SCOPE.json','A_METADATA_RESULT.json','A_SELECTIVE_INDEX_QA.json','A_INDEX_SYNTHETIC_ACCEPTANCE.json',
                 'B_SOURCE_MANIFEST.json','B_ROLES.json','B_FEATURE_PREFLIGHT.json','B_SYNTHETIC_ACCEPTANCE.json','PREREGISTRATION_B.zh-CN.md'):
        files.append(b.HERE/name)
    code={p.relative_to(b.ROOT).as_posix():b.sha(p) for p in sorted(set(files))}
    b.write(b.HERE/'B_EXECUTION_REGISTRATION.json',dict(status='REGISTERED_BEFORE_OFFICIAL_IQ_DECODE',registered=b.now(),
        maximum_new_neural_fits=0,W3_C_authorized=False,checkpoints=b.CHECKPOINTS,files_sha256=code,
        cohort=dict(TX=96,source_RX=6,source_days=[1,2],raw_packets_per_cell=32,total_packets=36864),
        methods=list(b.METHODS),candidate='rank8',threshold_relative_error_improvement=.15,
        audit_note='Must be tracked and unchanged relative to HEAD before extract; all other gates are bound in the Chinese preregistration.'))
    print('Sealed',len(code),'code/evidence files; commit before extract')


if __name__=='__main__':main()
