"""Freeze engineering checkpoint inputs from seals; never opens a checkpoint or IQ."""
from pathlib import Path
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
DESIGN=ROOT/'research/spectral_scale_review'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def new(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    P=ROOT/'research/periodic_span';C=ROOT/'research/standard_method_competition_c'
    originals=read(DESIGN/'FROZEN_REFERENCES.json');p_seal=read(P/'TRAINING_SEAL.json')
    p_evidence=read(P/'RUN_EVIDENCE.json');p_eval=read(P/'EVALUATION_MANIFEST.json')
    c_seal=read(C/'STAGE_C_SEAL.json');c_evidence=read(C/'RUN_EVIDENCE.json');c_eval=read(C/'EVALUATION_MANIFEST.json')
    original_C=Path('D:/crossatten/worktrees/training-instance-attribution/research/standard_method_competition_c')
    records={}
    for view in ('V0','V1'):
        for s in range(3):
            key=f'{view}/S{s}';old=originals[key];pjob=old['job'];cjob=f'{view}_S{s}_wd1e3_ERM_EMA'
            if p_seal['runs'][pjob]['checkpoints']['selected']!=old['sha256']:raise ValueError('P0 reference differs from original design')
            pdir=P/'artifacts'/pjob;cdir=original_C/'artifacts'/cjob
            records[key]={
                'F0':dict(origin=old['path'],sha256=old['sha256'],copy_relative=f'artifacts/references/F0_{view}_S{s}.pt',
                    checkpoint_val_prediction=dict(path=str(pdir/'selected_val.npz'),sha256=p_evidence[pjob]['COMPLETE']['files_sha256']['selected_val.npz']),
                    source_selection_prediction=dict(path=str(pdir/'selected_selection.npz'),sha256=p_eval['files'][pjob+'/selected_selection.npz'])),
                'E0':dict(origin=str(cdir/'C1_selected.pt'),sha256=c_seal['runs'][cjob]['checkpoints']['C1_selected.pt'],copy_relative=f'artifacts/references/E0_{view}_S{s}.pt',
                    checkpoint_val_prediction=dict(path=str(cdir/'C1_selected_val.npz'),sha256=c_evidence[cjob]['COMPLETE']['files_sha256']['C1_selected_val.npz']),
                    source_selection_prediction=dict(path=str(cdir/'C1_selected_selection.npz'),sha256=c_eval['files'][cjob+'/C1_selected_selection.npz']))}
    new(HERE/'REFERENCE_INPUTS.json',dict(status='TWELVE_SOURCE_REFERENCES_PINNED_FROM_ORIGINAL_SEALS',references=records,
        checkpoints_opened_during_registration=0,source_array_reads=0,
        original_seal_sha256=dict(P0=sha(P/'TRAINING_SEAL.json'),EMA=sha(C/'STAGE_C_SEAL.json'))))
    value=dict(status='ENGINEERING_ONLY_NO_SCIENTIFIC_EXECUTION_AUTHORIZATION',
        design_commit='21212557782a96382446be0181c648bf7b5678b8',design_sha256=sha(DESIGN/'REGISTRATION.json'),
        reference_registry_sha256=sha(HERE/'REFERENCE_INPUTS.json'),
        allowed_now=dict(source_checkpoint_files=12,restore_checkpoint_copies=True,synthetic_inputs=True,synthetic_optimizer_steps=True),
        forbidden_now=dict(source_IQ=True,source_metadata_arrays=True,source_prediction_decoding=True,scientific_gradient_fits=True,target_or_new_RX_access=True),
        user_scope='First complete executor engineering and CUDA acceptance; obtain explicit execution authorization before the original12 scientific fits.',
        future_scientific_fits=12,scientific_fits_authorized=0,
        code_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (HERE/'common.py',HERE/'prepare_scope.py',HERE/'recover.py')},
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip())
    new(HERE/'ENGINEERING_SCOPE.json',value)
    print(json.dumps(dict(status=value['status'],checkpoint_restore_files=12,scientific_fits_authorized=0)))


if __name__=='__main__':main()
