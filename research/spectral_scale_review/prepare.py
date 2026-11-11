"""Freeze a design and future job table. Provides no training entry point."""
from __future__ import annotations
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    evidence=read(HERE/'EVIDENCE.json');qa=read(HERE/'SYNTHETIC_ACCEPTANCE.json')
    if evidence['status']!='REVIEW_COMPLETE_NO_TRAINING' or qa['status']!='PASS' or qa['tests']!=12:raise ValueError('Review and synthetic acceptance required')
    spec=importlib.util.spec_from_file_location('registered_spectral_design',HERE/'models.py')
    M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
    counts={}
    for arm in M.ARMS:
        model=M.build(arm,24101,26101)
        counts[arm]=dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    if counts!={'F0':dict(total=82426,trainable=0),'F1':dict(total=85627,trainable=3201),'F2':dict(total=85627,trainable=3201)}:raise ValueError('Strict parameter equality failed')
    previous=ROOT/'research/periodic_span'
    reg=read(previous/'REGISTRATION.json');seal=read(previous/'TRAINING_SEAL.json')
    write_new(HERE/'FUTURE_SOURCE_PLAN.json',read(previous/'SOURCE_PLAN.json'))
    refs={};jobs=[]
    for view in ('V0','V1'):
        for stream in range(3):
            name=f'{view}_S{stream}_P0';key=f'{view}/S{stream}'
            refs[key]=dict(job=name,path=str(previous/'artifacts'/name/'selected.pt'),
                sha256=seal['runs'][name]['checkpoints']['selected'],role='Frozen P0 source-selected by checkpoint_val CE; no file opened this turn')
            for arm in ('F1','F2'):
                jobs.append(dict(id=f'{view}_S{stream}_{arm}',view=view,stream=f'S{stream}',arm=arm,
                    frozen_reference=key,side_initialization_seed=26101+stream,order_seed=24201+stream,stochastic_seed=26301,
                    epochs=40,batch=128,lr=.001,weight_decay=.001,updates_per_epoch=127,max_gradient_updates=5080,
                    checkpoint_candidate_epochs=list(range(41)),auxiliary_epochs=[10,40],
                    selected='minimum RX-equal checkpoint_val CE over epochs0-40; earliest tie',
                    trainable_parameters=3201,backbone_and_BN_frozen=True))
    write_new(HERE/'FROZEN_REFERENCES.json',refs)
    write_new(HERE/'PROPOSED_JOBS.json',jobs)
    C=ROOT/'research/standard_method_competition_c'
    ema=read(C/'EVALUATION_MANIFEST.json')
    members={n:h for n,h in ema['files'].items() if 'wd1e3' in n and n.endswith('/C1_selected_selection.npz')}
    if len(members)!=6:raise ValueError('Exactly six frozen EMA readouts required')
    write_new(HERE/'EMA_REFERENCE.json',dict(role='E0 auxiliary frozen strong reference; not another trainable candidate',
        original_manifest_path='research/standard_method_competition_c/EVALUATION_MANIFEST.json',
        original_manifest_sha256=sha(C/'EVALUATION_MANIFEST.json'),members=members,
        decoded_this_turn=False,source_weighting='same12RX/3fixed streams; CE-selected C1 wd=.001, fit-only calibrated BN'))
    files=[HERE/n for n in ('audit.py','models.py','prepare.py','PREREGISTRATION.zh-CN.md')]
    files.append(ROOT/'tests/test_spectral_scale_review.py')
    # Freeze current backbone dependencies already audited during the previous run.
    inherited={n:h for n,h in reg['code_sha256'].items() if n.startswith(('lib/','13-mixer-cfo-mechanisms/','17-rx-distortion/'))}
    inherited['09-phase-invariance/phase_features.py']=sha(ROOT/'09-phase-invariance/phase_features.py')
    for name,digest in inherited.items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen P0 source changed')
    inherited.update({p.relative_to(ROOT).as_posix():sha(p) for p in files})
    evidence_files=('REVIEW_SCOPE.json','EVIDENCE.json','SYNTHETIC_ACCEPTANCE.json','ENGINEERING_NOTES.json',
        'FUTURE_SOURCE_PLAN.json','FROZEN_REFERENCES.json','EMA_REFERENCE.json','PROPOSED_JOBS.json')
    value=dict(status='DESIGN_PREREGISTERED_EXECUTION_CLOSED',registered_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        current_turn=dict(scientific_fits_authorized=0,scientific_fits_executed=0,IQ_reads=0,checkpoint_reads=0,feature_cache_reads=0,optimizer_steps=0),
        unique_candidate='F2; frequency multiscale token residual on a fully frozen P0',controls=['F0 frozen P0','F1 ordinary same-parameter/same-maximum-span frequency residual','E0 frozen original EMA'],
        parameters=counts,proposed_future_fits=12,proposed_future_epochs=480,proposed_future_updates=60960,
        future_execution_requires_separate_user_authorization_and_committed_engineering_addendum=True,
        all_previous_STOPs_unchanged=True,source_input_whitelist=reg['source_input_whitelist'],
        primary_contrast='F2-F1',necessary_additional_comparisons=['F2-F0','F2-E0'],
        gates=dict(minimum_accuracy_delta=.01,conditional_RX_interval_lower_above=0,both_views_positive=True,
            minimum_positive_streams=2,minimum_paired_RX_accuracy_delta=-.02,maximum_supported_TX_RX_risk_increase=.05,support_per_stream=50),
        bootstrap=dict(unit='12 observed source-selection RX after averaging3 fixed streams',replicates=10000,seed=24501),
        scales_fixed_before_any_new_performance_output=[3,33],ordinary_control_kernel=33,
        token_centers='3.5+8*j',nominal_frequency_bin_spans=dict(P0=32,F2_fine=20,F2_wide=50,F1_wide=50),
        frozen_backbone_rollback_parity=dict(frozen_F0_bitwise=True,sealed_P0_max_abs_logit=1e-6,sealed_P0_argmax_identical=True),
        code_sha256=inherited,artifact_sha256={name:sha(HERE/name) for name in evidence_files},
        oracle_scope='87.0252% is only a label-oracle between fixed P0/P2; it does not bound F2 or justify its scales.')
    write_new(HERE/'REGISTRATION.json',value)
    print(json.dumps(dict(status=value['status'],current_scientific_fits=0,proposed_future_fits=12,parameters=counts)))


if __name__=='__main__':main()
