"""Synthetic CUDA acceptance, then freeze code/protocol without scientific IQ reads."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE))
import engine as E


def engineering():
    path=HERE/'ENGINEERING_QA.json'
    if path.exists():raise FileExistsError('Keep original synthetic acceptance')
    prior=sorted((HERE/'artifacts').glob('synthetic*')) if (HERE/'artifacts').exists() else []
    synthetic_root=HERE/'artifacts'/f'synthetic_attempt_{len(prior)+1:03d}'
    synthetic_root.mkdir(parents=True)
    E.write(synthetic_root/'ATTEMPT.json',dict(started_at=E.now(),scientific_IQ_reads=0,
        prior_attempt_directories=[p.relative_to(HERE).as_posix() for p in prior]))
    E.A.setup(24101)
    rng=np.random.default_rng(75001)
    def fixture(n):
        return dict(x=rng.normal(size=(n,2,256)).astype(np.float32),y=np.arange(n)%10,
            rx=np.arange(n)%4,day=np.ones(n,dtype=np.int64),keys=np.asarray([f'synthetic|{i}' for i in range(n)]),pool_ids=np.arange(n))
    train,val=fixture(40),fixture(20)
    job=dict(id='SYNTHETIC',epochs=2,batch=16,weight_decay=.001,order_seed=24201,stochastic_seed=24301,max_gradient_updates=6)
    outputs={};checks={};models={a:E.M.build(a,24101).cuda().eval() for a in E.M.ARMS}
    x=torch.as_tensor(val['x'],device='cuda')
    with torch.no_grad():
        logits={a:m(x) for a,m in models.items()}
        for a in E.M.ARMS:checks['tokens_'+a]=list(models[a].tokens(x).shape)==[20,32,64]
        checks['P1_P2_initial_prediction_exact']=torch.equal(logits['P1'],logits['P2'])
        checks['P2_N19_initial_prediction_exact']=torch.equal(logits['P2'],logits['N19'])
    del models,logits,x
    for arm in E.M.ARMS:
        out=synthetic_root/arm
        if out.exists():raise FileExistsError('Synthetic output exists')
        outputs[arm]=E.fit_loop(dict(job,arm=arm),train,val,E.M.build(arm,24101).cuda(),torch.device('cuda'),out,['rx0','rx1','rx2','rx3'])
    checks['all_endpoint_replays']=all(all(r['val_inference_replay'].values()) for r in outputs.values())
    histories={a:E.read(synthetic_root/a/'HISTORY.json') for a in E.M.ARMS}
    checks['same_sample_orders']=all(len({histories[a][k]['canonical_order_sha256'] for a in E.M.ARMS})==1 for k in range(2))
    checks['all_24_synthetic_updates']=sum(r['updates'] for r in outputs.values())==24
    if not all(checks.values()):raise ValueError('Synthetic GPU acceptance failed')
    value=dict(status='PASS',checks=checks,synthetic_gradient_trajectories=4,synthetic_gradient_updates=24,scientific_IQ_reads=0,
        synthetic_output_directory=synthetic_root.relative_to(HERE).as_posix(),
        prior_engineering_attempt_directories=[p.relative_to(HERE).as_posix() for p in prior],
        prior_engineering_failure='First synthetic fixture used NumPy Windows int32 labels; corrected explicit torch.int64 label conversion before any scientific input read.' if prior else None,
        parameters={a:r['active_parameters'] for a,r in outputs.items()},synthetic_outputs=outputs,
        environment=dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability())))
    E.write(path,value)
    print({k:value[k] for k in ('status','parameters','synthetic_gradient_updates')})


def register():
    if (HERE/'REGISTRATION.json').exists():raise FileExistsError('Registration exists')
    qa=E.read(HERE/'ENGINEERING_QA.json')
    if qa['status']!='PASS' or not all(qa['checks'].values()):raise ValueError('Acceptance required')
    parent=ROOT/'research/standard_method_competition_c'
    original=E.read(parent/'PRE_REGISTRATION.json')
    reg={k:original[k] for k in ('source_input_whitelist','fit_rx','rx_names','source_cell_sha256',
                                 'forbidden_historical_held_rx','forbidden_target_rx','allowed_days')}
    E.write(HERE/'SOURCE_PLAN.json',E.read(parent/'SOURCE_PLAN.json'))
    jobs=[]
    for view in ('V0','V1'):
        for stream in range(3):
            for arm in E.M.ARMS:
                jobs.append(dict(id=f'{view}_S{stream}_{arm}',view=view,stream=f'S{stream}',arm=arm,
                    initialization_seed=24101+stream,order_seed=24201+stream,stochastic_seed=24301,
                    epochs=40,batch=128,weight_decay=.001,updates_per_epoch=127,max_gradient_updates=5080))
    E.write(HERE/'JOBS.json',jobs)
    files=set(HERE.glob('*.py'))|{HERE/'PREREGISTRATION.zh-CN.md',ROOT/'tests/test_periodic_span.py'}
    for directory in ('lib/xa','lib/legacy_rfhstu','13-mixer-cfo-mechanisms','17-rx-distortion'):
        files.update((ROOT/directory).glob('*.py'))
    for relative in ('research/controlled_training_a/runner.py','research/controlled_training_a/analyze.py',
                     'research/standard_method_competition_c/engine.py','research/standard_method_competition_c/methods.py',
                     'research/standard_method_competition_c/analyze.py'):
        files.add(ROOT/relative)
    reg.update(schema_version=1,stage='PERIODIC_SPAN',status='REGISTERED_FOR_24_ARCHITECTURE_FITS',registered_at=E.now(),
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        authorization='User requests independent onset-free periodic-span source competition and explicitly says continue; fixed-window STOP remains.',
        primary_endpoint='selected',auxiliary_endpoints=['epoch10','final40'],maximum_scientific_fits=24,
        scientific_fits_executed_at_registration=0,parameters=qa['parameters'],environment=qa['environment'],
        epochs=40,batch=128,weight_decay=.001,weight_decay_grid=False,streams=3,source_views=2,
        bootstrap_replicates=10000,bootstrap_seed=24501,checkpoint_selection='RX-equal checkpoint_val CE strict minimum, earliest tie',
        total_scientific_gradient_epochs=960,total_scientific_gradient_updates=121920,
        code_sha256={p.relative_to(ROOT).as_posix():E.sha(p) for p in sorted(files)},
        artifacts_sha256={name:E.sha(HERE/name) for name in ('SOURCE_PLAN.json','JOBS.json','ENGINEERING_QA.json')},
        prior_fixed_window_result_sha256=E.sha(ROOT/'research/periodic_hstem/AUDIT_RESULT.json'),
        parent_C_registration_sha256=E.sha(parent/'PRE_REGISTRATION.json'),
        negative_control='N19 is fixed before training and never selected as a tuned candidate',
        primary_contrast='P2-P1',secondary_comparisons=['P2-P0','P2-N19','P1-P0','N19-P0'],
        selection_role_requires_committed_all24_seal=True,CPU_acceptance_tests=11,no_retries_or_additional_seeds=True)
    # Do not carry C's own execution/artifact registration into the new stage.
    for field in ('artifact_sha256','evidence_sha256','scope_this_turn','ema_decay','group_dro_step_size','ema_shadow_readouts','total_arm_candidates'):
        reg.pop(field,None)
    E.write(HERE/'REGISTRATION.json',reg)
    print(dict(status=reg['status'],scientific_fits=0,jobs=len(jobs),parameters=reg['parameters']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['engineering','register']);a=p.parse_args()
    {'engineering':engineering,'register':register}[a.command]()
