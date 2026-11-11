"""Full40-epoch CUDA loop on synthetic random-backbone fixtures, never source IQ."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

import common as H
import kernel as K

HERE=H.HERE


def main():
    if (HERE/'CUDA_ACCEPTANCE.json').exists():raise FileExistsError('Keep original CUDA acceptance evidence')
    cpu=H.read(HERE/'CPU_ACCEPTANCE.json')
    if cpu['status']!='PASS':raise ValueError('CPU engineering acceptance required')
    recovery=H.read(HERE/'REFERENCE_RECOVERY.json')
    if recovery['status']!='PASS_TWELVE_CHECKPOINTS_RESTORED_SYNTHETIC_CUDA_PARITY':raise ValueError('All12 source references must be recovered first')
    if (HERE/'EXECUTION_AUTHORIZATION.json').exists():raise ValueError('This command is engineering acceptance before scientific authorization')
    H.A.setup(26101)
    prior=sorted((HERE/'artifacts').glob('synthetic_attempt_*'))
    out=HERE/'artifacts'/f'synthetic_attempt_{len(prior)+1:03d}'
    out.mkdir(parents=True)
    H.write(out/'ATTEMPT.json',dict(started_at=H.now(),source_IQ_reads=0,scientific_fits=0,synthetic_only=True))
    rng=np.random.default_rng(27602)
    def fixture(n):
        return dict(x=rng.normal(size=(n,2,256)).astype(np.float32),y=(np.arange(n)%10).astype(np.int64),
            rx=(np.arange(n)%2).astype(np.int64),day=np.ones(n,dtype=np.int64),
            keys=np.asarray([f'synthetic|{i}' for i in range(n)]),pool_ids=np.arange(n,dtype=np.int64))
    train,val=fixture(24),fixture(20)
    job=dict(id='CUDA_SYNTHETIC',epochs=40,batch=8,lr=.001,weight_decay=.001,
        stochastic_seed=26301,order_seed=24201,max_gradient_updates=120)
    outputs={};histories={};initials={};checks={};start=time.perf_counter()
    for arm in ('F1','F2'):
        # Randomly initialized frozen backbone, not a recovered scientific model.
        model=H.M.build(arm,24101,26101).cuda()
        outputs[arm]=K.fit_loop(dict(job,arm=arm),train,val,model,torch.device('cuda'),out/arm,['synthetic0','synthetic1'])
        H.write(out/arm/'COMPLETE.json',outputs[arm])
        histories[arm]=H.read(out/arm/'HISTORY.json');initials[arm]=H.read(out/arm/'INITIAL.json')
        with np.load(out/arm/'GRADIENT_TRACE.npz',allow_pickle=False) as f:
            values=f['values'];columns=list(f['columns'])
            checks[arm+'/all120_updates_logged']=values.shape==(120,len(K.TRACE_COLUMNS))
            checks[arm+'/finite_every_step']=bool(np.isfinite(values).all()) and bool((values[:,columns.index('all_task_gradients_finite')]==1).all())
            checks[arm+'/task_side_gradient_starts_zero']=float(values[0,columns.index('task_side_gradient_l2')])==0.
            checks[arm+'/alpha_gradient_starts_live']=float(values[0,columns.index('task_alpha_gradient')])!=0.
            checks[arm+'/later_task_side_gradient_nonzero']=bool((values[1:,columns.index('task_side_gradient_l2')]>0).any())
            checks[arm+'/Adam_decay_is_separate_from_task_gradient']=float(values[0,columns.index('side_parameter_delta_l2')])>0.
            checks[arm+'/no_backbone_gradient_every_step']=bool((values[:,columns.index('backbone_gradients_absent')]==1).all())
        checks[arm+'/frozen_parameters_and_BN_every_epoch']=all(r['backbone_and_BN_unchanged'] for r in histories[arm])
        checks[arm+'/alpha0_exact_rollback_every_epoch']=all(r['alpha0_rollback_bitwise'] for r in histories[arm])
        checks[arm+'/all_endpoint_prediction_replays']=all(outputs[arm]['validation_endpoint_replay'].values())
        scores=[initials[arm]['epoch0_val_rx_ce']]+[r['val_rx_ce'] for r in histories[arm]]
        checks[arm+'/correct_epoch0_to40_checkpoint_choice']=outputs[arm]['best_epoch']==min(range(41),key=lambda i:scores[i])
        checks[arm+'/optimizer_only3201']=initials[arm]['optimizer_parameter_elements']==3201
        del model;torch.cuda.empty_cache()
    for field in ('frozen_backbone_sha256','common_residual_sha256','rng_sha256','scheduler_sha256'):
        checks['paired_initial/'+field]=initials['F1'][field]==initials['F2'][field]
    for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
        checks['paired40epoch/'+field]=all(histories['F1'][k][field]==histories['F2'][k][field] for k in range(40))
    if not all(checks.values()):raise ValueError('Full-loop CUDA synthetic acceptance failed')
    value=dict(status='PASS_FULL40_EPOCH_CUDA_SYNTHETIC_EXECUTOR',checks=checks,check_count=len(checks),
        synthetic_trajectories=2,synthetic_optimizer_updates=240,synthetic_gradient_epochs=80,
        backbone_source='random construction only; recovered source P0/EMA were used only in separate forward parity checks',
        outputs=outputs,synthetic_artifact_directory=out.relative_to(HERE).as_posix(),prior_attempts=[p.relative_to(HERE).as_posix() for p in prior],
        wall_seconds=time.perf_counter()-start,scientific_fits=0,source_IQ_reads=0,source_prediction_decodes=0,
        environment=dict(torch=str(torch.__version__),numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),FP32=True,TF32=False,deterministic=True),time=H.now())
    H.write(HERE/'CUDA_ACCEPTANCE.json',value)
    print(dict(status=value['status'],checks=len(checks),scientific_fits=0,synthetic_updates=240))


if __name__=='__main__':main()
