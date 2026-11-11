"""Reproduce the pre-execution synthetic CUDA acceptance; no scientific inputs."""
import importlib.util
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent


def data(n,seed):
    return dict(x=np.random.default_rng(seed).normal(size=(n,2,256)).astype(np.float32),
         y=np.arange(n,dtype=np.int64)%10,rx=np.arange(n,dtype=np.int64)%12,day=np.ones(n,dtype=np.int64),
         keys=np.asarray([f'synthetic-{seed}-{i}' for i in range(n)]),pool_ids=np.arange(n,dtype=np.int64))


def main():
    spec=importlib.util.spec_from_file_location('C_gpu_acceptance',HERE/'engine.py')
    E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
    root=HERE/'artifacts/engineering_cuda'
    if root.exists():raise FileExistsError('Keep original engineering attempt')
    root.mkdir(parents=True);results={}
    for name,family in (('ERM','ERM_EMA'),('REPLAY','ERM_EMA'),('DRO','RX_GDRO')):
        E.A.setup(24101);model=E.A.build_model(24101)
        job=dict(id='GPU_SYNTH_'+name,family=family,readouts=['C0','C1'] if family=='ERM_EMA' else ['C2'],
             batch=128,epochs=40,weight_decay=.001,stochastic_seed=24301,order_seed=24201,max_gradient_updates=80)
        results[name]=E.fit_loop(job,data(173,150),data(128,151),model,torch.device('cuda'),root/name,[f'synthetic-rx-{i}' for i in range(12)])
        del model
    histories={name:E.read(root/name/'HISTORY.json') for name in results}
    checks=dict(full_student_and_EMA_cuda_replay=histories['ERM']==histories['REPLAY'],
         same_initialization_across_families=E.read(root/'ERM/INITIAL.json')==E.read(root/'DRO/INITIAL.json'),
         same_relative_orders=all(a['relative_order_sha256']==b['relative_order_sha256'] for a,b in zip(histories['ERM'],histories['DRO'])),
         fixed_updates=all(r['updates']==80 for r in results.values()),
         all_readout_inference_replays=all(v['exact'] for r in results.values() for v in r['val_inference_replay'].values()),
         EMA_calibration_preserves_rng_and_student=all(x['calibration_preserved_training_state'] for h in histories.values() for x in h))
    if not all(checks.values()):raise RuntimeError('Synthetic CUDA acceptance failed')
    E.write(HERE/'ENGINEERING_GPU_QA.json',dict(status='PASS',checks=checks,synthetic_only=True,
         scientific_IQ_or_targets_read=0,scientific_fits=0,synthetic_backbone_trajectories=3,synthetic_updates_total=240,
         parameters=results['ERM']['active_parameters'],device=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
         cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version()))


if __name__=='__main__':main()
