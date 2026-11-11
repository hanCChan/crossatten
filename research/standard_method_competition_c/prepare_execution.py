"""Freeze execution-only addendum; no scientific arrays/weights are opened."""
import hashlib
import json
from pathlib import Path
import sys
import torch
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=HERE/'EXECUTION_REGISTRATION.json'
    if path.exists():raise FileExistsError('Keep original execution registration')
    gpu=json.loads((HERE/'ENGINEERING_GPU_QA.json').read_text(encoding='utf-8'))
    if gpu['status']!='PASS' or not all(gpu['checks'].values()):raise ValueError('Synthetic CUDA acceptance failed')
    reg=json.loads((HERE/'PRE_REGISTRATION.json').read_text(encoding='utf-8'))
    for relative,digest in reg['code_sha256'].items():
        if sha(ROOT/relative)!=digest:raise ValueError('Scientific preregistration code changed')
    files=[HERE/n for n in ('engine.py','analyze.py','prepare_execution.py','engineering_gpu.py','EXECUTION_ADDENDUM.zh-CN.md')]
    files.extend(ROOT/'tests'/n for n in ('test_standard_method_competition_c_engine.py','test_standard_method_competition_c_analysis.py'))
    environment=dict(torch_version=str(torch.__version__),numpy_version=np.__version__,python_version=sys.version,
         cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),device_name=torch.cuda.get_device_name(),
         device_capability=list(torch.cuda.get_device_capability()))
    value=dict(status='REGISTERED_FOR_FROZEN_24_JOB_EXECUTION',registration_sha256=sha(HERE/'PRE_REGISTRATION.json'),
         scientific_protocol_commit='bebf6bb',authorization='Current user task explicitly requests execution after synthetic acceptance and committed addendum.',
         prior_registration_only_scope_superseded_for_execution=True,maximum_scientific_gradient_attempts=24,
         methods_grid_seeds_views_selection_budget_unchanged=True,code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
         environment=environment,synthetic_CPU_tests_passed=34,synthetic_GPU_QA_sha256=sha(HERE/'ENGINEERING_GPU_QA.json'),
         synthetic_GPU_trajectories=3,synthetic_GPU_updates=240,scientific_IQ_or_target_reads_before_acceptance=0,
         source_preflight_required_before_training=True,all24_readouts_commit_seal_required_before_source_selection=True,
         historical_held6_access=False,target8_or_dates3_4_access=False,new_receiver_confirmation_access=False,
         no_retry_or_additional_seeds=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(status=value['status'],torch=environment['torch_version'],synthetic_CPU_tests=34,scientific_fits=0)))


if __name__=='__main__':main()
