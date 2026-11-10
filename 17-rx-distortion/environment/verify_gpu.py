"""Synthetic runtime check only. No IQ data, model selection or experiments."""
import json,sys,time
print('PYTHON_STARTED',sys.executable,flush=True)
start=time.perf_counter()
import torch
print('TORCH_IMPORTED',torch.__version__,flush=True)
available=torch.cuda.is_available();print('CUDA_AVAILABLE',available,flush=True)
if not available:raise RuntimeError('CUDA must be available before scientific GPU work')
x=torch.randn(16,32,device='cuda',requires_grad=True);head=torch.nn.Linear(32,10).cuda()
loss=torch.nn.functional.cross_entropy(head(x),torch.arange(16,device='cuda')%10);loss.backward();torch.cuda.synchronize()
if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(x.grad).all()):raise RuntimeError('Nonfinite GPU forward/backward')
print(json.dumps(dict(status='PASS',python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0),gpu_forward_backward=True,seconds=time.perf_counter()-start,scientific_data_access=False)),flush=True)
