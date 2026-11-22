"""Data-free structural/resource gate and exact epoch-state recovery check."""
import gc
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from s26_common import HERE,P,write,now
from s26_models import build,geometry,shared_hash
from s26_data import authorize
from s26_training import setup,save_resume,restore,learning_rate
from xa.frontends import awgn


def main():
    setup(26026);audits={}
    for arm in ['H0','P0','LP']:
        model=build(arm,26026).eval()
        with torch.no_grad():
            assert model.net.embedder.cnn_stem(torch.randn(2,5,8192)).shape==(2,32,64)
            with FlopCounterMode(display=False) as counter:
                assert model(torch.randn(2,2,8192)).shape==(2,10)
        audits[arm]=geometry(model);audits[arm]['partial_macs_per_packet']=counter.get_total_flops()/4
        assert audits[arm]['partial_macs_per_packet']>0
    assert audits['LP']['params']==138346 and audits['LP']['cnn_receptive_field']==350
    assert audits['LP']['tokens']==32 and audits['LP']['token_stride']==256 and audits['LP']['first_token_center']==127.5
    assert audits['LP']['bn_layers']==2 and audits['H0']['bn_layers']==5
    for seed in [2808879044,2602222937,1569434962,3016222868,2242899550,1787476561]:
        assert shared_hash(build('LP',seed))==shared_hash(build('P0',seed))==shared_hash(build('H0',seed))
    denied=0
    for rx,fold,role in [(P['confirm_receivers'][0],0,'source'),(P['blind_receivers_never_open'][0],0,'source'),
                         (P['folds'][0]['dev_val'][0],0,'source'),(P['folds'][0]['dev_val'][0],0,'dev_report_only')]:
        try:authorize(rx,fold,role)
        except PermissionError:denied+=1
    assert denied==4
    model=build('LP',26026).cuda().train();x=torch.randn(8,2,8192,device='cuda');y=torch.arange(8,device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
    order=torch.Generator(device='cuda').manual_seed(26026);noise=torch.Generator(device='cuda').manual_seed(26027)
    def update(m,opt,og,ng,step):
        ids=torch.randperm(8,generator=og,device='cuda');xx=awgn(x[ids],0,50,ng)
        for group in opt.param_groups:group['lr']=learning_rate(step,125)
        loss=F.cross_entropy(m(xx),y[ids]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        assert torch.isfinite(loss)
    update(model,optimizer,order,noise,0)
    with tempfile.TemporaryDirectory(prefix='s26-resume-') as directory:
        path=Path(directory)/'resume.pt';save_resume(path,model,optimizer,order,noise)
        update(model,optimizer,order,noise,1)
        expected={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        alternative=build('LP',26026).cuda().train();opt2=torch.optim.AdamW(alternative.parameters(),lr=1e-3,weight_decay=5e-4)
        og2=torch.Generator(device='cuda');ng2=torch.Generator(device='cuda')
        restore(path,alternative,opt2,og2,ng2);update(alternative,opt2,og2,ng2,1)
        assert all(torch.equal(v,alternative.state_dict()[k].cpu()) for k,v in expected.items())
    del model,alternative,optimizer,opt2;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=build('LP',26026).cuda().train();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=5e-4)
    loss=F.cross_entropy(model(torch.randn(64,2,8192,device='cuda')),torch.arange(64,device='cuda')%10)
    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step()
    with torch.no_grad():assert model.eval()(torch.randn(128,2,8192,device='cuda')).shape==(128,10)
    write(HERE/'GATE.json',dict(status='PASS',time=now(),architecture=audits,
        synthetic_train_batch=64,synthetic_eval_batch=128,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        exact_resume_after_optimizer_noise_permutation_state=True,forbidden_role_checks=denied,
        shared_initialization_checks=6,new_scientific_fits=0,raw_data_reads=0,tf32_matmul=False,tf32_cudnn=True))
    print('PASS: architecture, shared initialization, forbidden roles, finite gradients, resource batch64/128, exact resumed step.',flush=True)


if __name__=='__main__':main()
