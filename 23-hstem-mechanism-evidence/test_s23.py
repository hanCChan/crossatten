"""Test transformations and metrics, not a new scientific fit."""
import torch
from s23_math import build,operator,materialize_s1,fold_h0,equivalence,shifted,difference
from s23_run import local_gradients

def run():
    torch.set_num_threads(2);T=operator();singular=torch.linalg.svdvals(T)
    assert torch.linalg.matrix_rank(T)==256 and .38<float(singular.min())<.40
    w=torch.randn(2,256,dtype=torch.float64,requires_grad=True);v=w@T.T;loss=v.square().sum();gw,gv=torch.autograd.grad(loss,[w,v]);assert torch.allclose(gw,gv@T,atol=1e-12,rtol=1e-12)
    for device in ['cpu','cuda']:
        x=torch.randn(3,2,8192,device=device)
        for arm,convert in [('S1',materialize_s1),('H0',fold_h0)]:
            model=build(arm,123).to(device).eval()
            # Exercise nontrivial BN running statistics and affine parameters.
            if arm=='H0':
                with torch.no_grad():
                    for m in model.net.embedder.cnn_stem.tokenize:
                        if isinstance(m,torch.nn.BatchNorm1d):m.running_mean.uniform_(-.2,.2);m.running_var.uniform_(.5,2);m.weight.uniform_(.5,1.5);m.bias.uniform_(-.1,.1)
            other=convert(model)
            with torch.no_grad():result=equivalence(other(x),model(x))
            assert result['allclose'] and result['predictions_equal']==len(x),(device,arm,result)
            if arm=='S1':assert type(other.net.embedder.cnn_stem.tokenize[0]) is torch.nn.Conv1d
            else:assert sum(isinstance(m,torch.nn.BatchNorm1d) for m in other.net.embedder.cnn_stem.tokenize)==0
        for amount in [-4,-1,1,4]:
            rolled=shifted(x,amount,'circular');assert torch.equal(shifted(rolled,-amount,'circular'),x)
            assert torch.allclose(rolled.square().sum(-1),x.square().sum(-1),rtol=1e-6)
            zero=shifted(x,amount,'zero');assert torch.count_nonzero(zero[...,:amount] if amount>0 else zero[...,amount:])==0
    assert abs(difference(torch.tensor([1j]),torch.tensor([0j]))['max_abs']-1)<1e-12
    model=build('S1',123).cuda().eval();rows,chain=local_gradients(model,torch.randn(3,2,8192,device='cuda'),torch.tensor([0,1,2],device='cuda'),'S1',0,201)
    assert rows and chain['relative_l2']<1e-4
    print('PASS: CPU/CUDA materialization and nontrivial BN folding, predictions, chain rule, shifts and complex-error metrics;zero optimizer steps',flush=True)

if __name__=='__main__':run()
