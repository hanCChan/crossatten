import sys,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from s24_common import write_json
from models import build,hsic
from data import preprocess

def main():
    torch.set_num_threads(4);torch.manual_seed(2401)
    model=build('fan').cuda();model.train()
    x=torch.randn(8,2,8192,device='cuda');x=preprocess(x,'native','fan')
    h,logits=model(x);term=hsic(h,x);loss=F.cross_entropy(logits,torch.arange(8,device='cuda')%10)+0.004*term.cuda();loss.backward()
    good=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    torch.cuda.synchronize()
    # Compare CPU original to the existing mathematically identical device-safe port.
    from xa.paper_models import hsic_cca
    a=torch.randn(8,20,requires_grad=True);b=torch.randn(8,33)
    orig=hsic(a,b);g1=torch.autograd.grad(orig,a)[0]
    port=hsic_cca(a,b);g2=torch.autograd.grad(port,a)[0]
    same=torch.allclose(orig,port,atol=1e-5,rtol=1e-5) and torch.allclose(g1,g2,atol=1e-5,rtol=1e-5)
    write_json(HERE/'IB_GATE.json',dict(status='PASS' if good and same else 'FAIL',params=sum(p.numel() for p in model.parameters()),
               input_shape=list(x.shape),hidden_shape=list(h.shape),logits_shape=list(logits.shape),position_embedding_shape=list(model.patch_embedding.pos_embed.shape),
               original_hsic=float(term),original_vs_existing_hsic_max_gradient_error=float((g1-g2).abs().max()),
               finite_backward=good,original_hsic_preserved=True,optimizer_steps=0,scientific_fits=0))
    assert good and same
    print('8192/patch512/31tokens FAN and original CPU HSIC gradient gate PASS.')

if __name__=='__main__':main()
