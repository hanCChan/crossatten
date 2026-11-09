"""Actual immutable official/random backbones; synthetic CPU BF16 gradients only."""
import gc,time
import torch
from s16_core import HERE,write,now,read
from s16_models import build,parameter_audit

def run():
    torch.set_num_threads(2);results=[]
    for arm in ['R3_pretrained_frozen','R4_random_frozen','R5_pretrained_lora','R6_random_lora']:
        started=time.perf_counter();m=build(arm,17001);x=torch.randn(1,32,64)
        with torch.autocast('cpu',dtype=torch.bfloat16):logits=m(x).float();loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0]))
        if not torch.isfinite(loss):raise ValueError('Full official configuration nonfinite loss')
        loss.backward();trainable=[p for p in m.parameters() if p.requires_grad]
        norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
        if not float(m.project.weight.grad.abs().sum())>0 or not float(m.cls.grad.abs().sum())>0:raise ValueError('Missing RF bridge gradient')
        if any(p.grad is not None for p in m.backbone.parameters() if not p.requires_grad):raise ValueError('Frozen base received gradients')
        if any(p.dtype!=torch.float32 for p in trainable):raise ValueError('Trainable weights must be FP32')
        if m.backbone.get_input_embeddings() is not None:raise ValueError('Unused vocabulary retained')
        if 'lora' in arm:
            b=[p for n,p in m.named_parameters() if 'lora_B' in n]
            if not any(p.grad is not None and float(p.grad.abs().sum())>0 for p in b):raise ValueError('Missing LoRA B gradient')
        row=dict(arm=arm,status='PASS',seconds=time.perf_counter()-started,logit_shape=list(logits.shape),loss=float(loss.detach()),grad_norm=float(norm),audit=parameter_audit(m))
        results.append(row);print(arm,'PASS',round(row['seconds'],2),flush=True)
        del m,x,logits,loss,trainable;gc.collect()
    write(HERE/'OFFICIAL_CPU_SMOKE.json',dict(time=now(),status='PASS',device='CPU BF16 autocast; synthetic random RF tokens only',
        revision=read(HERE/'MODEL_MANIFEST.json')['revision'],results=results,no_gpu_use=True,scientific_accuracy_evidence=False))

if __name__=='__main__':run()
