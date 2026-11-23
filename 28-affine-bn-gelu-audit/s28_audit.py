"""Real frozen-checkpoint affine and Source recovery audit; no optimizer."""
import copy
import gc
from pathlib import Path
import numpy as np
import torch
from s28_core import HERE,ROOT,read,write,sha,now,validate,event
from s28_models import build,geometry
from s28_affine import MaterializedTokenizer,error_record
from s28_data import source_validation
from s27b_training import setup
from xa.train import stable_seed
import train as canonical

def compare_batches(model,operator,x,batch,atol,rtol):
    tokenizer=model.net.embedder.cnn_stem.tokenize
    sums={name:dict(max_abs=0.,max_relative=0.,sum_square=0.,failing_elements=0,elements=0) for name in ['tokens','logits','naive','naive_interior','shared_kernel_with_position_bias']}
    positions=torch.zeros(32,dtype=torch.float64);changed=0;naive_changed=0;logits_all=[]
    def collect(name,a,b):
        q=error_record(a,b,atol,rtol);s=sums[name]
        s['max_abs']=max(s['max_abs'],q['max_abs']);s['max_relative']=max(s['max_relative'],q['max_relative'])
        s['sum_square']+=float((a-b).double().square().sum());s['failing_elements']+=q['failing_elements'];s['elements']+=q['elements']
    for start in range(0,len(x),batch):
        xx=x[start:start+batch];captured=[]
        h=tokenizer.register_forward_pre_hook(lambda m,args:captured.append(args[0].detach()))
        with torch.no_grad():original=model(xx)
        h.remove();z=captured[0]
        with torch.no_grad():a=tokenizer(z);b=operator(z);c=operator.naive(z);d=operator.naive(z,position_bias=True)
        collect('tokens',a,b);collect('naive',a,c);collect('naive_interior',a[:,:,1:31],c[:,:,1:31]);collect('shared_kernel_with_position_bias',a,d)
        positions=torch.maximum(positions,(a-b).abs().amax((0,1)).cpu().double())
        model.net.embedder.cnn_stem.tokenize=operator
        try:
            with torch.no_grad():converted=model(xx)
        finally:model.net.embedder.cnn_stem.tokenize=tokenizer
        collect('logits',original,converted);changed+=int((original.argmax(1)!=converted.argmax(1)).sum())
        # Audit full naive logits on the balanced FP64 probe only; all Source
        # FP32 tokenizer/converted-logit checks remain complete.
        if len(x)==50:
            class Naive(torch.nn.Module):
                def forward(self,v):return operator.naive(v)
            model.net.embedder.cnn_stem.tokenize=Naive()
            try:
                with torch.no_grad():n=model(xx)
            finally:model.net.embedder.cnn_stem.tokenize=tokenizer
            naive_changed+=int((original.argmax(1)!=n.argmax(1)).sum())
        logits_all.append(original.cpu().float().numpy())
    for s in sums.values():
        s['rms']=(s.pop('sum_square')/s['elements'])**.5;s['allclose']=s['failing_elements']==0;s['atol']=atol;s['rtol']=rtol
    return dict(errors=sums,corrected_per_position_max_abs=positions.tolist(),packets=len(x),changed_predictions=changed,
        naive_changed_predictions_on_balanced_probe=naive_changed if len(x)==50 else None),np.concatenate(logits_all)

def main():
    reg=validate();setup(28028);references=[];affine=[];structures=[]
    for f in (0,1):
        data=source_validation(f);x=torch.as_tensor(data['x'],device='cuda')
        ids=[np.flatnonzero((data['rx']==r)&(data['y']==t))[0] for r in range(5) for t in range(10)]
        for q in [q for q in reg['references'] if q['fold']==f]:
            folder=Path(q['checkpoint']).parent
            for name,h in q['sha256'].items():assert sha(folder/name)==h
            seed=stable_seed('P-H5x160',f,q['seed']);setup(seed);model=build(q['arm'],seed).cuda().eval()
            model.load_state_dict(torch.load(q['checkpoint'],map_location='cuda',weights_only=True),strict=True)
            shape=geometry(model);structures.append(dict(id=q['id'],**shape,tokenizer_modules=[dict(index=i,kind=type(m).__name__) for i,m in enumerate(model.net.embedder.cnn_stem.tokenize)]))
            if q['arm']=='LA':
                op=MaterializedTokenizer(model.net.embedder.cnn_stem.tokenize)
                dest=HERE/'converted'/q['id']/'operator_float64.pt';dest.parent.mkdir(parents=True,exist_ok=True)
                torch.save(op.state_dict(),dest)
                original_tokenizer=copy.deepcopy(model.net.embedder.cnn_stem.tokenize).cpu().double().eval()
                rng=torch.Generator().manual_seed(28028);z=torch.randn(6,32,8192,generator=rng,dtype=torch.float64)
                z[0].zero_();z[2]*=100;z[3:].zero_();z[3,:,0]=1;z[4,:,-1]=1;z[5,:,256]=1
                with torch.no_grad():synthetic=error_record(original_tokenizer(z),op(z),1e-9,1e-9)
                float64_model=copy.deepcopy(model).double();op64=copy.deepcopy(op).cuda()
                f64,_=compare_batches(float64_model,op64,x[ids].double(),10,1e-9,1e-9)
                del float64_model,op64,original_tokenizer,z;gc.collect();torch.cuda.empty_cache()
                f32,logits=compare_batches(model,op.float().cuda(),x,128,1e-4,1e-4)
                affine.append(dict(id=q['id'],fold=f,seed=q['seed'],operator=op.description(),converted_sha256=sha(dest),
                    synthetic_FP64=synthetic,source_FP64=f64,source_FP32=f32,source_probe_rows_sha256=__import__('hashlib').sha256(data['rows'][ids].tobytes()).hexdigest()))
            else:logits=canonical.predict(model,x).cpu().numpy()
            old=read(folder/'metrics.json');score=canonical.score(logits,data['y'],data['rx'])
            assert score==old['source_validation'],q['id']
            y=torch.as_tensor(data['y'],device='cuda');rx=torch.as_tensor(data['rx'],device='cuda');lg=torch.as_tensor(logits,device='cuda')
            ce=float(torch.stack([torch.nn.functional.cross_entropy(lg[rx==r],y[rx==r]) for r in range(5)]).mean())
            assert abs(ce-old['best_source_ce'])<1e-7,(q['id'],ce,old['best_source_ce'])
            references.append(dict(id=q['id'],arm=q['arm'],fold=f,seed=q['seed'],source_metrics_exact=True,source_accuracy=score['summary']['acc_mean'],
                source_ce=ce,archived_ce=old['best_source_ce'],checkpoint_sha256=sha(q['checkpoint'])))
            for name,h in q['sha256'].items():assert sha(folder/name)==h
            print('Read-only Source recovery',q['id'],'PASS',flush=True)
            del model,logits,lg,y,rx;gc.collect();torch.cuda.empty_cache()
        del x,data;gc.collect();torch.cuda.empty_cache()
    numerical_pass=all(q['synthetic_FP64']['allclose'] and all(q[mode]['errors'][kind]['allclose'] for mode in ['source_FP64','source_FP32'] for kind in ['tokens','logits']) for q in affine)
    write(HERE/'AFFINE_AUDIT.json',dict(time=now(),status='PASS' if numerical_pass else 'NUMERICAL_CONTRACT_NOT_MET',models=affine,
        independent_frozen_models=6,all_token_positions_included=True,original_models_changed=False,RF_roles=['audit_source_val'],scientific_fits=0,
        scope='Conditional affine tokenizer, fixed eval BN. This establishes a representation equivalence, not that direct optimization learns the same function.'))
    write(HERE/'SOURCE_RECOVERY.json',dict(time=now(),status='PASS',models=references,model_count=24,source_predictions=48000,Dev_reads=0,scientific_fits=0))
    write(HERE/'STRUCTURE_AUDIT.json',dict(time=now(),models=structures,new_arm=read(HERE/'GATE.json')['architecture']['H2_LA']))
    validate();print('No-fit audit complete. Numerical contract:',numerical_pass,flush=True)

if __name__=='__main__':main()
