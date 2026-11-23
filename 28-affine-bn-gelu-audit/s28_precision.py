"""Fixed post-failure precision diagnostic; original failures are not replaced."""
import gc
from pathlib import Path
import numpy as np
import torch
from s28_core import HERE,read,write,sha,now,validate,committed
from s28_models import build
from s28_affine import MaterializedTokenizer
from s28_data import source_validation
from s28_audit import compare_batches
from s27b_training import setup
from xa.train import stable_seed

def main():
    reg=validate();plan=read(HERE/'PRECISION_PLAN.json');committed(HERE/'PRECISION_PLAN.json')
    assert sha(__file__)==plan['script_sha256'] and sha(HERE/'AFFINE_AUDIT.json')==plan['original_audit_sha256']
    assert not (HERE/'PRECISION_DIAGNOSTIC.json').exists();rows=[]
    for f in (0,1):
        data=source_validation(f);ids=[np.flatnonzero((data['rx']==r)&(data['y']==t))[0] for r in range(5) for t in range(10)]
        x=torch.as_tensor(data['x'][ids],device='cuda')
        for q in [q for q in reg['references'] if q['arm']=='LA' and q['fold']==f]:
            setup(stable_seed('P-H5x160',f,q['seed']));model=build('LA',stable_seed('P-H5x160',f,q['seed'])).cuda().eval()
            model.load_state_dict(torch.load(q['checkpoint'],map_location='cuda',weights_only=True))
            op=MaterializedTokenizer(model.net.embedder.cnn_stem.tokenize)
            original=next(t for t in read(HERE/'AFFINE_AUDIT.json')['models'] if t['id']==q['id'])
            path=HERE/'converted'/q['id']/'operator_float64.pt';assert sha(path)==original['converted_sha256']
            op.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));op=op.float().cuda()
            results={};predictions={}
            try:
                for flag in [True,False]:
                    torch.backends.cudnn.allow_tf32=flag
                    results[str(flag)],predictions[flag]=compare_batches(model,op,x,10,1e-4,1e-4)
            finally:torch.backends.cudnn.allow_tf32=True
            rows.append(dict(id=q['id'],packets=50,original_audit_preserved=True,cuDNN_TF32=results,
                original_native_vs_strict_prediction_changes=int(np.sum(predictions[True].argmax(1)!=predictions[False].argmax(1)))))
            assert sha(q['checkpoint'])==q['sha256']['model.pt']
            print('Fixed precision diagnostic',q['id'],results['False']['errors']['logits']['allclose'],flush=True)
            del model,op;gc.collect();torch.cuda.empty_cache()
        del x,data;gc.collect();torch.cuda.empty_cache()
    write(HERE/'PRECISION_DIAGNOSTIC.json',dict(time=now(),status='COMPLETE_NO_FIT_DIAGNOSTIC',models=rows,
        source_packets_per_model=50,models_checked=6,original_native_contract_still_failed=True,
        strict_probe_all_pass=all(q['cuDNN_TF32']['False']['errors'][k]['allclose'] for q in rows for k in ['tokens','logits']),
        scientific_fits=0,optimizer_updates=0,Dev_reads=0,scope=plan['scope']))

if __name__=='__main__':main()
