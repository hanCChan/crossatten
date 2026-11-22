"""Common GPU report-only inference after a committed six-model seal."""
from pathlib import Path
import time
import numpy as np
import torch
from s26_common import HERE,P,read,write,sha,validate,identity,now
from s26_data import cell,check_seal
from s26_models import build
from s26_training import setup
from xa.train import stable_seed
import train as canonical


def evaluate():
    registration=validate();check_seal();records=[];recovery=[]
    for fold in (0,1):
        for rx in P['folds'][fold]['dev_val']:
            data=cell(rx,fold,'dev_report_only')
            for arm in ['H0','P0','LP']:
                for seed in (201,202,203):
                    name=f'{arm}_f{fold}_s{seed}';out=HERE/'predictions'/name;out.mkdir(parents=True,exist_ok=True)
                    checkpoint=HERE/'runs'/name/'model.pt' if arm=='LP' else Path(next(
                        r['checkpoint'] for r in registration['references'] if r['arm']==arm and r['fold']==fold and r['seed']==seed))
                    setup(stable_seed('P-H5x160',fold,seed));model=build(arm,stable_seed('P-H5x160',fold,seed)).cuda().eval()
                    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True),strict=True)
                    logits=canonical.predict(model,data['x']).cpu().numpy();y=data['y']
                    row=dict(arm=arm,fold=fold,seed=seed,rx=rx,checkpoint_sha256=sha(checkpoint),**canonical.rx_metrics(logits,y))
                    records.append(row)
                    np.savez_compressed(out/(rx+'.npz'),logits=logits,y=y,rows=data['rows'])
                    if arm!='LP':
                        with np.load(checkpoint.parent/'preds_dev_val.npz',allow_pickle=False) as old:
                            np.testing.assert_array_equal(y,old['y_'+rx])
                            recovery.append(dict(arm=arm,fold=fold,seed=seed,rx=rx,
                                logits_bit_exact=bool(np.array_equal(logits,old[rx])),
                                max_absolute_logit_difference=float(np.max(np.abs(logits-old[rx]))),
                                changed_argmax_packets=int(np.sum(logits.argmax(1)!=old[rx].argmax(1)))))
                    del model;torch.cuda.empty_cache()
            print('Report-only evaluated fold',fold,rx,'all9 models',flush=True)
    assert len(records)==90
    write(HERE/'EVALUATION.json',dict(status='COMPLETE_REPORT_ONLY',time=now(),records=records,
        packet_predictions=180000,fitted_new_models=6,reused_models=12,new_training_steps=0,
        protocol='Common GPU FP32 network, original frontend, TF32 matmul off/cudnn on, batch128; all cohorts retained.',
        baseline_recovery=recovery,seal_sha256=sha(HERE/'seal.json')))


def costs():
    validate();setup(26026);rows=[]
    for arm in ['H0','P0','LP']:
        model=build(arm,26026).cuda().eval();batches={}
        for batch in [1,128]:
            x=torch.randn(batch,2,8192,device='cuda')
            with torch.no_grad():
                for _ in range(3):model(x)
                samples=[]
                for _ in range(15):
                    torch.cuda.synchronize();start=time.perf_counter();model(x);torch.cuda.synchronize()
                    samples.append((time.perf_counter()-start)*1000)
            batches[str(batch)]=dict(p50_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),samples=samples)
        rows.append(dict(arm=arm,params=sum(p.numel() for p in model.parameters()),batches=batches))
        del model;torch.cuda.empty_cache()
    write(HERE/'COSTS.json',dict(status='COMPLETE',time=now(),gpu=torch.cuda.get_device_name(0),rows=rows,
        scope='Separate common S26 session; synthetic GPU-resident IQ-to-logits, three warmups15 samples, no host transfer. Not pooled with older costs.')))


if __name__=='__main__':evaluate();costs()
