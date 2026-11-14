"""Prespecified auxiliary CE/relation training; downstream uses unchanged C loop."""
import copy,math,random,time
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from p_models import C,A,M,build_aux,pooled,relation,balanced_batches,transfer


from p_fit import fit_aux

def train():
    reg=validate();accept=read(HERE/'P_DATA_ACCEPTANCE.json');assert accept['status']=='PASS'
    if (HERE/'P_TRAINING_ATTEMPT.json').exists():raise RuntimeError('Keep original formal training attempt; no automatic scientific retries')
    for n in ('P_AUX_TRAIN.npz','P_DOWNSTREAM_TRAIN.npz'):
        if sha(ART/n)!=accept['cache_sha256'][n]:raise ValueError('Accepted training cache changed')
    with np.load(ART/'P_AUX_TRAIN.npz',allow_pickle=False) as p:
        aux={role:{k:p[f'{role}_{k}'] for k in ('x','y','rx','day','keys','pool_ids')} for role in ('train','val')}
    with np.load(ART/'P_DOWNSTREAM_TRAIN.npz',allow_pickle=False) as p:
        downstream={role:{k:p[f'{role}_{k}'] for k in ('x','y','rx','day','keys','pool_ids')} for role in ('train','val')}
    write('P_TRAINING_ATTEMPT.json',dict(status='RUNNING',time=now(),registration_sha256=sha(HERE/'EXECUTION_REGISTRATION.json'),
        registered_commit=git('rev-parse','HEAD'),authorized_maximum_fits=6,registered_actual_new_fits=6))
    completed=[]
    try:
        for job in reg['jobs']:
            out=ART/job['id'];seed=job['seed'];arm=job['arm']
            if job['stage']=='aux':
                assert arm=='A1';lam=0.
                result=fit_aux(job,aux['train'],aux['val'],out,lam)
            else:
                if out.exists():raise FileExistsError('Keep existing downstream attempt')
                out.mkdir(parents=True);write(out/'ATTEMPT.json',dict(job=job,time=now()))
                pre=read(ART/f'{arm}_S{seed}_aux'/'COMPLETE.json')['selected_checkpoint'];checked(pre,'load_pretraining_EMA_for_transfer',job['id'])
                ck=torch.load(pre['path'],map_location='cpu',weights_only=False);A.setup(24101+seed);model=transfer(ck['model'],seed)
                recipe=read(HERE/'DATA_PLAN.json')['A0'][str(seed)]['recipe'].copy();recipe.update(id=job['id'])
                result=C.fit_loop(recipe,downstream['train'],downstream['val'],model,torch.device('cuda'),out,read(HERE/'DATA_PLAN.json')['source']['rx_names'])
                result.update(job=job,pretraining_checkpoint_sha256=pre['sha256'],selected_checkpoint=record(out/'C1_selected.pt'),
                    selected_predictions=record(out/'C1_selected_val.npz'))
                write(out/'COMPLETE.json',result);del model;torch.cuda.empty_cache()
            completed.append(dict(job=job['id'],complete=record(out/'COMPLETE.json'),selected_checkpoint=result['selected_checkpoint'],
                selected_predictions=result['selected_predictions'],updates=result['updates']))
            write('P_PROGRESS.json',dict(status='RUNNING',completed=completed,new_fits=len(completed),time=now()))
        assert len(completed)==6 and sum(c['updates'] for c in completed)==28200
        write('P_MODEL_SEAL.json',dict(status='ALL_SIX_NEW_FITS_COMPLETE_BEFORE_HELD_SOURCE_READOUT',time=now(),runs=completed,
            A0_references=read(HERE/'DATA_PLAN.json')['A0'],new_fits=6,A0_reused=3,new_gradient_updates=28200,
            registration_sha256=sha(HERE/'EXECUTION_REGISTRATION.json'),held_source_IQ_rows_read=0,aux24_IQ_rows_read=0))
        print('SEALED 6 new fits; 28,200 updates; no held-source/aux24 readout',flush=True)
    except Exception as error:
        write('P_TRAINING_FAILURE.json',dict(time=now(),error_type=type(error).__name__,error=str(error),completed=completed,
            keep_original_attempt=True,no_automatic_scientific_retry=True));raise


if __name__=='__main__':train()
