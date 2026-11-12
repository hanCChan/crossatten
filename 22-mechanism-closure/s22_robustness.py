"""Bounded source-validation stress test; no model selection or new fitting."""
import itertools
import numpy as np
import torch
from s22_core import HERE,read,write,now,baseline_path
from s22_models import build
from s22_data import shen_train
from s22_evaluate import sealed,previous
from s22_report import csv_write
from xa.frontends import awgn
from xa.train import stable_seed

def shift_zero(x,amount):
    out=torch.zeros_like(x)
    if amount>0:out[...,amount:]=x[...,:-amount]
    elif amount<0:out[...,:amount]=x[...,-amount:]
    else:out.copy_(x)
    return out

def run():
    reg=sealed();rows=[]
    for fold in [0,1]:
        _,va=shen_train(fold);x=torch.as_tensor(va['x'],device='cuda');y=va['y']
        for arm,seed in itertools.product(['H0','P0'],[201,202,203]):
            path=baseline_path(arm,fold,seed);cfg=read(path/'config.json');previous.setup(cfg['data_seed'])
            model=build(arm,cfg['data_seed']).cuda();model.load_state_dict(torch.load(path/'model.pt',map_location='cuda',weights_only=True),strict=True)
            for condition in reg['robustness']['conditions']:
                if condition=='clean':perturbed=x
                elif condition.startswith('shift'):
                    perturbed=shift_zero(x,int(condition[5:]))
                else:
                    snr=int(condition[3:]);generator=torch.Generator(device='cuda').manual_seed(stable_seed('S22-robust',fold,seed,condition))
                    perturbed=awgn(x,snr,snr,generator)
                logits=previous.predict(model,perturbed).cpu().numpy()
                for rx in np.unique(va['rx']):
                    ids=va['rx']==rx;m=previous.rx_metrics(logits[ids],y[ids]);rows.append(dict(arm=arm,fold=fold,seed=seed,condition=condition,source_rx_index=int(rx),accuracy_pct=100*m['acc'],macro_f1_pct=100*m['macro_f1']))
            del model
        del x;torch.cuda.empty_cache()
    csv_write('robustness_source.csv',rows);write(HERE/'ROBUSTNESS.json',dict(time=now(),scope='Source validation only;fixed H0/P0;zero-padded shifts;input-added AWGN nominal dB;not independent RX generalization',rows=len(rows),conditions=reg['robustness']['conditions']))

if __name__=='__main__':run()
