import importlib.util,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import ROOT,PHASE,read,digest,h0_records

def h0_model(jid,device):
    record=h0_records()[jid]
    sys.path.insert(0,str(ROOT/'13-mixer-cfo-mechanisms'))
    spec=importlib.util.spec_from_file_location('_s24_budget_h0',ROOT/'13-mixer-cfo-mechanisms/models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.build('A1_mha',record['data_seed']).to(device).eval()
    path=Path(record['path'])/'model.pt'
    if digest(path)!=record['artifacts']['model.pt']['sha256']:raise ValueError('frozen H0 changed')
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True),strict=True)
    return model

def metrics(logits,y):
    y=np.asarray(y,dtype=np.int64)
    cm=np.bincount(y*10+logits.argmax(1),minlength=100).reshape(10,10)
    tp=cm.diagonal();f1=2*tp/np.maximum(cm.sum(0)+cm.sum(1),1)
    return dict(acc=float(tp.sum()/cm.sum()),macro_f1=float(f1.mean()),ce=float(F.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y))),confusion=cm.tolist())
