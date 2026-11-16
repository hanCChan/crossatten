import importlib,sys,types,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent;ROOT=STAGE.parent
sys.path.insert(0,str(STAGE));sys.path.insert(0,str(ROOT/'lib'))
from s24_common import verified_source
from s24_dann import ShenDANN,Reverse
from xa.frontends import shen_cis

class SourceCE(ShenDANN):
    def __init__(self):
        super().__init__(strict_ib=False)
        del self.rx_input;del self.rx_classifier
    def forward(self,x):
        return self.classifier(F.relu(self.tx_input(self.features(shen_cis(x)))))

def upstream_fan():
    repo='BeechburgPieStar/IB-RFF'
    paths=['backbones/FAN.py','backbones/Adaptive_Filter_Module.py','backbones/Embedding_Layer.py',
           'backbones/Bottleneck_Layer/DSCNN.py','backbones/Bottleneck_Layer/ICNN.py','backbones/Bottleneck_Layer/MLP.py']
    for path in paths:verified_source(repo,path)
    name='_s24_fan'
    if name not in sys.modules:
        pkg=types.ModuleType(name);pkg.__path__=[str(STAGE/'_upstream'/repo/'backbones')];sys.modules[name]=pkg
    return importlib.import_module(name+'.FAN').FAN

def hsic(hidden,inputs):
    global _hsic
    if '_hsic' not in globals():
        ns={};exec(compile(verified_source('BeechburgPieStar/IB-RFF','utils/hsic.py'),'pinned_original_hsic.py','exec'),ns)
        _hsic=ns['hsic_normalized_cca']
    return _hsic(hidden.flatten(1).cpu(),inputs.flatten(1).cpu(),sigma=5.0)

def build(method):
    if method in ('dann','ce'):
        # Both controls have identical encoder/TX initialization; CE excludes RX loss.
        return SourceCE() if method=='ce' else ShenDANN(strict_ib=False)
    if method in ('fan','ib'):
        return upstream_fan()(seq_len=8192,in_chans=2,patch_size=512,embed_dim=128,mlp_ratio=2.0,
                              dr=0.5,num_classes=10,bottleneck_type='DSCNN',use_SAF=True,use_FAF=True)
    if method=='ask':
        ns={};exec(compile(verified_source('BeechburgPieStar/ASKNet-RFF','Lora/backbones/ResCISNet.py'),'pinned_ResCISNet.py','exec'),ns)
        with torch.device('cuda'):return ns['ResCISNet'](num_classes=10)
    raise ValueError(method)

def forward(model,method,x,epoch=1):
    if method in ('dann','ce'):
        feat=model.features(shen_cis(x))
        tx=model.classifier(F.relu(model.tx_input(feat)))
        rx=model.rx_classifier(F.relu(model.rx_input(Reverse.apply(feat)))) if method=='dann' else None
        return tx,rx,None
    if method in ('fan','ib'):
        hidden,tx=model(x);return tx,None,hidden
    return model(x,tau=5.0+(0.1-5.0)*min(epoch/200,1)),None,None
