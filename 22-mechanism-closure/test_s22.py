"""Structural, gradient, rank and downstream integration checks before fitting."""
import itertools
import json
from pathlib import Path
import tempfile
import numpy as np
from s22_models import verify_architecture,parameterization_audit,ARMS
from s22_report import gate,equivalent,ce_value
from s22_core import read,write

def synthetic_evaluation():
    # Exercise the real evaluator's directory creation, baseline identity, hooks,
    # archives and summaries without reading any Shen data or training a model.
    import torch
    import s22_evaluate as e
    saved={name:getattr(e,name) for name in ['HERE','sealed','baseline_path','build','shen_cell','roles','PROTOCOL']}
    previous={name:getattr(e.previous,name) for name in ['setup','predict']}
    try:
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'reference';source.mkdir();e.HERE=root/'new_stage'
            labels=np.arange(10,dtype=np.int64);logits=torch.eye(10)*3
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.net=torch.nn.Module();self.net.encoder=torch.nn.Module();self.net.encoder.final_norm=torch.nn.Identity()
                def cuda(self):return self
                def load_state_dict(self,*args,**kwargs):return None
            model=Model();torch.save({},source/'model.pt');write(source/'config.json',dict(data_seed=123))
            expected={}
            for role in ['dev_val','confirm']:
                np.savez_compressed(source/f'preds_{role}.npz',rx=logits.numpy(),y_rx=labels)
                per={'rx':e.previous.rx_metrics(logits.numpy(),labels)};expected[role]=dict(summary=e.previous.summarize(per),per_rx=per)
            write(source/'target_metrics.json',expected)
            e.sealed=lambda:dict(references={'H0_f0_s201':dict(sha256={})});e.baseline_path=lambda *args:source;e.build=lambda *args:model
            e.shen_cell=lambda *args:dict(x=np.zeros((10,2,8192),dtype=np.float32),y=labels)
            e.roles=lambda fold:dict(dev_val=['rx']);e.PROTOCOL=dict(confirm_receivers=['rx'])
            e.previous.setup=lambda seed:None
            def predict(model,x):
                model.net.encoder.final_norm(torch.arange(10*2*64,dtype=torch.float32).reshape(10,2,64));return logits
            e.previous.predict=predict
            # map_location=cuda is immaterial for the empty tensor state.
            e.evaluate('H0',0,201)
            out=e.HERE/'runs/reused__H0__f0__s201';assert (out/'diagnostics_dev_val.npz').exists()
            assert len(read(out/'evaluation.json')['baseline_prediction_recovery'])==2
    finally:
        for name,value in saved.items():setattr(e,name,value)
        for name,value in previous.items():setattr(e.previous,name,value)

def synthetic_report():
    import s22_report as r
    import s22_evaluate as e
    saved_here=r.HERE;saved_seal=e.sealed
    try:
        with tempfile.TemporaryDirectory() as folder:
            r.HERE=Path(folder);e.sealed=lambda:{}
            for arm,fold,seed in itertools.product(ARMS,[0,1],[201,202,203]):
                out=r.outpath(arm,fold,seed);out.mkdir(parents=True);metrics={}
                for role in ['dev_val','confirm']:
                    z={};per={}
                    for rx in ['a','b']:
                        y=np.repeat(np.arange(10),4);lg=np.eye(10,dtype=np.float32)[y]*3;n={'H0':1,'P0':6,'H1':2,'H2':5,'S1':4,'S2':3}[arm]
                        lg[:n]=0;lg[np.arange(n),(y[:n]+1)%10]=3
                        z[rx]=lg;z['y_'+rx]=y;z['margin_'+rx]=np.where(lg.argmax(1)==y,3.,-3.);z['distances_'+rx]=np.abs(np.arange(10)[:,None]-np.arange(10)[None,:])
                        per[rx]=e.previous.rx_metrics(lg,y)
                    np.savez_compressed(out/f'diagnostics_{role}.npz',**z);metrics[role]=dict(per_rx=per)
                write(out/'evaluation.json',dict(metrics=metrics))
            r.report();d=read(r.HERE/'decision.json');assert len(d['signals'])==4 and all(d['signals'].values());assert (r.HERE/'tx_rx_deltas.csv').exists()
            import s22_paper as paper
            original_paper_here,original_paper_folder=paper.HERE,paper.PAPER
            try:
                paper.HERE=r.HERE;paper.PAPER=r.HERE/'paper';paper.PAPER.mkdir()
                costs=[dict(arm=a,params=134266 if a in ['P0','H1','S1'] else 133690,counted_macs=1e7,batch=b,p50_ms=2.0 if b==1 else 15.0) for a,b in itertools.product(ARMS,[1,128])]
                write(r.HERE/'costs.json',dict(rows=costs));(r.HERE/'robustness_source.csv').write_text('synthetic_only\n',encoding='utf-8')
                paper.result_assets();paper.write_draft()
                assert (paper.PAPER/'mechanism_results.png').stat().st_size>1000
                assert 'S22' in (paper.PAPER/'MANUSCRIPT_DRAFT.zh-CN.md').read_text(encoding='utf-8')
            finally:paper.HERE, paper.PAPER=original_paper_here,original_paper_folder
    finally:r.HERE=saved_here;e.sealed=saved_seal

def run():
    verify_architecture(cuda=True);audit=parameterization_audit();assert len(audit['proposed_center_crop']['inactive_base_taps'])==62
    s=dict(mean_delta_pp=.5,positive_pairs=5,ci95_pp=[.1,.8],fold_delta_pp={'0':.2,'1':.8},worst_receiver_delta_pp=-2)
    assert gate(s) and not gate(dict(s,positive_pairs=4)) and not gate(dict(s,ci95_pp=[0,.8]))
    assert equivalent(dict(s,ci95_pp=[-.3,.4],fold_delta_pp={'0':.1,'1':.2})) and not equivalent(s)
    assert abs(ce_value(np.zeros((2,10)),np.array([0,1]))-np.log(10))<1e-12
    synthetic_evaluation();synthetic_report()
    print('PASS:CPU/CUDA shape,rank,all-tap gradients,affine gradients,initial pairing,output directories,real evaluator and full synthetic reporting',flush=True)

if __name__=='__main__':run()
