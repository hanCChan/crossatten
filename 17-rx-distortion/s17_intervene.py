"""D1: fixed probes, TX preservation proxies and matched-EVM noise controls."""
import numpy as np
import torch
from s17_core import HERE,read,write,sha,validate,now,commit
from s17_models import build
from s17_data import load
from s17_augment import perturb,as_complex
from s17_train import run_id,profiles_path,setup,score
from xa.train import stable_seed

@torch.no_grad()
def probe(model,data,family,tau,profiles,seed,control=False):
    model.eval();logits=[];evm=[]
    g=torch.Generator(device='cuda').manual_seed(seed);n=torch.Generator(device='cuda').manual_seed(seed+1)
    for i in range(0,len(data['y']),128):
        x=torch.as_tensor(np.array(data['x'][i:i+128],copy=True),device='cuda');y=perturb(x,family,tau,profiles,g,n,control=control)
        evm.extend(((as_complex(y)-as_complex(x)).abs().square().mean(1)/as_complex(x).abs().square().mean(1)).sqrt().cpu().tolist())
        logits.append(model(y).cpu().numpy())
    return np.concatenate(logits),float(np.mean(evm))

def tx_accuracy(logits,y):return {int(t):float((logits[y==t].argmax(1)==t).mean()) for t in np.unique(y)}

def run():
    validate();rows=[];jobs=[j for j in read(HERE/'jobs_pilot.json') if j['arm']=='M0']
    for j in jobs:
        out=HERE/'runs'/run_id(j);cfg=read(out/'config.json');setup(cfg['data_seed']);m=build(j['dataset'],cfg['data_seed']).cuda()
        m.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True));data,meta=load(j);va=data['val'];pool=read(profiles_path(j))['profiles']
        baseline,_=probe(m,va,'identity',0,pool,180170+j['fold']);b=score(baseline,va,meta['rx_names']);tx0=tx_accuracy(baseline,va['y'])
        expected=read(out/'metrics.json')['source_validation']['summary']['acc_mean']
        if abs(expected-b['summary']['acc_mean'])>1e-10:raise ValueError('Frozen intervention baseline differs from selected model')
        for family in ['global_gain','global_phase','amp','iq','phase']:
            levels=[1.] if family.startswith('global') else [.25,.5,1.]
            for tau in levels:
                for control in ([False,True] if family in ['amp','iq','phase'] else [False]):
                    for draw in [0,1]:
                        seed=stable_seed('S17-D1',j['fold'],family,tau,draw)
                        lg,evm=probe(m,va,family,tau,pool,seed,control);s=score(lg,va,meta['rx_names']);t=tx_accuracy(lg,va['y'])
                        rows.append(dict(fold=j['fold'],family=family,tau=tau,draw=draw,noise_control=control,evm=evm,
                            delta_acc_pp=100*(s['summary']['acc_mean']-b['summary']['acc_mean']),delta_f1_pp=100*(s['summary']['f1_mean']-b['summary']['f1_mean']),
                            worst_tx_delta_pp=100*min(t[k]-tx0[k] for k in tx0),summary=s['summary']))
        del m;torch.cuda.empty_cache();print('D1 fold',j['fold'],'complete',flush=True)
    selected={};eligible=read(HERE/'decision_d0.json')['eligible']
    for family in ['amp','iq']:
        safe=[]
        for tau in [.25,.5,1.]:
            v=[r for r in rows if r['family']==family and r['tau']==tau and not r['noise_control']]
            folds=[dict(acc=np.mean([r['delta_acc_pp'] for r in v if r['fold']==f]),f1=np.mean([r['delta_f1_pp'] for r in v if r['fold']==f]),
                worst_tx=np.mean([r['worst_tx_delta_pp'] for r in v if r['fold']==f])) for f in range(4)]
            if all(s['acc']>=-1 and s['f1']>=-1 and s['worst_tx']>=-5 for s in folds) and np.mean([r['evm'] for r in v])>=.002:safe.append(tau)
        if family in eligible and safe:selected[family]=max(safe)
    invariant=all(abs(r['delta_acc_pp'])<=.1 for r in rows if r['family'].startswith('global'))
    if not invariant:selected={}
    result=dict(time=now(),status='GO_PILOT' if selected else 'STOP_INTERVENTION_GATE' if invariant else 'STOP_INVARIANCE_CONTROL',selected=selected,d0_eligible=eligible,
        invariance_controls_passed=invariant,
        interpretation='Frozen-model Accuracy/F1/worst-TX are preservation proxies, not proof that physical TX information is unchanged',
        phase_training_eligible=False,held_source_read=False,real_target_access=False)
    write(HERE/'D1_interventions.json',dict(rows=rows));write(HERE/'decision_d1.json',result)
    commit([HERE/'D1_interventions.json',HERE/'decision_d1.json'],'S17 freeze fixed interventions, negative controls and TX-preservation gate')

if __name__=='__main__':run()
