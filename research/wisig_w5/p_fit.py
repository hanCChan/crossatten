"""Prespecified auxiliary CE/relation training; downstream uses unchanged C loop."""
import copy,math,random,time
import numpy as np
import torch
import torch.nn.functional as F
from common import *
from p_models import C,A,M,build_aux,pooled,relation,balanced_batches,transfer


def fit_aux(job,train,val,out,lam):
    out=Path(out)
    if out.exists():raise FileExistsError('Keep every auxiliary fit attempt: '+str(out))
    out.mkdir(parents=True);write(out/'ATTEMPT.json',dict(job=job,lambda_used=lam,started=now()))
    seed=job['seed'];A.setup(24101+seed);model=build_aux(seed)
    torch.manual_seed(41601+seed);np.random.seed(41601+seed);random.seed(41601+seed)
    device=torch.device('cuda');opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.1,patience=5,min_lr=1e-6);shadow=M.make_ema(model)
    train_tensor={k:torch.as_tensor(train[k],device=device) for k in ('x','y','rx')}
    val_tensor={k:torch.as_tensor(val[k],device=device) for k in ('x','y','rx')}
    classes=len(np.unique(train['y']));calibration_order=np.concatenate(balanced_batches(classes,41301+seed,0))
    calibration_tensor=train_tensor['x'][torch.as_tensor(calibration_order,device=device)]
    initial=A.semantic_hash(model.state_dict())
    if 'initial_state_sha256' in job and initial!=job['initial_state_sha256']:raise ValueError('Auxiliary initialization differs from accepted gradient probe')
    write(out/'INITIAL.json',dict(model_sha256=initial,seed=seed,head_classes=72))
    best=math.inf;best_epoch=0;history=[];updates=0
    costs=dict(gradient_seconds=0.,validation_seconds=0.,BN_calibration_seconds=0.,BN_samples=0,BN_passes=0,EMA_updates=0)
    started=time.perf_counter()
    for epoch in range(1,job['epochs']+1):
        model.train();t=time.perf_counter();ce_total=rel_total=spread_min=0.;seen=0;spread_min=math.inf
        batches=balanced_batches(classes,41301+seed,epoch);order=np.concatenate(batches)
        for ids in batches:
            idx=torch.as_tensor(ids,device=device);x=train_tensor['x'][idx];y=train_tensor['y'][idx]
            h=pooled(model,x);ce=F.cross_entropy(model.classifier(h),y);rel,structure=relation(h.reshape(8,6,4,64))
            spread=float(structure['minimum_relative_spread'])
            if spread<1e-6:raise ValueError('Auxiliary TX geometry completely collapsed; scientific attempt stopped')
            objective=ce+lam*rel
            if not torch.isfinite(objective):raise ValueError('Nonfinite auxiliary objective')
            opt.zero_grad(set_to_none=True);objective.backward();opt.step();shadow.update_parameters(model)
            updates+=1;costs['EMA_updates']+=1;seen+=len(ids);ce_total+=float(ce.detach())*len(ids);rel_total+=float(rel.detach())*len(ids);spread_min=min(spread_min,spread)
        torch.cuda.synchronize();costs['gradient_seconds']+=time.perf_counter()-t
        model.eval();t=time.perf_counter()
        with torch.no_grad():v=torch.cat([model(x) for x in val_tensor['x'].split(192)])
        raw_ce=C.rx_ce(v,val_tensor['y'],val_tensor['rx']);costs['validation_seconds']+=time.perf_counter()-t
        before=A.semantic_hash(model.state_dict());shadow_before=A.semantic_hash(shadow.state_dict());rng_before=A.semantic_hash(torch.cuda.get_rng_state())
        t=time.perf_counter();candidate,cal=M.calibrated_ema_candidate(shadow,calibration_tensor.split(192),role='fit_train')
        torch.cuda.synchronize();costs['BN_calibration_seconds']+=time.perf_counter()-t;costs['BN_samples']+=cal['samples'];costs['BN_passes']+=1
        if before!=A.semantic_hash(model.state_dict()) or shadow_before!=A.semantic_hash(shadow.state_dict()) or rng_before!=A.semantic_hash(torch.cuda.get_rng_state()):raise ValueError('Auxiliary EMA calibration changed training state')
        t=time.perf_counter();logits=C.predict_tensor(candidate,val_tensor['x'],192)
        ema_ce=C.rx_ce(torch.as_tensor(logits,device=device),val_tensor['y'],val_tensor['rx']);costs['validation_seconds']+=time.perf_counter()-t
        scheduler.step(raw_ce)
        if ema_ce<best:
            best,best_epoch=ema_ce,epoch
            torch.save(dict(model=A.cpu_tree(candidate.state_dict()),epoch=epoch,checkpoint_val_rx_ce=ema_ce,job=job,lambda_used=lam),out/'selected.pt')
            C.save_predictions(out/'selected_val.npz',logits,val,read(HERE/'DATA_PLAN.json')['aux_RX'])
        row=dict(epoch=epoch,updates=updates,train_CE=ce_total/seen,train_relation=rel_total/seen,
            lambda_used=lam,minimum_relative_spread=spread_min,raw_val_RX_CE=raw_ce,EMA_val_RX_CE=ema_ce,best_epoch=best_epoch,
            lr_next=opt.param_groups[0]['lr'],order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),student_sha256=A.semantic_hash(model.state_dict()))
        history.append(row);write(out/'HISTORY.json',history)
        write('P_LIVE.json',dict(time=now(),job=job['id'],stage='aux_pretraining',**row))
        if epoch==1 or epoch%5==0:print(job['id'],'epoch',epoch,'CE',round(ce_total/seen,4),'rel',round(rel_total/seen,4),'EMA-val',round(ema_ce,4),flush=True)
        del candidate
    expected=job['epochs']*classes*6*48//192;assert updates==expected
    torch.save(dict(student=A.cpu_tree(model.state_dict()),EMA=A.cpu_tree(shadow.state_dict()),optimizer=A.cpu_tree(opt.state_dict()),scheduler=scheduler.state_dict(),updates=updates),out/'final_state.pt')
    selected=torch.load(out/'selected.pt',map_location='cpu',weights_only=False);model.load_state_dict(selected['model']);model.eval()
    replay=C.predict_tensor(model,val_tensor['x'],192)
    with np.load(out/'selected_val.npz',allow_pickle=False) as p:assert np.array_equal(replay,p['logits'])
    complete=dict(status='COMPLETE',job=job,updates=updates,best_epoch=best_epoch,best_aux_val_RX_CE=best,
        parameter_count=sum(p.numel() for p in model.parameters()),initial_state_sha256=initial,lambda_used=lam,
        seconds=time.perf_counter()-started,costs=costs,selected_validation_bitwise_replay=True,
        selected_checkpoint=record(out/'selected.pt'),selected_predictions=record(out/'selected_val.npz'),
        final_state=record(out/'final_state.pt'),history=record(out/'HISTORY.json'))
    write(out/'COMPLETE.json',complete);del model,shadow,train_tensor,val_tensor,calibration_tensor;torch.cuda.empty_cache()
    return complete


