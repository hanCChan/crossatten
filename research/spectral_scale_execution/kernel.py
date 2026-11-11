"""One unchanged fit kernel for synthetic acceptance and the later12 authorized fits."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
import numpy as np

import common as H

TRACE_COLUMNS=('update','epoch','batch','ce','lr','alpha_before','alpha_after',
    'task_alpha_gradient','task_side_gradient_l2','side_parameter_delta_l2','all_task_gradients_finite',
    'all_updated_parameters_finite','backbone_gradients_absent')


def choose_checkpoint(best_ce,best_epoch,new_ce,epoch):
    if not math.isfinite(new_ce):raise ValueError('Nonfinite checkpoint validation CE')
    return (new_ce,epoch,True) if new_ce<best_ce else (best_ce,best_epoch,False)


def snapshot(model,opt,scheduler,order,epoch,best_ce,best_epoch,updates):
    return H.cpu_tree(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
        rng=H.A.rng_state(order),epoch=epoch,best_ce=best_ce,best_epoch=best_epoch,updates=updates))


def trainable_parameters(model):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if sum(p.numel() for _,p in named)!=3201 or any(not n.startswith('residual.') for n,_ in named):raise ValueError('Only3201 registered residual parameters may train')
    if any(p.requires_grad for p in model.backbone.parameters()):raise ValueError('Backbone unexpectedly trainable')
    return named


def save_readout(out,endpoint,model,logits,val,names,epoch,ce):
    torch.save(dict(model=H.cpu_tree(model.state_dict()),epoch=epoch,checkpoint_val_rx_ce=ce,
        alpha=float(model.residual.alpha.detach())),out/(endpoint+'.pt'))
    H.C.save_predictions(out/(endpoint+'_val.npz'),logits,val,names)


def fit_loop(job,train,val,model,device,out,rx_names):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name in ('orders','epoch_states'):(out/name).mkdir(exist_ok=True)
    torch.manual_seed(job['stochastic_seed']);np.random.seed(job['stochastic_seed']);random.seed(job['stochastic_seed'])
    named=trainable_parameters(model)
    alpha=model.residual.alpha
    if float(alpha.detach())!=0.:raise ValueError('Registered alpha initialization is zero')
    side=[p for n,p in named if n!='residual.alpha']
    opt=torch.optim.Adam([p for _,p in named],lr=job['lr'],weight_decay=job['weight_decay'])
    if {id(p) for group in opt.param_groups for p in group['params']} & {id(p) for p in model.backbone.parameters()}:raise ValueError('Backbone entered optimizer')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.1,patience=5,min_lr=1e-6)
    order=torch.Generator(device=device).manual_seed(job['order_seed'])
    tensors={role:{k:torch.as_tensor(d[k],device=device,dtype=torch.float32 if k=='x' else torch.int64) for k in ('x','y','rx')}
             for role,d in (('train',train),('val',val))}
    frozen_hash=H.semantic_hash(model.backbone.state_dict())
    generator=torch.Generator(device=device).manual_seed(27699)
    probe=torch.randn(8,2,256,generator=generator,device=device)
    reference_probe=H.C.predict_tensor(model,probe,8)
    before_rng=H.semantic_hash(H.A.rng_state(order))
    logits=H.C.predict_tensor(model,tensors['val']['x'],job['batch'])
    ce=H.C.rx_ce(torch.as_tensor(logits,device=device),tensors['val']['y'],tensors['val']['rx'])
    if before_rng!=H.semantic_hash(H.A.rng_state(order)):raise ValueError('Initial validation changed RNG')
    best,best_epoch,updates=ce,0,0
    save_readout(out,'epoch0',model,logits,val,rx_names,0,ce)
    save_readout(out,'selected',model,logits,val,rx_names,0,ce)
    initial=snapshot(model,opt,scheduler,order,0,best,best_epoch,0)
    torch.save(initial,out/'initial.pt')
    H.write(out/'INITIAL.json',dict(model_sha256=H.semantic_hash(model.state_dict()),frozen_backbone_sha256=frozen_hash,
        common_residual_sha256=H.semantic_hash({k:v for k,v in model.residual.state_dict().items() if k.startswith(('reduce.','wide.','output.')) or k=='alpha'}),
        rng_sha256=H.semantic_hash(initial['rng']),optimizer_sha256=H.semantic_hash(initial['optimizer']),
        scheduler_sha256=H.semantic_hash(initial['scheduler']),epoch0_val_rx_ce=ce,alpha=0.,
        trainable_parameters=[dict(name=n,shape=list(p.shape),elements=p.numel()) for n,p in named],
        optimizer_parameter_elements=sum(p.numel() for group in opt.param_groups for p in group['params'])))
    np.savez_compressed(out/'SAMPLE_KEYS.npz',train_keys=train['keys'],val_keys=val['keys'],train_ids=train['pool_ids'],val_ids=val['pool_ids'])
    history=[];trace=[];started=time.perf_counter();training_seconds=validation_seconds=rollback_seconds=0.
    # Every successful optimizer update retains both task gradients and actual
    # parameter movement, which can include coupled Adam weight decay at alpha0.
    with (out/'GRADIENT_TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as journal:
        for epoch in range(1,job['epochs']+1):
            t=time.perf_counter();model.train();lr=opt.param_groups[0]['lr']
            if model.backbone.training or any(m.training for m in model.backbone.modules() if isinstance(m,torch.nn.BatchNorm1d)):raise ValueError('Frozen backbone/BN is in training mode')
            permutation=torch.randperm(len(train['y']),generator=order,device=device)
            positions=permutation.cpu().numpy()
            if not np.array_equal(np.sort(positions),np.arange(len(positions))):raise ValueError('Training permutation does not cover every packet once')
            np.save(out/'orders'/f'epoch_{epoch:02d}.npy',positions)
            ce_total,correct=0.,0
            for batch_index,pos in enumerate(permutation.split(job['batch']),1):
                logits=model(tensors['train']['x'][pos]);labels=tensors['train']['y'][pos]
                loss=F.cross_entropy(logits,labels)
                if not torch.isfinite(loss):raise ValueError('Nonfinite task CE')
                opt.zero_grad(set_to_none=True);loss.backward()
                if any(p.grad is None for _,p in named):raise ValueError('Unused trainable parameter')
                finite=all(bool(torch.isfinite(p.grad).all()) for _,p in named)
                frozen_gradients=all(p.grad is None for p in model.backbone.parameters())
                if not finite or not frozen_gradients:raise ValueError('Nonfinite task gradient or forbidden backbone gradient')
                side_grad=torch.cat([p.grad.detach().flatten() for p in side])
                side_before=torch.cat([p.detach().flatten() for p in side]).clone()
                alpha_before=float(alpha.detach());grad_alpha=float(alpha.grad.detach())
                grad_side=float(side_grad.norm())
                if updates==0 and (alpha_before!=0 or grad_side!=0):raise ValueError('Initial alpha0 task-gradient identity failed')
                opt.step();updates+=1
                side_after=torch.cat([p.detach().flatten() for p in side])
                updated_finite=all(bool(torch.isfinite(p).all()) for _,p in named)
                row=(updates,epoch,batch_index,float(loss.detach()),lr,alpha_before,float(alpha.detach()),grad_alpha,grad_side,
                    float((side_after-side_before).norm()),float(finite),float(updated_finite),float(frozen_gradients))
                trace.append(row);journal.write(json.dumps(dict(zip(TRACE_COLUMNS,row)),allow_nan=False)+'\n')
                if not updated_finite:raise ValueError('Nonfinite updated residual parameter')
                ce_total+=float(loss.detach())*len(pos);correct+=int((logits.argmax(1)==labels).sum())
            journal.flush()
            if device.type=='cuda':torch.cuda.synchronize()
            training_seconds+=time.perf_counter()-t
            if H.semantic_hash(model.backbone.state_dict())!=frozen_hash:raise ValueError('Frozen parameters or BN buffers changed during training')
            t=time.perf_counter();before_rng=H.semantic_hash(H.A.rng_state(order))
            logits=H.C.predict_tensor(model,tensors['val']['x'],job['batch'])
            ce=H.C.rx_ce(torch.as_tensor(logits,device=device),tensors['val']['y'],tensors['val']['rx'])
            validation_seconds+=time.perf_counter()-t
            t=time.perf_counter();copy_model=copy.deepcopy(model)
            with torch.no_grad():copy_model.residual.alpha.zero_()
            rollback=H.C.predict_tensor(copy_model,probe,8)
            rollback_exact=np.array_equal(rollback,reference_probe)
            del copy_model
            rollback_seconds+=time.perf_counter()-t
            if not rollback_exact or H.semantic_hash(model.backbone.state_dict())!=frozen_hash:raise ValueError('alpha0 rollback does not preserve frozen F0')
            if before_rng!=H.semantic_hash(H.A.rng_state(order)):raise ValueError('Validation/rollback advanced training RNG')
            best,best_epoch,improved=choose_checkpoint(best,best_epoch,ce,epoch)
            # epoch0 participates in checkpoint choice but does NOT step plateau.
            scheduler.step(ce)
            endpoints=(['selected'] if improved else [])+(['epoch10'] if epoch==10 else [])+(['final40'] if epoch==job['epochs'] else [])
            for endpoint in endpoints:save_readout(out,endpoint,model,logits,val,rx_names,epoch,ce)
            saved=snapshot(model,opt,scheduler,order,epoch,best,best_epoch,updates)
            torch.save(saved,out/'epoch_states'/f'epoch_{epoch:02d}.pt')
            record=dict(epoch=epoch,updates=updates,train_ce=ce_total/len(train['y']),train_accuracy=correct/len(train['y']),
                val_rx_ce=ce,best_epoch=best_epoch,alpha=float(alpha.detach()),lr_used=lr,lr_next=opt.param_groups[0]['lr'],
                relative_order_sha256=H.semantic_hash(positions),canonical_order_sha256=H.semantic_hash(train['keys'][positions]),
                class_day_order_sha256=H.semantic_hash(np.stack((train['y'][positions],train['day'][positions]),1)),
                frozen_backbone_sha256=frozen_hash,backbone_and_BN_unchanged=True,alpha0_rollback_bitwise=rollback_exact,
                model_sha256=H.semantic_hash(saved['model']),state_sha256=H.semantic_hash(saved))
            history.append(record);H.write(out/'HISTORY.json',history)
            H.write(out/'LIVE.json',dict(time=H.now(),job=job['id'],elapsed_seconds=time.perf_counter()-started,**record))
            if epoch==1 or epoch%5==0:print(job['id'],f'epoch={epoch}',f'CE={ce:.6f}',f'best={best_epoch}',f'alpha={float(alpha.detach()):+.7f}',flush=True)
    if updates!=job['max_gradient_updates']:raise ValueError('Fixed update budget failed')
    np.savez_compressed(out/'GRADIENT_TRACE.npz',columns=np.asarray(TRACE_COLUMNS),values=np.asarray(trace,dtype=np.float64))
    replay={}
    endpoints=('epoch0','selected','final40')+(('epoch10',) if job['epochs']>=10 else ())
    for endpoint in endpoints:
        ck=torch.load(out/(endpoint+'.pt'),map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
        prediction=H.C.predict_tensor(model,tensors['val']['x'],job['batch'])
        with np.load(out/(endpoint+'_val.npz'),allow_pickle=False) as f:replay[endpoint]=np.array_equal(prediction,f['logits'])
        if not replay[endpoint]:raise ValueError('Saved endpoint prediction replay failed')
    return dict(status='COMPLETE',job=job,epochs=len(history),updates=updates,best_epoch=best_epoch,best_val_rx_ce=best,
        frozen_backbone_sha256=frozen_hash,initial_task_side_gradient_l2=trace[0][8],initial_task_alpha_gradient=trace[0][7],
        first_side_parameter_delta_l2=trace[0][9],all_task_gradients_finite=all(row[10]==1 for row in trace),
        all_backbone_gradients_absent=all(row[12]==1 for row in trace),validation_endpoint_replay=replay,
        active_parameters=3201,total_parameters=sum(p.numel() for p in model.parameters()),
        costs=dict(training_seconds=training_seconds,validation_seconds=validation_seconds,rollback_probe_seconds=rollback_seconds,total_loop_seconds=time.perf_counter()-started),
        checkpoints={e:H.sha(out/(e+'.pt')) for e in endpoints})
