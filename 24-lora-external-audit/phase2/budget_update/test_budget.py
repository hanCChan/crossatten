import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import json
from unittest.mock import patch
import numpy as np
import torch
import common
from common import HERE,PHASE,STAGE,read,atomic_json,jobid,P
from inference import metrics,h0_model

def main():
    torch.set_num_threads(2);r=read(PHASE/'pilot_registry.json');checks=[]
    assert len(common.cohort_jobs('baseline48',r))==48
    assert len(common.cohort_jobs('native_ask6',r))==6
    assert len(common.cohort_jobs('retained54',r))==54
    checks.append('cohort partitions48+6 retain both folds/all seeds; whole Strict ASK arm excluded')
    active={jobid(j) for j in common.cohort_jobs('retained54',r)}
    with patch('common.policy',return_value=(dict(active_job_ids=active),r)):
        for j in r['jobs']:
            if j['method']=='ask' and j['view']=='strict_ib':
                try:common.authorized_training(j)
                except PermissionError:pass
                else:raise AssertionError('deferred training accepted')
        checks.append('all six deferred Strict ASK jobs denied by amended training entrypoint')
        with patch('numpy.load',side_effect=AssertionError('waveform IO happened before rejection')):
            for rx in P['blind_receivers_never_open']+P['confirm_receivers']+P['folds'][0]['source']:
                try:common.dev_cell('ce__native__f0__s201',rx,'baseline48')
                except PermissionError:pass
                else:raise AssertionError('forbidden/wrong-fold RX opened')
            try:common.dev_cell('ask__strict_ib__f0__s201','rtl_2','baseline48')
            except PermissionError:pass
            else:raise AssertionError('deferred prediction accepted')
        checks.append('blind/confirm/source-role/deferred-job denied before waveform IO')
        with patch('common.committed',side_effect=PermissionError('uncommitted')):
            try:common.dev_cell('ce__native__f0__s201','rtl_2','baseline48')
            except PermissionError:pass
            else:raise AssertionError('uncommitted seal accepted')
        checks.append('legal Dev requires committed canonical cohort seal')
    y=np.tile(np.arange(10),2);logits=np.eye(10,dtype=np.float32)[y]*10
    a=metrics(logits,y);assert a['acc']==1. and a['macro_f1']==1.
    logits[y==0]=np.eye(10,dtype=np.float32)[1]*10
    b=metrics(logits,y);assert abs(b['acc']-.9)<1e-12 and abs(b['macro_f1']-0.8666666666666666)<1e-12
    order=np.random.default_rng(241118).permutation(len(y));c=metrics(logits[order],y[order])
    assert b['acc']==c['acc'] and b['macro_f1']==c['macro_f1']
    checks.append('10-class Accuracy/Macro-F1 hand examples and packet order invariance')
    model=h0_model('h0__native__f0__s201','cpu')
    iq=torch.from_numpy(np.load(STAGE/'local/source_probe.npz')['iq'][:2].copy())
    with torch.no_grad():
        batch=model(iq);single=torch.cat([model(iq[i:i+1]) for i in range(2)])
    error=float((batch-single).abs().max())
    assert torch.allclose(batch,single,atol=1e-4,rtol=1e-4) and torch.equal(batch.argmax(1),single.argmax(1))
    checks.append('frozen H0 CPU strict weight load and source-probe batch/single consistency')
    atomic_json(HERE/'TESTS.json',dict(status='PASS',checks=checks,source_probe_packets=2,h0_batch_single_max_abs=error,
                dev_reads=0,confirm_blind_reads=0,optimizer_steps=0))
    print('PASS',len(checks),'budget/permission/metrics/frozen-inference tests; no Dev packets opened.')

if __name__=='__main__':main()
