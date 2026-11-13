"""Synthetic acceptance, then admitted fit-only IQ, A0 replay and initial scales."""
import argparse,io,unittest
import numpy as np
import torch
from common import *
from p_models import A,C,build_aux,initial_gradient_scale,transfer
from p_fit import fit_aux
from p_data import auxiliary,downstream


def synth_data(samples,seed):
    rng=np.random.default_rng(seed);classes=8;rx=6
    y=np.repeat(np.arange(classes),rx*samples);r=np.tile(np.repeat(np.arange(rx),samples),classes)
    n=len(y);t=np.arange(256)[None,:]
    z=np.exp(2j*np.pi*(3+y[:,None]*5)*t/256)
    x=np.stack([z.real,z.imag],1)+rng.normal(size=(n,2,256))*.25
    x=(x/np.sqrt((x[:,0]**2+x[:,1]**2).mean(1))[:,None,None]).astype(np.float32)
    return dict(x=x,y=y.astype(np.int64),rx=r.astype(np.int64),day=np.tile(np.repeat([1,2],samples//2),classes*rx).astype(np.int64),
        keys=np.array([f'synthetic_{seed}_{i}' for i in range(n)]),pool_ids=np.arange(n))


def synthetic():
    validate('ADMISSION_REGISTRATION.json');committed([HERE/n for n in ('p_models.py','p_fit.py','p_data.py','p_accept.py','test_p.py')])
    import test_p
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream).run(unittest.defaultTestLoader.loadTestsFromModule(test_p))
    if not tests.wasSuccessful() or tests.testsRun!=7:raise ValueError(stream.getvalue())
    train=synth_data(48,42501);val=synth_data(16,42502);A.setup(24101)
    model=build_aux(0);scale=initial_gradient_scale(model,train,0);del model
    results={}
    for arm in ('A1','A2'):
        job=dict(id='engineering_'+arm,stage='aux',arm=arm,seed=0,epochs=1,synthetic_only=True)
        results[arm]=fit_aux(job,train,val,ART/job['id'],0 if arm=='A1' else scale['lambda_value'])
    assert results['A1']['initial_state_sha256']==results['A2']['initial_state_sha256']
    histories={a:read(ART/('engineering_'+a)/'HISTORY.json') for a in results}
    assert histories['A1'][0]['order_sha256']==histories['A2'][0]['order_sha256']
    ck=torch.load(ART/'engineering_A2/selected.pt',map_location='cpu',weights_only=False);m=transfer(ck['model'],0)
    assert sum(p.numel() for p in m.parameters())==82426
    for key,value in m.state_dict().items():
        if not key.startswith('classifier.'):assert torch.equal(value.cpu(),ck['model'][key])
    write('P_SYNTHETIC_QA.json',dict(status='PASS',CPU_geometry_and_sampler_tests=7,
        GPU_auxiliary_optimization_updates=sum(r['updates'] for r in results.values()),GPU_synthetic_fits=2,
        paired_initialization_and_sample_order_exact=True,relation_gradient_nonzero=True,
        geometry_collapse_guard_passed=True,EMA_BN_calibration_preserves_state=True,head_transfer_preserves_shared_state=True,
        lambda_example=scale,official_IQ_reads=0,scientific_fits=0,engineering_results=results))
    print('PASS seven synthetic tests, 24 GPU updates, exact matching orders and EMA transfer',flush=True)


def data():
    validate('ADMISSION_REGISTRATION.json')
    files=[HERE/n for n in ('P_SYNTHETIC_QA.json','p_models.py','p_fit.py','p_data.py','p_accept.py','test_p.py')];committed(files)
    assert read(HERE/'P_SYNTHETIC_QA.json')['status']=='PASS'
    if (HERE/'P_DATA_ATTEMPT.json').exists():raise RuntimeError('Keep initial admitted data preparation attempt')
    write('P_DATA_ATTEMPT.json',dict(time=now(),commit=git('rev-parse','HEAD'),role='aux72 and V0 fit12 only; no aux24 or V0 held-source values'))
    plan=read(HERE/'DATA_PLAN.json');aux=auxiliary(plan);ds=downstream(plan)
    tr,va=ds['fit_train'],ds['checkpoint_val'];assert len(tr['y'])==16246 and len(va['y'])==6954
    assert not set(tr['keys'])&set(va['keys'])
    assert len(aux['train']['y'])==20736 and len(aux['val']['y'])==6912
    assert not set(aux['train']['keys'])&set(aux['val']['keys'])
    assert len(set(aux['train']['keys']))==20736 and len(set(aux['val']['keys']))==6912
    ART.mkdir(exist_ok=True)
    for name,parts in (('P_AUX_TRAIN.npz',aux),('P_DOWNSTREAM_TRAIN.npz',{'train':tr,'val':va})):
        np.savez_compressed(ART/name,**{role+'_'+k:v for role,d in parts.items() for k,v in d.items()})
    replay=[];scales={}
    for seed in range(3):
        reference=plan['A0'][str(seed)];A.setup(24101+seed);model=A.build_model(24101+seed)
        initial=read(checked(reference['initialization'],'A0_initial_state_metadata',str(seed)))
        assert A.semantic_hash(model.state_dict())==initial['model_sha256']
        with np.load(checked(reference['sample_keys'],'A0_original_sample_roles',str(seed)),allow_pickle=False) as p:
            for r,d in (('train',tr),('val',va)):
                assert np.array_equal(p[r+'_keys'],d['keys']) and np.array_equal(p[r+'_ids'],d['pool_ids'])
        ck=torch.load(checked(reference['checkpoint'],'A0_frozen_selected_EMA',str(seed)),map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);before=A.semantic_hash(model.state_dict())
        logits=C.predict_tensor(model,torch.as_tensor(va['x'],device='cuda'),128)
        with np.load(checked(reference['validation'],'A0_original_fit_validation_predictions',str(seed)),allow_pickle=False) as p:
            assert np.array_equal(logits,p['logits']) and np.array_equal(va['keys'],p['keys']) and np.array_equal(va['pool_ids'],p['pool_ids'])
        assert before==A.semantic_hash(model.state_dict())
        replay.append(dict(seed=seed,status='PASS_EXACT_A0_REUSE',initial_model_exact=True,all_train_validation_keys_exact=True,
            validation_logits_bitwise_exact=True,frozen_model_state_unchanged=True))
        del model;A.setup(24101+seed);probe=build_aux(seed)
        scales[str(seed)]=initial_gradient_scale(probe,aux['train'],seed);del probe;torch.cuda.empty_cache()
    write('P_INITIAL_GRADIENT_SCALES.json',dict(status='FIT_ONLY_INITIAL_PROBES_NO_OPTIMIZER_UPDATES',scales=scales,
        source='First four fixed balanced fit batches per seed; no auxiliary val, aux24 or source-selection values'))
    write('P_DATA_ACCEPTANCE.json',dict(status='PASS',time=now(),A0_strict_reuse=replay,aux_fit_packets=20736,aux_val_packets=6912,
        downstream_train_packets=16246,downstream_checkpoint_val_packets=6954,aux24_IQ_rows_read=0,source_selection_IQ_rows_read=0,
        cache_sha256={n:sha(ART/n) for n in ('P_AUX_TRAIN.npz','P_DOWNSTREAM_TRAIN.npz')},
        initial_scales_sha256=sha(HERE/'P_INITIAL_GRADIENT_SCALES.json'),new_optimizer_updates=0,new_scientific_fits=0))
    print('PASS admitted training data; three bitwise A0 replays; three frozen initial gradient scales',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('synthetic','data'));args=p.parse_args();globals()[args.command]()
