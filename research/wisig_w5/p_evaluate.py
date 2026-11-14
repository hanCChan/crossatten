"""Only sealed models may read internal held-source IQ or the 24 auxiliary TX."""
import argparse,shutil,time
import numpy as np
import torch
from common import *
from p_models import A,C,build_aux,pooled
from p_data import downstream,auxiliary


def register():
    validate();committed([HERE/'P_MODEL_SEAL.json'])
    seal=read(HERE/'P_MODEL_SEAL.json');assert seal['new_fits']==6 and seal['new_gradient_updates']==28200
    assert read(HERE/'P_TRAINING_QA.json')['status']=='PASS_PRE_HELD_TRAINING_AND_TRANSFER_AUDIT'
    files=[HERE/n for n in ('p_evaluate.py','p_analyze.py','metrics.py','p_data.py','p_models.py','common.py','P_MODEL_SEAL.json',
                            'DATA_PLAN.json','PREREGISTRATION.zh-CN.md','EXECUTION_REGISTRATION.json','P_TRAINING_QA.json')]
    write('P_EVALUATION_REGISTRATION.json',dict(status='SEALED_MODELS_AND_FIXED_ENDPOINTS_BEFORE_NEW_HELD_SOURCE_READOUT',time=now(),
        files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},expected_predictions=6,
        held_source_packets_per_model=11500,aux24_diagnostic_packets=0,
        primary='RX-equal Accuracy, three paired seeds, predefined three W5-A gates',
        auxiliary_TX_diagnostic='Not authorized in W5-A',
        forbidden_actual_target_or_D3_D4_or_historical_held6_IQ=True))
    print('Registered sealed evaluation for six matched source-selection predictions; commit before predict',flush=True)


def latency(model):
    g=torch.Generator(device='cuda').manual_seed(42551);x=torch.randn(128,2,256,device='cuda',generator=g);records={}
    with torch.inference_mode():
        for n in (1,128):
            for _ in range(20):model(x[:n])
            torch.cuda.synchronize();events=[]
            for _ in range(100):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();model(x[:n]);b.record();events.append((a,b))
            torch.cuda.synchronize();ms=np.array([a.elapsed_time(b) for a,b in events])
            records[str(n)]=dict(mean_ms=float(ms.mean()),median_ms=float(np.median(ms)),p95_ms=float(np.quantile(ms,.95)),iterations=100,warmup=20)
    return dict(batch=records,scope='Entire frozen model incl. RF frontend; fixed synthetic CUDA-resident input; no host/disk copy',
        parameters=sum(p.numel() for p in model.parameters()),device=torch.cuda.get_device_name(),precision='FP32; TF32 false')


def predict():
    validate('P_EVALUATION_REGISTRATION.json')
    if (HERE/'P_EVALUATION_ATTEMPT.json').exists():raise RuntimeError('Keep original sealed evaluation attempt')
    seal=read(HERE/'P_MODEL_SEAL.json');plan=read(HERE/'DATA_PLAN.json')
    for run in seal['runs']:
        for field in ('complete','selected_checkpoint','selected_predictions'):checked(run[field],'verify_training_seal',run['job'])
    write('P_EVALUATION_ATTEMPT.json',dict(time=now(),commit=git('rev-parse','HEAD'),
        registration_sha256=sha(HERE/'P_EVALUATION_REGISTRATION.json'),model_seal_sha256=sha(HERE/'P_MODEL_SEAL.json')))
    A.setup(24101);data=downstream(plan,held=True)['source_selection'];x=torch.as_tensor(data['x'],device='cuda')
    names=np.array([plan['source']['rx_names'][int(r)] for r in data['rx']]);predictions={};costs={};states=[]
    runmap={r['job']:r for r in seal['runs']}
    for seed in range(3):
        # Rotate the identical-architecture latency order across seeds.
        methods=list(('A0','A1'));methods=methods[seed%2:]+methods[:seed%2]
        for arm in methods:
            key=f'{arm}/{seed}';path=ART/f'P_{arm}_S{seed}_selection.npz';A.setup(24101+seed);m=A.build_model(24101+seed)
            if arm=='A0':
                ref=plan['A0'][str(seed)];checkpoint=checked(ref['checkpoint'],'A0_frozen_latency_checkpoint',key)
                old=checked(ref['selection'],'reuse_A0_sealed_source_selection_predictions',key)
                with np.load(old,allow_pickle=False) as p:
                    assert np.array_equal(p['y'],data['y']) and np.array_equal(p['rx'],names) and np.array_equal(p['day'],data['day'])
                    assert np.array_equal(p['keys'],data['keys']) and np.array_equal(p['pool_ids'],data['pool_ids'])
                shutil.copyfile(old,path)
            else:checkpoint=checked(runmap[f'{arm}_S{seed}_downstream']['selected_checkpoint'],'selected_downstream_EMA_readout',key)
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);m.eval().requires_grad_(False)
            before=A.semantic_hash(m.state_dict())
            if arm!='A0':C.save_predictions(path,C.predict_tensor(m,x,128),data,plan['source']['rx_names'])
            costs[key]=latency(m);assert costs[key]['parameters']==82426
            after=A.semantic_hash(m.state_dict());assert before==after
            predictions[key]=record(path);states.append(dict(model=key,before_state_sha256=before,after_state_sha256=after,unchanged=True))
            del m;torch.cuda.empty_cache();print('Saved source-selection predictions',key,flush=True)
    del x
    write('P_INFERENCE_COST.json',dict(status='COMPLETE',records=costs,architecture_identical=True,latency_is_conditional_not_an_efficiency_claim=True))
    write('P_PREDICTION_RECEIPT.json',dict(status='ALL_SIX_DOWNSTREAM_PREDICTIONS_FROZEN',time=now(),
        predictions=predictions,downstream_frozen_states=states,auxiliary_frozen_states=[],
        model_seal_sha256=sha(HERE/'P_MODEL_SEAL.json'),
        evaluation_registration_sha256=sha(HERE/'P_EVALUATION_REGISTRATION.json'),
        forbidden_actual_target_D3_D4_historical_held6_IQ_reads=0,neural_fits=0))
    print('PREDICTIONS AND FEATURES SEALED; final metrics have not been computed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('register','predict'));args=p.parse_args();globals()[args.command]()
