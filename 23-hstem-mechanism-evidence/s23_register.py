"""Register bounded zero-fit analysis before any new source inference."""
import torch
from s23_core import HERE,ROOT,read,write,sha,now,git,baseline_path

def register():
    if (HERE/'registration.json').exists():raise ValueError('Registration already exists')
    references={}
    for arm in ['H0','P0','H1','H2','S1','S2']:
        for fold in [0,1]:
            for seed in [201,202,203]:
                p=baseline_path(arm,fold,seed);metric=read(p/'metrics.json')
                assert metric['epochs']==200 and metric['model_sha256']==sha(p/'model.pt')
                references[f'{arm}_f{fold}_s{seed}']=dict(path=str(p),sha256={name:sha(p/name) for name in ['model.pt','config.json','metrics.json','history.csv']})
    files=[p for directory in ['lib','13-mixer-cfo-mechanisms','15-hstem-attribution','21-hstem-mechanisms','22-mechanism-closure',HERE.name] for p in ((ROOT/directory).rglob('*.py') if directory=='lib' else (ROOT/directory).glob('*.py'))]
    files+=[ROOT/'configs/protocol.json',ROOT/'15-hstem-attribution/architecture_search.json']
    write(HERE/'registration.json',dict(stage='S23',time=now(),base_git=git('rev-parse','HEAD'),maximum_new_fits=0,maximum_new_architecture_candidates=0,references=references,
        data='Only original source-validation40/TX/RX,2000 packets/fold;source HDF5 schema/attribute names only with zero waveform rows;no source-training analysis,dev,confirm,blind or WiSig raw samples',
        equivalence=dict(models=['S1 materialized as standard256 Conv1d','H0 with exactly three tokenizer Conv-BN pairs folded'],fits_per_conversion=6,full_source_validation=True,atol=1e-4,rtol=1e-4,report=['max/rms/relative logit error','prediction agreement','CE','structure','same learned function vs different training path'],failure='Record failure;do not loosen tolerance'),
        kernels=dict(models=['P0','S1','H1'],nfft=1024,bands_cycles_per_feature_sample=[0,.125,.25,.5],scope='Effective first grouped temporal kernel;not physical RF transfer function'),
        local_gradients=dict(models=['P0','S1','H1','H0','S2','H2'],data='First source-validation packet in each RXxTX cell;50 packets/fold',mode='Frozen checkpoint eval-mode CE sensitivity;not recorded historical training gradients',optimizer_steps=0),
        timing=dict(models=['H0','P0'],shifts=[-4,-1,1,4],kinds=['zero','circular'],full_source_validation=True,fft_locked_diagnostics='Circular +/-4 with only the FFT-magnitude channel replaced by its unshifted counterpart;internal intervention,not physical input',layer_probe_packets=50,metrics=['Accuracy/F1/CE/true-class margin','perRX/TX errors','relative feature response and eligible integer-stride equivariance','raw boundary energy and closure','frontend commutation and Fourier shift theorem']),
        interpretation=dict(boundary_dominant='Circular4 mean drop <=5pp and <=20% of zero4 mean drop;descriptive signal only',zero_boundary_insufficient='Circular4 mean drop >=20pp with positive drops in >=5/6 fits for both signs;not proof of physical acquisition error',physical_validity='If synchronization/timestamps/adjacent samples unavailable,never equate circular or zero shifts with real capture errors',decision='A:write within demonstrated aligned-protocol scope,or B:separate new-method proposal only if a practically grounded defect is supported;do not start new fits'),
        latency=dict(precision='FP32',batches=[1,128],repeats=3,warmup=10,measurements=50,scope='Resident IQ to logits;conditional desktop GPU timing'),
        trajectory='All36 archived200-epoch histories;no new checkpoint selection;best epochs audited only',
        frontend_fact='Actual FFT input is normalized magnitude,not explicit FFT phase;time-domain atan2 phase is a different channel;time/frequency indices are concatenated',
        paper1_source=dict(path='refs/OOB_IOTJ.pdf',sha256=sha(ROOT/'refs/OOB_IOTJ.pdf'),read_only=True),
        stop='S23 is the last bounded mechanism analysis;no automatic S24,optimizer,new seed,new loss,augmentation or anti-alias training',
        environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0)),uncertainty='Exploratory source-validation evidence;receiver units;no independent confirmation or causal hardware attribution',
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},artifacts_sha256={}))
    print('Registered zero fits;36 immutable checkpoints;source-only probes and fixed equivalence tolerances',flush=True)

if __name__=='__main__':register()
