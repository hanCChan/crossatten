"""Freeze twelve scientific fits and the data-free deviation before any training."""
import torch
from s22_core import HERE,ROOT,read,write,sha,now,baseline_path,git
from s22_models import verify_architecture,parameterization_audit,REFERENCES

def register():
    if (HERE/'registration.json').exists():raise ValueError('Registration already exists')
    write(HERE/'architecture_audit.json',verify_architecture(cuda=True))
    write(HERE/'parameterization_audit.json',parameterization_audit())
    references={}
    for arm in REFERENCES:
        for fold in [0,1]:
            for seed in [201,202,203]:
                p=baseline_path(arm,fold,seed);metric=read(p/'metrics.json')
                assert metric['epochs']==200 and metric['model_sha256']==sha(p/'model.pt')
                extra=['evaluation.json','diagnostics_dev_val.npz','diagnostics_confirm.npz'] if arm in ['H1','H2'] else ['target_metrics.json','preds_dev_val.npz','preds_confirm.npz']
                references[f'{arm}_f{fold}_s{seed}']=dict(path=str(p),sha256={name:sha(p/name) for name in ['model.pt','config.json','metrics.json','history.csv']+extra})
    jobs=[dict(phase='p1',dataset='Shen',arm=arm,fold=fold,seed=seed) for fold in [0,1] for seed in [201,202,203] for arm in ['S1','S2']]
    write(HERE/'jobs_p1.json',jobs)
    files=list((ROOT/'lib').rglob('*.py'))+[p for directory in ['13-mixer-cfo-mechanisms','15-hstem-attribution','21-hstem-mechanisms',HERE.name] for p in (ROOT/directory).glob('*.py')]+[ROOT/'configs/protocol.json',ROOT/'15-hstem-attribution/architecture_search.json']
    write(HERE/'registration.json',dict(stage='S22',time=now(),base_git=git('rev-parse','HEAD'),maximum_new_fits=12,jobs=jobs,references=references,
        primary_comparisons=[['H1','S1'],['S1','P0'],['H0','S2'],['S2','H2']],
        primary='RX equal dev_val accuracy',gate=dict(practical_delta_pp=.5,minimum_positive_pairs=5,fold_directions='both positive',receiver_bootstrap_lower='strictly >0',worst_receiver_delta_pp=-2,equivalence='conditional RX95% interval entirely inside [-0.5,+0.5]pp and both fold mean differences within +/-0.5pp; otherwise do not declare equivalence'),
        intervention_S1='Fixed linear align_corners=False interpolation 256->340, scale sqrt(256/340), then linear340->256; stride256,padding0; same learned kernel, biases, group8 and 80->64 projection',
        design_deviation='Original center-crop is rank194 with62 inactive taps (19840 nominal parameters disconnected). Replaced before training or target reads with rank256 round-trip smoothing. This is not the originally proposed center-crop S1. No trained crop candidate.',
        intervention_S2='Replace precisely three tokenizer BN by channel affine with weight1,bias0; preserve352 affine parameters; no BN running buffers or batch statistics in these three modules',
        training=dict(recipe='Unmodified S13 Shen200',precision='FP32',batch=64,epochs=200,awgn='U[0,50)dB',optimizer='AdamW1e-3,wd5e-4,warmup5/cosine',checkpoint='minimum source RX equal CE',paired_seeds=[201,202,203],torch=torch.__version__,gpu=torch.cuda.get_device_name(0)),
        secondary=['RX equal macroF1 and cross-entropy','perRX and TXxRX errors/margins','normalized TX centroid distances','params/MACs/resident FP32 batch1,batch128 latency'],
        robustness=dict(models=['H0','P0'],data='All source-validation packets only;not unseen-RX evidence',conditions=['clean','snr20','snr10','snr0','shift-4','shift-1','shift1','shift4'],random_stream='stable_seed(S22-robust,fold,seed,condition);same for H0/P0',shift='zero-padded sample shifts;not circular roll',use='descriptive only;no model selection or tuning'),
        recovery='Before fit:24 baselines strict load and exact source metrics/CE. After all12 new checkpoints committed:216 archived baseline target arrays bit-exact; new target evaluation only then.',
        access='No blind RX or WiSig data. Existing dev primary; historical confirm descriptive. Blind qualification from existing docs/logs only.',
        stop='Only12 new fits. No H3, crop fit, kernels, losses, anti-alias training, seed replacement, blind evaluation or outcome-driven model replacement. Technical failures preserved; no automatic scientific retry.',
        uncertainty='Receiver-cluster conditional bootstrap,5000 draws fixed seed14014; seeds/folds not independent receivers; four predeclared comparisons reported together without multiplicity correction',
        limitations=['Round-trip is an invertible kernel reparameterization; it changes initialization/smoothing/optimization and frequency response, not only support','H1-S1 is not a pure support intervention; physical receptive field is global because TrueIB/FFT','Affine-only does not isolate BN training statistics vs running statistics vs optimization dynamics','CUDA linear interpolation backward may be nondeterministic under original warn_only setting; shared inputs/noise seeds remain paired','Historical confirm and paper1 blind history are not new independent evidence'],
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},artifacts_sha256={name:sha(HERE/name) for name in ['architecture_audit.json','parameterization_audit.json','jobs_p1.json']}))
    print('Registered12 fits,24 baseline identities,full-rank S1 and affine S2. Commit required before training.',flush=True)

if __name__=='__main__':register()
