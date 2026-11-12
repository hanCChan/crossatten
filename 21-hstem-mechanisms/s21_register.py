"""Freeze the complete twelve-fit experiment before any optimization."""
import torch
from s21_core import HERE, ROOT, read, write, sha, now, baseline_path, git
from s21_models import verify_architecture

def register():
    if (HERE/'registration.json').exists(): raise RuntimeError('Registration already exists')
    audit=verify_architecture(); write(HERE/'architecture_audit.json',audit)
    references={}
    for arm in ['H0','P0']:
        for fold in [0,1]:
            for seed in [201,202,203]:
                p=baseline_path(arm,fold,seed); metric=read(p/'metrics.json')
                assert sha(p/'model.pt')==metric['model_sha256']
                assert metric['epochs']==200
                references[f'{arm}_f{fold}_s{seed}']=dict(path=str(p),sha256={name:sha(p/name) for name in ['model.pt','config.json','metrics.json','history.csv','target_metrics.json','preds_dev_val.npz','preds_confirm.npz']})
    jobs=[dict(phase='p1',dataset='Shen',arm=arm,fold=fold,seed=seed) for fold in [0,1] for seed in [201,202,203] for arm in ['H1','H2']]
    write(HERE/'jobs_p1.json',jobs)
    files=list((ROOT/'lib').rglob('*.py'))+list((ROOT/'13-mixer-cfo-mechanisms').glob('*.py'))+list((ROOT/'15-hstem-attribution').glob('*.py'))+list(HERE.glob('*.py'))+[ROOT/'configs/protocol.json',ROOT/'15-hstem-attribution/architecture_search.json']
    record=dict(stage='S21',time=now(),base_git=git('rev-parse','HEAD'),jobs=jobs,maximum_new_fits=12,
        environment=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),precision='FP32',training_loop='unchanged S13 train.fit',checkpoint='minimum source RX equal CE',epochs=200,batch=64,awgn='U[0,50)',order_seed='stable_seed(P-H5x160,fold,seed); CUDA generator',seed_pairing=[201,202,203]),
        interventions=dict(H1='Interpolate learned grouped [80,4,256] kernel to340, linear align_corners=False, scale sqrt(256/340), stride256,padding42; same biases/pointwise conv',H2='Replace exactly three tokenizer BatchNorm1d with Identity; retain all GELU, convolutions, strides, head',H3='Registered conditional candidate only; prohibited in this twelve-fit run'),
        primary=dict(metric='RX equal dev_val accuracy',recovery='(Acc(H1)-Acc(P0))/(Acc(H0)-Acc(P0)); ratio of aggregate means',signal='recovery >=0.50 and mean H1-P0 >0 in both folds',denominator_rule='if H0-P0 <=0 or <0.25pp, recovery undefined; report raw differences',H2='Report H2-H0 paired change; descriptive important-BN signal if mean <=-1pp, at least5/6 negative, both folds negative',stop='Stop after12, even if unclear. Do not automatically run H3, anti-alias models, kernel sweeps, losses, or WiSig fits'),
        secondary=['RX equal macro-F1','per RX and TXxRX error deltas','true class logit margin including misclassified packets','cross-RX TX-centroid distance structure, descriptive only','params, counted MACs, resident FP32 latency'],
        evaluation='Seal and commit all12 source-CE checkpoints before any new target data read. Existing dev RX primary. Historical confirm descriptive only. No blind RX.',
        inference_recovery='Before fitting: strict-load all12 baseline weights and recover source validation CE/accuracy. After sealing: reproduce archived float32 dev_val/confirm predictions exactly at batch128; fail closed on mismatch.',
        uncertainty='RX-cluster bootstrap conditional on fitted seeds; fold/seed repeats are not independent receivers; no multiplicity correction; no causal identification from margin or geometry',
        limitations=['H1 changes filter frequency response, interpolation smoothness and MACs as well as local support','H2 tests joint effect of BN normalization, affine parameters and optimization; does not prove RX removal','Local CNN support is not raw-IQ physical receptive field because of TrueIB and FFT','No new blind target evidence; novelty is not established by these ablations'],
        references=references,code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},
        artifacts_sha256={name:sha(HERE/name) for name in ['architecture_audit.json','jobs_p1.json']})
    write(HERE/'registration.json',record)
    print('Registered12 fits; baseline artifacts and model geometry verified; commit required before execution.',flush=True)

if __name__=='__main__': register()
