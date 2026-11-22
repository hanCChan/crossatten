"""Freeze one chosen arm and six jobs before any S26 source training."""
from pathlib import Path
import subprocess
import torch
from s26_common import HERE,ROOT,read,sha,write,now


def main():
    assert not (HERE/'registration.json').exists()
    assert read(HERE/'GATE.json')['status']=='PASS'
    parent=read(ROOT/'25-tifs-integration/registration.json')
    code=list(HERE.glob('s26_*.py'))+list((ROOT/'lib').rglob('*.py'))
    code+=[ROOT/folder/name for folder,names in {
        '13-mixer-cfo-mechanisms':['train.py','models.py','core.py','data.py','metrics.py','wifi.py'],
        '15-hstem-attribution':['s15_models.py','s15_core.py'],
        '21-hstem-mechanisms':['s21_models.py','s21_core.py']}.items() for name in names]
    evidence=[ROOT/'configs/protocol.json',ROOT/'15-hstem-attribution/architecture_search.json',
        ROOT/'25-tifs-integration/registration.json',ROOT/'25-tifs-integration/LONG_PATCH_AUDIT.json',
        ROOT/'25-tifs-integration/FINAL_AUDIT.json']
    jobs=[dict(arm='LP',fold=f,seed=s) for f in (0,1) for s in (201,202,203)]
    for r in parent['models']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
    write(HERE/'registration.json',dict(stage='S26_LONG_PATCH_CONTROL',time=now(),
        base_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        authority='2026-11-22 user asks to execute the attached follow-up plan, including the single Long-Patch six-fit control. Prior S25 no-fit restriction remains historical, not changed retroactively.',
        maximum_new_fits=6,jobs=jobs,architecture=dict(arm='LP',groups=8,mid=64,kernel=340,stride=256,padding=42,
            params=138346,reference_h0_params=133690,relative_parameter_increase=.034826838207794086,
            local_support=350,token_shape=[32,64],bn_layers=2,rank_upper_bound=64),
        prior_exposure='All S15-S25 target results have been observed. This is a new developmental control, not an independent confirmatory experiment.',
        selection='The single groups8/width64 candidate proposed in S25; no candidate accuracy search, extra arms or replacement seeds.',
        training=dict(epochs=200,batch=64,optimizer='AdamW',lr=.001,weight_decay=.0005,warmup_epochs=5,
            schedule='Original per-step cosine',augmentation='Original AWGN0..50dB',checkpoint='minimum clean source RX-equal CE',
            data_seed='stable_seed(P-H5x160,fold,seed)',permutation_rng='cuda seed=data_seed',noise_rng='cuda seed=data_seed+1',
            tokenizer_rng='CPU stable_seed(S26-LongPatch,data_seed); CPU/CUDA training RNG otherwise preserved',
            tf32_matmul=False,tf32_cudnn=True,recovery='Atomic epoch model/AdamW/RNG/history snapshot; continue same seed/epoch, no restart search'),
        permissions=dict(source='Original200 packets/TX/RX,160train40val,5sourceRX/fold',
            dev='Report only after all6 checkpoint/config/history hashes sealed and committed; LP/H0/P0 common GPU inference',
            confirm=False,old_blind=False,new_independent_capture=False,ask_strict_ib=False),
        statistics=dict(primary='H0 minus LP RX-equal Accuracy pp',secondary='LP minus P0 Accuracy pp; F1, perRX/TX and costs descriptive',
            pairing='Same fold/seed and10 unique held RX; average3seeds withinRX',bootstrap_replicates=10000,bootstrap_seed=26026,
            uncertainty='Conditional RX bootstrap only; no new acquisition/architecture-selection uncertainty or multiplicity correction',
            practical_rule='For bounded consistent H0 advantage: mean>=0.5pp, conditional lowerCI>0, at least5/6 positive fit differences, both fold means positive. Otherwise report criterion not met; not proof of equality.',
            interpretation='LP matching H0 narrows hierarchy attribution; H0 advantage limits support-only sufficiency within this arm. Width, grouping factorization, BN, nonlinear depth and compute remain confounded in either outcome.'),
        references=parent['models'],source_caches=parent['source_caches'],
        canonical_training_sha256=sha(ROOT/'13-mixer-cfo-mechanisms/train.py'),
        gate_sha256=sha(HERE/'GATE.json'),torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in code},
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence}))
    print('Frozen: one Long-Patch arm, six fits; commit registration/code before source IO.',flush=True)


if __name__=='__main__':main()
