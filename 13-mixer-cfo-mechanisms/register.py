"""Pre-outcome registration and bounded job matrices."""
import subprocess
import sys
from pathlib import Path
from core import HERE, ROOT, git, once, sha, now
from models import ARMS, build, nparams


def main():
    base = "f3c9729a0d2b4bcbafafcaf28683b3f5f5770a1f"
    if subprocess.run(['git','merge-base','--is-ancestor',base,'HEAD'],cwd=ROOT).returncode:
        raise ValueError('S13 must descend from the checked S12 branch')
    arms={name:dict(cfg,params=nparams(build(name,123))) for name,cfg in ARMS.items()}
    target=arms['A3_rfhstu']['params']
    for name in ['A1_mha_matched','A2_transformer_matched']:
        assert abs(arms[name]['params']/target-1)<.05
    code=list(HERE.glob('*.py'))+list((HERE/'tests').glob('*.py'))
    code+=list((ROOT/'lib/xa').glob('*.py'))+list((ROOT/'lib/legacy_rfhstu').glob('*.py'))
    code+=[ROOT/'09-phase-invariance/phase_features.py',ROOT/'configs/protocol.json']
    registration=dict(stage='S13',created_at=now(),base_commit=base,branch=git('branch','--show-current'),
        authorization='User requested attached plan execution; S13 schedules P0/P1 and conditional P2. P3 needs a separate source-only UDA protocol after mechanism evidence.',
        proposal_sha256=sha(HERE/'PROPOSAL_ORIGINAL.txt'),arms=arms,
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(code)},
        p1=dict(folds=[0,1],seeds=[201,202,203],formal_fits=48,epochs=200,batch=64,
                optimizer='AdamW lr1e-3 wd5e-4 warmup5/cosine',awgn_db=[0,50],
                checkpoint='minimum RX-equal source packet validation CE',evaluation_batch=128,
                target_rule='all source checkpoints sealed and committed before dev_val/used-confirm access',
                paired_seed='sha256(P-H5x160|fold|seed)',primary='RX-equal accuracy; every arm reported',
                primary_pairs=[['A3_rfhstu',a] for a in ['A0_cnn','A1_mha','A1_mha_matched','A2_transformer','A2_transformer_matched','A4_softmax_gate','A5_silu_gate']],
                descriptive_gate=dict(mean_delta_pp=1,positive_pairs=5,pairs=6,worst_receiver_delta_pp=-2),
                limitations=['receiver bootstrap conditional on fitted seeds; no multiple-comparison correction',
                             'matched transformer FFN32 is a bottleneck; natural FFN256 also reported',
                             'MHA matched width80 changes attention head width; outputs remain32x64',
                             'SiLU-score control only changes RF mixer score, not a full original HSTU reproduction']),
        p0=dict(datasets=['ManyRx','ManySig'],folds=[0,1,2,3],group='first',seed=15001,
                fresh_source_fits=8,epochs_cap=40,batch=128,optimizer='Adam lr1e-3 Plateau(.1,5), earlystop10',
                data='legal source RX days1-2; internal 70/30 fit/packet validation; held-source for diagnostics only',
                historical_weights='No download/replay. New source diagnostic fits explicitly labeled fresh.',
                diagnostic_samples_per_tx_rx_day=16,feature_pool_bins=8,
                probes='balanced within-cell 50/50 packet split; TX-held-RX transfer; TX-conditioned RX logistic probe',
                rho='unit-length pooled features; equal TX/RX/day cell centroid scatters',
                interventions=['clean-fit-only BN recalibration','1% relative stem perturbation','1% relative mixer perturbation'],
                no_adaptation_claim='P3 and held-RX fitting not executed; sensitivity is not proof of shallow adaptation'),
        cfo=dict(fs_hz=25e6,lag=20,prefix=[0,160],confidence_threshold=.7,
                 shifts_hz=[-10000,-5000,0,5000,10000],added_noise_snr_db=[30,20,10],
                 estimator_gate=dict(min_reliable_fraction=.7,max_added20db_median_error_hz=500,
                                     max_added20db_p95_error_hz=2000),
                 interpretation='observed CFO mixes TX/RX/channel effects; synthetic additivity is not natural CFO ground truth'),
        p2=dict(status='CONDITIONAL_NOT_REGISTERED_FOR_EXECUTION',variants=['B0','B1_cfo','B2_corrected','B2_dual','B2_dual_raw'],
                prospective_pilot_cap=40,pilot_seed=16001,pilot_group='first',epochs_cap=40,
                admission=dict(manyrx_mean_pp=1,manyrx_positive_folds=3,worst_rx_pp=-2,
                               manysig_mean_pp=0,manysig_folds_ge_minus1pp=3),
                confirmation='separate source-only registration; fresh seeds16011/16012, last held-source group, paired new controls',
                target='no WiSig target access implemented in S13; any final CRD requires source confirmation and committed separate final matrix'),
        p3=dict(status='NOT_RUN',reason='independent UDA/TTA protocol deferred until source mechanism evidence and separate registration'),
        historical_preservation='No writes outside S13 except Git branch/commits; no changes to prior STOPs',
        execution=dict(gpu=0,concurrent_fits=1,device='RTX4060 Laptop 8GB',precision='FP32',automatic_retry=False))
    once(HERE/'registration.json',registration)
    p1=[dict(phase='p1',dataset='Shen',arm=a,fold=f,seed=s) for f in [0,1] for s in [201,202,203] for a in ARMS]
    p0=[dict(phase='p0',dataset=d,arm='B0',fold=f,group='first',seed=15001) for d in ['ManyRx','ManySig'] for f in range(4)]
    smoke=[dict(phase='smoke',dataset='Shen',arm=a,fold=0,seed=13001) for a in ARMS]
    smoke += [dict(phase='smoke',dataset=d,arm='B0',fold=0,group='first',seed=13001) for d in ['ManyRx','ManySig']]
    for name,jobs in [('p1',p1),('p0',p0),('smoke',smoke)]:once(HERE/f'jobs_{name}.json',jobs)
    print(f'Registered P1={len(p1)}, P0={len(p0)}, smoke={len(smoke)}. Commit registration/code BEFORE data execution.')


if __name__=='__main__':main()
