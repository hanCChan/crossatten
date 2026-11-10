"""Prospective follow-up after D0; preserves the original S17 stop."""
import sys
from s17b_core import HERE,ROOT,read,write,sha,now,frozen,commit,validate,committed

def design():
    assert read(HERE/'ENGINEERING_TESTS.json')['status']=='PASS'
    assert read(ROOT/'17-rx-distortion/environment/WINDOWS_GPU_RECOVERY.json')['status']=='PASS'
    jobs=[dict(phase='pilot',dataset='ManyRx',arm='M0',fold=f,group='first',seed=18101,split_seed=16001) for f in range(4)]
    frozen(HERE/'jobs_pilot.json',jobs);frozen(HERE/'registration_pilot.json',dict(time=now(),stage='baseline-only before D1',matrix_sha256=sha(HERE/'jobs_pilot.json'),selected_tau=None))
    frozen(HERE/'registration_baseline.json',read(HERE/'registration_pilot.json'))
    code=list(HERE.glob('*.py'))+list((ROOT/'17-rx-distortion').glob('*.py'))+list((ROOT/'13-mixer-cfo-mechanisms').glob('*.py'))+list((ROOT/'lib').rglob('*.py'))+[ROOT/'09-phase-invariance/phase_features.py']
    evidence=[HERE/'README.zh-CN.md',HERE/'ENGINEERING_TESTS.json',ROOT/'17-rx-distortion/decision_d0.json',ROOT/'17-rx-distortion/environment/WINDOWS_GPU_RECOVERY.json']+list((ROOT/'17-rx-distortion').glob('D0_fold*.json'))
    frozen(HERE/'registration.json',dict(time=now(),stage='S17B',
        rationale='Post-D0 phase-only follow-up authorized by ongoing receiver-mechanism work; original amplitude/IQ stop is preserved',
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(code)},evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence},
        d1=dict(levels=[.25,.5,1.],draws=2,controls=['same EVM AWGN','same EVM global phase'],
            physical=dict(fir_taps=17,pad_length=512,known_delay=8,residual_l1=.25,coefficient_bound_radians=.15),
            preservation=dict(all_folds=True,max_accuracy_drop_pp=1,max_f1_drop_pp=1,max_worst_tx_drop_pp=5),
            bounds=dict(min_mean_evm=.002,max_evm_q95=.10,max_frequency_ripple_q95_db=.25),
            source_phase_subspace=dict(source='Frozen S17 training-only RX profiles',rank=3,feature_std_floor=1e-4,
                coverage_min=.10,alignment_min=.15,alignment_advantage_over_noise_min=.02,required_folds=3,
                clean_magnitude_support_fixed=True),select='largest preregistered dose meeting all gates; no extra levels'),
        training=dict(model='HStem+ordinary MHA 82426 ManyRx parameters, no new trainable modules',epochs_max=40,batch=128,
            optimizer='Adam1e-3;sourceCE plateau .1/patience5/min1e-6;earlywait10',precision='FP32',augmentation_probability=.5,
            seed=18101,pilot_split=16001,checkpoints='source receiver-equal CE; all source checkpoints sealed before held-source reads'),
        pilot_gate=dict(controls=['M0','N_phase','G_phase'],manyrx_mean_gain_pp=1,positive_folds=3,worst_paired_rx_pp=-2,rx_ci_lower_gt=0),
        confirmation=dict(rounds=[18111,18112],group='last',datasets=['ManyRx','ManySig'],manysig_mean_pp=0,manysig_folds_ge_minus1=3),
        maximum_formal_fits=dict(pilot=16,confirmation=64,total=80),real_target_access=False,no_automatic_fit_retry=True))

def expand(phase):
    validate()
    if phase=='pilot':
        decision=HERE/'decision_d1.json';committed(decision);tau=read(decision)['selected_tau']
        jobs=read(HERE/'jobs_pilot.json');seed=18101;group='first';datasets=['ManyRx']
    else:
        prior='pilot' if phase=='confirm1' else 'confirm1';decision=HERE/f'decision_{prior}.json';committed(decision)
        if not read(decision)['admitted']:raise ValueError('Previous source gate stopped')
        tau=read(HERE/'registration_pilot.json')['selected_tau'];seed=18111 if phase=='confirm1' else 18112;group='last';datasets=['ManyRx','ManySig']
        jobs=[dict(phase=phase,dataset=d,arm='M0',fold=f,group=group,seed=seed) for d in datasets for f in range(4)]
    if tau is None:raise ValueError('D1 stopped')
    for d in datasets:
        for f in range(4):
            for arm,mode in [('W_phase','physical'),('N_phase','noise'),('G_phase','global')]:
                jobs.append(dict(phase=phase,dataset=d,arm=arm,control=mode,family='phase',tau=tau,fold=f,group=group,seed=seed,
                    **({'split_seed':16001} if phase=='pilot' else {})))
    write(HERE/f'jobs_{phase}.json',jobs);write(HERE/f'registration_{phase}.json',dict(time=now(),selected_tau=tau,
        matrix_sha256=sha(HERE/f'jobs_{phase}.json'),previous_decision_sha256=sha(decision),original_s17_stop_changed=False))
    commit([HERE/f'jobs_{phase}.json',HERE/f'registration_{phase}.json'],f'S17B preregister {phase} comparisons before method fitting')

if __name__=='__main__':design() if sys.argv[1]=='design' else expand(sys.argv[1])
