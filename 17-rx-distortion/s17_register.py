"""Freeze baseline and deterministic conditional matrices before outcomes."""
import sys
from s17_core import HERE,ROOT,read,write,sha,now,frozen,commit,validate,committed

def design():
    code=list(HERE.glob('*.py'))+list((ROOT/'13-mixer-cfo-mechanisms').glob('*.py'))+list((ROOT/'lib').rglob('*.py'))+[ROOT/'09-phase-invariance/phase_features.py']
    jobs=[dict(phase='pilot',dataset='ManyRx',arm='M0',fold=f,group='first',seed=18001,split_seed=16001) for f in range(4)]
    refs=[]
    for f in range(4):
        p=ROOT/'16-rf-qwen3/final_evidence'/f'pilot__ManyRx__R0_mha__f{f}__first__s17001'/'metrics.json'
        refs.append(dict(fold=f,metrics=p.relative_to(ROOT).as_posix(),sha256=sha(p),scope='source_validation only; S16 fixed RF stem+MHA; split16001'))
    frozen(HERE/'jobs_pilot.json',jobs)
    frozen(HERE/'registration_pilot.json',dict(time=now(),version='baselines-planned-after-D0',families={},matrix_sha256=sha(HERE/'jobs_pilot.json')))
    frozen(HERE/'registration_baseline.json',read(HERE/'registration_pilot.json'))
    frozen(HERE/'registration.json',dict(time=now(),stage='S17',code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(code)},
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [HERE/'README.zh-CN.md',HERE/'SOURCES.md',HERE/'ENGINEERING_TESTS.json',HERE/'CPU_STATS_TESTS.json',HERE/'ENVIRONMENT_BLOCK.json']},
        d0_frozen_references=refs,
        baseline_architecture='Unchanged HStem first-order frequency features + ordinary MHA; end-to-end FP32; no extra learned modules',
        protocol=dict(epochs_max=40,batch=128,optimizer='Adam1e-3',scheduler='plateau factor.1 patience5 min1e-6',early_stop_wait=10,
            checkpoint='source RX-equal CE',source_partition='70/30 split16001 pilot; confirmation split equals independent seed',days=[1,2],real_target=False),
        d0=dict(families=['amp','phase','iq'],profiles_source_train_only=True,error_reference_source_validation=True,cell_min_packets=5,rx_centering='TX/day cell centering, not packet ratios',
            permutations=199,stable_fold_thresholds=dict(rx_eta2=.1,cross_day_corr=.4,cross_tx_half_corr=.4,rx_probe_times_chance=2,p=.01),
            stable_folds=3,error_rho_median=.2,positive_error_rho_folds=3,phase_train_eligible=False),
        d1=dict(levels=[.25,.5,1.],draws=2,max_acc_drop_pp=1,max_f1_drop_pp=1,max_worst_tx_drop_pp=5,min_evm=.002,
            all_folds_required=True,select='largest fixed level passing every fold; never tune on held source RX',
            negative_controls=['global gain','global phase','same-EVM proper complex AWGN']),
        augmentation=dict(probability=.5,amp_db_cap=3,iq_beta_cap=.05,linear_fir_taps=17,fft_length=512,known_delay_removed=8,fir_identity_residual_l1_cap=.25,
            iq='(z+beta conjugate(z))/sqrt(1+abs(beta)^2)',amplitude_profile='differences between source RX-associated TX/day-centered log spectra'),
        source_gate=dict(manyrx_min_gain_pp=1,positive_folds=3,worst_paired_rx_pp=-2,conditional_ci_lower_gt=0,
            comparisons=['M0 clean','same-family same-EVM noise control'],confirmation_manysig_mean_pp=0,confirmation_manysig_folds_ge_minus1=3),
        confirmations=[dict(phase='confirm1',seed=18011,group='last'),dict(phase='confirm2',seed=18012,group='last')],
        max_formal_fits=dict(pilot=20,confirmation=80,total=100),no_automatic_retry=True,
        scope='Receiver-associated observational proxies plus bounded interventions; channel/TX/RX effects are not uniquely identifiable'))

def expand(phase):
    validate()
    if phase=='pilot':
        decision=HERE/'decision_d1.json';committed(decision);families=read(decision)['selected']
        if not families:raise ValueError('D1 stopped')
        jobs=read(HERE/'jobs_pilot.json');seed=18001;group='first';datasets=['ManyRx']
    else:
        prior='pilot' if phase=='confirm1' else 'confirm1';decision=HERE/f'decision_{prior}.json';committed(decision)
        selected=read(decision)['selected'];families={f:read(HERE/'registration_pilot.json')['families'][f] for f in selected}
        if not families:raise ValueError('Previous source gate stopped')
        seed=18011 if phase=='confirm1' else 18012;group='last';datasets=['ManyRx','ManySig'];jobs=[]
        for dataset in datasets:
            for fold in range(4):jobs.append(dict(phase=phase,dataset=dataset,arm='M0',fold=fold,group=group,seed=seed))
    for dataset in datasets:
        for fold in range(4):
            for family,tau in sorted(families.items()):
                for control in [False,True]:jobs.append(dict(phase=phase,dataset=dataset,arm=('N_' if control else 'W_')+family,
                    family=family,tau=tau,noise_control=control,fold=fold,group=group,seed=seed,**({'split_seed':16001} if phase=='pilot' else {})))
    write(HERE/f'jobs_{phase}.json',jobs)
    write(HERE/f'registration_{phase}.json',dict(time=now(),families=families,decision_sha256=sha(decision),matrix_sha256=sha(HERE/f'jobs_{phase}.json'),
        baseline_registration_sha256=sha(HERE/'registration_baseline.json'),source_only=True))
    commit([HERE/f'jobs_{phase}.json',HERE/f'registration_{phase}.json'],f'S17 freeze conditional {phase} matrix before method fitting')

if __name__=='__main__':design() if sys.argv[1]=='design' else expand(sys.argv[1])
