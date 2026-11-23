"""Freeze a no-fit audit and an explicitly inactive six-fit factorial plan."""
import csv
from pathlib import Path
import subprocess
from s28_core import HERE,ROOT,BASE,P,read,write,sha,now
from xa.train import stable_seed

def main():
    assert not (HERE/'registration.json').exists()
    gate=read(HERE/'GATE.json');assert gate['status']=='PASS' and gate['scientific_fits']==0
    assert sha(HERE/'PLAN_C.original.json')==sha(ROOT/'27-mechanism-identification/PLAN_C.json')
    refs=[]
    for q in read(ROOT/'27b-tokenizer-backend-interaction/registration.json')['references']:
        refs.append(dict(id=q['id'],arm=q['arm'],fold=q['fold'],seed=q['seed'],checkpoint=q['checkpoint'],sha256=q['artifacts_sha256']))
    prior=read(ROOT/'23-hstem-mechanism-evidence/registration.json')['references']
    seal=read(ROOT/'27-mechanism-identification/seal.json')['models']
    for f in (0,1):
        for s in (201,202,203):
            name=f'H2_f{f}_s{s}';q=prior[name]
            refs.append(dict(id=name,arm='H2',fold=f,seed=s,checkpoint=str(Path(q['path'])/'model.pt'),sha256=q['sha256']))
            name=f'LA_f{f}_s{s}';folder=Path('D:/crossatten/worktrees/s27-mechanism-identification/27-mechanism-identification/runs')/name
            refs.append(dict(id=name,arm='LA',fold=f,seed=s,checkpoint=str(folder/'model.pt'),sha256=seal[name]))
    metadata=[]
    for q in refs:
        folder=Path(q['checkpoint']).parent
        for file,h in q['sha256'].items():assert sha(folder/file)==h,(q['id'],file)
        cfg=read(folder/'config.json');metric=read(folder/'metrics.json')
        assert cfg['data_seed']==stable_seed('P-H5x160',q['fold'],q['seed']) and cfg['batch']==64 and cfg['epochs_budget']==200
        assert metric['epochs']==200 and cfg['train_packets']==8000 and cfg['val_packets']==2000
        with (folder/'history.csv').open(encoding='utf-8',newline='') as f:history=list(csv.DictReader(f))
        assert len(history)==200
        best=min(history,key=lambda x:float(x['source_val_ce']))
        assert int(best['epoch'])==metric['best_epoch'] and abs(float(best['source_val_ce'])-metric['best_source_ce'])<1e-10
        metadata.append(dict(id=q['id'],fold=q['fold'],seed=q['seed'],data_seed=cfg['data_seed'],epochs=200,source_train=8000,
            source_validation=2000,batch=64,source_ce_checkpoint_verified=True,best_epoch=metric['best_epoch'],sha256=q['sha256']))
    write(HERE/'REFERENCE_AUDIT.json',dict(time=now(),status='PASS',models=metadata,RF_reads=0,scientific_fits=0,
        history='All prior packets and Dev exposure are historical development evidence; no independent confirmation is created.'))
    a=dict(time=now(),stage='S28_A_BN_BY_GELU',status='REGISTERED_NOT_ACTIVATED',exploratory_after_observing_S27B=True,
        planned_new_fits=6,automatic_fit_budget=0,activation_required=True,
        jobs=[dict(arm='H2_LA',fold=f,seed=s) for f in (0,1) for s in (201,202,203)],
        cells={'BN_GELU':'H0: six frozen models','BN_noGELU':'LA: six frozen S27-A models',
               'noBN_GELU':'H2: six frozen S21 models','noBN_noGELU':'H2_LA: six planned new fits, none started'},
        architecture=gate['architecture']['H2_LA'],intervention='Only tokenizer indices1/4/7 BN and2/5/8 GELU become Identity; original convolutions, shared-stem BN/LeakyReLU and MHA are retained.',
        initialization='Same full state as H2 at each paired data_seed; all surviving state equals H0; no RNG draw on Identity replacement.',
        training=read(ROOT/'27-mechanism-identification/registration.json')['training'],
        permissions=dict(source='Original5RX/fold; first160/last40 fixedselectedrows/TX;8000train2000val',
            dev='Report only after all6 new models/config/history sealed and committed; same24-model cohort, no model or hyperparameter choices.',
            confirm=False,blind=False,C=False,ASK_strict=False,IQ_only=False),
        primary=dict(formula='(H0-LA)-(H2-H2_LA)',unit='accuracy percentage points',pairing='Same fold,seed,RX',
            receiver_units=10,bootstrap_replicates=10000,bootstrap_seed=28028,aggregation='Average3seeds peruniqueRX before RX resampling',
            directional_rule='Two-sided exploratory: abs(mean)>=0.5pp; conditional95%CI excludes0;>=5/6pairs and bothfoldmeans have the same direction as the mean.',
            caveat='No multiplicity-corrected confirmatory pvalue; intervals condition on existing seed/sample/architecture choices. Non-significance is not equivalence.'),
        secondary=['Allfourcell source/dev accuracy,CE,macroF1,worstRX,perTXrecall','All6fit and10RX interactions',
                   'H2_LA versus LP, LA versus LP and H0 versus LP; descriptive only','Source learning and source-minus-Dev gaps, not a replacement for C'],
        interpretation='BN includes learned affine parameters, batch statistics, running statistics and optimization. Removing BN changes352 parameters; this is a factorial procedure comparison, not a unique normalization cause.',
        stop='Exactly6 newfits only after separate explicit authorization and committed execution activation. No retuning, extra seeds, C/IQ-only/ASKStrict/S28-B expansion.')
    write(HERE/'PLAN_A.json',a)
    c=read(HERE/'PLAN_C.original.json')
    for split in c['splits']:
        f=split['fold'];ranking=sorted(P['folds'][f]['source'],key=lambda rx:__import__('hashlib').sha256(f'S27C-HeldRX-v1|{f}|{rx}'.encode()).hexdigest())
        assert ranking==split['deterministic_ranking'] and ranking[0]==split['held_source_receiver']
        assert set(split['training_receivers'])==set(P['folds'][f]['source'])-{ranking[0]}
    assert c['automatic_fit_budget']==0 and c['planned_new_fits']==12
    write(HERE/'C_READINESS.json',dict(time=now(),status='ORIGINAL_PLAN_PRESERVED_NOT_ACTIVATED',original_plan_sha256=sha(HERE/'PLAN_C.original.json'),
        planned_fits=12,automatic_fit_budget=0,splits=c['splits'],C_fits=0,training_packets=6400,source_validation_packets=1600,
        held_evaluation_packets=2000,steps_per_epoch=100,total_steps_per_fit=20000,source_seed=c['source_seed'],
        source_only_selection=True,all_twelve_sealed_before_held=True,fresh_initialization_required=True,
        receiver_units=2,population_bootstrap=False,original_consistent_pilot_signal=c['statistics']['consistent_pilot_signal'],
        held_only_geometry_caveat='One held RX has no across-RX scatter degrees of freedom. Held-only D=0 by construction and Jc cannot identify RX invariance; mark this diagnostic not identifiable rather than interpreting a large epsilon ratio. Jw and TX discrimination remain descriptive.',
        activation='Separate explicit user authorization and frozen execution adapter required; Gate validates architecture, seeds, roles and100-step schedule, not an executed C cohort.'))
    code=list(HERE.glob('s28_*.py'))+[HERE/'launch.py']+list((ROOT/'lib').rglob('*.py'))
    code += [ROOT/folder/file for folder,files in {
        '13-mixer-cfo-mechanisms':['train.py','models.py','core.py','data.py','metrics.py','wifi.py'],
        '15-hstem-attribution':['s15_models.py','s15_core.py'],'21-hstem-mechanisms':['s21_models.py','s21_core.py'],
        '26-long-patch-control':['s26_models.py','s26_common.py'],'27-mechanism-identification':['s27_models.py','s27_common.py'],
        '27b-tokenizer-backend-interaction':['s27b_training.py','s27b_common.py','s27b_data.py','s27b_models.py']}.items() for file in files]
    evidence=[HERE/file for file in ['REQUEST.original.zh-CN.txt','PLAN_C.original.json','PLAN_A.json','C_READINESS.json','GATE.json','REFERENCE_AUDIT.json']]
    evidence += [ROOT/file for file in ['configs/protocol.json','15-hstem-attribution/architecture_search.json',
        '23-hstem-mechanism-evidence/registration.json','27-mechanism-identification/seal.json','27b-tokenizer-backend-interaction/registration.json','27b-tokenizer-backend-interaction/RESULTS.json']]
    # Reporting code is intentionally excluded; mathematical/data/gate code is frozen.
    code=[p for p in code if p.name not in ['s28_report.py']]
    write(HERE/'registration.json',dict(time=now(),stage='S28_0_READ_ONLY_AFFINE_AND_FACTORIAL_READINESS',base_git=BASE,
        authority='User asks to execute attached S28 proposal. Its explicit final instruction requires no-fit audit, preregistration and Gate first; new scientific fitting requires separate explicit authorization.',
        scientific_fit_budget=0,synthetic_gate_optimizer_updates=gate['synthetic_optimizer_updates'],A_planned_fits=6,C_planned_fits=12,S28B_planned_fits=0,
        RF_access='Original legal Source validation only; no Dev,Confirm,blind or held-C evaluation in this phase.',
        numeric_contract=dict(FP64_atol=1e-9,FP64_rtol=1e-9,FP32_atol=1e-4,FP32_rtol=1e-4,
            source_FP32='All2000SourceValpackets/fold for each frozenLA model; all32positions and final logits, batch128.',
            source_FP64='First validation packet in eachTXxRXcell: balanced50/fold, batch10; full logits and allTokenpositions.',
            synthetic='Six fixed samples: zero,N(0,1),100*N(0,1),left/right/interior impulses; generator28028.',
            failure_policy='Report all errors and numerical failures without relaxing tolerances, cropping tokens or selecting favorable seeds.'),
        references=refs,source_caches=read(ROOT/'27b-tokenizer-backend-interaction/registration.json')['source_caches'],
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in code},evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence}))
    print('Registered no-fit S28 audit; reserved A6/C12, execution budgets0. Commit before Source IO.',flush=True)

if __name__=='__main__':main()
