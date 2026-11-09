"""Freeze design first, then deterministic resource/pilot/confirmation matrices."""
import sys
from pathlib import Path
from s16_core import HERE,ROOT,S13,read,write,sha,now,frozen,commit,validate,committed

PILOT_CANDIDATES=['R1_qwen_small','R2_transformer','R3_pretrained_frozen','R5_pretrained_lora']

def sha_record(paths):return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}

def design():
    if (HERE/'design.json').exists():raise ValueError('Design already frozen')
    code=list(HERE.glob('*.py'))+list((ROOT/'13-mixer-cfo-mechanisms').glob('*.py'))+list((ROOT/'lib').rglob('*.py'))
    evidence=[HERE/n for n in ['MODEL_MANIFEST.json','MODEL_ARCHITECTURE_AUDIT.json','small_architecture_match.json','requirements-qwen3.txt','ENVIRONMENT.json','ENGINEERING_TESTS.json','OFFICIAL_CPU_SMOKE.json','README.zh-CN.md','proposal_1.txt','proposal_2.txt']]
    refs=[]
    for fold in range(4):
        p=S13/'runs'/f'p2__ManyRx__B0__f{fold}__first__s16001'
        refs.append(dict(fold=fold,path=str(p),checkpoint_sha256=sha(p/'model.pt'),config_sha256=sha(p/'config.json')))
    frozen(HERE/'design.json',dict(time=now(),stage='S16',base_s14='fc4a3cbeafcfc9bb18aa69125809b779fa974df4',
        code_sha256=sha_record(code),evidence_sha256=sha_record(evidence),front_references=refs,
        model_revision=read(HERE/'MODEL_MANIFEST.json')['revision'],pilot_seed=17001,source_split_seed=16001,
        protocol=dict(frozen_source_pretrained_rf=True,raw_input=[2,256],rf_tokens=[32,64],qwen_tokens=33,cls='last',causal=True,
            effective_batch=128,epochs=20,checkpoint='minimum source RX-equal CE',precision='BF16 autocast; FP32 trainable weights',
            optimizer='AdamW 1e-3 wd5e-4 warmup2 cosine20 clip1',lora=dict(rank=8,alpha=16,dropout=0,targets=['q_proj','v_proj']),
            real_target=False,days=[1,2],training_and_validation_partition='same70/30 split as source RF tokenizer'),
        source_gate=dict(manyrx_mean_pp=1,positive_folds=3,ci95_lower_gt=0,worst_rx_delta_pp=-2,
            confirmation_manysig_mean_pp=0,confirmation_manysig_folds_ge_minus1=3,bootstrap=5000,rx_cluster=True),
        confirmation_rounds=[dict(phase='confirm1',seed=17011,front_phase='front_r1'),dict(phase='confirm2',seed=17012,front_phase='front_r2')],
        maximum_new_fits=dict(pilot_heads=28,confirmation_heads=128,confirmation_rf_fronts=16),
        resource_gate=dict(microbatches=[16,32,64,128],max_allocated_gib=5.75,max_reserved_gib=6.5,max_projected_fit_seconds=7200),
        selection='at most one small-architecture candidate and one pretrained candidate; no promotion of confirmation controls',
        no_automatic_fit_retry=True))

def pilot():
    d=validate('design.json');p0=HERE/'P0_ENGINEERING.json';committed(p0);p=read(p0)
    arms=['R0_mha','R0_hstu','R1_qwen_small','R2_transformer']
    if p['large_models_allowed']:arms+=['R3_pretrained_frozen','R4_random_frozen','R5_pretrained_lora']
    jobs=[dict(phase='pilot',dataset='ManyRx',arm=a,fold=f,seed=17001,split_seed=16001,group='first') for f in range(4) for a in arms]
    frozen(HERE/'jobs_pilot.json',jobs)
    candidates=[a for a in PILOT_CANDIDATES if a in arms]
    frozen(HERE/'registration_pilot.json',dict(time=now(),candidates=candidates,matrix_sha256=sha(HERE/'jobs_pilot.json'),engineering_sha256=sha(p0)))
    frozen(HERE/'registration.json',dict(**d,large_microbatch=p['large_microbatch'],design_sha256=sha(HERE/'design.json'),p0_sha256=sha(p0)))
    commit([HERE/n for n in ['jobs_pilot.json','registration_pilot.json','registration.json']],'S16 preregister resource-qualified source-only paired pilot')

def confirmation(phase):
    validate();previous='pilot' if phase=='confirm1' else 'confirm1'
    decision=HERE/f'decision_{previous}.json';committed(decision);selected=read(decision)['selected']
    if not selected:raise ValueError('Source gate stopped; no confirmation')
    seed=17011 if phase=='confirm1' else 17012;front_phase='front_r1' if phase=='confirm1' else 'front_r2'
    from s16_reports import CONTROLS
    arms={'R0_mha','R0_hstu'}
    for a in selected:arms.add(a);arms.update(CONTROLS[a])
    if 'R5_pretrained_lora' in selected:arms.update(['R4_random_frozen','R6_random_lora'])
    jobs=[dict(phase=phase,dataset=d,arm=a,fold=f,seed=seed,split_seed=seed,group='last',front_phase=front_phase)
        for d in ['ManyRx','ManySig'] for f in range(4) for a in sorted(arms)]
    fronts=[dict(phase=front_phase,dataset=d,arm='B0',fold=f,seed=seed,group='last') for d in ['ManyRx','ManySig'] for f in range(4)]
    frozen(HERE/f'jobs_{phase}.json',jobs);frozen(HERE/f'jobs_{front_phase}.json',fronts)
    frozen(HERE/f'registration_{phase}.json',dict(time=now(),candidates=selected,seed=seed,group='last',previous_decision_sha256=sha(decision),
        matrix_sha256=sha(HERE/f'jobs_{phase}.json'),front_matrix_sha256=sha(HERE/f'jobs_{front_phase}.json')))
    commit([HERE/f'jobs_{phase}.json',HERE/f'jobs_{front_phase}.json',HERE/f'registration_{phase}.json'],f'S16 preregister independent {phase} and new leakage-free RF frontends')

if __name__=='__main__':
    if sys.argv[1]=='design':design()
    elif sys.argv[1]=='pilot':pilot()
    elif sys.argv[1] in ['confirm1','confirm2']:confirmation(sys.argv[1])
    else:raise ValueError('Unknown registration step')
