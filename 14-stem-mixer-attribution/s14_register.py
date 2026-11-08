"""Design before S13 completes; final registration only after the full frozen evidence."""
import sys
from s14_core import HERE,ROOT,S13,FROZEN,read,write,frozen,sha,now,git,committed,commit,validate,s13_id

def code_manifest():
    paths=list(HERE.glob('*.py'))+list((HERE/'tests').glob('*.py'))
    # Include the unmodified S13 loop/frontends/models imported by the adapter.
    paths+=list(FROZEN.glob('*.py'))+list((ROOT/'lib/xa').glob('*.py'))+list((ROOT/'lib/legacy_rfhstu').glob('*.py'))
    paths+=[ROOT/'configs/protocol.json',ROOT/'09-phase-invariance/phase_features.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)}

def design():
    if (HERE/'design.json').exists():raise ValueError('Existing design must not be silently rewritten')
    value=dict(stage='S14',time=now(),status='DESIGN_FROZEN_WAITING_FOR_S13',live_s13=str(S13),
        branch='exp/s14-stem-mixer',source_base=git('rev-parse','HEAD'),code_sha256=code_manifest(),
        mandatory=dict(arms=['E1_patch_mha','E2_patch_hstu'],folds=[0,1],seeds=[201,202,203],new_fits=12,
            reused_arms=['A1_mha','A3_rfhstu'],recipe='Exact S13 frozen fit function; same RNG streams, AdamW warm5/cos200, CE, batch64, AWGN[0,50], source RX-equal checkpoint, evaluation batch128',
            initialization='Only tokenizer is replaced; common stem, mixer, input projection, position, preLN, finalLN and classifier match reused S13 arm exactly',
            parameter_matching='Patch and HStem differ in capacity and receptive field. No matched-Stem control is scheduled; do not claim a uniquely isolated parameter-efficiency cause.'),
        mechanism=dict(arms=['G1_no_u','G2_no_rel'],max_fits=12,
            admission='S13 dev_val only: HSTU vs all4 natural/matched MHA/Transformer controls has mean>=1pp, >=5/6 positive, worstRX>=-2pp and conditional RX CI lower>0; heuristic with no multiplicity correction',
            inactive_parameters='Projections/bias allocations retained for exact single-switch comparison; report active vs allocated counts'),
        cfo_diagnostics=dict(training_fits=0,source_only=True,phases=['p2','p2_confirmation'],
            shifts_hz=[-10000,-5000,-2000,0,2000,5000,10000],confidence_threshold=.7,
            abs_cfo_edges_hz=[0,2000,5000,10000,50000],windows_start=[0,10,20,40,60,80],lag_checks=[20,40,60,80],
            sample_policy='All held-source packets; groups fixed from unperturbed estimates. All named source extreme cells audited, none removed.',
            truth='No true natural CFO; observed CFO and coherence strata, never called ground-truth estimator-error bins'),
        tta=dict(status='LITERATURE_AND_CODE_FEASIBILITY_ONLY',training_fits=0,reason='Independent UDA/TTA direction; separate protocol before any adaptation'),
        execution=dict(start_condition='S13 COMPLETE_BOUNDED_SOURCE_WORKFLOW and parent PID exited; all source/target artifacts and decisions exist and are committed',
            concurrency=1,automatic_retry=False,do_not_stop_or_restart_s13=True,do_not_write_s13=True,
            target_rule='S14 new source checkpoints all sealed/committed before own dev_val/used-confirm evaluation; never blind or WiSig target/date3-4'),
        cost=dict(inference_repeats=3,precision='FP32',warmup=20,trials=100,
            implementation_controls=['Identical MHA parameters with PyTorch SDPA inference only; no new training or target-based backend selection'],
            caveat='FFT/elementwise FLOPs incomplete; actual SDPA backend hardware dependent; no GPU benchmarks before S13 exit'))
    frozen(HERE/'design.json',value)
    frozen(HERE/'planned_jobs.json',[dict(phase='p1',dataset='Shen',arm=a,fold=f,seed=s) for f in [0,1] for s in [201,202,203] for a in value['mandatory']['arms']])
    return value

def finalize():
    if (HERE/'registration.json').exists():return validate()
    plan=validate('design.json')
    status=read(S13/'progress.json')
    if status['status']!='COMPLETE_BOUNDED_SOURCE_WORKFLOW':raise ValueError('S13 is incomplete')
    for name in ['decision_p1_source.json','p1_summary.json']:
        committed(S13/name,S13.parent)
    from s14_reports import s13_comparison
    comparison=s13_comparison();gate=comparison['mechanism_gate']
    reused={}
    seals=read(S13/'decision_p1_source.json')['checkpoints']
    for arm in ['A1_mha','A3_rfhstu']:
        for fold in [0,1]:
            for seed in [201,202,203]:
                name=s13_id(arm,fold,seed);out=S13/'runs'/name
                if sha(out/'model.pt')!=seals[name]:raise ValueError('S13 reference checkpoint no longer sealed')
                reused[name]=dict(model_sha256=seals[name],config_sha256=sha(out/'config.json'),target_metrics_sha256=sha(out/'target_metrics.json'))
    jobs=read(HERE/'planned_jobs.json')
    if gate['admitted']:jobs+=[dict(phase='p1',dataset='Shen',arm=a,fold=f,seed=s) for f in [0,1] for s in [201,202,203] for a in ['G1_no_u','G2_no_rel']]
    cfo_models=[]
    for phase in ['p2','p2_confirmation']:
        path=S13/f'jobs_{phase}.json'
        if not path.exists():continue
        for job in read(path):
            name=f"{phase}__{job['dataset']}__{job['arm']}__f{job['fold']}__{job['group']}__s{job['seed']}"
            out=S13/'runs'/name;m=read(out/'metrics.json')
            if sha(out/'model.pt')!=m['model_sha256']:raise ValueError('CFO checkpoint differs')
            cfo_models.append(dict(job=job,run=name,model_sha256=m['model_sha256'],config_sha256=sha(out/'config.json'),metrics_sha256=sha(out/'metrics.json')))
    value=dict(stage='S14',time=now(),status='REGISTERED_AFTER_COMPLETE_S13',s13_final_git_sha=git('rev-parse','HEAD',cwd=S13.parent),
        design_sha256=sha(HERE/'design.json'),code_sha256=plan['code_sha256'],jobs=jobs,reused=reused,
        mechanism_gate=gate,cfo_diagnostics=plan['cfo_diagnostics'],cfo_models=cfo_models,tta=plan['tta'],
        source_inputs_read_only=True,comparison='Full factorial contrasts and paired RX-bootstrap interaction; confirm reporting only, not new blind')
    frozen(HERE/'registration.json',value);frozen(HERE/'jobs_p1.json',jobs)
    commit([HERE/'registration.json',HERE/'jobs_p1.json',HERE/'ATTENTION_COMPARISON_S13.md',HERE/'ATTENTION_PAIRED_DELTAS.png',HERE/'attention_comparison_s13.json',HERE/'attention_paired_receiver_deltas.csv'],
        'S14 preregister paired Stem x Mixer tests after complete S13 evidence')
    return validate()

if __name__=='__main__':design() if len(sys.argv)>1 and sys.argv[1]=='design' else finalize()
