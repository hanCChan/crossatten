"""Commit the explicit 24-Nov A+C authorization before any new RF IO/fit."""
from pathlib import Path
import subprocess
from ac_common import HERE,ROOT,BASE,P,A,C,read,write,sha,now

def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==BASE
    assert not (HERE/'registration.json').exists() and read(HERE/'GATE.json')['status']=='PASS'
    assert sha(HERE/'PLAN_A.original.json')==sha(ROOT/'28-affine-bn-gelu-audit/PLAN_A.json')
    assert sha(HERE/'PLAN_C.original.json')==sha(ROOT/'27-mechanism-identification/PLAN_C.json')
    jobs=[dict(stage='A',**q) for q in A['jobs']]+[dict(stage='C',**q) for q in C['jobs']]
    assert len(jobs)==18
    write(HERE/'ACTIVATION.json',dict(time=now(),base_git=BASE,authorized_scientific_fits=18,A_fits=6,C_fits=12,
        user_authority='2026-11-24 user explicitly authorizes A+C,18scientific fits,fromde7d010; A6 then C12 regardless of A outcome.',
        A_plan_sha256=sha(HERE/'PLAN_A.original.json'),C_plan_sha256=sha(HERE/'PLAN_C.original.json'),
        initial_S28_zero_fit_record_preserved=True,delay_all_target_inference_until_both_cohorts_seal=True,
        no_result_adaptive_C_modification=True,other_scientific_fits=0,stop_after_18=True,
        initialization_interpretation='A uses PLAN_A top-level invariant: fullstate equalsH2, survivingstate equalsH0; removedBNstate is absent. C freshly initialized from S27C-H4x160.',
        C_Jc_scope='Held-only singleRX acrossRXscatter is not identifiable; preserve algebraic regularized ratio separately without treating it as invariance.',
        decisions='A and C designs do not adapt to each other. Shared data/history mean no claim of statistically independent datasets.',
        resume='Complete epoch current/bestmodel,optimizer,allRNG,history atomic snapshot; PAUSE_REQUESTED saves and stops at epoch boundaries. No replacement seeds.'))
    refs=read(ROOT/'28-affine-bn-gelu-audit/registration.json')['references']
    for q in refs:
        folder=Path(q['checkpoint']).parent
        for file,h in q['sha256'].items():assert sha(folder/file)==h
        q['predictions']={}
        for rx in P['folds'][q['fold']]['dev_val']:
            if q['arm']=='H0':path=folder/'preds_dev_val.npz';y='y_'+rx;logit=rx
            elif q['arm']=='H2':path=folder/'diagnostics_dev_val.npz';y='y_'+rx;logit=rx
            elif q['arm']=='LA':path=Path('D:/crossatten/worktrees/s27-mechanism-identification/27-mechanism-identification/predictions')/q['id']/(rx+'.npz');y='y';logit='logits'
            else:path=Path('D:/crossatten/worktrees/s26-long-patch-control/26-long-patch-control/predictions')/q['id']/(rx+'.npz');y='y';logit='logits'
            q['predictions'][rx]=dict(path=str(path),sha256=sha(path),y_key=y,logits_key=logit)
    code=list(HERE.glob('ac_*.py'))+[HERE/'resume.py']+list((ROOT/'lib').rglob('*.py'))
    code += [ROOT/folder/file for folder,files in {
        '13-mixer-cfo-mechanisms':['train.py','models.py','core.py','data.py','metrics.py','wifi.py'],
        '15-hstem-attribution':['s15_models.py','s15_core.py'],'21-hstem-mechanisms':['s21_models.py','s21_core.py'],
        '25-tifs-integration':['geometry_stats.py'],'26-long-patch-control':['s26_models.py','s26_common.py'],
        '27-mechanism-identification':['s27_models.py','s27_common.py'],
        '27b-tokenizer-backend-interaction':['s27b_training.py','s27b_common.py','s27b_models.py','s27b_data.py','s27b_features.py'],
        '28-affine-bn-gelu-audit':['s28_models.py','s28_core.py']}.items() for file in files]
    evidence=[HERE/f for f in ['ACTIVATION.json','PLAN_A.original.json','PLAN_C.original.json','GATE.json','ENGINEERING_UPDATES.json','AUTHORIZATION.zh-CN.md']]
    evidence += [ROOT/f for f in ['configs/protocol.json','15-hstem-attribution/architecture_search.json',
        '28-affine-bn-gelu-audit/SUMMARY.json','28-affine-bn-gelu-audit/AFFINE_AUDIT.json','28-affine-bn-gelu-audit/FINAL_AUDIT.json',
        '28-affine-bn-gelu-audit/registration.json','27b-tokenizer-backend-interaction/registration.json']]
    prior=read(ROOT/'27b-tokenizer-backend-interaction/registration.json')
    write(HERE/'registration.json',dict(time=now(),stage='S28_A_PLUS_ORIGINAL_S27_C_EXECUTION',base_git=BASE,maximum_new_fits=18,jobs=jobs,
        stages=dict(A=6,C=12),reference_models=24,references=refs,source_caches=prior['source_caches'],reference_streams_A=prior['reference_streams'],
        training_steps=dict(A=25000,C=20000),total_retained_optimizer_steps=390000,selection='Source RX-equal clean CE only',
        permissions='A5SourceRX/C4SourceRX; no Dev or heldC until all18 models sealed. Confirm/blind forbidden.',
        target_predictions=dict(A=300000,C_held=24000),source_predictions=dict(A=60000,C=19200),C_geometry_vectors=72000,
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in code},evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence}))
    print('Registered18 authorized fits; immutable A/C plans, adapter hashes and roles. Commit and push before Source IO.',flush=True)

if __name__=='__main__':main()
