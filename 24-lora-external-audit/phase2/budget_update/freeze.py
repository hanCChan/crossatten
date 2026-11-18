"""Freeze a budget-only amendment BEFORE target evaluation; retain original registry."""
from pathlib import Path
from common import HERE,PHASE,ROOT,STAGE,P,read,digest,atomic_json,validate_registry,jobid,now

def main():
    path=HERE/'AMENDMENT.json'
    if path.exists():raise RuntimeError('Never overwrite a frozen amendment')
    r=validate_registry();active=[jobid(j) for j in r['jobs'] if not(j['method']=='ask' and j['view']=='strict_ib')]
    deferred=[jobid(j) for j in r['jobs'] if j['method']=='ask' and j['view']=='strict_ib']
    if len(active)!=54 or len(deferred)!=6:raise ValueError('unexpected matrix')
    if any((PHASE/'runs'/jid/'config.json').exists() for jid in deferred):raise ValueError('Strict-IB ASK already started; inspect instead of silently deferring')
    records=[]
    for j in r['jobs']:
        p=PHASE/'runs'/jobid(j)/'metrics.json'
        if p.exists():records.append(dict(job=jobid(j),sha256=digest(p)))
    old_access=PHASE/'access.jsonl'
    import json
    events=[json.loads(line) for line in old_access.read_text(encoding='utf-8').splitlines()]
    if any(e['role']!='source' for e in events):raise ValueError('Dev already exposed before amendment')
    if read(HERE/'TESTS.json')['status']!='PASS':raise ValueError('engineering tests did not pass')
    code=list(HERE.glob('*.py'))+[PHASE/'launch.ps1',ROOT/'13-mixer-cfo-mechanisms/models.py',ROOT/'13-mixer-cfo-mechanisms/core.py']
    obj=dict(status='FROZEN_USER_BUDGET_AMENDMENT',time=now(),
             authority='User 2026-11-18 request: finish current ASK, retain Native two folds x three seeds, defer all six Strict-IB ASK jobs; prioritize complete CE/DANN/FAN/IB report-only evaluation.',
             reason='Compute budget and research priority, not Dev accuracy; prior source validation outcomes were visible and are disclosed.',
             original_registry_sha256=digest(PHASE/'pilot_registry.json'),original_registry_retained_unchanged=True,
             active_job_ids=active,deferred_job_ids=deferred,active_fit_budget=54,original_fit_budget=60,
             resume_deferred_only_after_explicit_user_authorization=True,
             keep_current_worker_uninterrupted=True,completed_metrics_at_amendment=records,
             no_prior_new_dev_reads=True,prior_source_access_events=len(events),
             evaluate_now='All four methods, both views, both folds, all three seeds (48), plus frozen H0 six; committed cohort seal required before any Dev packet.',
             evaluate_later='All six ASK Native checkpoints together after cohort seal; no partial seed reporting.',
             no_native_training_changes_after_seeing_baseline_dev=True,no_parameter_or_selector_changes=True,
             confirm_blind_always_forbidden=True,no_new_hstem_or_architecture_search=True,
             analysis=dict(primary='RX-equal packet accuracy',secondary=['RX-equal macro-F1','worst RX seed mean','fit SD','seed SD','within-fold seed SD'],
                           pairs=['H0-CE','H0-DANN','H0-FAN','H0-IB','H0-ASK Native','DANN-CE','IB-FAN'],
                           cluster_unit='Unique RX identity, average three seeds first',bootstrap_replicates=10000,bootstrap_seed=24016,
                           uncertainty='Conditional on fixed seeds and historically developed RX, no multiplicity adjustment',
                           devices='CPU FP32 network for immediate baseline/H0 evaluation while preserved GPU training runs; original float64 frontends retained; ASK later GPU FP32',
                           all_target_packets_clean=True,no_target_augmentation_adaptation_or_fusion=True),
             tests_sha256=digest(HERE/'TESTS.json'),code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in code})
    atomic_json(path,obj)
    atomic_json(HERE/'DEFERRED.json',dict(status='DEFERRED_NOT_RUN',jobs=deferred,reason='Whole input-budget arm deferred by user compute budget; no Dev-based stopping',accuracy=None,automatic_start_allowed=False))
    print('Frozen draft:54 retained,6 deferred; commit before any Dev evaluation.')

if __name__=='__main__':main()
