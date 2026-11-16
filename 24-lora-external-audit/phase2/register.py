import sys,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;STAGE=HERE.parent;ROOT=STAGE.parent
sys.path.insert(0,str(STAGE))
from s24_common import write_json
from execution import digest,read,jobid

def main():
    if (HERE/'pilot_registry.json').exists():raise RuntimeError('Registry already exists; never overwrite')
    assert read(HERE/'DANN_PARITY.json')['status']=='PASS'
    assert read(HERE/'IB_GATE.json')['status']=='PASS'
    assert read(HERE/'ASK_GATE_b128.json')['status']=='PASS'
    jobs=[]
    for method in ['fan','ib','ce','dann','ask']:
        for view in ['native','strict_ib']:
            for fold in [0,1]:
                for seed in [201,202,203]:
                    shen=method in ('ce','dann')
                    jobs.append(dict(method=method,view=view,fold=fold,seed=seed,batch=64 if shen else 128,
                       optimizer='SGD' if shen else 'Adam',lr=0.001,weight_decay=0.,momentum=0.9 if shen else None,
                       epochs=500 if shen else 200,early_stop=20 if shen or method=='ask' else 10,
                       scheduler_patience=10 if shen or method=='ask' else 5,scheduler_factor=.2 if shen else .1,
                       min_lr=1e-6,alpha=.004 if method=='ib' else 0.,awgn=[0.,49.] if shen else None,
                       checkpoint='minimum_source_RX_equal_TX_CE',validation_noise='one fixed realization, separate seed stream' if shen else 'none',
                       sampler='one seeded no-replacement permutation per epoch, source train only',
                       strict_order='raw IQ -> source training AWGN (CE/DANN only) -> common IB projection if strict -> method original preprocessing',
                       precision='FP32, complex64 ASK/SAF, TF32 disabled; no AMP',last_checkpoint_every_epochs=5))
    code=list(HERE.glob('*.py'))+[STAGE/'s24_dann.py',STAGE/'s24_common.py']+list((ROOT/'lib/xa').glob('*.py'))+list((ROOT/'lib/legacy_rfhstu').glob('*.py'))
    artifacts=[STAGE/p for p in ['UPSTREAM.json','PACKETS.csv','DATA_AUDIT.json','HISTORY_AUDIT.json']]
    artifacts += [ROOT/'configs/protocol.json']+[HERE/p for p in ['gate_policy.json','DANN_PARITY.json','IB_GATE.json','ASK_GATE_b128.json','ENGINEERING_TESTS.json']]
    assert read(HERE/'ENGINEERING_TESTS.json')['status']=='PASS'
    obj=dict(stage='S24_phase2',date='2026-11-16',status='PILOT_AUTHORIZED_FORMAL_PER_METHOD_FROZEN_ONLY',
      maximum_scientific_fit_initializations=60,maximum_unique_jobs=60,includes_full_native_pilots=True,
      pilot_ids=[f'{m}__native__f0__s201' for m in ['fan','ib','ce','dann','ask']],jobs=jobs,
      no_score_threshold_for_pilot=True,no_automatic_failed_fit_reinitialization=True,
      promotion_rule='Pilot is first registered matrix cell, exact recipe/code unchanged; reuse avoids duplicate scientific fits',
      methods_passed_gates=['ce','dann','fan','ib','ask'],source_only=True,
      validation='RX-equal TX CE for every method, including DANN and CE control; source RX head loss never selects checkpoints',
      dann_change_disclosure='Correct official RX validation labels; epoch permutation instead of repeated random batch sampling; fixed full source validation AWGN; best TX CE restore. DANN and CE controls share all these rules.',
      ib_change_disclosure='Shen8192 adapted, patch512/stride256/31tokens/dropout.5; alpha .004 fixed both views. Historical target development was exposed; no new-blind or optimal-alpha claim. Exact original CPU HSIC function retained including alpha0 diagnostics.',
      ask_change_disclosure='Full official default architecture, batch128, candidates unchanged. Fixed packet/split and source-only checkpointing adapt original drift task.',
      evaluation='Dev only after every one of 60 registered cells COMPLETE and global seal committed; no confirm/blind ever',
      recovery='Atomic best checkpoints; full optimizer/scheduler/RNG every5epochs and on failure; resume same initialization, preserve attempt logs. Technical code repair requires explicit amendment, never target-guided tuning.',
      reporting='RX clustered paired comparisons; seeds are repeats, RX roles swap across folds; Native and Strict-IB tables distinct',
      artifact_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in artifacts},
      code_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in code})
    write_json(HERE/'pilot_registry.json',obj)
    print('Registered 60 cells including 5 full source pilots; no optimizer steps launched by registration.')

if __name__=='__main__':main()
