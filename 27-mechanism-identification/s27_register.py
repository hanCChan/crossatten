"""Freeze A execution and nonexecuting B/C designs before this stage's data IO."""
from pathlib import Path
import subprocess
import torch
from s27_common import HERE,ROOT,P,read,write,sha,now
from xa.train import stable_seed


def main():
    assert not (HERE/'registration.json').exists()
    assert read(HERE/'GATE.json')['status']=='PASS'
    assert read(HERE/'PLAN_B.json')['automatic_fit_budget']==read(HERE/'PLAN_C.json')['automatic_fit_budget']==0
    prior=read(ROOT/'25-tifs-integration/registration.json');refs=[]
    for rec in prior['models']:
        if rec['arm']!='H0':continue
        q=dict(rec);folder=Path(q['checkpoint']).parent
        q['prediction_format']='legacy_bundle'
        q['predictions_sha256']={rx:sha(folder/'preds_dev_val.npz') for rx in P['folds'][q['fold']]['dev_val']}
        q['artifacts_sha256']={n:sha(folder/n) for n in ['model.pt','config.json','metrics.json','history.csv']}
        assert sha(q['checkpoint'])==q['checkpoint_sha256'];refs.append(q)
    previous=Path('D:/crossatten/worktrees/s26-long-patch-control/26-long-patch-control')
    seal=read(previous/'seal.json')
    for f in (0,1):
        for s in (201,202,203):
            name=f'LP_f{f}_s{s}';folder=previous/'runs'/name
            for n,h in seal['models'][name].items():assert sha(folder/n)==h
            refs.append(dict(id=name,arm='LP',fold=f,seed=s,data_seed=stable_seed('P-H5x160',f,s),params=138346,
                checkpoint=str(folder/'model.pt'),checkpoint_sha256=sha(folder/'model.pt'),
                prediction_format='per_receiver_file',prediction_directory=str(previous/'predictions'/name),
                predictions_sha256={rx:sha(previous/'predictions'/name/(rx+'.npz')) for rx in P['folds'][f]['dev_val']},
                artifacts_sha256={n:sha(folder/n) for n in ['model.pt','config.json','metrics.json','history.csv']}))
    code=[HERE/'resume.py']+list(HERE.glob('s27_*.py'))+list((ROOT/'lib').rglob('*.py'))
    code += [ROOT/folder/name for folder,names in {
        '13-mixer-cfo-mechanisms':['train.py','models.py','core.py','data.py','metrics.py','wifi.py'],
        '15-hstem-attribution':['s15_models.py','s15_core.py'],
        '21-hstem-mechanisms':['s21_models.py','s21_core.py'],
        '26-long-patch-control':['s26_models.py','s26_common.py']}.items() for name in names]
    evidence=[HERE/'PLAN_B.json',HERE/'PLAN_C.json',HERE/'GATE.json',HERE/'ENGINEERING_CORRECTION_001.json',
        ROOT/'configs/protocol.json',ROOT/'15-hstem-attribution/architecture_search.json',
        ROOT/'21-hstem-mechanisms/decision.json',ROOT/'25-tifs-integration/registration.json',
        ROOT/'26-long-patch-control/registration.json',ROOT/'26-long-patch-control/FINAL_AUDIT.json']
    jobs=[dict(arm='LA',fold=f,seed=s) for f in (0,1) for s in (201,202,203)]
    write(HERE/'registration.json',dict(stage='S27_A_LINEAR_ACTIVATION',time=now(),
        base_git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        authority='2026-11-23 user requests execution of attached S27 plan. Authorizes sixA fits after preregistration/Gate; B/C/IQ-only/ASKStrict not automatically authorized. HistoricalS21H3 no-execution record remains unchanged.',
        maximum_new_fits=6,jobs=jobs,not_started_planned_fits=dict(B=12,C=12,D='conditional_unregistered_no_auto_budget'),
        architecture=dict(arm='LA',historical_definition='S21 H3, previously unexecuted',
            intervention='Exactly tokenize.2/.5/.8 GELU replaced withIdentity; no parameter/state/RNG changes at initialization.',
            params=133690,local_support=350,token_shape=[32,64],bn_layers=5,
            confounds='Training BN remains batch/spatial dependent. Intervention changes forward activations, gradient flow, learned weights and BN trajectories jointly; not pure inference-only or physicalRF mechanism.'),
        training=dict(epochs=200,batch=64,optimizer='AdamW',lr=.001,weight_decay=.0005,warmup_epochs=5,
            schedule='Original per-step cosine',augmentation='Original AWGN0..50dB',checkpoint='minimum clean source RX-equal CE',
            data_seed='stable_seed(P-H5x160,fold,seed)',permutation_rng='cuda data_seed',noise_rng='cuda data_seed+1',
            initialization='Bit-identical entire initial H0 state_dict for each paired seed, no randomdraw onGELU replacement',
            tf32_matmul=False,tf32_cudnn=True,recovery='Atomic full-epoch model/optimizer/allRNG recovery only; no replacement seeds'),
        permissions=dict(source='Original5RX/fold,200selectedrows/TX,first160train/last40val;8000train2000val',
            dev='Only after all6 A checkpoints/config/history sealed and committed; commonH0/LA/LP GPU inference. All10RX retained.',
            confirm=False,old_blind=False,B=False,C=False,IQ_only=False,ask_strict_ib=False),
        packet_identity='PerRX selected rows/labels/IQ SHA256 in source_caches, with fixed first160/last40perTX source_split. No resampling packets.',
        prior_exposure='Post-S26 developmental intervention, not independent confirmation. Historical dev exposure and checkpoint-selection reuse fully disclosed.',
        statistics=dict(primary='H0 minusLA RX-equal accuracypp',secondary='LAminusLP andH0minusLP accuracy;F1,worstRX,sourceCE/sourceacc,costs descriptive',
            pairing='Samefold/seed/RX; average3seeds per each of10uniqueRX before resampling',bootstrap_replicates=10000,bootstrap_seed=27027,
            practical_rule='mean>=0.5pp,conditional lowerCI>0,>=5/6positivefit differences,bothfold meanspositive',
            interpretation='Rulemet supports usefulness of3GELU underthisarchitecture/training. Notmet does not prove equivalence/unimportance. LAnearLP is descriptive, no equivalence test. Source/Dev differences cannot replace C heldRX study.',
            uncertainty='Conditional RX bootstrap, omitsfreshseed/acquisition/architectureselection uncertainty; allcontrasts reported, no multiplicity-corrected confirmatorypvalues'),
        outcomes=dict(positive='H0 reliablyhigher thanLA: retaining3GELU is beneficial for this complete optimization procedure; nonlinear expressivity/BN/optimization remain inseparable.',
            negative='LA matches/exceeds H0 under descriptive rule: GELU necessity unestablished, revisit stagewise linear mixing/BN/optimization; no ad hoc activation search.',
            uncertain='MixedRX/seed/fold signs or small effects: retainall; no retuning or addedseeds.',
            source_vs_dev='Report both and gap; larger Dev change with small source change motivates C but does not identify RX-specific suppression.'),
        stop='Stop after6A fits andfullreport. NoautomaticB/C/IQ/ASKStrict, no improving architecture fromDev scores.',
        references=refs,source_caches=prior['source_caches'],canonical_training_sha256=sha(ROOT/'13-mixer-cfo-mechanisms/train.py'),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in code},
        evidence_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in evidence}))
    print('Registered6A fits; B/C executionbudget0; commit before source IO.',flush=True)


if __name__=='__main__':main()
