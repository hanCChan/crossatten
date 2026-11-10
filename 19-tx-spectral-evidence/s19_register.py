"""Register fixed TCSE design before source evidence or any new formal fit."""
from s19_core import HERE,ROOT,HISTORY,read,write,sha,now,git
from s19_data import locations
from s19_metrics import gates


def run():
    if (HERE/'registration.json').exists():raise RuntimeError('Already registered')
    if (HERE/'access.jsonl').exists() or list(HERE.glob('evidence_f*.json')) or (HERE/'runs').exists():
        raise RuntimeError('Scientific access before registration')
    if read(HERE/'ENGINEERING_TESTS.json')['status']!='PASS' or read(HERE/'S18_VERIFIED.json')['status']!='PASS':
        raise ValueError('Required engineering/history verification missing')
    derived={'s19_verify.py','s19_reports.py','s19_costs.py'}
    code=[p for p in HERE.glob('*.py') if p.name not in derived]+list((ROOT/'lib').rglob('*.py'))+[ROOT/'13-mixer-cfo-mechanisms'/p for p in ['core.py','models.py','wifi.py','metrics.py']]+[
          ROOT/'17-rx-distortion'/p for p in ['s17_core.py','s17_models.py']]+[ROOT/'09-phase-invariance/phase_features.py']
    old=[HISTORY/p for p in ['17b-phase-intervention/final_evidence/MANIFEST.json','17-rx-distortion/decision_d0.json','17b-phase-intervention/decision_d1.json',
          '15-hstem-attribution/decision.json','16-rf-qwen3/STATUS.json','10-tifs-validation/final_comparisons.csv']]
    for fold in range(4):
        cache,run=locations(fold);old.append(cache/'fit_manifest.json');old.extend(run/p for p in ['config.json','metrics.json'])
    jobs=[dict(fold=f,arm=a,seed=18101,split_seed=16001,reused=a=='B0') for f in range(4) for a in ['B0','B1','B2','B3','B4']]
    write(HERE/'jobs_pilot.json',jobs)
    docs=['README.zh-CN.md','historical_matrix.md','S18_VERIFIED.json','ENGINEERING_TESTS.json','jobs_pilot.json']
    write(HERE/'registration.json',dict(time=now(),stage='S19',base_commit=git('rev-parse','HEAD'),
          code_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(code))},
          design_sha256={p:sha(HERE/p) for p in docs},historical_sha256={p.relative_to(HISTORY).as_posix():sha(p) for p in old},
          data=dict(dataset='ManyRx',folds=list(range(4)),fit_rx=18,held_source_rx=6,source_days=[1,2],source_split_seed=16001,baseline_seed=18101),
          spectral=dict(bands=8,bins=256,transition_bins_per_side=2,modulation_alpha=.2,teacher_attenuation=.5,retained_spectrum='full legal WiSig FFT'),
          evidence=dict(train_cap_per_cell=16,sampling_seed='19001+fold',lambda_value=.1,quantile=.25,
              controls=['per-packet EVM-matched global amplitude','per-packet EVM-matched global phase'],selection_from_controls=False),
          arms=dict(B0='reused HStem+ordinary MHA',B1='static8band filter',B2='272parameter power gate',B3='272parameter TCSE CE only',B4='same TCSE with evidence CE'),
          training=dict(epochs_max=40,batch=128,optimizer='Adam1e-3 wd0',scheduler='Plateau(.1,5,min1e-6)',earlywait=10,precision='FP32',
              init='identical backbone random state to frozen B0; no teacher weight warm-start',checkpoints='fit-source RX-equal CE'),
          gates=gates(),new_formal_fit_budget=20,planned_new_fits=16,reused_fits=4,
          derived_only_scripts_excluded_from_primary_analysis=sorted(derived),
          all_checkpoints_sealed_before_held_source=True,real_target_access=False,no_parameter_search=True,no_automatic_fit_retry=True))


if __name__=='__main__':run()
