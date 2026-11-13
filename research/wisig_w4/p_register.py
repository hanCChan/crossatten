"""Bind accepted fit-only scales and the twelve fixed neural fits before updates."""
from common import *


def main():
    validate('ADMISSION_REGISTRATION.json')
    accept=read(HERE/'P_DATA_ACCEPTANCE.json');assert accept['status']=='PASS'
    assert all(r['status']=='PASS_EXACT_A0_REUSE' for r in accept['A0_strict_reuse'])
    assert accept['aux24_IQ_rows_read']==accept['source_selection_IQ_rows_read']==accept['new_optimizer_updates']==0
    assert read(HERE/'P_SYNTHETIC_QA.json')['status']=='PASS' and (HERE/'E_RESULT.json').exists()
    if (HERE/'P_TRAINING_ATTEMPT.json').exists():raise RuntimeError('No registration after scientific updates')
    scales=read(HERE/'P_INITIAL_GRADIENT_SCALES.json')['scales'];jobs=[]
    for stage in ('aux','downstream'):
        for seed in range(3):
            for arm in ('A1','A2'):
                jobs.append(dict(id=f'{arm}_S{seed}_{stage}',arm=arm,seed=seed,stage=stage,epochs=40,
                    expected_updates=4320 if stage=='aux' else 5080,
                    initial_state_sha256=scales[str(seed)]['initial_state_sha256'] if stage=='aux' else 'Pretrained encoder plus exact A0 random 10-class head'))
    assert len(jobs)==12 and sum(j['expected_updates'] for j in jobs)==56400
    files=list(HERE.glob('*.py'))
    for name in ('PREREGISTRATION.zh-CN.md','ADMISSION_REGISTRATION.json','DATA_PLAN.json','E_RESULT.json','P_SYNTHETIC_QA.json',
                 'P_DATA_ACCEPTANCE.json','P_INITIAL_GRADIENT_SCALES.json','P_AUX72_READS.json','P_DOWNSTREAM_FIT_READS.json'):
        files.append(HERE/name)
    historical=read(ROOT/'research/wisig_w3/B_EXECUTION_REGISTRATION.json')['files_sha256']
    files.extend(ROOT/rel for rel in historical if rel.endswith('.py') and not rel.startswith('research/wisig_w3/'))
    files.append(ROOT/'research/wisig_w3/selective_pickle.py')
    write('EXECUTION_REGISTRATION.json',dict(status='REGISTERED_TWELVE_FITS_WITH_THREE_EXACT_A0_REUSES',time=now(),
        historical_base='598ba17',jobs=jobs,initial_gradient_scales=scales,actual_new_fits=12,maximum_authorized_new_fits=15,A0_reused=3,
        total_new_gradient_updates=56400,auxiliary_EMA_BN_calibration_order='Fixed balanced epoch0 order, seed41301+seed, 192 per batch',
        auxiliary_sampler='Epoch seed41301+seed+1009*epoch; 12 cycles, within-cell permutation and per-cycle TX permutation',
        new_target_or_D3_D4_or_historical_held6_IQ_authorized=False,second_source_view_training_authorized=False,
        files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}))
    print('REGISTERED 12 fits / 56,400 updates; no held-source or aux24 readout; commit before train',flush=True)


if __name__=='__main__':main()
