"""Metadata-only W4 admission before reading new IQ or paired prediction values."""
import hashlib
from common import *


def plan():
    source=read(ROOT/'research/standard_method_competition_c/PRE_REGISTRATION.json')
    view=read(ROOT/'research/standard_method_competition_c/SOURCE_PLAN.json')['views'][0]
    w3=read(ROOT/'research/wisig_w3/B_SOURCE_MANIFEST.json')
    assert view['id']=='V0' and set(w3['source_RX'])<=set(view['fit_rx'])
    assert not set(w3['source_RX'])&set(view['source_selection_rx']+source['forbidden_target_rx']+source['forbidden_historical_held_rx'])
    aux=sorted({c['tx'] for c in w3['cells']},key=lambda t:hashlib.sha256(('W4-TX-v1|'+t).encode()).hexdigest())
    train,hold=aux[:72],aux[72:]
    assert len(aux)==96 and len(train)==72 and len(hold)==24 and not set(aux)&set(w3['excluded_TX'])
    cells=w3['cells'];assert len(cells)==1152 and all(c['rows']==[0,32] and c['eq']==0 and c['day'] in (1,2) for c in cells)
    references={};jobs=read(ROOT/'research/standard_method_competition_c/JOBS.json')
    refs=read(ROOT/'research/spectral_scale_execution/REFERENCE_INPUTS.json')['references']
    for seed in range(3):
        tag=f'V0/S{seed}';r=refs[tag]['E0'];folder=Path(r['origin']).parent
        assert sha(r['origin'])==r['sha256']
        references[str(seed)]=dict(checkpoint=dict(path=r['origin'],sha256=r['sha256']),
            validation=r['checkpoint_val_prediction'],selection=r['source_selection_prediction'],
            initialization=record(folder/'INITIAL.json'),sample_keys=record(folder/'SAMPLE_KEYS.npz'),
            recipe=next(j for j in jobs if j['id']==f'V0_S{seed}_wd1e3_ERM_EMA'))
    write('DATA_PLAN.json',dict(status='METADATA_PASS_NO_IQ_VALUES_DECODED',historical_base='598ba17',source=source,view=view,
        aux_train_TX=train,aux_diagnostic_TX=hold,aux_RX=w3['source_RX'],excluded_target_TX=w3['excluded_TX'],
        source_manifest=dict(path=str(ROOT/'research/wisig_w3/B_SOURCE_MANIFEST.json'),sha256=sha(ROOT/'research/wisig_w3/B_SOURCE_MANIFEST.json')),
        raw_pickle=dict(path=w3['path'],sha256=w3['sha256'],bytes=w3['bytes']),cells=cells,
        aux_fit_rows=[0,24],aux_val_rows=[24,32],diagnostic_rows=[0,32],
        aux_fit_packets=20736,aux_val_packets=6912,aux_diagnostic_packets=9216,A0=references,
        raw_historical_held6_or_real_target_access=False))
    pairs=[];manifest=read(S20/'MANIFEST.json')
    for fold in range(4):
        ask=S20/'runs'/f'ASK__f{fold}__first__s18101';hst=S19/'runs'/f'ManyRx__B0__f{fold}__first__s18101'
        ac=read(ask/'config.json');hc=read(hst/'config.json')
        cache=Path(f'D:/crossatten/worktrees/s17-rx-distortion/17b-phase-intervention/cache/ManyRx_f{fold}_first_s16001')
        fm=read(cache/'fit_manifest.json')
        assert fm['source_only'] and set(ac['fit_rx'])==set(fm['fit_rx']) and ac['held_rx']==fm['held_rx']
        assert not set(ac['fit_rx'])&set(ac['held_rx'])
        assert hc['job']['dataset']=='ManyRx' and hc['job']['fold']==fold and hc['data_seed']==ac['data_seed']
        assert hc['source_split_seed']==ac['split_seed']==16001
        ah=read(ask/'held_metrics.json');hh=read(hst/'held_metrics.json')
        assert ah['held_sources']==hh['held_sources'] and ah['rx_names']==hh['rx_names']==fm['rx_names']
        inputs={}
        for name,directory in (('ASK',ask),('HStem',hst)):
            hm=read(directory/'held_metrics.json');m=read(directory/'metrics.json')
            held=record(directory/'held_predictions.npz');val=record(directory/'source_val_predictions.npz')
            assert held['sha256']==hm['prediction_sha256']==manifest['local_predictions'][f'{"ASK" if name=="ASK" else "B0"}_f{fold}']
            assert val['sha256']==m['source_predictions_sha256' if name=='ASK' else 'prediction_sha256']
            inputs[name]=dict(held=held,fit_validation=val,config=record(directory/'config.json'),held_metadata=record(directory/'held_metrics.json'))
        pairs.append(dict(fold=fold,fit_RX=ac['fit_rx'],held_RX=ac['held_rx'],rx_names=ah['rx_names'],
            held_source_cells=ah['held_sources'],source_manifest=record(cache/'fit_manifest.json'),
            train_indices=dict(path=str(cache/'train_ids.npy'),sha256=fm['sha256']['train_ids.npy']),
            validation_indices=dict(path=str(cache/'val_ids.npy'),sha256=fm['sha256']['val_ids.npy']),models=inputs))
    write('E_INPUTS.json',dict(status='METADATA_PASS_SAME_SOURCE_CELLS_AND_TRAINING_EXCLUSION',pairs=pairs,
        identity_limitation='No explicit packet IDs in historical NPZ; canonical provenance/order-derived keys must be checked at readout.',
        raw_IQ_reads=0,neural_fits=0,only_saved_S20_source_predictions=True))
    files=[HERE/'PREREGISTRATION.zh-CN.md',HERE/'DATA_PLAN.json',HERE/'E_INPUTS.json',HERE/'prepare.py',HERE/'common.py']
    write('ADMISSION_REGISTRATION.json',dict(status='METADATA_AND_SCOPE_FROZEN_BEFORE_W4_SCIENTIFIC_READOUT',time=now(),
        maximum_new_neural_fits=15,maximum_pretrain_fits=6,maximum_downstream_fits=9,expected_A0_reused=3,
        files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},forbidden_real_target_or_D3_D4=True,
        E_only_historical_prediction_exception=True))
    print('PASS W4 metadata: 72/24 auxiliary TX, exact V0 source roles, four S20 prediction pairs')


if __name__=='__main__':plan()
