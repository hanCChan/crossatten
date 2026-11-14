"""Post-fit, pre-held independent checks of transfer, optimizer reset and pairing."""
import numpy as np
import torch
from common import *
from p_models import A


def main():
    validate();seal=read(HERE/'P_MODEL_SEAL.json');plan=read(HERE/'DATA_PLAN.json');reg=read(HERE/'EXECUTION_REGISTRATION.json')
    assert seal['new_fits']==6 and len(seal['runs'])==6 and seal['new_gradient_updates']==28200
    assert seal['held_source_IQ_rows_read']==seal['aux24_IQ_rows_read']==0
    evidence={};checks=[];histories={};initials={}
    for run in seal['runs']:
        for key in ('complete','selected_checkpoint','selected_predictions'):assert sha(run[key]['path'])==run[key]['sha256']
        job=run['job'];folder=Path(run['complete']['path']).parent;h=read(folder/'HISTORY.json');histories[job]=h
        ck=torch.load(run['selected_checkpoint']['path'],map_location='cpu',weights_only=False)
        assert len(h)==40 and h[-1]['updates']==(4320 if job.endswith('_aux') else 5080)
        if job.endswith('_aux'):
            best=int(np.argmin([row['EMA_val_RX_CE'] for row in h]))+1
            assert ck['epoch']==best and min(row['minimum_relative_spread'] for row in h)>=1e-6
            seed=int(job.split('_')[1][1:]);arm=job[:2];scale=reg['initial_gradient_scales'][str(seed)]
            assert read(folder/'INITIAL.json')['model_sha256']==scale['initial_state_sha256']
            assert ck['lambda_used']==(0 if arm=='A1' else scale['lambda_value'])
            evidence[job]=dict(history=h,initial=read(folder/'INITIAL.json'),selected_epoch=ck['epoch'])
        else:
            best=int(np.argmin([row['val_rx_ce']['C1'] for row in h]))+1;assert ck['epoch']==best
            assert all(row['calibration_preserved_training_state'] for row in h)
            initials[job]=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
            evidence[job]=dict(history=h,initial=read(folder/'INITIAL.json'),selected_epoch=ck['epoch'])
        checks.append(dict(job=job,all_checkpoint_hashes_pass=True,fixed_40_epochs=True,selected_epoch_is_first_minimum=True))
    transfer_checks=[]
    for seed in range(3):
        ref=plan['A0'][str(seed)];folder=Path(ref['checkpoint']['path']).parent
        baseline=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
        assert A.semantic_hash(baseline['student'])==read(ref['initialization']['path'])['model_sha256']
        original_history=read(folder/'HISTORY.json')
        complete=read(folder/'COMPLETE.json');assert sha(folder/'HISTORY.json')==complete['files_sha256']['HISTORY.json']
        evidence[f'A0_S{seed}_downstream']=dict(history=original_history,initial=read(ref['initialization']['path']),selected_epoch=complete['best_epochs']['C1'])
        for arm in ('A1',):
            job=f'{arm}_S{seed}_downstream';state=initials[job];aux=torch.load(ART/f'{arm}_S{seed}_aux/selected.pt',map_location='cpu',weights_only=False)['model']
            for key,value in state['student'].items():
                assert torch.equal(value,baseline['student'][key] if key.startswith('classifier.') else aux[key]),key
            assert state['optimizer']['state']=={} and state['optimizer']['param_groups']==baseline['optimizer']['param_groups']
            assert state['scheduler']==baseline['scheduler']
            for key in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
                assert [r[key] for r in histories[job]]==[r[key] for r in original_history]
            transfer_checks.append(dict(model=job,encoder_and_BN_equal_selected_auxiliary_EMA=True,
                ten_class_head_equal_A0_random_initialization=True,optimizer_and_scheduler_reset_equal_A0=True,
                all_40_training_orders_exactly_equal_A0=True))
    role_meta={}
    for name,prefix in (('P_AUX_TRAIN.npz','aux'),('P_DOWNSTREAM_TRAIN.npz','downstream')):
        with np.load(ART/name,allow_pickle=False) as p:
            role_meta.update({prefix+'_'+k:p[k] for k in p.files if not k.endswith('_x')})
    np.savez_compressed(ART/'P_ROLE_METADATA.npz',**role_meta)
    write(ART/'P_TRAINING_EVIDENCE.json',dict(records=evidence))
    result=dict(status='PASS_PRE_HELD_TRAINING_AND_TRANSFER_AUDIT',time=now(),own_code_sha256=sha(__file__),
        fits=checks,transfers=transfer_checks,W4_A1_fixed_algorithm=True,
        A0_reuses=3,new_fits=6,total_new_updates=28200,held_source_or_aux24_IQ_reads=0,
        role_metadata=record(ART/'P_ROLE_METADATA.npz'),history_evidence=record(ART/'P_TRAINING_EVIDENCE.json'))
    write('P_TRAINING_QA.json',result);print('PASS 6 fixed fits, three exact EMA transfers/head/optimizer resets and 40-epoch A0 order matches',flush=True)


if __name__=='__main__':main()
