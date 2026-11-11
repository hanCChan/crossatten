"""Audit engineering evidence and synthetic states; never decode scientific arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

import common as H
import executor as E
import kernel as K

HERE,ROOT=H.HERE,H.ROOT


def audit():
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('Engineering QA failed: '+name)
    check('scientific_authorization_absent',not (HERE/'EXECUTION_AUTHORIZATION.json').exists())
    check('scientific_attempts_zero',not list((HERE/'artifacts/scientific').glob('*')))
    check('scientific_access_log_absent',not (HERE/'SCIENTIFIC_ACCESS_LOG.jsonl').exists())
    check('source_preflight_not_started',not (HERE/'SOURCE_PREFLIGHT_ATTEMPT.json').exists())
    try:E.scientific_context()
    except PermissionError:blocked=True
    else:blocked=False
    check('source_entry_rejected_before_access',blocked)
    H.validate_design();jobs=H.validate_budget(H.read(H.DESIGN/'PROPOSED_JOBS.json'))
    check('future_budget_unchanged',sum(j['max_gradient_updates'] for j in jobs)==60960 and sum(j['epochs'] for j in jobs)==480)
    # These manifests are read as bytes/JSON; no old IQ or prediction arrays.
    for stage in ('periodic_hstem','periodic_span','spectral_scale_review'):
        directory=ROOT/'research'/stage
        for name,item in H.read(directory/'ARTIFACT_MANIFEST.json')['files'].items():
            check('old_stage_unchanged:'+stage+'/'+name,H.sha(directory/name)==item['sha256'])
    cpu=H.read(HERE/'CPU_ACCEPTANCE.json');cuda=H.read(HERE/'CUDA_ACCEPTANCE.json')
    check('CPU_eight_tests',cpu['status']=='PASS' and cpu['tests']==8 and cpu['failures']==cpu['errors']==0)
    check('CUDA_full40_epoch_acceptance',cuda['status']=='PASS_FULL40_EPOCH_CUDA_SYNTHETIC_EXECUTOR' and all(cuda['checks'].values()))
    check('CUDA_budget_synthetic_only',(cuda['synthetic_optimizer_updates'],cuda['synthetic_gradient_epochs'],cuda['scientific_fits'],cuda['source_IQ_reads'],cuda['source_prediction_decodes'])==(240,80,0,0,0))
    registry=H.read(HERE/'REFERENCE_INPUTS.json')['references'];recovery=H.read(HERE/'REFERENCE_RECOVERY.json')
    check('twelve_reference_records',len(recovery['references'])==12)
    expected_paths=set()
    for key,group in registry.items():
        for role,item in group.items():
            proof=recovery['references'][key+'/'+role];copy=HERE/item['copy_relative']
            expected_paths.update((str(Path(item['origin']).resolve()),str(copy.resolve())))
            check('reference_SHA:'+key+'/'+role,H.sha(copy)==proof['copy_sha256']==proof['origin_sha256']==item['sha256'])
            check('reference_CUDA_parity:'+key+'/'+role,proof['max_abs_original_vs_frozen_logit']<=1e-6 and proof['identical_argmax'])
            check('reference_freeze_and_tokens:'+key+'/'+role,proof['all_backbone_parameters_frozen'] and proof['frozen_buffers_unchanged'] and proof['token_shape']==[128,32,64])
            if role=='F0':check('reference_alpha0_bitwise:'+key,proof['alpha0_branch_bitwise']=={'F1':True,'F2':True})
            for which in ('checkpoint_val_prediction','source_selection_prediction'):
                # Existence only: byte identities came from the prior sealed manifests.
                check('archive_available_without_decoding:'+key+'/'+role+'/'+which,Path(item[which]['path']).is_file())
    accesses=[json.loads(line) for line in (HERE/'ENGINEERING_ACCESS_LOG.jsonl').read_text(encoding='utf-8').splitlines()]
    check('only24_registered_checkpoint_accesses',len(accesses)==24 and {str(Path(r['path']).resolve()) for r in accesses}==expected_paths)
    check('only_checkpoint_actions',all(Path(r['path']).suffix=='.pt' and r['action'] in ('verify_original_checkpoint_SHA_and_copy','hash_and_load_registered_source_checkpoint') for r in accesses))
    check('source_parity_pending',recovery['source_checkpoint_val_parity']=='PENDING_EXPLICIT_EXECUTION_AUTHORIZATION')
    check('no_source_data_in_recovery',recovery['source_IQ_reads']==recovery['source_prediction_decodes']==recovery['scientific_gradient_fits']==0)
    root=HERE/cuda['synthetic_artifact_directory'];local_files={}
    for path in sorted(root.rglob('*')):
        if path.is_file():local_files[path.relative_to(HERE).as_posix()]=dict(bytes=path.stat().st_size,sha256=H.sha(path))
    histories={};initials={}
    for arm in ('F1','F2'):
        out=root/arm;initials[arm]=H.read(out/'INITIAL.json');history=H.read(out/'HISTORY.json');histories[arm]=history
        with np.load(out/'GRADIENT_TRACE.npz',allow_pickle=False) as f:columns=list(f['columns']);trace=f['values']
        rows=[json.loads(v) for v in (out/'GRADIENT_TRACE.jsonl').read_text(encoding='utf-8').splitlines()]
        check(arm+'/complete240_trace_half',trace.shape==(120,len(K.TRACE_COLUMNS)) and columns==list(K.TRACE_COLUMNS))
        check(arm+'/trace_journal_matches',np.array_equal(trace,np.asarray([[r[c] for c in columns] for r in rows])))
        check(arm+'/all_values_finite',np.isfinite(trace).all())
        for c in ('all_task_gradients_finite','all_updated_parameters_finite','backbone_gradients_absent'):
            check(arm+'/'+c,(trace[:,columns.index(c)]==1).all())
        check(arm+'/initial_task_gradient_and_decay',trace[0,8]==0 and trace[0,7]!=0 and trace[0,9]>0)
        check(arm+'/later_task_side_gradient',(trace[1:,8]>0).any())
        initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
        initial_model=initial['model'];backbone={k.removeprefix('backbone.'):v for k,v in initial_model.items() if k.startswith('backbone.')}
        frozen=H.semantic_hash(backbone)
        check(arm+'/initial_backbone_identity',frozen==initials[arm]['frozen_backbone_sha256'])
        check(arm+'/initial_alpha_zero',float(initial_model['residual.alpha'])==0)
        with np.load(out/'SAMPLE_KEYS.npz',allow_pickle=False) as f:keys={k:f[k] for k in f.files}
        check(arm+'/synthetic_keys_only',all(str(k).startswith('synthetic|') for k in np.r_[keys['train_keys'],keys['val_keys']]))
        check(arm+'/all40_epochs',len(history)==40)
        for row in history:
            epoch=row['epoch'];state=torch.load(out/'epoch_states'/f'epoch_{epoch:02d}.pt',map_location='cpu',weights_only=False)
            check(f'{arm}/epoch{epoch}/full_state_hash',H.semantic_hash(state)==row['state_sha256'])
            state_backbone={k.removeprefix('backbone.'):v for k,v in state['model'].items() if k.startswith('backbone.')}
            check(f'{arm}/epoch{epoch}/frozen_state_including_BN',H.semantic_hash(state_backbone)==frozen)
            check(f'{arm}/epoch{epoch}/alpha_trace',float(state['model']['residual.alpha'])==row['alpha']==trace[epoch*3-1,6])
            positions=np.load(out/'orders'/f'epoch_{epoch:02d}.npy',allow_pickle=False)
            check(f'{arm}/epoch{epoch}/permutation',np.array_equal(np.sort(positions),np.arange(24)))
            check(f'{arm}/epoch{epoch}/order_hash',H.semantic_hash(positions)==row['relative_order_sha256'] and H.semantic_hash(keys['train_keys'][positions])==row['canonical_order_sha256'])
        complete=H.read(out/'COMPLETE.json')
        scores=[initials[arm]['epoch0_val_rx_ce']]+[r['val_rx_ce'] for r in history]
        check(arm+'/checkpoint0_to40_earliest_min',complete['best_epoch']==min(range(41),key=lambda i:scores[i]))
        for endpoint,digest in complete['checkpoints'].items():check(arm+'/checkpoint_SHA/'+endpoint,H.sha(out/(endpoint+'.pt'))==digest)
    for field in ('frozen_backbone_sha256','common_residual_sha256','rng_sha256','scheduler_sha256'):
        check('paired_initial/'+field,initials['F1'][field]==initials['F2'][field])
    for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
        check('paired40epoch/'+field,all(histories['F1'][i][field]==histories['F2'][i][field] for i in range(40)))
    return dict(status='PASS',check_count=len(checks),checks=checks,scientific_fits=0,source_IQ_reads=0,source_prediction_decodes=0,
        synthetic_state_decodes=82,synthetic_optimizer_updates_by_verifier=0,time=H.now()),local_files


def verify_public():
    ex=H.read(HERE/'EXECUTION_REGISTRATION.json')
    for name,digest in ex['code_sha256'].items():
        if H.sha(ROOT/name)!=digest:raise ValueError('Execution code changed:'+name)
    for name,digest in ex['artifact_sha256'].items():
        if H.sha(HERE/name)!=digest:raise ValueError('Engineering evidence changed:'+name)
    for name,item in H.read(HERE/'ARTIFACT_MANIFEST.json')['files'].items():
        if H.sha(HERE/name)!=item['sha256']:raise ValueError('Published engineering file changed:'+name)
    with zipfile.ZipFile(HERE/'EXECUTION_CODE_BYTES.zip') as z:
        if set(z.namelist())!=set(ex['code_sha256']):raise ValueError('Incomplete code byte archive')
        for name,digest in ex['code_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Archived code bytes differ:'+name)
    manifest=H.read(HERE/'SYNTHETIC_ARCHIVE.json')
    with zipfile.ZipFile(HERE/'SYNTHETIC_TRACES.zip') as z:
        if set(z.namelist())!=set(manifest['files']):raise ValueError('Synthetic trace archive member mismatch')
        for name,item in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=item['sha256']:raise ValueError('Synthetic trace archive changed')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--record',action='store_true');parser.add_argument('--public',action='store_true');args=parser.parse_args()
    value,local=audit()
    if args.record:
        if (HERE/'ENGINEERING_QA.json').exists():raise FileExistsError('Keep original engineering QA evidence')
        H.write(HERE/'ENGINEERING_QA.json',value)
        H.write(HERE/'LOCAL_SYNTHETIC_MANIFEST.json',dict(files=local,source_checkpoints_included=False,source_IQ_included=False))
    else:
        if local!=H.read(HERE/'LOCAL_SYNTHETIC_MANIFEST.json')['files']:raise ValueError('Local synthetic artifacts changed')
    if args.public:verify_public()
    print(dict(status='PASS',checks=value['check_count'],public_archive_checked=args.public,scientific_fits=0,source_IQ_reads=0))


if __name__=='__main__':main()
