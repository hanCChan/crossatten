"""Registered source-only end-to-end competition, then a separate sealed scoring."""
from pathlib import Path
import argparse
import contextlib
import copy
import json
import math
import subprocess
import time
import traceback
import numpy as np
import torch
from w2_models import A,C,QualityGate,packed,ROOT
from run_w1 import digest,write,HERE


def register():
    if (HERE/'W2_REGISTRATION.json').exists():raise FileExistsError('Keep original W2 registration')
    assert A.read(HERE/'W1_RESULT.json')['status']=='W1_PASS_PROCEED_TO_FROZEN_W2_DESIGN'
    assert A.read(HERE/'W2_ENGINEERING.json')['status']=='PASS'
    source=A.read(ROOT/'research/standard_method_competition_c/PRE_REGISTRATION.json')
    plan=A.read(ROOT/'research/standard_method_competition_c/SOURCE_PLAN.json')
    refs=A.read(ROOT/'research/spectral_scale_execution/REFERENCE_INPUTS.json')['references']
    oldjobs=A.read(ROOT/'research/standard_method_competition_c/JOBS.json')
    references={};jobs=[]
    for view in ('V0','V1'):
        for k in range(3):
            stream='S'+str(k);r=refs[view+'/'+stream]['E0'];folder=Path(r['origin']).parent
            records={}
            for name,key in [('C1_selected.pt','checkpoint'),('SAMPLE_KEYS.npz','sample_keys'),('INITIAL.json','initial')]:
                p=folder/name;records[key]=dict(path=str(p),sha256=digest(p))
            if records['checkpoint']['sha256']!=r['sha256']:raise ValueError('Historical EMA checkpoint hash differs')
            records['val_predictions']=r['checkpoint_val_prediction']
            records['selection_predictions']=r['source_selection_prediction']
            old=next(j for j in oldjobs if j['id']==f'{view}_{stream}_wd1e3_ERM_EMA')
            records['recipe']=old;references[view+'/'+stream]=records
            for arm in ('B1','B2','N0'):
                job=copy.deepcopy(old);job.update(id=f'{view}_{stream}_{arm}',arm=arm,
                    shuffle_seeds=dict(fit_train=29101+k,checkpoint_val=29201+k,source_selection=29301+k))
                jobs.append(job)
    local_files=['w2_models.py','w2_engineering.py','w2_execute.py','W2_PROTOCOL.zh-CN.md','quality.py']
    dependencies=['research/standard_method_competition_c/engine.py','research/standard_method_competition_c/methods.py',
        'research/controlled_training_a/runner.py','13-mixer-cfo-mechanisms/wifi.py','17-rx-distortion/s17_models.py','lib/xa/explore_models.py']
    result=dict(status='REGISTERED_18_END_TO_END_FITS',date='2026-11-13',authorization='参考方案执行，可以根据现实成果，独立思考',
        W1_result_sha256=digest(HERE/'W1_RESULT.json'),engineering_sha256=digest(HERE/'W2_ENGINEERING.json'),
        source=source,views=plan['views'],references=references,jobs=jobs,
        code_sha256={n:digest(HERE/n) for n in local_files},dependency_sha256={n:digest(ROOT/n) for n in dependencies},
        environment=dict(torch=str(torch.__version__),numpy=np.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name()),
        total_new_scientific_fits=18,total_new_gradient_updates=91440,base_reused_fits=6)
    write('W2_REGISTRATION.json',result)
    print('Registered18 scientific fits / 91440 updates',flush=True)


def validate():
    p=HERE/'W2_REGISTRATION.json';A.require_committed(p)
    r=A.read(p)
    if r['status']!='REGISTERED_18_END_TO_END_FITS' or len(r['jobs'])!=18:raise ValueError('Unregistered budget')
    for name,h in r['code_sha256'].items():
        if digest(HERE/name)!=h:raise ValueError('Registered W2 code changed: '+name)
    for name,h in r['dependency_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Dependency changed: '+name)
    if digest(HERE/'W2_ENGINEERING.json')!=r['engineering_sha256']:raise ValueError('Engineering evidence changed')
    return r


def checked(record,action,job=None):
    p=Path(record['path']);h=digest(p)
    if h!=record['sha256']:raise ValueError('Input changed: '+str(p))
    with (HERE/'W2_ACCESS_LOG.jsonl').open('a',encoding='utf8') as f:
        f.write(json.dumps(dict(path=str(p),action=action,job=job,sha256=h))+'\n')
    return p


def metadata(reg):
    source=reg['source'];pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids','val_ids'):
        pool[name]=np.load(checked(source['source_input_whitelist'][name],'metadata'),allow_pickle=False)
    pool['keys']=A.make_keys(pool['fit_y'],pool['fit_rx'],pool['fit_day'],source['rx_names'],source['source_cell_sha256'])
    if set(pool['fit_day'])!={1,2}:raise ValueError('Date scope differs')
    return pool


def materialize(reg,pool,view,roles):
    ids=C.role_ids(reg['source'],view,pool)
    array=np.load(checked(reg['source']['source_input_whitelist']['fit_x'],'mmap_source_'+','.join(roles),view['id']),mmap_mode='r')
    if array.shape!=(34668,2,256):raise ValueError('IQ schema differs')
    data={}
    for role in roles:
        i=ids[role]
        data[role]=dict(x=np.array(array[i],copy=True),y=pool['fit_y'][i].astype('int64'),rx=pool['fit_rx'][i].astype('int64'),
            day=pool['fit_day'][i].astype('int64'),keys=pool['keys'][i],pool_ids=i.astype('int64'))
    return data


def preflight(reg,pool):
    evidence=[]
    for view in reg['views']:
        data=materialize(reg,pool,view,('fit_train','checkpoint_val'))
        for k in range(3):
            key=view['id']+'/S'+str(k);ref=reg['references'][key]
            with np.load(checked(ref['sample_keys'],'baseline_role_keys',key),allow_pickle=False) as p:
                for role,label in [('fit_train','train'),('checkpoint_val','val')]:
                    if not np.array_equal(data[role]['keys'],p[label+'_keys']) or not np.array_equal(data[role]['pool_ids'],p[label+'_ids']):
                        raise ValueError('B0 sample roles differ')
            A.setup(24101+k);m=QualityGate('B2',24101+k).cuda().eval()
            init=A.read(checked(ref['initial'],'baseline_initialization',key))
            if A.semantic_hash(m.backbone.state_dict())!=init['model_sha256']:raise ValueError('B0 backbone initialization differs')
            checkpoint=torch.load(checked(ref['checkpoint'],'baseline_selected_EMA',key),map_location='cpu',weights_only=False)
            m.backbone.load_state_dict(checkpoint['model'])
            x=torch.as_tensor(data['checkpoint_val']['x'],device='cuda')
            logits=C.predict_tensor(m.backbone,x,128)
            with np.load(checked(ref['val_predictions'],'baseline_prediction_replay',key),allow_pickle=False) as p:
                if not np.array_equal(p['pool_ids'],data['checkpoint_val']['pool_ids']) or not np.array_equal(p['logits'],logits):
                    raise ValueError('B0 selected EMA replay differs')
            evidence.append(dict(reference=key,exact_roles=True,exact_common_initialization=True,EMA_prediction_bitwise_replay=True))
            del m,x
    write('W2_PREFLIGHT.json',dict(status='PASS',references=evidence,new_scientific_fits=0))
    return evidence


def train(reg):
    if (HERE/'W2_TRAINING_ATTEMPT.json').exists():raise FileExistsError('No automatic retry of scientific training')
    pool=metadata(reg);preflight(reg,pool)
    # Gate must be committed before the first scientific update.
    subprocess.run(['git','-C',str(ROOT),'add',str(HERE/'W2_PREFLIGHT.json')],check=True)
    subprocess.run(['git','-C',str(ROOT),'-c','user.name=Codex','-c','user.email=codex@openai.com','commit','-m','Verify exact EMA baseline identity before quality-gate fits'],check=True,stdout=subprocess.DEVNULL)
    write('W2_TRAINING_ATTEMPT.json',dict(status='RUNNING',registration_sha256=digest(HERE/'W2_REGISTRATION.json'),time=time.time()))
    completed=[]
    try:
        for view in reg['views']:
            data=materialize(reg,pool,view,('fit_train','checkpoint_val'))
            for job in [j for j in reg['jobs'] if j['view']==view['id']]:
                out=HERE/'artifacts'/job['id']
                if out.exists():raise FileExistsError(out)
                out.mkdir(parents=True)
                train_data=copy.deepcopy(data['fit_train']);val=copy.deepcopy(data['checkpoint_val'])
                train_data['x'],tp=packed(train_data['x'],job['arm'],job['shuffle_seeds']['fit_train'])
                val['x'],vp=packed(val['x'],job['arm'],job['shuffle_seeds']['checkpoint_val'])
                np.savez_compressed(out/'CONTEXT_PAIRING.npz',train=tp,val=vp)
                A.setup(job['initialization_seed']);m=QualityGate(job['arm'],job['initialization_seed']).cuda()
                write('W2_LIVE.json',dict(status='RUNNING',job=job['id'],completed=len(completed),total=18))
                with (out/'TRAINING.log').open('x',encoding='utf8',buffering=1) as f,contextlib.redirect_stdout(f):
                    result=C.fit_loop(job,train_data,val,m,torch.device('cuda',0),out,reg['source']['rx_names'])
                # Named gradients and an unchanged backbone are NOT required:
                # this experiment explicitly trains the complete classifier.
                result['files_sha256']={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}
                A.write(out/'COMPLETE.json',result)
                completed.append(dict(id=job['id'],complete_sha256=digest(out/'COMPLETE.json'),updates=result['updates'],parameters=result['active_parameters']))
                write('W2_PROGRESS.json',dict(status='RUNNING',completed=completed,total=18))
                print('COMPLETE',job['id'],len(completed),'/18',flush=True)
                del m,train_data,val
        write('W2_TRAINING_SEAL.json',dict(status='ALL_18_FITS_COMPLETE_SCORING_NOT_STARTED',registration_sha256=digest(HERE/'W2_REGISTRATION.json'),
            completed=completed,new_scientific_fits=18,gradient_updates=sum(v['updates'] for v in completed),source_selection_scoring=0))
        print('ALL18_FITS_COMPLETE_READY_TO_COMMIT_SEAL',flush=True)
    except Exception:
        write('W2_FAILURE.json',dict(status='FAILED_NO_AUTOMATIC_RETRY',completed=completed,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['register','train']);args=parser.parse_args()
    if args.action=='register':register()
    else:train(validate())
