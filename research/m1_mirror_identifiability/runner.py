"""Preregistered M1-A physics only. Scientific arrays closed until committed registration."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import core as C

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(16<<20):h.update(b)
    return h.hexdigest()
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def new(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def access(p,action):
    with (HERE/'ACCESS_LOG.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(time=now(),path=str(p),action=action))+'\n')
def committed(p):
    relative=Path(p).resolve().relative_to(ROOT).as_posix()
    b=subprocess.check_output(['git','-C',str(ROOT),'show','HEAD:'+relative])
    if b.replace(b'\r\n',b'\n')!=Path(p).read_bytes().replace(b'\r\n',b'\n'):raise ValueError('Uncommitted prerequisite:'+relative)


def register():
    source=ROOT/'research/standard_method_competition_c/PRE_REGISTRATION.json';plan=ROOT/'research/standard_method_competition_c/SOURCE_PLAN.json'
    for p in (source,plan):access(p,'read_prior_protocol_JSON_only')
    r,p=read(source),read(plan)
    common=sorted(set.intersection(*(set(v['fit_rx']) for v in p['views'])))
    forbidden=set.union(*(set(v['source_selection_rx']) for v in p['views']))
    assert len(common)==6 and not set(common)&forbidden
    code={q.relative_to(ROOT).as_posix():sha(q) for q in sorted(HERE.glob('*.py'))}
    code['tests/test_m1_mirror.py']=sha(ROOT/'tests/test_m1_mirror.py')
    documents={q.name:sha(q) for q in sorted(HERE.glob('*.md'))}
    history={}
    for stage,files in {
        'periodic_hstem':['ARTIFACT_MANIFEST.json','AUDIT_RESULT.json'],
        'periodic_span':['ARTIFACT_MANIFEST.json','DECISION.json'],
        'spectral_scale_review':['ARTIFACT_MANIFEST.json','REGISTRATION.json'],
        'spectral_scale_execution':['SCIENCE_ARTIFACT_MANIFEST.json','SCIENCE_STATUS.json','DECISION.json'],
        'wisig_raw_eq_m0':['PUBLIC_SOURCE_MANIFEST.json','LOCAL_METADATA_REVIEW.json','DECISION.json']}.items():
        for name in files:
            q=ROOT/'research'/stage/name;access(q,'hash_historical_document_only');history[q.relative_to(ROOT).as_posix()]=sha(q)
    # Fixed relevant S09/S17/S19 documents/code; no diagnostics replay or payload.
    for name in ('09-phase-invariance/RESULTS.md','17-rx-distortion/s17_diagnostics.py','17-rx-distortion/D0_RESULTS.zh-CN.md','17-rx-distortion/decision_d0.json','19-tx-spectral-evidence/RESULTS.zh-CN.md','lib/xa/wisig.py','13-mixer-cfo-mechanisms/data.py'):
        q=ROOT/name
        if q.is_file():access(q,'hash_history_or_preparation_text');history[name]=sha(q)
    prior=read(ROOT/'research/wisig_raw_eq_m0/PUBLIC_SOURCE_MANIFEST.json')
    snapshots={}
    for item in prior['sources']:
        if item['repo']=='WiSig-dataset/wisig-process-raw':
            q=ROOT/'research/wisig_raw_eq_m0'/item['snapshot'];access(q,'read_official_text_snapshot')
            b=q.read_bytes();blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
            if blob!=item['sha']:raise ValueError('Official snapshot differs from pinned Git blob')
            snapshots[q.relative_to(ROOT).as_posix()]=dict(sha256=sha(q),git_blob=blob,url=item['url'])
    reg=dict(status='REGISTERED_M1_A_ONLY',base_commit='81ec47637594e764141a4774f6481a481b6d8ccd',registered_at=now(),
        user_authorization='M1-A design preregistration, synthetic tests and bounded source physics; no classifier or network fitting; publish this new research branch',
        code_sha256=code,document_sha256=documents,history_sha256=history,official_snapshots=snapshots,
        parent_protocol_sha256={source.relative_to(ROOT).as_posix():sha(source),plan.relative_to(ROOT).as_posix():sha(plan)},
        input_whitelist={k:r['source_input_whitelist'][k] for k in ('fit_x','fit_y','fit_rx','fit_day','train_ids')},
        rx_names=r['rx_names'],audit_rx=common,days=[1,2],sample_prefix='M1-A-v1',per_cell=16,min_cell=8,
        source_role='intersection of both views fit_train; no source_selection in either view',
        fs=25_000_000,window=[20,180],fit=[20,100],validation=[100,180],phase_step=.25,
        condition_max=10,fit_nmse_max=.15,validation_nmse_max=.15,cfo_abs_hz_max=200000,q_abs_max=.5,a_abs_min=.1,
        overall_valid_min=.9,rx_day_valid_min=.8,advantage_min=.02,bootstrap=dict(clusters=6,draws=10000,seed=27102),
        synthetic=dict(seed=27101,packets=1728,SNR_dB=[10,20,30],cfo_hz=[-100000,0,100000],shifts=[0,.5,3],replicates=8,valid_min=.9,noise_tolerance='6*sqrt(2)*sigma/sigma_min(U_fit)'),
        forbidden=['checkpoint_val IQ','source_selection IQ or predictions','historical held6','target8','days3-4','Full unknown RX','weights','embedding','cross-RX packet pairing','classification training','M1-B','M1-C'],
        maximum_classifier_fits=0,maximum_neural_updates=0,automatic_rescue=False)
    reg['user_request_sha256']=sha(HERE/'USER_REQUEST.txt')
    new(HERE/'REGISTRATION.json',reg)
    print(dict(status='REGISTERED_NO_SCIENTIFIC_ARRAY_READS',RX=common))


def validate():
    path=HERE/'REGISTRATION.json'
    if not path.exists():raise PermissionError('Committed M1-A registration required before source IO')
    committed(path);r=read(path)
    for group in (r['code_sha256'],r['history_sha256'],r['parent_protocol_sha256']):
        for name,digest in group.items():
            if sha(ROOT/name)!=digest:raise ValueError('Frozen code/evidence changed:'+name)
    for name,digest in r['document_sha256'].items():
        if sha(HERE/name)!=digest:raise ValueError('Frozen design changed:'+name)
    return r


def synthetic():
    r=validate()
    if (HERE/'SYNTHETIC_RESULTS.json').exists():raise FileExistsError('Keep original synthetic evidence')
    import unittest
    spec=importlib.util.spec_from_file_location('m1_tests',ROOT/'tests/test_m1_mirror.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))
    if not result.wasSuccessful():raise ValueError('Engineering suite failed')
    rng=np.random.default_rng(27101);est=C.Estimator();groups=[];max_ratio=0.;recoveries=0;rx_only=[]
    for tx in (0j,.08*np.exp(.4j)):
        for rx in (0j,.06*np.exp(-.7j)):
            for cfo in (-100000,0,100000):
                for shift in (0,.5,3):
                    for channel in (((0,1+0j),),((0,1+0j),(2,.2*np.exp(.8j)))):
                        for snr in (10,20,30):
                            clean=C.forward(np.arange(256),tx,rx,cfo,shift,channel)
                            sigma=float(np.sqrt(np.mean(abs(clean)**2)*10**(-snr/10)))
                            count=0;nmse=[]
                            for _ in range(8):
                                y=clean+sigma/np.sqrt(2)*(rng.normal(size=256)+1j*rng.normal(size=256));out=est.packet(y)
                                count+=out['valid'];nmse.append(out.get('models',{}).get('WL',{}).get('held_nmse',1.))
                                if cfo==0 and len(channel)==1:
                                    s=C.template(C.N[:80],shift);U=np.stack([s,s.conj()],1)
                                    fit=C.svd_fit(U,y[20:100]);h=channel[0][1]
                                    true=np.array([np.exp(.3j)*h+rx*np.exp(-.3j)*h.conjugate()*tx.conjugate(),np.exp(.3j)*h*tx+rx*np.exp(-.3j)*h.conjugate()])
                                    error=float(np.linalg.norm(fit['theta']-true));tol=6*np.sqrt(2)*sigma/np.linalg.svd(U,compute_uv=False)[-1]
                                    max_ratio=max(max_ratio,error/tol);recoveries+=1
                                    if error>tol:raise ValueError('Registered noise-dependent recovery tolerance failed')
                            groups.append(dict(tx_present=tx!=0,rx_present=rx!=0,cfo_hz=cfo,shift=shift,channel_taps=len(channel),snr_db=snr,packets=8,valid=count,held_nmse_mean=float(np.mean(nmse))))
            if tx==0 and rx!=0:
                s=C.template(C.N[:80]);U=np.stack([s,s.conj()],1)
                fit=C.svd_fit(U,C.forward(C.N[:80],tx,rx))
                rx_only.append(dict(effective_b_abs=float(abs(fit['theta'][1])),true_TX_nu=0,tx_iq_estimate=None))
    valid=sum(g['valid'] for g in groups);total=sum(g['packets'] for g in groups)
    diagnostics=[]
    for f in (0,100,1000,100000):
        U=C.full_design(C.N[:80],0,f);fit=C.svd_fit(U,C.forward(C.N[:80],.08,.06,f),max_condition=1e8)
        diagnostics.append(dict(cfo_hz=f,rank=fit['rank'],condition=fit['condition'],identifiable_at_known_nuisance=fit['valid']))
    n=C.N;U=np.stack([C.template(n-d) for d in range(12)],1);W=np.c_[U,U.conj()]
    rank_L=int(np.linalg.matrix_rank(U,tol=1e-10));rank_WL=int(np.linalg.matrix_rank(W,tol=1e-10))
    proof=dict(short_conjugate_halfshift_max_error=float(np.max(abs(C.template(n).conj()+1j*C.template(n+10)))),ordinary_channel_rank=rank_L,widely_linear_channel_rank=rank_WL,
        conjugate_in_channel_span_residual=float(np.linalg.norm(U@np.linalg.lstsq(U,C.template(n).conj(),rcond=1e-12)[0]-C.template(n).conj())/np.linalg.norm(C.template(n))))
    passes=valid/total>=.9 and max_ratio<=1 and proof['short_conjugate_halfshift_max_error']<1e-10 and rank_L==rank_WL==12
    value=dict(status='PASS' if passes else 'STOP_SYNTHETIC_SCOPE_NOT_RELIABLE',engineering_tests=result.testsRun,engineering_pass=True,
        packets=total,valid=valid,valid_fraction=valid/total,groups=groups,effective_coefficient_recovery_cases=recoveries,max_error_over_registered_tolerance=max_ratio,
        RX_only_false_TX_attribution_counterexamples=rx_only,full_design_diagnostics=diagnostics,structural_proofs=proof,
        true_TX_recovery='NOT_IDENTIFIABLE_NOT_RETURNED',physical_fingerprint_separability='NOT_ASSESSED',source_IQ_reads=0,classifier_fits=0,time=now())
    new(HERE/'SYNTHETIC_RESULTS.json',value);print(dict(status=value['status'],synthetic_packets=total,valid_fraction=value['valid_fraction']))


def load_source_metadata(r):
    validate()
    s=read(HERE/'SYNTHETIC_RESULTS.json')
    if s['status']!='PASS':raise PermissionError('Synthetic gate not passed; source IQ remains closed')
    committed(HERE/'SYNTHETIC_RESULTS.json');pool={}
    for name in ('fit_y','fit_rx','fit_day','train_ids'):
        item=r['input_whitelist'][name];p=Path(item['path']);access(p,'hash_and_decode_boundary_metadata:'+name)
        if sha(p)!=item['sha256']:raise ValueError('Input identity changed')
        pool[name]=np.load(p,allow_pickle=False)
    return pool


def summarize(rows,r):
    def group(items):
        ok=[x for x in items if x['metrics']['valid']]
        reasons={}
        for x in items:
            for reason in x['metrics']['reasons']:reasons[reason]=reasons.get(reason,0)+1
        return dict(packets=len(items),valid=len(ok),valid_fraction=len(ok)/len(items) if items else 0.,
            advantage_mean=float(np.mean([x['metrics']['advantage'] for x in items])) if items else 0.,failure_reasons=reasons,
            q_abs_mean_valid=float(np.mean([x['metrics']['q_abs'] for x in ok])) if ok else None)
    rx={name:group([x for x in rows if x['rx']==name]) for name in r['audit_rx']}
    days={str(d):group([x for x in rows if x['day']==d]) for d in (1,2)}
    rxday=[dict(rx=name,day=d,**group([x for x in rows if x['rx']==name and x['day']==d])) for name in r['audit_rx'] for d in (1,2)]
    cells=[dict(rx=name,day=d,tx=t,**group([x for x in rows if x['rx']==name and x['day']==d and x['tx']==t])) for name in r['audit_rx'] for d in (1,2) for t in range(10)]
    values=np.array([rx[name]['advantage_mean'] for name in sorted(rx)])
    draws=np.random.default_rng(27102).integers(0,6,(10000,6));ci=np.quantile(values[draws].mean(1),[.025,.975]).tolist()
    overall=group(rows);observed=[x for x in cells if x['packets']]
    linear_gain=float(np.mean([np.mean([x['metrics'].get('delta_L_WL',0.) if x['metrics']['valid'] else 0. for x in rows if x['rx']==name]) for name in r['audit_rx']]))
    gates=dict(overall_valid_at_least90=overall['valid_fraction']>=.9,every_RX_day_at_least80=all(x['valid_fraction']>=.8 for x in rxday),
        present_cell_support_at_least8=bool(observed) and all(x['packets']>=8 for x in observed),
        all_RX_days_present=all(x['packets']>0 for x in rxday),mean_validation_advantage_at_least2percent=float(values.mean())>=.02,
        all_RX_advantage_positive=bool((values>0).all()),both_days_advantage_positive=all(x['advantage_mean']>0 for x in days.values()),
        RX_conditional_interval_lower_positive=ci[0]>0,WL_beats_L_on_held_segment=linear_gain>0)
    return dict(overall=overall,per_rx=rx,per_day=days,per_rx_day=rxday,cells=cells,mean_advantage_RX_equal=float(values.mean()),
        advantage_RX_conditional95=ci,linear_to_WL_RX_equal_gain=linear_gain,gates=gates,
        decision='M1_A_PASSED_B_REQUIRES_SEPARATE_AUTHORIZATION' if all(gates.values()) else 'STOP_EXPLICIT_MIRROR_ESTIMATOR')


def source():
    r=validate()
    if (HERE/'SOURCE_ATTEMPT.json').exists():raise FileExistsError('Keep original source attempt; no rescue/retry')
    new(HERE/'SOURCE_ATTEMPT.json',dict(status='STARTED',time=now(),registration_sha256=sha(HERE/'REGISTRATION.json')))
    try:
        pool=load_source_metadata(r);y,rx,day=[pool['fit_'+k] for k in ('y','rx','day')];train=pool['train_ids']
        if not(y.shape==rx.shape==day.shape==(34668,)) or set(day)!={1,2}:raise ValueError('Source pool scope mismatch')
        selected=[]
        for name in r['audit_rx']:
            code=r['rx_names'].index(name)
            for t in range(10):
                for d in (1,2):
                    ids=train[(rx[train]==code)&(y[train]==t)&(day[train]==d)]
                    selected.extend(sorted(map(int,ids),key=lambda i:hashlib.sha256(f'M1-A-v1|{i}'.encode()).digest())[:16])
        ids=np.array(selected,dtype=np.int64)
        if not np.isin(ids,train).all() or len(np.unique(ids))!=len(ids):raise ValueError('Invalid source sample')
        item=r['input_whitelist']['fit_x'];p=Path(item['path']);access(p,'hash_registered_source_file_bytes')
        if sha(p)!=item['sha256']:raise ValueError('Source IQ digest changed')
        array=np.load(p,mmap_mode='r',allow_pickle=False)
        if array.shape!=(34668,2,256) or array.dtype!=np.float32:raise ValueError('Source IQ layout mismatch')
        access(p,'copy_only_registered_common_fit_train_rows:'+str(len(ids)))
        samples=np.array(array[ids],copy=True);est=C.Estimator();rows=[]
        for j,(i,x) in enumerate(zip(ids,samples)):
            row=dict(pool_row=int(i),rx=r['rx_names'][int(rx[i])],tx=int(y[i]),day=int(day[i]),metrics=est.packet(x[0].astype(float)+1j*x[1]))
            rows.append(row)
            if (j+1)%200==0:print('source packets audited',j+1,flush=True)
        result=summarize(rows,r)
        from scipy.stats import spearmanr
        usable=[x['metrics'] for x in rows if 'models' in x['metrics']];correlations={}
        for name,feature in [('abs_CFO',lambda m:abs(m['cfo_hz'])),('fit_nmse',lambda m:m['models']['WL']['fit_nmse']),('channel_proxy_cv',lambda m:m['channel_proxy_cv'])]:
            stat=spearmanr([feature(m) for m in usable],[int(m['valid']) for m in usable]);correlations[name]=None if not np.isfinite(stat.statistic) else float(stat.statistic)
        new(HERE/'PACKET_DIAGNOSTICS.json',dict(rows=rows,contains_IQ=False,tx_parameters_separated=False))
        new(HERE/'SOURCE_VALIDITY.json',dict(status='COMPLETED',**result,correlations_with_validity_descriptive=correlations,
            registration_sha256=sha(HERE/'REGISTRATION.json'),packet_diagnostics_sha256=sha(HERE/'PACKET_DIAGNOSTICS.json'),
            source_selection_reads=0,checkpoint_val_IQ_reads=0,classifier_fits=0,time=now()))
        new(HERE/'DECISION.json',dict(status=result['decision'],gates=result['gates'],M1_B_authorized=False,M1_C_authorized=False,
            neural_updates=0,classifier_fits=0,interpretation='Only the registered gauge-defined effective estimator; no TX/RX causal separation or TX separability conclusion',closed=True))
        print(result['decision'])
    except Exception:
        new(HERE/'SOURCE_FAILURE.json',dict(status='STOP_ENGINEERING_OR_INPUT_FAILURE',traceback=traceback.format_exc(),automatic_retry=False,time=now()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','synthetic','source']);args=p.parse_args()
    {'register':register,'synthetic':synthetic,'source':source}[args.command]()
