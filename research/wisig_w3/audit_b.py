"""Independent W3-B replay from features and saved operators, without IQ or torch.

Checks normal equations on fit-TX and recomputes held-TX predictions directly.
This file is a post-result verifier, not part of the prespecified estimator.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
METHODS=('identity','shift','diagonal','rank2','rank8','shuffled_rank8')
checks=[]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,ok):
    if not bool(ok):raise AssertionError(name)
    checks.append(name)
def close(name,a,b,tol=2e-10):check(name,np.allclose(a,b,rtol=tol,atol=tol))
def rows(p):
    with Path(p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def derange(n,seed):
    rng=np.random.default_rng(seed);permutation=list(range(n))
    for end in range(n-1,0,-1):
        other=int(rng.integers(0,end));permutation[end],permutation[other]=permutation[other],permutation[end]
    check('shuffle_has_no_fixed_TX',all(i!=v for i,v in enumerate(permutation)))
    return permutation
def ci(c,v):
    rng=np.random.default_rng(31013);draw=rng.integers(0,len(c),(20000,len(c)))
    d=(c-v)[draw].mean(1);rel=1-v[draw].mean(1)/c[draw].mean(1)
    return np.quantile(d,[.025,.975]),np.quantile(rel,[.025,.975])


def main(portable=False):
    reg=read(HERE/'B_EXECUTION_REGISTRATION.json');result=read(HERE/'B_RESULT.json')
    for rel,digest in reg['files_sha256'].items():check('registered_hash:'+rel,sha(ROOT/rel)==digest)
    attempt=read(HERE/'B_FEATURE_ATTEMPT.json');receipt=read(HERE/'B_FEATURE_RECEIPT.json')
    digest=sha(HERE/'B_EXECUTION_REGISTRATION.json')
    check('registration_bound_before_IQ',attempt['registration_sha256']==receipt['registration_sha256']==result['registration_sha256']==digest)
    if not portable:
        frozen=subprocess.check_output(['git','-C',str(ROOT),'show',attempt['registered_commit']+':research/wisig_w3/B_EXECUTION_REGISTRATION.json'])
        check('registration_existed_at_recorded_commit',hashlib.sha256(frozen).hexdigest()==digest)
        for rel in reg['files_sha256']:
            blob=subprocess.check_output(['git','-C',str(ROOT),'show',attempt['registered_commit']+':'+rel])
            actual=(ROOT/rel).read_bytes()
            # Historical text uses Git's LF normalization / Windows CRLF checkout.
            # The exact working-file digest has already been checked above.
            same=blob==actual or (Path(rel).suffix in ('.py','.json','.md') and blob.replace(b'\r\n',b'\n')==actual.replace(b'\r\n',b'\n'))
            check('registered_blob:'+rel,same)
    manifest=read(HERE/'B_SOURCE_MANIFEST.json');rr=read(HERE/'B_ROLES.json');access=read(HERE/'B_IQ_ACCESS_LOG.json')
    check('no_neural_fit_or_C_authorization',reg['maximum_new_neural_fits']==result['neural_fits']==receipt['neural_fits']==0 and not reg['W3_C_authorized'] and not result['W3_C_authorized'])
    tx=rr['TX'];rx=rr['RX'];contexts=rr['contexts'];ti={t:i for i,t in enumerate(tx)}
    check('96_auxiliary_TX_6_source_RX',len(tx)==96 and len(set(tx))==96 and len(rx)==6)
    check('target_TX_union_excluded',not(set(tx)&set(manifest['excluded_TX'])) and len(manifest['excluded_TX'])==11)
    keys={c['key']:c for c in manifest['cells']}
    check('complete_source_grid',len(keys)==1152 and {(c['tx'],c['rx'],c['day']) for c in keys.values()}=={(t,r,d) for t in tx for r in rx for d in (1,2)})
    check('only_raw_first32_rows',all(c['eq']==0 and c['rows']==[0,32] and c['shape'][1:]==[256,2] for c in keys.values()))
    ordered=sorted(tx,key=lambda t:hashlib.sha256(('W3-TX-v1|'+t).encode()).hexdigest())
    for i,c in enumerate(contexts):
        f,h=(ordered[:48],ordered[48:]) if i<2 else (ordered[48:],ordered[:48])
        check('fixed_TX_direction:'+c['id'],c['fit_TX']==f and c['held_TX']==h and not set(f)&set(h))
        check('cross_day_roles:'+c['id'],(c['fit_day'],c['eval_day'])==((1,2) if i%2==0 else (2,1)))
    log={a['key']:a for a in access['cells']}
    check('read_only_manifest_cells',len(access['cells'])==len(log)==1152 and set(log)==set(keys))
    for key,a in log.items():
        c=keys[key];check('admitted_read:'+key,a['rows']==c['rows'] and a['offset']==c['offset'] and a['bytes']==32*256*2*np.dtype(c['dtype']).itemsize)
    check('36864_rows_and_no_forbidden_read',access['rows_read']==36864 and access['forbidden_RX_day_TX_rows_read']==0)
    for name,p in receipt['models'].items():
        pre=read(HERE/'B_FEATURE_PREFLIGHT.json')['models'][name]
        check('frozen_state:'+name,p['before_state_sha256']==p['after_state_sha256']==pre['state_sha256'] and p['every_module_eval'] and p['no_gradients'])
    feature_path=HERE/'artifacts/B_FEATURES.npz';check('feature_cache_hash',sha(feature_path)==receipt['feature_cache_sha256']==result['feature_cache_sha256'])
    for rel,d in result['output_sha256'].items():check('scientific_output_hash:'+rel,sha(HERE/rel)==d)
    f=np.load(feature_path,allow_pickle=False);ops=np.load(HERE/'artifacts/B_OPERATORS.npz',allow_pickle=False)
    check('feature_axis_identities',f['TX'].tolist()==tx and f['RX'].tolist()==rx and f['days'].tolist()==[1,2])
    table=rows(HERE/'B_HELD_TX_METRICS.csv');geo=rows(HERE/'B_PAIR_GEOMETRY.csv')
    key=lambda a:(a['model'],a['context'],a['source_RX'],a['destination_RX'],a['method'],a['held_TX'])
    lookup={key(a):a for a in table};gkey=lambda a:(a['model'],a['context'],a['source_RX'],a['destination_RX'],a['method'])
    gl={gkey(a):a for a in geo}
    check('complete_nonduplicated_metric_tables',len(table)==len(lookup)==69120 and len(geo)==len(gl)==1440)
    summaries={};pairs=[(s,d) for s in range(6) for d in range(6) if s!=d]
    for name in ('EMA','ERM'):
        z=f[name].astype(np.float64);check('valid_frozen_features:'+name,z.shape==(96,6,2,32,64) and np.isfinite(z).all())
        mu=z.mean(3);ee=np.empty((4,30,6,48));aa=np.empty_like(ee);mm=np.empty_like(ee);gg=np.empty((4,30,6))
        for cidx,c in enumerate(contexts):
            fitids=[ti[t] for t in c['fit_TX']];heldids=[ti[t] for t in c['held_TX']];fd=c['fit_day']-1;ed=c['eval_day']-1
            basis=ops[f'{name}_{cidx}_basis'];cloud=mu[fitids,:,fd].reshape(-1,64);centered=cloud-cloud.mean(0)
            covariance=centered.T@centered;vals=np.linalg.eigvalsh(covariance)[::-1][:8]
            close('PCA_orthonormal',basis.T@basis,np.eye(8))
            close('PCA_top_fit_only_eigenvalues',basis.T@covariance@basis,np.diag(vals),tol=1e-8)
            close('PCA_fit_only_eigenvectors',covariance@basis,basis@np.diag(vals),tol=1e-8)
            for pidx,(s,d) in enumerate(pairs):
                xf=mu[fitids,s,fd];yf=mu[fitids,d,fd];xh=mu[heldids,s,ed];yh=mu[heldids,d,ed]
                prototypes=z[heldids,d,ed,:16].mean(1);query=z[heldids,s,ed,16:]
                for midx,method in enumerate(METHODS):
                    prefix=f'{name}_{cidx}_{pidx}_{method}_';center=ops[prefix+'center'];shift=ops[prefix+'shift']
                    ys=yf[derange(48,31013+cidx*100+pidx)] if method=='shuffled_rank8' else yf
                    close('fit_TX_center',center,xf.mean(0));close('fit_TX_shift',shift,(ys-xf).mean(0))
                    xc=xf-center;residual=ys-xf-shift
                    if method=='diagonal':
                        slope=ops[prefix+'diagonal'];ss=(xc*xc).sum(0);ridge=max(.1*ss.mean(),1e-12)
                        close('diagonal_fit_normal_equation',slope*(ss+ridge),(xc*residual).sum(0))
                    elif method in ('rank2','rank8','shuffled_rank8'):
                        k=2 if method=='rank2' else 8;p=ops[prefix+'projection'];w=ops[prefix+'weight']
                        close('registered_PCA_dimension',p,basis[:,:k]);design=xc@p;gram=design.T@design;ridge=max(.1*np.trace(gram)/k,1e-12)
                        close('fit_TX_only_ridge_normal_equation',(gram+ridge*np.eye(k))@w,design.T@residual)
                    def transform(v):
                        if method=='identity':return v.copy()
                        vout=v+shift
                        if method=='shift':return vout
                        if method=='diagonal':return vout+(v-center)*slope
                        return vout+(v-center)@p@w
                    pred=transform(xh);err=((pred-yh)**2).sum(1)/((yh*yh).sum(1)+1e-12)
                    q=transform(query.reshape(-1,64)).reshape(48,16,64)
                    # Direct difference distances, independent of executor's dot-product expansion.
                    distances=((q[:,:,None,:]-prototypes[None,None,:,:])**2).sum(-1)
                    labels=np.arange(48)[:,None];acc=(distances.argmin(-1)==labels).mean(1)
                    td=distances[np.arange(48),:,np.arange(48)]
                    alt=distances.copy();alt[np.arange(48),:,np.arange(48)]=np.inf
                    margin=((alt.min(-1)-td)/(distances.mean(-1)+1e-12)).mean(1)
                    i,j=np.triu_indices(48,1)
                    gr=np.median(((pred[i]-pred[j])**2).sum(1)/np.maximum(((xh[i]-xh[j])**2).sum(1),1e-12))
                    expected=[lookup[(name,c['id'],rx[s],rx[d],method,t)] for t in c['held_TX']]
                    close('held_error_replay',err,[float(a['transport_error']) for a in expected])
                    close('prototype_accuracy_replay',acc,[float(a['labeled_prototype_accuracy']) for a in expected])
                    close('prototype_margin_replay',margin,[float(a['normalized_margin']) for a in expected])
                    close('TX_pair_geometry_replay',gr,float(gl[(name,c['id'],rx[s],rx[d],method)]['median_TX_squared_distance_ratio']))
                    ee[cidx,pidx,midx]=err;aa[cidx,pidx,midx]=acc;mm[cidx,pidx,midx]=margin;gg[cidx,pidx,midx]=gr
        given=result['models'][name];simple=ee[:,:,:4].mean((1,3)).argmin(1)
        control=np.stack([ee[c,:,simple[c]] for c in range(4)]);control_acc=np.stack([aa[c,:,simple[c]] for c in range(4)]);control_margin=np.stack([mm[c,:,simple[c]] for c in range(4)])
        rel=1-ee[:,:,4].mean()/control.mean();close('global_error_effect:'+name,rel,given['relative_error_reduction'])
        for ci_,c in enumerate(contexts):
            check('strongest_simple_reference',METHODS[simple[ci_]]==given['contexts'][ci_]['strongest_simple_control'])
            close('context_error_effect',1-ee[ci_,:,4].mean()/control[ci_].mean(),given['contexts'][ci_]['relative_error_reduction'])
        clusters=[]
        for t in tx:
            entries=[]
            for cidx,c in enumerate(contexts):
                if t in c['held_TX']:
                    h=c['held_TX'].index(t);entries.append([control[cidx,:,h].mean(),ee[cidx,:,4,h].mean(),ee[cidx,:,5,h].mean()])
            check('two_dates_per_TX_cluster',len(entries)==2);clusters.append(np.mean(entries,0))
        clusters=np.array(clusters)
        for field,col in (('vs_strongest_simple_control',0),('vs_shuffled_rank8',2)):
            interval,relative=ci(clusters[:,col],clusters[:,1])
            close('TX_bootstrap_error_CI:'+name+field,interval,given[field]['paired_error_reduction_CI95'])
            close('TX_bootstrap_relative_CI:'+name+field,relative,given[field]['relative_error_reduction_CI95'])
        positive=int(((control-ee[:,:,4]).mean((0,2))>0).sum());check('positive_RX_pairs:'+name,positive==given['positive_directed_RX_pairs'])
        close('geometry_median',np.median(gg[:,:,4]),given['median_TX_squared_distance_ratio'])
        close('geometry_minimum',gg[:,:,4].min(),given['minimum_TX_squared_distance_ratio'])
        close('prototype_delta',100*(aa[:,:,4]-control_acc).mean(),given['labeled_prototype_accuracy_delta_pp'])
        close('margin_delta',(mm[:,:,4]-control_margin).mean(),given['normalized_margin_delta'])
        for m,method in enumerate(METHODS):
            close('method_error_mean',ee[:,:,m].mean(),given['method_means'][method]['transport_error'])
            close('method_proto_mean',aa[:,:,m].mean(),given['method_means'][method]['labeled_prototype_accuracy'])
        summaries[name]=given
    e=summaries['EMA'];er=summaries['ERM']
    derived=[e['relative_error_reduction']>=.15,all(c['relative_error_reduction']>0 for c in e['contexts']),e['positive_directed_RX_pairs']>=20,
        e['vs_strongest_simple_control']['paired_error_reduction_CI95'][0]>0,e['vs_shuffled_rank8']['paired_error_reduction_CI95'][0]>0,
        e['median_TX_squared_distance_ratio']>=.9,e['minimum_TX_squared_distance_ratio']>=.8,e['labeled_prototype_accuracy_delta_pp']>=-1,
        e['worst_context_prototype_accuracy_delta_pp']>=-5,e['normalized_margin_delta']>=-.01,er['relative_error_reduction']>0]
    check('all_11_prespecified_gates_reproduced',derived==list(result['gates'].values()))
    check('correct_STOP_decision',not all(derived) and result['status']=='STOP_W3B_LINEAR_TRANSPORT_HYPOTHESIS')
    report=dict(status='PASS',checks_passed=len(checks),portable_mode=portable,
        own_code_sha256=sha(__file__),official_IQ_rows_read_by_audit=0,neural_fits=0,
        operators_refitted=0,all_fit_normal_equations_verified=True,
        all_69120_held_TX_rows_replayed=True,all_1440_geometry_rows_replayed=True,
        bootstrap_recomputed=True,scientific_decision=result['status'],
        note='Audit uses frozen features, saved parameters and independent distance arithmetic; no target data or neural libraries.')
    (HERE/('B_PORTABLE_AUDIT_QA.json' if portable else 'B_INDEPENDENT_AUDIT_QA.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--portable',action='store_true');main(p.parse_args().portable)
