"""Independent NumPy-only replay of W4 classification, gates, roles and diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,ast,csv,hashlib,json
from pathlib import Path,PureWindowsPath
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;ART=HERE/'artifacts';COUNT=0
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,label):
    global COUNT
    if not bool(ok):raise AssertionError(label)
    COUNT+=1
def near(a,b,label,tol=2e-9):require(np.allclose(a,b,atol=tol,rtol=tol),label)
def csv_rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def canon(p):return str(p).replace('\\','/')


def statistics(logits,y):
    pred=logits.argmax(1);f1=[]
    for label in range(10):
        true=y==label;positive=pred==label;tp=np.sum(true&positive);den=np.sum(true)+np.sum(positive)
        f1.append(2*tp/den if den else 0)
    l=logits.astype(np.float64);maxi=l.max(1)
    ce=maxi+np.log(np.exp(l-maxi[:,None]).sum(1))-l[np.arange(len(y)),y]
    return dict(accuracy_pct=100*float((pred==y).mean()),macro_F1_pct=100*float(np.mean(f1)),CE=float(ce.mean()))


def main(portable=False):
    index=read(HERE/'REPLAY_FILES.json') if (HERE/'REPLAY_FILES.json').exists() else {'files':[]}
    aliases={canon(v['source']):v for v in index['files']}
    def resolve(item):
        if portable:
            record=aliases.get(canon(item['path']));require(record is not None,'portable input has a relative alias')
            p=ROOT/record['relative']
        else:p=Path(item['path'])
        require(sha(p)==item['sha256'],'exact artifact SHA256:'+Path(p).name)
        return p
    for name in ('ADMISSION_REGISTRATION.json','E_REGISTRATION.json','EXECUTION_REGISTRATION.json','P_EVALUATION_REGISTRATION.json'):
        for rel,digest in read(HERE/name)['files_sha256'].items():require(sha(ROOT/rel)==digest,'bound code/evidence:'+rel)
    plan=read(HERE/'DATA_PLAN.json');aux_tx=sorted({c['tx'] for c in plan['cells']},key=lambda t:hashlib.sha256(('W4-TX-v1|'+t).encode()).hexdigest())
    require(plan['aux_train_TX']==aux_tx[:72] and plan['aux_diagnostic_TX']==aux_tx[72:] and len(aux_tx)==96,'fixed 72/24 TX split')
    require(not set(aux_tx)&set(plan['excluded_target_TX']),'excluded target TX union')
    require(set(plan['aux_RX'])<=set(plan['view']['fit_rx']),'auxiliary RX contained in V0 fit12')
    require(not set(plan['aux_RX'])&set(plan['view']['source_selection_rx']),'no auxiliary held-source RX')
    forbidden=set(plan['source']['forbidden_target_rx']+plan['source']['forbidden_historical_held_rx'])
    require(not set(plan['view']['fit_rx']+plan['view']['source_selection_rx'])&forbidden,'P dataset source-only physical RX')
    for name,ncells,allowed in (('P_AUX72_READS.json',864,set(aux_tx[:72])),('P_AUX24_READS.json',288,set(aux_tx[72:]))):
        log=read(HERE/name);require(len(log['cells'])==ncells,'exact auxiliary cell count')
        require(len({v['key'] for v in log['cells']})==ncells,'auxiliary cell keys unique')
        for row in log['cells']:
            tx,rx,day,eq=row['key'].split('|');require(tx in allowed and rx in plan['aux_RX'] and day in ('D1','D2') and eq=='raw' and row['rows']==[0,32],'only admitted auxiliary waveform rows')
    tq=read(HERE/'P_TRAINING_QA.json');meta=np.load(resolve(tq['role_metadata']),allow_pickle=False)
    for role in ('train','val'):
        ids=meta[f'aux_{role}_pool_ids'];y=meta[f'aux_{role}_y'];rx=meta[f'aux_{role}_rx'];day=meta[f'aux_{role}_day'];keys=meta[f'aux_{role}_keys']
        require(len(ids)==(20736 if role=='train' else 6912),'auxiliary train/validation counts')
        require(np.array_equal(y,ids//384) and np.array_equal(rx,(ids//64)%6) and np.array_equal(day,(ids//32)%2+1),'auxiliary row identity axes')
        require(np.all(ids%32<24) if role=='train' else np.all(ids%32>=24),'fixed packet-row split')
        expected=np.array([f'{aux_tx[int(t)]}|{plan["aux_RX"][int(r)]}|D{int(d)}|raw|{int(i%32)}' for t,r,d,i in zip(y,rx,day,ids)])
        require(np.array_equal(keys,expected) and len(set(keys))==len(keys),'auxiliary packet keys reconstructed')
    require(not set(meta['aux_train_keys'])&set(meta['aux_val_keys']),'auxiliary train and validation disjoint')
    for role,original in (('train','fit_train'),('val','checkpoint_val')):
        ids=meta[f'downstream_{role}_pool_ids'];require(hashlib.sha256(ids.tobytes()).hexdigest()==plan['view'][original]['indices_array_sha256'],'original C role indices exact')
    require(not set(meta['downstream_train_keys'])&set(meta['downstream_val_keys']),'downstream train and checkpoint-val disjoint')
    evidence=read(resolve(tq['history_evidence']))['records'];reg=read(HERE/'EXECUTION_REGISTRATION.json')
    for seed in range(3):
        s=reg['initial_gradient_scales'][str(seed)];probes=s['probes']
        expected=.1*np.mean([v['CE_gradient_norm'] for v in probes])/np.mean([v['relation_gradient_norm'] for v in probes])
        near(expected,s['lambda_value'],'fit-only initial gradient formula',1e-12)
        h1=evidence[f'A1_S{seed}_aux'];h2=evidence[f'A2_S{seed}_aux']
        require(h1['initial']['model_sha256']==h2['initial']['model_sha256']==s['initial_state_sha256'],'paired auxiliary initial state')
        require([v['order_sha256'] for v in h1['history']]==[v['order_sha256'] for v in h2['history']],'40 auxiliary order pairs')
        baseline=evidence[f'A0_S{seed}_downstream']['history']
        for arm in ('A1','A2'):
            pre=evidence[f'{arm}_S{seed}_aux'];down=evidence[f'{arm}_S{seed}_downstream']
            require(len(pre['history'])==len(down['history'])==40 and pre['history'][-1]['updates']==4320 and down['history'][-1]['updates']==5080,'fixed 40-epoch neural budgets')
            require(pre['selected_epoch']==int(np.argmin([v['EMA_val_RX_CE'] for v in pre['history']]))+1,'auxiliary checkpoint chosen only by its fit-RX val CE')
            require(down['selected_epoch']==int(np.argmin([v['val_rx_ce']['C1'] for v in down['history']]))+1,'downstream checkpoint first minimum')
            require(all(v['minimum_relative_spread']>=1e-6 for v in pre['history']),'no complete class-geometry collapse')
            for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
                require([v[field] for v in down['history']]==[v[field] for v in baseline],'all downstream training orders match A0')
    seal=read(HERE/'P_MODEL_SEAL.json');require(seal['new_fits']==12 and seal['new_gradient_updates']==56400 and len(seal['runs'])==12,'actual fit/update budget')
    for run in seal['runs']:
        for field in ('complete','selected_checkpoint','selected_predictions'):resolve(run[field])
        with np.load(resolve(run['selected_predictions']),allow_pickle=False) as v:
            l=v['logits'].astype(float);y=v['y'];r=v['rx'];s=l-l.max(1,keepdims=True)
            ce=np.log(np.exp(s).sum(1))-s[np.arange(len(y)),y]
            value=np.mean([ce[r==rx].mean() for rx in np.unique(r)])
            e=evidence[run['job']];h=e['history'][e['selected_epoch']-1]
            expected=h['EMA_val_RX_CE'] if run['job'].endswith('_aux') else h['val_rx_ce']['C1']
            near(value,expected,'selected checkpoint validation CE replay',2e-6)
    result=read(HERE/'P_RESULT.json');receipt=read(HERE/'P_PREDICTION_RECEIPT.json');rx_order=plan['view']['source_selection_rx'];preds={}
    for key,item in receipt['predictions'].items():
        with np.load(resolve(item),allow_pickle=False) as z:preds[key]={n:z[n] for n in z.files}
    require(len(preds)==9,'all nine matched downstream predictions')
    reference=preds['A0/0']
    for p in preds.values():
        require(all(np.array_equal(p[k],reference[k]) for k in ('keys','pool_ids','y','rx','day')),'downstream packet identities/labels equal')
        require(len(p['y'])==11468 and set(p['rx'])==set(rx_order) and set(p['day'])=={1,2},'held-source whitelist')
        require(hashlib.sha256(p['pool_ids'].tobytes()).hexdigest()==plan['view']['source_selection']['indices_array_sha256'],'held-source indices exact')
    rows=csv_rows(HERE/'P_RX_METRICS.csv');cells=csv_rows(HERE/'P_TX_RX_RISKS.csv');overall=csv_rows(HERE/'P_PACKET_METRICS.csv')
    require(len(rows)==54 and len(cells)==540 and len(overall)==9,'complete downstream metric tables')
    matrix={}
    for row in rows:
        p=preds[f'{row["arm"]}/{row["seed"]}'];m=p['rx']==row['RX'];stats=statistics(p['logits'][m],p['y'][m])
        require(int(row['packets'])==m.sum(),'RX support')
        for k,v in stats.items():near(v,float(row[k]),'independent RX '+k)
        matrix[row['arm'],int(row['seed']),row['RX']]=stats['accuracy_pct']
    for row in cells:
        p=preds[f'{row["arm"]}/{row["seed"]}'];m=(p['rx']==row['RX'])&(p['y']==int(row['TX']));n=int(m.sum())
        errors=int((p['logits'][m].argmax(1)!=p['y'][m]).sum()) if n else 0
        require(int(row['packets'])==n and int(row['errors'])==errors,'all TX/RX risk numerators and denominators')
        if n:near(100*errors/n,float(row['error_pct']),'TX/RX error rate')
        else:require(row['error_pct']=='','missing support is disclosed')
    for row in overall:
        p=preds[f'{row["arm"]}/{row["seed"]}']
        for k,v in statistics(p['logits'],p['y']).items():near(v,float(row[k]),'packet '+k)
    comp={}
    for a,b in (('A2','A1'),('A2','A0'),('A1','A0')):
        ds=np.array([np.mean([matrix[a,s,r]-matrix[b,s,r] for r in rx_order]) for s in range(3)])
        dr=np.array([np.mean([matrix[a,s,r]-matrix[b,s,r] for s in range(3)]) for r in rx_order]);expected=result['comparisons'][a+'_minus_'+b]
        near(ds,expected['per_seed_gain_pp'],'paired seed Accuracy effects');near(ds.mean(),expected['mean_gain_pp'],'primary mean effect')
        near(dr.min(),expected['worst_RX_seed_mean_gain_pp'],'worst receiver effect')
        draw=np.random.default_rng(42101).integers(0,6,size=(10000,6));ci=np.quantile(dr[draw].mean(1),[.025,.975])
        near(ci,expected['conditional_RX_interval']['CI95_pp'],'conditional RX bootstrap')
        comp[a,b]=(ds,dr)
    ds,dr=comp['A2','A1'];ds0,dr0=comp['A2','A0']
    gates=[ds.mean()>=1,np.all(ds>0),dr.min()>=-2,ds0.mean()>=0,dr0.min()>=-2]
    require(gates==list(result['gates'].values()),'all five registered gates independently reproduced')
    require(result['status']==('W4P_PASS_FIRST_SOURCE_VIEW_ONLY' if all(gates) else 'STOP_W4P_RELATIONAL_PRETRAINING_NOT_PROMOTED'),'correct scientific decision')
    for row in receipt['downstream_frozen_states']+receipt['auxiliary_frozen_states']:
        require(row['before_state_sha256']==row['after_state_sha256'] and row['unchanged'],'evaluation did not adapt frozen state')
    # Additional independent E and 24-TX checks follow below.
    audit_E(resolve)
    audit_auxiliary(resolve,receipt)
    report=dict(status='PASS',checks_passed=COUNT,portable_mode=portable,own_code_sha256=sha(__file__),
        original_IQ_reads=0,new_neural_fits=0,classification_and_RX_intervals_recomputed=True,
        packet_identity_and_role_metadata_rechecked=True,formal_neural_fits=12,formal_gradient_updates=56400,
        scientific_decision=result['status'],verification_scope='Predictions, frozen features, training histories, role metadata and exact checkpoint/file digests; not rerunning raw-IQ training.')
    (HERE/('PORTABLE_QA.json' if portable else 'INDEPENDENT_QA.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))


def audit_E(resolve):
    result=read(HERE/'E_RESULT.json');inputs=read(HERE/'E_INPUTS.json');p=ART/'E_PAIRED.npz'
    require(sha(p)==result['paired_prediction_cache_sha256'],'E paired cache SHA')
    data=np.load(p,allow_pickle=False);groups=csv_rows(HERE/'E_GROUP_METRICS.csv');effects=[]
    fusion=csv_rows(HERE/'E_FUSION_RX.csv');weights=read(HERE/'E_FIT_ONLY_FUSION_WEIGHTS.json')['weights']
    for pair in inputs['pairs']:
        f=pair['fold'];a=data[f'f{f}_HStem_logits'];b=data[f'f{f}_ASK_logits'];y=data[f'f{f}_HStem_y']
        names=data[f'f{f}_RX_names'];rx=names[data[f'f{f}_HStem_rx']]
        require(not set(pair['fit_RX'])&set(pair['held_RX']),'E model fit/held RX exclusion')
        for field in ('y','rx','day'):require(np.array_equal(data[f'f{f}_HStem_{field}'],data[f'f{f}_ASK_{field}']),'E labels/RX/day pairing')
        for model,item in pair['models'].items():
            for role,prefix in (('held',f'f{f}_{model}_'),('fit_validation',f'f{f}_fit_{model}_')):
                with np.load(resolve(item[role]),allow_pickle=False) as original:
                    for field in ('logits','y','rx','day'):require(np.array_equal(original[field],data[prefix+field]),'E cache equals original saved prediction bytes')
        ca=a.argmax(1)==y;cb=b.argmax(1)==y
        vectors=dict(hstem_accuracy=ca,asknet_accuracy=cb,hstem_wrong_asknet_right=~ca&cb,hstem_right_asknet_wrong=ca&~cb,
            both_wrong=~ca&~cb,both_right=ca&cb,oracle_accuracy=ca|cb)
        for row in [v for v in groups if int(v['fold'])==f]:
            mask=np.ones(len(y),bool)
            if row['level']!='TX':mask&=rx==row['RX']
            if row['level']!='RX':mask&=y==int(row['TX'])
            require(int(row['packets'])==mask.sum(),'E group support')
            if mask.any():
                for k,v in vectors.items():near(v[mask].mean(),float(row[k]),'E independent quadrant')
        mean=lambda v:np.mean([v[rx==r].mean() for r in pair['held_RX']])
        effects.append(100*(mean(ca|cb)-max(mean(ca),mean(cb))))
        # The fixed scalar is checked by convex optimality, using only fit RX.
        fa=data[f'f{f}_fit_HStem_logits'].astype(float);fb=data[f'f{f}_fit_ASK_logits'].astype(float)
        fy=data[f'f{f}_fit_HStem_y'];frx=names[data[f'f{f}_fit_HStem_rx']];alpha=weights[f]['alpha_HStem']
        require(set(frx)==set(pair['fit_RX']) and not set(frx)&set(pair['held_RX']),'E alpha fit-only RX')
        mix=alpha*fa+(1-alpha)*fb;mix-=mix.max(1,keepdims=True);probs=np.exp(mix);probs/=probs.sum(1,keepdims=True)
        derivative_rows=(probs*(fa-fb)).sum(1)-(fa-fb)[np.arange(len(fy)),fy]
        derivative=np.mean([derivative_rows[frx==r].mean() for r in pair['fit_RX']])
        require((alpha<1e-10 and derivative>=-1e-9) or (alpha>1-1e-10 and derivative<=1e-9) or abs(derivative)<1e-9,'E scalar convex KKT')
        models=dict(HStem=a,ASKNet=b,equal_logit=(a+b)/2,fit_CE_scalar=alpha*a+(1-alpha)*b)
        for row in [v for v in fusion if int(v['fold'])==f]:
            mask=rx==row['RX'];near((models[row['method']].argmax(1)[mask]==y[mask]).mean(),float(row['accuracy']),'E fusion accuracy')
        # Archived metadata records support canonical source-cache pairing.
        fm=read(resolve(pair['source_manifest']))
        require(fm['source_only'] and fm['fit_rx']==pair['fit_RX'] and fm['held_rx']==pair['held_RX'],'E source cache roles')
        for item in pair['models'].values():
            resolve(item['config']);hm=read(resolve(item['held_metadata']))
            require(hm['held_sources']==pair['held_source_cells'] and hm['rx_names']==pair['rx_names'],'E identical source-cell provenance')
        resolve(pair['train_indices']);resolve(pair['validation_indices'])
    near(np.mean(effects),result['primary_oracle_headroom_pp'],'E primary oracle headroom')
    require(np.mean(effects)>=3 and result['fusion_candidate_count']==1,'E fixed fusion budget gate')


def audit_auxiliary(resolve,receipt):
    data=np.load(resolve(receipt['auxiliary_features']),allow_pickle=False)
    result=read(HERE/'P_AUX24_RESULT.json');rows=csv_rows(HERE/'P_AUX24_PAIR_METRICS.csv');plan=read(HERE/'DATA_PLAN.json')
    require(data['TX'].tolist()==plan['aux_diagnostic_TX'] and data['RX'].tolist()==plan['aux_RX'],'aux24 feature identity axes')
    require(len(rows)==360,'complete 60-context auxiliary diagnostic per model')
    for seed in range(3):
        for arm in ('A1','A2'):
            key=f'{arm}_S{seed}';z=data[key].astype(float);require(z.shape==(24,6,2,32,64),'aux24 feature shape')
            matching=[r for r in rows if r['arm']==arm and int(r['seed'])==seed];acc=[];margin=[]
            for row in matching:
                s=plan['aux_RX'].index(row['prototype_RX']);d=plan['aux_RX'].index(row['query_RX'])
                sd=int(row['prototype_day'])-1;dd=int(row['query_day'])-1
                anchors=z[:,s,sd].mean(1);q=z[:,d,dd]
                # Direct differences, independent of primary dot-product distances.
                distances=((q[:,:,None,:]-anchors[None,None,:,:])**2).sum(-1)
                truth=np.arange(24);correct=distances.argmin(-1)==truth[:,None]
                own=distances[truth,:,truth];other=distances.copy();other[truth,:,truth]=np.inf
                m=((other.min(-1)-own)/(distances.mean(-1)+1e-12)).mean()
                near(100*correct.mean(),float(row['prototype_diagnostic_accuracy_pct']),'aux24 independent prototype decisions')
                near(m,float(row['normalized_margin']),'aux24 independent margin')
                acc.append(100*correct.mean());margin.append(m)
            near(np.mean(acc),result['summary'][key]['prototype_diagnostic_accuracy_pct'],'aux24 aggregate')
            near(np.mean(margin),result['summary'][key]['normalized_margin'],'aux24 aggregate margin')
            centers=z.mean(3);i,j=np.triu_indices(24,1);r,s=np.triu_indices(6,1);loss=[];spreads=[]
            for day in (0,1):
                c=centers[:,:,day].transpose(1,0,2);dist=((c[:,i]-c[:,j])**2).sum(-1);norm=dist/(dist.mean(1,keepdims=True)+1e-8)
                loss.append(((norm[r]-norm[s])**2).mean());spreads.extend((dist.mean(1)/(np.mean(c*c,axis=(1,2))+1e-8)).tolist())
            near(np.mean(loss),result['summary'][key]['normalized_RX_relation_MSE'],'aux24 relation geometry')
            near(min(spreads),result['summary'][key]['minimum_relative_class_spread'],'aux24 class spread')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--portable',action='store_true');main(p.parse_args().portable)
