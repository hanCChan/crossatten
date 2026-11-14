"""Independent NumPy-only replay of W5-A classification, gates, roles and diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,ast,csv,hashlib,itertools,json
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
    for name in ('ADMISSION_REGISTRATION.json','EXECUTION_REGISTRATION.json','P_EVALUATION_REGISTRATION.json'):
        for rel,digest in read(HERE/name)['files_sha256'].items():require(sha(ROOT/rel)==digest,'bound code/evidence:'+rel)
    old=ROOT/'research/wisig_w4'
    require(sha(HERE/'p_models.py')==sha(old/'p_models.py'),'exact W4 A1 model implementation')
    def funcs(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    require(funcs(HERE/'p_fit.py')['fit_aux']==funcs(old/'p_fit.py')['fit_aux'],'exact W4 auxiliary training AST')
    x=funcs(HERE/'p_data.py');y=funcs(old/'p_data.py')
    require(all(x[k]==y[k].replace('mmap_C_V0_admitted_IQ_rows','mmap_C_V1_admitted_IQ_rows') for k in y),'W4 data processing unchanged apart from V1 audit label')
    plan=read(HERE/'DATA_PLAN.json');aux_tx=plan['aux_train_TX']
    require(len(aux_tx)==72 and plan['aux_diagnostic_TX']==[],'fixed 72 TX; no additional diagnostic IQ')
    require(aux_tx==read(ROOT/'research/wisig_w4/DATA_PLAN.json')['aux_train_TX'],'W4 original 72 TX and order unchanged')
    qe=read(HERE/'QUALIFICATION_EVIDENCE.json');q=read(HERE/'QUALIFICATION.json')
    with np.load(resolve(qe['count_metadata']),allow_pickle=False) as v:
        txlist=v['TX'].tolist();rxlist=v['RX'].tolist();counts=v['counts']
    require(counts.shape==(4,41,174),'official counts tensor schema')
    combos=[]
    for group in itertools.combinations(q['candidate_RX'],6):
        eligible=[t for t in qe['auxiliary_TX_pool'] if min(counts[d,rxlist.index(r),txlist.index(t)] for r in group for d in (0,1))>=32]
        combos.append(dict(RX=list(group),eligible_aux_TX=len(eligible),original_72_supported=len(set(eligible)&set(aux_tx)),missing_original_TX=sorted(set(aux_tx)-set(eligible))))
    require(combos==q['all_six_RX_combinations'],'all seven receiver qualifications independently replayed')
    valid=[v for v in combos if v['original_72_supported']==72]
    require(len(valid)==1 and valid[0]['RX']==plan['aux_RX'],'unique legal receiver group preserving original 72 TX')
    require(not set(aux_tx)&set(plan['excluded_target_TX']),'excluded target TX union')
    require(set(plan['aux_RX'])<=set(plan['view']['fit_rx']),'auxiliary RX contained in V1 fit12')
    require(not set(plan['aux_RX'])&set(plan['view']['source_selection_rx']),'no auxiliary held-source RX')
    forbidden=set(plan['source']['forbidden_target_rx']+plan['source']['forbidden_historical_held_rx'])
    require(not set(plan['view']['fit_rx']+plan['view']['source_selection_rx'])&forbidden,'P dataset source-only physical RX')
    for name,ncells,allowed in (('P_AUX72_READS.json',864,set(aux_tx)),):
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
        s=reg['initial_gradient_scales'][str(seed)]
        require(s['lambda_value']==0.,'fixed auxiliary CE only')
        pre=evidence[f'A1_S{seed}_aux']
        require(pre['initial']['model_sha256']==s['initial_state_sha256'],'original W4 random initialization retained')
        baseline=evidence[f'A0_S{seed}_downstream']['history']
        for arm in ('A1',):
            pre=evidence[f'{arm}_S{seed}_aux'];down=evidence[f'{arm}_S{seed}_downstream']
            require(len(pre['history'])==len(down['history'])==40 and pre['history'][-1]['updates']==4320 and down['history'][-1]['updates']==5080,'fixed 40-epoch neural budgets')
            require(pre['selected_epoch']==int(np.argmin([v['EMA_val_RX_CE'] for v in pre['history']]))+1,'auxiliary checkpoint chosen only by its fit-RX val CE')
            require(down['selected_epoch']==int(np.argmin([v['val_rx_ce']['C1'] for v in down['history']]))+1,'downstream checkpoint first minimum')
            require(all(v['minimum_relative_spread']>=1e-6 for v in pre['history']),'no complete class-geometry collapse')
            for field in ('relative_order_sha256','canonical_order_sha256','class_day_order_sha256'):
                require([v[field] for v in down['history']]==[v[field] for v in baseline],'all downstream training orders match A0')
    seal=read(HERE/'P_MODEL_SEAL.json');require(seal['new_fits']==6 and seal['new_gradient_updates']==28200 and len(seal['runs'])==6,'actual fit/update budget')
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
    require(len(preds)==6,'all six matched downstream predictions')
    reference=preds['A0/0']
    for p in preds.values():
        require(all(np.array_equal(p[k],reference[k]) for k in ('keys','pool_ids','y','rx','day')),'downstream packet identities/labels equal')
        require(len(p['y'])==11500 and set(p['rx'])==set(rx_order) and set(p['day'])=={1,2},'held-source whitelist')
        require(hashlib.sha256(p['pool_ids'].tobytes()).hexdigest()==plan['view']['source_selection']['indices_array_sha256'],'held-source indices exact')
    rows=csv_rows(HERE/'P_RX_METRICS.csv');cells=csv_rows(HERE/'P_TX_RX_RISKS.csv');overall=csv_rows(HERE/'P_PACKET_METRICS.csv')
    require(len(rows)==36 and len(cells)==360 and len(overall)==6,'complete downstream metric tables')
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
    for a,b in (('A1','A0'),):
        ds=np.array([np.mean([matrix[a,s,r]-matrix[b,s,r] for r in rx_order]) for s in range(3)])
        dr=np.array([np.mean([matrix[a,s,r]-matrix[b,s,r] for s in range(3)]) for r in rx_order]);expected=result['comparisons'][a+'_minus_'+b]
        near(ds,expected['per_seed_gain_pp'],'paired seed Accuracy effects');near(ds.mean(),expected['mean_gain_pp'],'primary mean effect')
        near(dr.min(),expected['worst_RX_seed_mean_gain_pp'],'worst receiver effect')
        draw=np.random.default_rng(43101).integers(0,6,size=(10000,6));ci=np.quantile(dr[draw].mean(1),[.025,.975])
        near(ci,expected['conditional_RX_interval']['CI95_pp'],'conditional RX bootstrap')
        comp[a,b]=(ds,dr)
    ds,dr=comp['A1','A0']
    gates=[ds.mean()>=1,np.all(ds>0),dr.min()>=-2]
    require(gates==list(result['gates'].values()),'all three registered gates independently reproduced')
    require(result['status']==('W5A_PASS_CROSS_VIEW_SOURCE_REPLICATION' if all(gates) else 'STOP_W5A_A1_NOT_REPLICATED'),'correct scientific decision')
    for row in receipt['downstream_frozen_states']+receipt['auxiliary_frozen_states']:
        require(row['before_state_sha256']==row['after_state_sha256'] and row['unchanged'],'evaluation did not adapt frozen state')
    report=dict(status='PASS',checks_passed=COUNT,portable_mode=portable,own_code_sha256=sha(__file__),
        original_IQ_reads=0,new_neural_fits=0,classification_and_RX_intervals_recomputed=True,
        packet_identity_and_role_metadata_rechecked=True,formal_neural_fits=6,formal_gradient_updates=28200,
        scientific_decision=result['status'],verification_scope='Predictions, training histories, role metadata and exact checkpoint/file digests; not rerunning raw-IQ training.')
    (HERE/('PORTABLE_QA.json' if portable else 'INDEPENDENT_QA.json')).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,indent=2))



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--portable',action='store_true');main(p.parse_args().portable)
