"""Official cross-observation metadata audit and bounded source acquisition probe."""
from pathlib import Path
import ast
import csv
import hashlib
import io
import json
import pickle
import urllib.request
import numpy as np
import requests

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PUBLIC='https://raw.githubusercontent.com/WiSig-dataset/wisig-subset-creation/bc8c6bd352a8c184a571e6b4c2d7a890cdfff3d8/'
COUNT=Path('D:/crossatten/worktrees/training-instance-attribution/research/wisig_full_confirmation_metadata/artifacts/data_summary.metadata.pkl')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


class Counts(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
        if (module,name) not in allowed:raise pickle.UnpicklingError((module,name))
        return allowed[module,name]


def session():
    s=requests.Session();s.proxies.update({'http':'http://127.0.0.1:78','https':'http://127.0.0.1:78'});return s


def metadata():
    scope=json.loads((HERE/'ACCESS_SCOPE.json').read_text())
    artifacts=HERE/'artifacts';artifacts.mkdir(exist_ok=True)
    payload=COUNT.read_bytes()
    if len(payload)!=459700 or hashlib.sha256(payload).hexdigest()!='072ee403d2eed11a2f960a0cc18a7370afdc60fad96d3252ba83d66a91f34146':raise ValueError('Count metadata identity differs')
    counts=Counts(io.BytesIO(payload)).load()
    if set(counts)!={'capture_date_list','tx_list','rx_list','mat_date','mat_date_eq'}:raise ValueError('Count metadata schema')
    matrices={k:np.asarray(counts[k]) for k in ('mat_date','mat_date_eq')}
    if any(v.shape!=(4,41,174) or not np.isfinite(v).all() or (v<0).any() or (v!=np.floor(v)).any() for v in matrices.values()):raise ValueError('Invalid counts')
    s=session();sources={}
    for name in ('py/002_FullWiSig_create_compact_datasets.py','data_utilities.py'):
        r=s.get(PUBLIC+name,timeout=45);r.raise_for_status()
        if len(r.content)>100000:raise ValueError('Unexpected source size')
        p=artifacts/name.split('/')[-1];p.write_bytes(r.content)
        sources[name]=dict(url=PUBLIC+name,path=str(p),sha256=sha(p),bytes=len(r.content))
    tree=ast.parse((artifacts/'002_FullWiSig_create_compact_datasets.py').read_text(encoding='utf8'))
    current={};compact={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('tx_list','rx_list','capture_date_list','dataset_name','max_n'):
                    try:current[target.id]=ast.literal_eval(node.value)
                    except (ValueError,TypeError):pass
                    if target.id=='max_n' and current.get('dataset_name') in ('ManyTx','ManyRx','ManySig','SingleDay'):
                        compact[current['dataset_name']]={k:current[k] for k in ('tx_list','rx_list','capture_date_list','max_n')}
    assert len(compact['ManyTx']['tx_list'])==150 and len(compact['ManyTx']['rx_list'])==18
    historical=json.loads((ROOT/'05-wisig/out/05-1-ManyRx-manifest.json').read_text())
    assert compact['ManyRx']['tx_list']==historical['tx_list']
    target_tx=set(compact['ManyRx']['tx_list'])|set(compact['ManySig']['tx_list'])
    plan=json.loads((ROOT/'research/standard_method_competition_c/SOURCE_PLAN.json').read_text())
    source=next(v for v in plan['views'] if v['id']=='V0')['fit_rx']
    source_rx=sorted(set(source)&set(compact['ManyTx']['rx_list']))
    aux=sorted(set(compact['ManyTx']['tx_list'])-target_tx)
    records=[];qualified=[]
    for tx in aux:
        ti=counts['tx_list'].index(tx);minimum=1000000
        for rx in source_rx:
            ri=counts['rx_list'].index(rx)
            for di in (0,1):
                n=int(matrices['mat_date'][di,ri,ti]);minimum=min(minimum,n)
                records.append(dict(tx=tx,rx=rx,date=counts['capture_date_list'][di],day=di+1,raw_full_count=n,compact_expected_count=min(n,50),selected_rows=32 if n>=32 else 0))
        if minimum>=32:qualified.append(tx)
    with (HERE/'A_AUXILIARY_COUNT_GRID.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    gates=dict(six_source_RX=len(source_rx)>=6,at_least40_complete_auxiliary_TX=len(qualified)>=40,
               target_TX_disjoint=not(set(qualified)&target_tx),source_days_only=True)
    result=dict(status='METADATA_CROSS_OBSERVATIONS_PASS_PAYLOAD_NOT_YET_ADMITTED' if all(gates.values()) else 'STOP_A_METADATA_SUPPORT_INSUFFICIENT',
        gates=gates,source_RX=source_rx,all_ManyTx_RX=compact['ManyTx']['rx_list'],ManyTx_RX_outside_selected_source=sorted(set(compact['ManyTx']['rx_list'])-set(source_rx)),
        ManyRx_TX=compact['ManyRx']['tx_list'],ManySig_TX=compact['ManySig']['tx_list'],excluded_target_TX=sorted(target_tx),
        auxiliary_TX_pool=aux,qualified_auxiliary_TX=qualified,compact_sets=compact,
        Full_TX=counts['tx_list'],Full_RX=counts['rx_list'],dates=counts['capture_date_list'],
        expected_source_packets=len(qualified)*len(source_rx)*2*32,raw_IQ_reads=0,feature_reads=0,new_neural_fits=0,
        metadata_sources=sources,count_metadata_sha256=hashlib.sha256(payload).hexdigest(),
        packet_lineage='Official raw compact cell is first min(n,50) Full raw rows; namespace (TX,RX,date,raw,row). Target-TX and forbidden-RX/day cells are not decoded. Full raw/EQ packet pairing is NOT assumed.',
        independent_confirmation='Unchanged: extra Full RX have unresolved historical exposure, access remains closed; this is source-side TX transfer analysis only')
    write('A_METADATA_RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('status','gates','source_RX','expected_source_packets')},indent=2));print('auxiliary_TX_pool',len(aux),'complete_TX',len(qualified),flush=True)


def probe():
    url='https://drive.usercontent.google.com/download?id=17EnvGFoflJEh1xhFC8wx5fhCuPYhWt2l&export=download&confirm=t'
    s=session();out={}
    with s.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=45) as r:
        out=dict(url=url,status=r.status_code,headers={k:v for k,v in r.headers.items() if k.lower() in ('content-type','content-length','content-range','content-disposition','accept-ranges')},final_url=r.url)
        if r.status_code>=400 or 'text/html' in r.headers.get('Content-Type',''):
            out['diagnostic']=r.raw.read(3000).decode('utf8','replace')
    write('A_OFFICIAL_DOWNLOAD_PROBE.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':
    import sys
    metadata() if len(sys.argv)==1 or sys.argv[1]=='metadata' else probe()
