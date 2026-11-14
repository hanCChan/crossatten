"""W5 metadata admission; no NumPy waveform payload is read."""
import hashlib,importlib.util,io,itertools,json,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):(HERE/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    w4=read(ROOT/'research/wisig_w4/DATA_PLAN.json');meta=read(ROOT/'research/wisig_w3/A_METADATA_RESULT.json')
    view=read(ROOT/'research/standard_method_competition_c/SOURCE_PLAN.json')['views'][1];assert view['id']=='V1'
    m=load('W5_counts_helpers',ROOT/'research/wisig_w3/w3_a.py');payload=m.COUNT.read_bytes()
    assert hashlib.sha256(payload).hexdigest()==meta['count_metadata_sha256']
    counts=m.Counts(io.BytesIO(payload)).load();matrix=np.asarray(counts['mat_date'])
    candidates=sorted(set(view['fit_rx'])&set(meta['all_ManyTx_RX']));rows=[]
    for group in itertools.combinations(candidates,6):
        eligible=[t for t in meta['auxiliary_TX_pool'] if min(int(matrix[d,counts['rx_list'].index(r),counts['tx_list'].index(t)]) for r in group for d in (0,1))>=32]
        rows.append(dict(RX=list(group),eligible_aux_TX=len(eligible),original_72_supported=len(set(eligible)&set(w4['aux_train_TX'])),missing_original_TX=sorted(set(w4['aux_train_TX'])-set(eligible))))
    valid=[r for r in rows if r['original_72_supported']==72]
    write('QUALIFICATION.json',dict(status='PASS_UNIQUE_SAME_72TX_SIX_RX_COHORT' if len(valid)==1 else 'STOP_QUALIFICATION_NOT_UNIQUE_OR_INSUFFICIENT',
        V1_fit_RX=view['fit_rx'],V1_held_RX=view['source_selection_rx'],candidate_RX=candidates,all_six_RX_combinations=rows,count_metadata_sha256=meta['count_metadata_sha256'],IQ_values_read=0))
    if len(valid)!=1:raise RuntimeError('No unique legal six-RX cohort preserves all fixed 72TX')
    selected=valid[0]['RX'];assert not set(selected)&set(view['source_selection_rx'])
    sys.path.insert(0,str(ROOT/'research/wisig_w3'))
    from selective_pickle import index,ArrayRef
    from index_manytx import AuditReader
    raw=w4['raw_pickle'];p=Path(raw['path']);assert p.stat().st_size==raw['bytes']
    with p.open('rb') as f:reader=AuditReader(f);obj,_=index(reader)
    assert obj['tx_list']==meta['compact_sets']['ManyTx']['tx_list'] and obj['rx_list']==meta['all_ManyTx_RX']
    spans=[]
    for txdata in obj['data']:
        for rxdata in txdata:
            for daydata in rxdata:
                for v in daydata:assert isinstance(v,ArrayRef);spans.append(v.payload)
    spans.sort(key=lambda v:v.offset);j=0
    for start,end in reader.reads:
        while j<len(spans) and spans[j].offset+spans[j].length<=start:j+=1
        assert j==len(spans) or not (spans[j].length and start<spans[j].offset+spans[j].length and end>spans[j].offset)
    cells=[]
    for tx in w4['aux_train_TX']:
        for rx in selected:
            for d in (0,1):
                v=obj['data'][obj['tx_list'].index(tx)][obj['rx_list'].index(rx)][d][0]
                n=min(50,int(matrix[d,counts['rx_list'].index(rx),counts['tx_list'].index(tx)]))
                assert v.shape==(n,256,2) and n>=32 and v.order=='C'
                cells.append(dict(key=f'{tx}|{rx}|D{d+1}|raw',tx=tx,rx=rx,day=d+1,eq=0,date=obj['capture_date_list'][d],
                    shape=list(v.shape),dtype=v.dtype.str,order='C',offset=v.payload.offset,payload_bytes=v.payload.length,rows=[0,32]))
    refs=read(ROOT/'research/spectral_scale_execution/REFERENCE_INPUTS.json')['references'];jobs=read(ROOT/'research/standard_method_competition_c/JOBS.json');a0={}
    for seed in range(3):
        ref=refs[f'V1/S{seed}']['E0'];folder=Path(ref['origin']).parent
        rec=lambda p:dict(path=str(p),sha256=sha(p))
        a0[str(seed)]=dict(checkpoint=dict(path=ref['origin'],sha256=ref['sha256']),validation=ref['checkpoint_val_prediction'],selection=ref['source_selection_prediction'],
            initialization=rec(folder/'INITIAL.json'),sample_keys=rec(folder/'SAMPLE_KEYS.npz'),recipe=next(j for j in jobs if j['id']==f'V1_S{seed}_wd1e3_ERM_EMA'))
    plan=dict(status='W5A_FIXED_W4_72TX_ON_LEGAL_V1_RX',source=w4['source'],view=view,aux_train_TX=w4['aux_train_TX'],aux_diagnostic_TX=[],aux_RX=selected,
        excluded_target_TX=w4['excluded_target_TX'],raw_pickle=raw,cells=cells,A0=a0,aux_fit_rows=[0,24],aux_val_rows=[24,32])
    write('DATA_PLAN.json',plan);write('INDEX_QA.json',dict(status='PASS',numeric_arrays=len(spans),IQ_payload_intersections=0,
        metadata_bytes_read=sum(b-a for a,b in reader.reads),selected_cells=len(cells),selected_packets=len(cells)*32,all_counts_match=True))
    print('PASS same72TX, six legal V1 RX, 864 cells; numeric IQ read=0',flush=True)


if __name__=='__main__':main()
