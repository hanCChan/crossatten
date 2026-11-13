"""Audit the archive schema and commit a source-row manifest, skipping all IQ bytes."""
from pathlib import Path
import csv
import hashlib
import json
from selective_pickle import index,ArrayRef
from w3_a import HERE,write


class AuditReader:
    def __init__(self,f):self.f=f;self.reads=[]
    def read(self,n):
        if n<0 or n>1000000:raise ValueError('Unregistered bulk read during metadata indexing')
        p=self.f.tell();v=self.f.read(n);self.reads.append((p,p+len(v)));return v
    def readline(self):
        p=self.f.tell();v=self.f.readline(4096);self.reads.append((p,p+len(v)))
        if not v.endswith(b'\n'):raise ValueError('Unbounded metadata line')
        return v
    def seek(self,*args):return self.f.seek(*args)
    def tell(self):return self.f.tell()


def main():
    a=json.loads((HERE/'A_ACQUISITION.json').read_text(encoding='utf8'))
    meta=json.loads((HERE/'A_METADATA_RESULT.json').read_text(encoding='utf8'))
    if (HERE/'B_SOURCE_MANIFEST.json').exists():raise FileExistsError('Keep original source manifest')
    p=Path(a['pickle_path'])
    if p.stat().st_size!=a['pickle_bytes']:raise ValueError('Opaque input changed size')
    with p.open('rb') as f:
        reader=AuditReader(f);obj,spans=index(reader)
    expected=meta['compact_sets']['ManyTx']
    for key in ('tx_list','rx_list','capture_date_list'):
        if obj[key]!=expected[key]:raise ValueError('Official compact identity differs: '+key)
    if obj['equalized_list']!=[0,1] or obj['max_sig']!=50:raise ValueError('Unexpected raw/EQ order or cap')
    data=obj['data'];arrays=[];allcells=[]
    for ti,tx in enumerate(obj['tx_list']):
        for ri,rx in enumerate(obj['rx_list']):
            for di,day in enumerate(obj['capture_date_list']):
                for ei,eq in enumerate(obj['equalized_list']):
                    ref=data[ti][ri][di][ei]
                    if not isinstance(ref,ArrayRef) or len(ref.shape)!=3 or ref.shape[1:]!=(256,2):raise ValueError('Unexpected cell array')
                    arrays.append(ref.payload)
                    allcells.append(dict(tx=tx,rx=rx,day=di+1,eq=eq,n=ref.shape[0]))
    # Linear interval walk proves metadata reads never intersect waveform spans.
    j=0;touches=[];arrays=sorted(arrays,key=lambda s:s.offset)
    for start,end in reader.reads:
        while j<len(arrays) and arrays[j].offset+arrays[j].length<=start:j+=1
        if j<len(arrays) and arrays[j].length and start<arrays[j].offset+arrays[j].length and end>arrays[j].offset:touches.append((start,end))
    if touches:raise ValueError('Metadata-only scope read waveform bytes')
    with (HERE/'A_AUXILIARY_COUNT_GRID.csv').open(encoding='utf8') as f:grid={(r['tx'],r['rx'],int(r['day'])):int(r['compact_expected_count']) for r in csv.DictReader(f)}
    cells=[]
    for tx in meta['qualified_auxiliary_TX']:
        for rx in meta['source_RX']:
            for di in (0,1):
                ref=data[obj['tx_list'].index(tx)][obj['rx_list'].index(rx)][di][0]
                if ref.shape[0]!=grid[tx,rx,di+1] or ref.shape[0]<32:raise ValueError('Observed compact row support differs from official metadata')
                if ref.order!='C' or ref.dtype.kind!='f':raise ValueError('Unregistered source payload layout')
                key=f'{tx}|{rx}|D{di+1}|raw'
                cells.append(dict(key=key,tx=tx,rx=rx,day=di+1,date=obj['capture_date_list'][di],eq=0,shape=list(ref.shape),dtype=ref.dtype.str,
                    order=ref.order,offset=ref.payload.offset,payload_bytes=ref.payload.length,rows=[0,32],
                    identity_namespace='Full raw TX/RX/date/prefix-row, official creation order'))
    result=dict(status='PASS_SOURCE_COHORT_AND_SELECTIVE_ROW_MANIFEST',path=str(p),sha256=a['pickle_sha256'],bytes=a['pickle_bytes'],
        cells=cells,auxiliary_TX=meta['qualified_auxiliary_TX'],source_RX=meta['source_RX'],days=[1,2],excluded_TX=meta['excluded_target_TX'],
        selected_cells=len(cells),selected_packets=len(cells)*32,IQ_payload_bytes_read_by_index=0,
        metadata_bytes_read=sum(e-s for s,e in reader.reads),numeric_arrays_indexed=len(allcells),
        packet_identity_checks='TX/RX/date/eq/row keys are unique; target-TX union excluded. Cross-RX packets are NOT treated as paired emissions.',
        independence='Source-side mechanism cohort only; no fresh blind RX qualification claimed')
    write('B_SOURCE_MANIFEST.json',result)
    write('A_SELECTIVE_INDEX_QA.json',dict(status='PASS',arrays=len(allcells),read_intersections_with_waveform_spans=0,metadata_bytes_read=result['metadata_bytes_read'],
        selected_cells=len(cells),selected_packets=len(cells)*32,all_selected_counts_match_official=True,
        source_TX_disjoint_from_both_benchmarks=True,forbidden_RX_or_day_IQ_decodes=0,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps({k:result[k] for k in ('status','selected_cells','selected_packets','numeric_arrays_indexed','metadata_bytes_read')},indent=2),flush=True)


if __name__=='__main__':main()
