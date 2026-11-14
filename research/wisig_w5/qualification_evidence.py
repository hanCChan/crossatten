"""Archive counts metadata and independently reproducible receiver qualification."""
import io,itertools
import numpy as np
from common import *
def main():
    h=module('W5_metadata_counts',ROOT/'research/wisig_w3/w3_a.py')
    payload=h.COUNT.read_bytes();qual=read(HERE/'QUALIFICATION.json');assert hashlib.sha256(payload).hexdigest()==qual['count_metadata_sha256']
    counts=h.Counts(io.BytesIO(payload)).load();meta=read(ROOT/'research/wisig_w3/A_METADATA_RESULT.json')
    np.savez_compressed(ART/'COUNT_METADATA.npz',TX=np.array(counts['tx_list']),RX=np.array(counts['rx_list']),counts=np.asarray(counts['mat_date']))
    write('QUALIFICATION_EVIDENCE.json',dict(status='OFFICIAL_COUNTS_METADATA_ONLY',count_metadata=record(ART/'COUNT_METADATA.npz'),
        original_pickle_sha256=qual['count_metadata_sha256'],auxiliary_TX_pool=meta['auxiliary_TX_pool'],IQ_reads=0))
    print('Archived counts-only qualification evidence; no IQ')
if __name__=='__main__':main()
