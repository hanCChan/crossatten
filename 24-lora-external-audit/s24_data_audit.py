"""Audit ONLY exact source filenames, selected rows, and training/validation identity."""
import argparse
import csv
import json
from collections import Counter
import numpy as np
import h5py
from s24_common import HERE, P, PROTOCOL_PATH, sha, source_path, selected_rows, split_indices, unpack_iq, write_json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', type=Path, required=True)
    args = parser.parse_args()
    records, accesses, summaries, duplicate_groups = [], [], [], []
    seen, samples = {}, []
    for fold in P['folds']:
        for domain, rx in enumerate(fold['source']):
            path = source_path(args.raw_dir, rx, fold['fold'])
            accesses.append(dict(fold=fold['fold'], rx=rx, path=str(path), fields=['label','data'],
                                 purpose='phase1_source_identity_and_preprocessing', forbidden_metadata_read=False))
            with h5py.File(path, 'r') as h:
                labels = np.asarray(h['label'][:]).reshape(-1).astype(np.int64)
                if h['data'].shape != (len(labels), 16384): raise ValueError('raw data shape')
                rows = selected_rows(labels)
                y = labels[rows] - 31
                tr, va = split_indices(y, rows)
                train_set = set(tr.tolist())
                iq_hash = __import__('hashlib').sha256()
                for start in range(0, len(rows), 100):
                    iq = unpack_iq(h['data'][rows[start:start+100]])
                    iq_hash.update(iq.tobytes())
                    for j, x in enumerate(iq):
                        i = start + j
                        digest = sha(x.tobytes())
                        identity = f'{rx}_train.h5:{rows[i]}'
                        if digest in seen:
                            duplicate_groups.append([seen[digest], identity])
                        else: seen[digest] = identity
                        records.append(dict(fold=fold['fold'], rx=rx, source_domain=domain,
                                            tx_raw=int(y[i]+31), tx=int(y[i]), raw_row=int(rows[i]),
                                            split='train' if i in train_set else 'source_val',
                                            packet_id=identity, iq_float32_sha256=digest))
                    if start == 0: samples.append(iq[:2].copy())
            summaries.append(dict(fold=fold['fold'], rx=rx, selected=2000, train=len(tr), source_val=len(va),
                                  raw_shape=[len(labels), 16384], raw_file_bytes=path.stat().st_size,
                                  selected_rows_sha256=sha(rows.astype(np.int64).tobytes()),
                                  selected_labels_sha256=sha(y.astype(np.int64).tobytes()),
                                  selected_iq_float32_sha256=iq_hash.hexdigest()))
            print(f'fold{fold["fold"]} {rx}: {len(tr)} train + {len(va)} source_val', flush=True)
    with (HERE / 'PACKETS.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(records)
    write_json(HERE / 'ACCESS_AUDIT.json', dict(accesses=accesses, blind_reads=0, confirm_reads=0,
               evaluation_role_reads=0, note='Each file is opened only as source of its own fold; no model/evaluation is run.'))
    (HERE / 'local').mkdir(exist_ok=True)
    np.savez(HERE / 'local/source_probe.npz', iq=np.concatenate(samples))
    write_json(HERE / 'DATA_AUDIT.json', dict(protocol_sha256=sha(PROTOCOL_PATH.read_bytes()),
               records=len(records), unique_packet_ids=len({r['packet_id'] for r in records}),
               duplicate_float32_waveforms=duplicate_groups,
               passed=not duplicate_groups, files=summaries,
               hash_scope='Selected raw I/Q cast to float32, no normalization. No whole-HDF5 hash claimed.',
               byte_exact_not_near_duplicate_detection=True,
               sample_probe_packets=sum(len(s) for s in samples)))


if __name__ == '__main__': main()
