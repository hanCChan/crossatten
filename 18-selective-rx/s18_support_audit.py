"""Post-guard metadata audit: no IQ, checkpoints, features or performance reads."""
import numpy as np
from s18_core import HERE, read, write, sha, access, validate, now
from s18_data import locations
from xa.wisig import receivers


def audit():
    validate()
    rows = []
    for fold in range(4):
        cache, _ = locations(fold)
        manifest = read(cache/'fit_manifest.json')
        source, target = receivers('ManyRx', fold)
        if manifest['fit_rx'] != source[6:]: raise ValueError('Illegal cache roles')
        arrays = {}
        for key in ['fit_y', 'fit_rx', 'fit_day', 'train_ids', 'val_ids']:
            path = cache/f'{key}.npy'
            if sha(path) != manifest['sha256'][path.name]: raise ValueError('Changed source metadata')
            arrays[key] = np.load(path)
            access(fold=fold, kind='source_support_metadata', path=str(path), sha256=sha(path), rx=source[6:], days=[1, 2])
        y, rx, day = [arrays[f'fit_{key}'] for key in ['y', 'rx', 'day']]
        observed_rx = sorted(np.unique(rx))
        if {manifest['rx_names'][int(r)] for r in observed_rx} != set(source[6:]): raise ValueError('Illegal RX')
        for part in ['train', 'val']:
            ids = arrays[f'{part}_ids']
            counts = np.array([[[int(np.sum((y[ids] == t) & (rx[ids] == r) & (day[ids] == d))) for d in [1, 2]] for r in observed_rx] for t in range(10)])
            missing = [dict(tx=int(t), rx=manifest['rx_names'][int(observed_rx[r])], day=int(d+1), packets=int(counts[t, r, d])) for t, r, d in np.argwhere(counts < 4)]
            rows.append(dict(fold=fold, part=part, minimum=int(counts.min()), maximum=int(counts.max()),
                             cells=int(counts.size), cells_zero=int(np.sum(counts == 0)), cells_below4=int(np.sum(counts < 4)),
                             below4=missing, counts=counts.tolist(), rx_names=[manifest['rx_names'][int(r)] for r in observed_rx]))
    result = dict(time=now(), stage='S18-D0-support', reason='Original D0 guard stopped before learned feature extraction',
                  learned_features_read=False, interventions_run=0, scientific_rerun=False, original_attempt_retained=True, rows=rows)
    write(HERE/'SUPPORT_AUDIT.json', result)
    for row in rows: print(row['fold'], row['part'], 'min', row['minimum'], 'zero', row['cells_zero'], 'below4', row['cells_below4'], flush=True)


if __name__ == '__main__': audit()
