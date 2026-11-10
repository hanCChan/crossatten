"""Read ONLY sealed S17B fit caches. No raw data, held RX or dates3-4 API."""
import numpy as np
from s18_core import HISTORY, read, sha, access
from xa.wisig import receivers


def locations(fold):
    if fold not in range(4):
        raise ValueError('ManyRx folds0-3 only')
    base = HISTORY/'17b-phase-intervention'
    return (base/'cache'/f'ManyRx_f{fold}_first_s16001',
            base/'runs'/f'pilot__ManyRx__M0__f{fold}__first__s18101')


def load(fold):
    cache, run = locations(fold)
    manifest = read(cache/'fit_manifest.json')
    source, target = receivers('ManyRx', fold)
    expected_held = source[:6]
    expected_fit = source[6:]
    if manifest['fit_rx'] != expected_fit or manifest['held_rx'] != expected_held or not manifest['source_only']:
        raise ValueError('Cache roles disagree with frozen source protocol')
    pool = {}
    for key in ['x', 'y', 'rx', 'day']:
        name = f'fit_{key}.npy'
        if sha(cache/name) != manifest['sha256'][name]:
            raise ValueError('Changed source cache: '+name)
        pool[key] = np.load(cache/name, mmap_mode='r')
        access(fold=fold, dataset='ManyRx', kind='fit_cache', path=str(cache/name), sha256=manifest['sha256'][name], rx=expected_fit, days=[1, 2])
    for part in ['train', 'val']:
        name = f'{part}_ids.npy'
        if sha(cache/name) != manifest['sha256'][name]:
            raise ValueError('Changed source split: '+name)
    tr = np.load(cache/'train_ids.npy')
    va = np.load(cache/'val_ids.npy')
    for ids in [tr, va]:
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(pool['y']):
            raise ValueError('Invalid source split indices')
    if len(np.intersect1d(tr, va)) or len(np.unique(np.r_[tr, va])) != len(pool['y']):
        raise ValueError('Source split overlap or incomplete coverage')
    actual = {manifest['rx_names'][int(r)] for r in np.unique(pool['rx'])}
    if actual != set(expected_fit) or actual & set(target+expected_held) or set(np.unique(pool['day'])) != {1, 2}:
        raise ValueError('Illegal receiver/date in fit cache')
    if set(np.unique(pool['y'])) != set(range(10)) or pool['x'].shape != (len(pool['y']), 2, 256):
        raise ValueError('Unexpected TX/input layout')
    cfg = read(run/'config.json')
    expected_job = dict(phase='pilot', dataset='ManyRx', arm='M0', fold=fold, group='first', seed=18101, split_seed=16001)
    if cfg['job'] != expected_job or cfg['source_split_seed'] != 16001 or cfg['active_params'] != 82426:
        raise ValueError('Frozen model protocol mismatch')
    return pool, tr, va, manifest, run, cfg


def estimation_ids(pool, ids, fold, cap=48):
    """Deterministic capped sampling by training-only TX/RX/day cells."""
    rng = np.random.default_rng(18001+fold)
    selected = []
    for t in range(10):
        for r in sorted(np.unique(pool['rx'])):
            for d in (1, 2):
                cell = ids[(pool['y'][ids] == t) & (pool['rx'][ids] == r) & (pool['day'][ids] == d)]
                if len(cell) < 4:
                    raise ValueError('Insufficient estimation cell support')
                selected.extend(rng.permutation(cell)[:cap])
    return np.array(selected, dtype=np.int64)


def halves(y, rx, day):
    """Disjoint even/odd packet halves within each TX/RX/day cell."""
    a, b = [], []
    for t in np.unique(y):
        for r in np.unique(rx):
            for d in np.unique(day):
                ids = np.flatnonzero((y == t) & (rx == r) & (day == d))
                a.extend(ids[::2]); b.extend(ids[1::2])
    return np.asarray(a), np.asarray(b)
