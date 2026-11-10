"""Only registered fit18/date1-2 caches, then sealed held-source cells."""
import numpy as np
from s19_core import HERE, HISTORY, SOURCE_CELLS, read, write, sha, access, committed
from xa.wisig import receivers


def locations(fold):
    if fold not in range(4): raise ValueError('Only ManyRx folds0-3 registered')
    base = HISTORY/'17b-phase-intervention'
    return base/'cache'/f'ManyRx_f{fold}_first_s16001', base/'runs'/f'pilot__ManyRx__M0__f{fold}__first__s18101'


def roles(fold):
    source, target = receivers('ManyRx', fold)
    return source[6:], source[:6], target


def load_fit(fold):
    cache, run = locations(fold)
    manifest = read(cache/'fit_manifest.json')
    fit, held, target = roles(fold)
    if manifest['fit_rx'] != fit or manifest['held_rx'] != held or not manifest['source_only']:
        raise ValueError('Changed or illegal source roles')
    pool = {}
    for key in ['x', 'y', 'rx', 'day']:
        path = cache/f'fit_{key}.npy'
        if sha(path) != manifest['sha256'][path.name]: raise ValueError('Source cache hash mismatch')
        pool[key] = np.load(path, mmap_mode='r', allow_pickle=False)
        access(fold=fold, kind='fit_cache', dataset='ManyRx', path=str(path), sha256=manifest['sha256'][path.name], rx=fit, days=[1, 2])
    indices = {}
    for part in ['train', 'val']:
        path = cache/f'{part}_ids.npy'
        if sha(path) != manifest['sha256'][path.name]: raise ValueError('Changed source split')
        ids = np.load(path, allow_pickle=False)
        if ids.dtype.kind not in 'iu' or ids.ndim != 1 or len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(pool['y']):
            raise ValueError('Invalid source split')
        indices[part] = ids
    if len(np.intersect1d(indices['train'], indices['val'])) or not np.array_equal(np.sort(np.r_[indices['train'], indices['val']]), np.arange(len(pool['y']))):
        raise ValueError('Source split must be complete and disjoint')
    if {manifest['rx_names'][int(r)] for r in np.unique(pool['rx'])} != set(fit) or set(np.unique(pool['day'])) != {1, 2}:
        raise ValueError('Forbidden RX or date in fit data')
    if pool['x'].shape != (len(pool['y']), 2, 256) or set(np.unique(pool['y'])) != set(range(10)):
        raise ValueError('Unexpected packet or class dimensions')
    data = {part:{key:np.array(value[ids], copy=True) for key,value in pool.items()} for part,ids in indices.items()}
    return data, manifest, run, indices


def teacher(fold):
    _, run = locations(fold)
    records = read(HISTORY/'17b-phase-intervention/final_evidence/MANIFEST.json')['records']
    refs = [r for r in records if r['job']['fold'] == fold]
    if len(refs) != 1: raise ValueError('Ambiguous B0 checkpoint')
    cfg = read(run/'config.json')
    expected = dict(phase='pilot', dataset='ManyRx', arm='M0', fold=fold, group='first', seed=18101, split_seed=16001)
    if cfg['job'] != expected or cfg['active_params'] != 82426 or cfg['source_split_seed'] != 16001:
        raise ValueError('Historical B0 protocol mismatch')
    for name in ['config.json', 'metrics.json']:
        if sha(run/name) != refs[0]['files_sha256'][name]: raise ValueError('Changed historical B0 metadata')
    if sha(run/'model.pt') != refs[0]['checkpoint_sha256']: raise ValueError('Changed B0 weights')
    access(fold=fold, kind='source_teacher_checkpoint', path=str(run/'model.pt'), sha256=refs[0]['checkpoint_sha256'])
    return run, cfg, refs[0]['checkpoint_sha256']


def evidence_ids(data, fold, cap=16):
    rng = np.random.default_rng(19001+fold)
    ids = []
    for t in range(10):
        for r in sorted(np.unique(data['rx'])):
            for d in [1, 2]:
                found = np.flatnonzero((data['y'] == t) & (data['rx'] == r) & (data['day'] == d))
                if len(found): ids.extend(rng.permutation(found)[:cap])
    return np.array(ids, dtype=np.int64)


def load_held(fold):
    seal = HERE/'seal_pilot.json'
    committed(seal)
    if not read(seal)['all_checkpoints_complete']: raise ValueError('Unsealed phase')
    fit, held, target = roles(fold)
    if set(held) & set(target+fit): raise ValueError('Illegal held-source role')
    values = {k:[] for k in ['x','y','rx','day']}
    names = sorted(fit+held)
    sources = []
    for name in held:
        for day in [1, 2]:
            path = SOURCE_CELLS/'ManyRx'/f'rx_{name}_day{day}.npz'
            digest = sha(path)
            access(fold=fold, kind='held_source_cell', dataset='ManyRx', path=str(path), sha256=digest, rx=name, day=day, seal_sha256=sha(seal))
            with np.load(path, allow_pickle=False) as raw:
                x, y = raw['x'], raw['y']
            if x.shape != (len(y), 2, 256) or np.any(y < 0) or np.any(y >= 10): raise ValueError('Illegal held cell dimensions')
            values['x'].append(x); values['y'].append(y)
            values['rx'].append(np.full(len(y), names.index(name), dtype=np.int64))
            values['day'].append(np.full(len(y), day, dtype=np.int64))
            sources.append(dict(rx=name, day=day, sha256=digest))
    return {k:np.concatenate(v) for k,v in values.items()}, names, sources
