"""Independent post-result S19 QA: cached predictions/metadata, no IQ or forward.

Checkpoint bytes are hashed, never deserialized. Teacher margins, conservative
TX evidence, physical consistency, every saved TX score, source gates and access
roles are regenerated. Historical experiment files remain read-only.
"""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from s19_core import HERE, ROOT, HISTORY, S18, SOURCE_CELLS, read, sha, write, now, validate, committed, git, job_id
from s19_data import roles, locations, evidence_ids
from s19_evidence import aggregate_contributions
from s19_metrics import score, decide

ARMS = ('B0', 'B1', 'B2', 'B3', 'B4')


def require(value, message):
    if not value:
        raise AssertionError(message)


def same(actual, expected, path='value', atol=1e-7, rtol=1e-7):
    """Nested comparisons retain missing-cell None and exact label/string roles."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), path+': keys')
        for key, value in expected.items():
            same(actual[key], value, path+'.'+str(key), atol, rtol)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        require(len(actual) == len(expected), path+': length')
        for i, value in enumerate(expected):
            same(actual[i], value, f'{path}[{i}]', atol, rtol)
    elif isinstance(expected, (float, np.floating)):
        require(np.isfinite(actual) and np.isfinite(expected), path+': nonfinite')
        require(np.isclose(actual, expected, atol=atol, rtol=rtol), path+': float mismatch')
    elif isinstance(expected, (int, np.integer, bool, np.bool_)):
        require(actual == expected, path+': integer/bool mismatch')
    else:
        require(actual == expected, path+': mismatch')


def margins(logits, y):
    logits = np.asarray(logits)
    require(logits.shape == (len(y), 10) and np.isfinite(logits).all(), 'finite [N,10] teacher logits')
    other = logits.copy()
    other[np.arange(len(y)), y] = -np.inf
    return logits[np.arange(len(y)), y]-other.max(axis=1)


def source_metadata(fold):
    """Only source labels and IDs; fit_x.npy is never opened by this audit."""
    cache, run = locations(fold)
    manifest = read(cache/'fit_manifest.json')
    fit, held, target = roles(fold)
    require(manifest['fit_rx'] == fit and manifest['held_rx'] == held and manifest['source_only'], 'source manifest roles')
    arrays = {}
    for key in ('fit_y', 'fit_rx', 'fit_day', 'train_ids', 'val_ids'):
        path = cache/(key+'.npy')
        require(sha(path) == manifest['sha256'][path.name], 'source metadata SHA '+key)
        arrays[key] = np.load(path, allow_pickle=False)
    n = len(arrays['fit_y'])
    train, val = arrays['train_ids'], arrays['val_ids']
    require(np.array_equal(np.sort(np.r_[train, val]), np.arange(n)), 'complete disjoint source split')
    require(not np.intersect1d(train, val).size, 'source train/val nonoverlap')
    require({manifest['rx_names'][int(r)] for r in np.unique(arrays['fit_rx'])} == set(fit), 'fit18 identities')
    require(set(np.unique(arrays['fit_day'])) == {1, 2}, 'source dates1-2')
    require(set(np.unique(arrays['fit_y'])) == set(range(10)), 'source ten TX classes')
    require(not set(fit) & set(held+target), 'fit/held/target separation')
    return arrays, manifest, run


def expected_masks():
    """NumPy implementation of the registered eight raised-sine subbands."""
    mask = np.zeros((8, 256), dtype=np.float64)
    for j in range(8):
        mask[j, j*32:(j+1)*32] = 1
    offsets = np.arange(-2, 2)
    right = .5*(1+np.sin(np.pi*(offsets+.5)/4))
    for j in range(8):
        bins = (j*32+offsets) % 256
        mask[:, bins] = 0
        mask[(j-1) % 8, bins] = 1-right
        mask[j, bins] = right
    return mask


def independent_q(drop, y, rx, day):
    """Independently reconstruct day-balanced TX/RX lower quartiles."""
    result = []
    for t in range(10):
        receivers = []
        for r in sorted(np.unique(rx)):
            cells = []
            for d in (1, 2):
                selected = (y == t) & (rx == r) & (day == d)
                if np.any(selected): cells.append(np.asarray(drop[selected], dtype=np.float64).mean(0))
            if cells: receivers.append(np.mean(cells, axis=0))
        require(len(receivers) >= 2, 'TX lower quartile has >=2 source RX')
        result.append(np.quantile(np.asarray(receivers), .25, axis=0, method='linear'))
    q = np.asarray(result)
    epsilon = 1e-6*max(float(q.mean()), 1e-6)
    return q, (q+epsilon)/(q+epsilon).sum(1, keepdims=True), epsilon


def verify_evidence(fold, metadata, manifest, teacher_digest):
    record = read(HERE/f'evidence_f{fold}.json')
    path = HERE/'cache'/f'evidence_f{fold}.npz'
    require(sha(path) == record['cache_sha256'], 'evidence cache SHA')
    require(record['teacher_sha256'] == teacher_digest and record['evidence_role'] == 'train_only', 'source teacher evidence role')
    summary = record['summary']
    require(summary['bands'] == 8 and summary['attenuation'] == .5 and summary['max_cell_samples'] == 16, 'fixed evidence design')
    train = metadata['train_ids']
    data = {key:metadata['fit_'+key][train] for key in ('y', 'rx', 'day')}
    selected = evidence_ids(data, fold, cap=16)
    with np.load(path, allow_pickle=False) as cache:
        same(cache['selected_train_indices'], selected, 'evidence relative IDs', atol=0, rtol=0)
        same(cache['selected_global_indices'], train[selected], 'evidence global IDs', atol=0, rtol=0)
        same(cache['val_global_indices'], metadata['val_ids'], 'evidence forbidden validation IDs', atol=0, rtol=0)
        require(not np.intersect1d(cache['selected_global_indices'], cache['val_global_indices']).size, 'evidence excludes validation')
        require(hashlib.sha256(cache['selected_global_indices'].tobytes()).hexdigest() == record['global_indices_sha256'], 'evidence ID hash')
        y, rx, day = (cache[k] for k in ('y', 'rx', 'day'))
        for key in ('y', 'rx', 'day'):
            same(cache[key], data[key][selected], 'evidence labels '+key, atol=0, rtol=0)
        require(record['packets'] == summary['packets'] == len(y), 'evidence packet counts')
        clean_margin = margins(cache['clean_logits'], y)
        for prefix, key in (('', 'masked_logits'), ('amplitude_', 'amplitude_logits'), ('phase_', 'phase_logits')):
            logits = cache[key]
            require(logits.shape == (len(y), 8, 10) and np.isfinite(logits).all(), 'finite counterfactual logits')
            signed = np.column_stack([clean_margin-margins(logits[:, j], y) for j in range(8)])
            same(cache[prefix+'signed_delta'], signed, prefix+'signed margin delta', atol=1e-6)
            same(cache[prefix+'drop'], np.maximum(signed, 0), prefix+'nonnegative margin drop', atol=1e-6)
        q, bar_q, epsilon = independent_q(cache['drop'], y, rx, day)
        same(cache['q'], q, 'independent q', atol=1e-10, rtol=1e-10)
        same(summary['q'], q, 'summary q', atol=1e-10, rtol=1e-10)
        same(cache['bar_q'], bar_q, 'float32 cached bar_q', atol=1e-7)
        same(summary['bar_q'], bar_q, 'float64 summary bar_q', atol=1e-10, rtol=1e-10)
        same(summary['epsilon'], epsilon, 'q smoothing epsilon', atol=1e-15)
        full = aggregate_contributions(cache['drop'], y, rx, day)
        for key, value in full.items(): same(summary[key], value, 'full contribution summary.'+key)
        same(cache['masks'], expected_masks(), 'registered band masks', atol=1e-6)
        masks = cache['masks']
        require(np.all((masks >= 0) & (masks <= 1)), 'nonnegative bounded masks')
        same(masks.sum(0), np.ones(256), 'mask partition of unity', atol=1e-6)
        evm, retained = cache['evm'], cache['retained_energy']
        require(evm.shape == retained.shape == (len(y), 8), 'EVM energy shape')
        require(np.isfinite(evm).all() and np.isfinite(retained).all(), 'finite physical descriptors')
        require(np.all((evm >= -1e-6) & (evm <= .5+1e-6)), 'attenuation EVM bound')
        require(np.all((retained >= .25-1e-6) & (retained <= 1+1e-6)), 'attenuation retained-energy bound')
        # For gain 1-.5*m, retained=1-E[m]+EVM^2 and E[m^2]=4EVM^2.
        mu = 1+evm.astype(float)**2-retained
        second = 4*evm.astype(float)**2
        require(np.all(mu >= -1e-5) and np.all(mu <= 1+1e-5), 'bounded weighted mask moments')
        require(np.all(second >= mu**2-1e-5) and np.all(second <= mu+1e-5), 'mask moment physical consistency')
        same(mu.sum(1), np.ones(len(y)), 'Parseval mask partition consistency', atol=1e-5)
        same(cache['amplitude_evm'], evm, 'amplitude EVM-matched control', atol=1e-6)
        same(cache['phase_evm'], evm, 'phase EVM-matched control', atol=1e-6)
        impulse = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(-.5*masks, axes=-1), axis=-1), axes=-1)
        impulse_energy = abs(impulse)**2
        tail = impulse_energy[:, abs(np.arange(256)-128) > 4].sum(1)/impulse_energy.sum(1)
        same(cache['residual_impulse_tail_fraction'], tail, 'circular-filter tail descriptor', atol=1e-6)
        artifacts = summary['artifacts']
        expected = dict(evm_mean=float(evm.mean()), evm_q95=float(np.quantile(evm, .95)),
                        retained_energy_mean=float(retained.mean()), retained_energy_min=float(retained.min()),
                        amplitude_evm_max_error=float(np.max(abs(cache['amplitude_evm']-evm))),
                        phase_evm_max_error=float(np.max(abs(cache['phase_evm']-evm))),
                        residual_impulse_tail_fraction=cache['residual_impulse_tail_fraction'].tolist())
        for key, value in expected.items(): same(artifacts[key], value, 'artifact summary.'+key)
        for prefix in ('amplitude', 'phase'):
            for suffix in ('signed', 'drop'):
                key = prefix+'_'+suffix+'_mean'
                source = prefix+('_signed_delta' if suffix == 'signed' else '_drop')
                same(summary['controls'][key], float(cache[source].mean()), 'control summary.'+key)
        same(summary['clean_source_accuracy'], float((cache['clean_logits'].argmax(1) == y).mean()), 'teacher train accuracy')
    return dict(fold=fold, cache_sha256=record['cache_sha256'], packets=record['packets'], source_training_only=True,
                q_recomputed=True, all_counterfactual_logits_checked=True, energy_matched_controls_checked=True)


def verify_predictions(path, expected, names, allowed_rx, metadata=None):
    with np.load(path, allow_pickle=False) as cache:
        require(set(cache.files) == {'logits', 'y', 'rx', 'day'}, 'prediction cache schema')
        logits = cache['logits']
        data = {key:cache[key] for key in ('y', 'rx', 'day')}
        require(logits.shape == (len(data['y']), 10) and np.isfinite(logits).all(), 'finite prediction logits')
        require(all(v.ndim == 1 and len(v) == len(logits) for v in data.values()), 'prediction labels have one value per logit')
        require(data['y'].dtype.kind in 'iu' and np.all((data['y'] >= 0) & (data['y'] < 10)), 'declared TX labels')
        require(data['rx'].dtype.kind in 'iu' and np.all((data['rx'] >= 0) & (data['rx'] < len(names))), 'RX index bounds')
        require(set(np.unique(data['day'])) == {1, 2}, 'prediction source dates1-2')
        require({names[int(r)] for r in np.unique(data['rx'])} == set(allowed_rx), 'prediction receiver role whitelist')
        if metadata is not None:
            for key in data: same(data[key], metadata[key], 'prediction provenance '+key, atol=0, rtol=0)
        actual = score(logits, data, names)
        same(expected, actual, 'score independently regenerated', atol=1e-9, rtol=1e-9)
        # Additional float64 math checks avoid relying only on the scorer.
        double = logits.astype(np.float64)
        ce = logsumexp(double, axis=1)-double[np.arange(len(double)), data['y']]
        true_margin = margins(double, data['y'])
        for rid in np.unique(data['rx']):
            selected = data['rx'] == rid
            recorded = expected['per_rx'][names[int(rid)]]
            cm = np.bincount(data['y'][selected]*10+logits[selected].argmax(1), minlength=100).reshape(10, 10)
            same(recorded['confusion'], cm, 'independent confusion', atol=0, rtol=0)
            same(recorded['ce'], float(ce[selected].mean()), 'independent float64 CE', atol=2e-6)
            same(recorded['margin'], float(true_margin[selected].mean()), 'independent float64 margin', atol=2e-6)
        return {key:data[key].copy() for key in data}


def verify_jobs(fold, metadata, manifest, seal, historical_record):
    _, original_run = locations(fold)
    original_cfg, original_metrics = read(original_run/'config.json'), read(original_run/'metrics.json')
    fit, held, target = roles(fold)
    val_data = {key:metadata['fit_'+key][metadata['val_ids']] for key in ('y', 'rx', 'day')}
    config_by_arm, history_by_arm, held_reference = {}, {}, None
    held_rows = []
    checkpoints = []
    for arm in ARMS:
        jid = job_id(fold, arm); output = HERE/'runs'/jid
        cfg, metrics = read(output/'config.json'), read(output/'metrics.json')
        held_metrics = read(output/'held_metrics.json')
        require(cfg['fold'] == metrics['fold'] == held_metrics['fold'] == fold, 'job fold')
        require(cfg['arm'] == metrics['arm'] == held_metrics['arm'] == arm, 'job arm')
        require(metrics['status'] == 'COMPLETE', 'completed fit before held evaluation')
        for filename in ('config.json', 'metrics.json'):
            require(sha(output/filename) == seal['metadata_sha256'][jid][filename], 'sealed metadata '+jid+' '+filename)
        ref = seal['checkpoints'][jid]
        expected_path = original_run/'model.pt' if arm == 'B0' else output/'model.pt'
        require(Path(ref['path']).resolve() == expected_path.resolve() and ref['fold'] == fold and ref['arm'] == arm, 'sealed checkpoint path/role')
        digest = sha(expected_path)
        require(digest == ref['sha256'] == metrics['model_sha256'] == held_metrics['model_sha256'], 'checkpoint SHA chain')
        seed = cfg.get('seed', cfg.get('job', {}).get('seed'))
        require(cfg['data_seed'] == original_cfg['data_seed'] and seed == 18101, 'paired data/model seed')
        require(cfg['batch'] == 128 and cfg['epochs_budget'] == 40 and cfg['precision'] == 'FP32', 'same training budget/batch/precision')
        if arm == 'B0':
            require(cfg['reused'] and metrics['reused'], 'B0 is reuse, not new fit')
            require(cfg['original_config_sha256'] == sha(original_run/'config.json'), 'B0 config provenance')
            require(digest == historical_record['checkpoint_sha256'], 'B0 historical checkpoint SHA')
            require(cfg['initial_state_sha256'] == original_cfg['initial_state_sha256'], 'B0 initial backbone metadata')
            for name in ('config.json', 'metrics.json'):
                require(sha(original_run/name) == historical_record['files_sha256'][name], 'B0 original metadata unchanged')
            require(not (output/'attempt.json').exists() and not (output/'model.pt').exists(), 'no duplicate B0 fit/checkpoint')
            same(metrics['source_validation']['summary']['acc'], original_metrics['source_validation']['summary']['acc_mean'], 'historical B0 accuracy')
        else:
            require(cfg['split_seed'] == 16001 and cfg['initial_backbone_sha256'] == original_cfg['initial_state_sha256'], 'paired backbone/split')
            require(cfg['train_indices_sha256'] == hashlib.sha256(metadata['train_ids'].tobytes()).hexdigest(), 'training index hash')
            require(cfg['val_indices_sha256'] == hashlib.sha256(metadata['val_ids'].tobytes()).hexdigest(), 'validation index hash')
            require(cfg['evidence_lambda'] == (.1 if arm == 'B4' else 0.), 'only B4 evidence loss')
            require(cfg['evidence_sha256'] == read(HERE/f'evidence_f{fold}.json')['cache_sha256'], 'same frozen evidence reference')
            require(cfg['active_params'] == 82426+(8 if arm == 'B1' else 272), 'declared active model parameter count')
            require(read(output/'attempt.json')['status'] == 'COMPLETE' and not (output/'failure.json').exists(), 'fit completed without retry/failure')
            with (output/'history.csv').open(encoding='utf-8', newline='') as file:
                history = list(csv.DictReader(file))
            require(len(history) == metrics['epochs'] and 1 <= len(history) <= 40, 'epoch budget respected')
            require([int(r['epoch']) for r in history] == list(range(1, len(history)+1)), 'sequential training epochs')
            best = min(history, key=lambda r:float(r['source_val_ce']))
            require(int(best['epoch']) == metrics['best_epoch'], 'source CE selects checkpoint')
            same(float(best['source_val_ce']), metrics['best_source_ce'], 'source minimum CE', atol=1e-12)
            same(metrics['source_validation']['summary']['ce'], metrics['best_source_ce'], 'prediction checkpoint source CE', atol=1e-6)
            require(all(len(row['order_sha256']) == 64 for row in history), 'epoch order digests present')
            history_by_arm[arm] = history
        require(sha(output/'source_val_predictions.npz') == metrics['prediction_sha256'], 'source predictions SHA')
        verify_predictions(output/'source_val_predictions.npz', metrics['source_validation'], manifest['rx_names'], fit, val_data)
        require(sha(output/'held_predictions.npz') == held_metrics['prediction_sha256'], 'held predictions SHA')
        require(held_metrics['rx_names'] == sorted(fit+held), 'held RX-index names')
        this_held = verify_predictions(output/'held_predictions.npz', held_metrics['held_source'], held_metrics['rx_names'], held, held_reference)
        if held_reference is None: held_reference = this_held
        require(datetime.fromisoformat(held_metrics['time']) > datetime.fromisoformat(seal['time']), 'held metrics after seal')
        sources = held_metrics['held_sources']
        require(len(sources) == 12 and {(r['rx'], r['day']) for r in sources} == {(r, d) for r in held for d in (1, 2)}, 'held cell role set')
        config_by_arm[arm] = cfg
        held_rows.append(held_metrics)
        checkpoints.append(dict(job_id=jid, fold=fold, arm=arm, checkpoint_sha256=digest,
                                source_predictions_checked=True, held_predictions_checked=True))
    require(config_by_arm['B3']['initial_full_state_sha256'] == config_by_arm['B4']['initial_full_state_sha256'], 'B3/B4 full initial state identical')
    for arm in ('B2', 'B3', 'B4'):
        left, right = history_by_arm['B1'], history_by_arm[arm]
        require(all(a['order_sha256'] == b['order_sha256'] for a, b in zip(left, right)), 'paired epoch sample orders '+arm)
    for key in ('data_seed', 'split_seed', 'batch', 'epochs_budget', 'precision', 'optimizer', 'initial_backbone_sha256', 'train_indices_sha256', 'val_indices_sha256'):
        require(len({config_by_arm[a][key] for a in ARMS[1:]}) == 1, 'same formal-fit recipe '+key)
    return held_rows, checkpoints


def verify_access(seal):
    entries = [json.loads(line) for line in (HERE/'access.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(entries) == 168, 'exact registered source access count168')
    kinds = {name:[r for r in entries if r['kind'] == name] for name in ('fit_cache', 'source_teacher_checkpoint', 'held_source_cell', 'sealed_checkpoint')}
    require(set(kinds) == {r['kind'] for r in entries}, 'access type whitelist')
    require({k:len(v) for k,v in kinds.items()} == dict(fit_cache=80, source_teacher_checkpoint=20, held_source_cell=48, sealed_checkpoint=20), 'access counts by role')
    for fold in range(4):
        fit, held, target = roles(fold); cache, run = locations(fold)
        manifest = read(cache/'fit_manifest.json')
        rows = [r for r in kinds['fit_cache'] if r['fold'] == fold]
        require(len(rows) == 20, 'five fit-pool loads per fold')
        for key in ('x', 'y', 'rx', 'day'):
            path = cache/f'fit_{key}.npy'
            selected = [r for r in rows if Path(r['path']).resolve() == path.resolve()]
            require(len(selected) == 5, 'source fit path whitelist')
            for row in selected:
                require(row['dataset'] == 'ManyRx' and row['rx'] == fit and row['days'] == [1, 2], 'source access roles')
                require(row['sha256'] == manifest['sha256'][path.name], 'source loader hashes match sealed manifest')
        teachers = [r for r in kinds['source_teacher_checkpoint'] if r['fold'] == fold]
        require(len(teachers) == 5, 'five source teacher provenance accesses')
        for row in teachers:
            require(Path(row['path']).resolve() == (run/'model.pt').resolve(), 'teacher checkpoint whitelist')
            require(row['sha256'] == seal['checkpoints'][job_id(fold, 'B0')]['sha256'], 'teacher/checkpoint SHA')
        cells = [r for r in kinds['held_source_cell'] if r['fold'] == fold]
        require(len(cells) == 12 and {(r['rx'], r['day']) for r in cells} == {(r, d) for r in held for d in (1, 2)}, 'held source cells once per fold')
        for row in cells:
            require(row['dataset'] == 'ManyRx', 'held dataset whitelist')
            expected = SOURCE_CELLS/'ManyRx'/f"rx_{row['rx']}_day{row['day']}.npz"
            require(Path(row['path']).resolve() == expected.resolve(), 'held source path whitelist')
            require(datetime.fromisoformat(row['time']) > datetime.fromisoformat(seal['time']), 'held cell first read after seal')
            require(row['seal_sha256'] == sha(HERE/'seal_pilot.json'), 'held access seal hash')
        model_rows = [r for r in kinds['sealed_checkpoint'] if r['fold'] == fold]
        require(len(model_rows) == 5 and {r['arm'] for r in model_rows} == set(ARMS), 'each sealed model evaluated once')
        for row in model_rows:
            ref = seal['checkpoints'][job_id(fold, row['arm'])]
            require(Path(row['path']).resolve() == Path(ref['path']).resolve() and row['sha256'] == ref['sha256'], 'evaluated checkpoint sealed SHA')
            require(row['seal_sha256'] == sha(HERE/'seal_pilot.json'), 'checkpoint access seal SHA')
            require(datetime.fromisoformat(row['time']) > datetime.fromisoformat(seal['time']), 'sealed-model access chronology')
    return dict(entries=len(entries), by_kind={k:len(v) for k,v in kinds.items()}, forbidden_role_entries=0,
                real_target_entries=0, many_sig_entries=0, shen_entries=0, held_reads_after_seal=True)


def verify_s18():
    path = S18/'18-selective-rx'
    manifest = read(path/'MANIFEST.json')
    require(len(manifest['files_sha256']) == 43, 'S18 full archive has43 files')
    for name, digest in manifest['files_sha256'].items():
        require(sha(path/name) == digest, 'S18 original archive unchanged: '+name)
    require(read(path/'STATUS.json')['status'] == 'STOP_D0_MECHANISM_GATE', 'S18 STOP preserved')
    require(read(HERE/'S18_VERIFIED.json')['commit'].startswith('933d4a6'), 'S18 expected commit verified before S19')
    return dict(verified_files=43, manifest_sha256=sha(path/'MANIFEST.json'), original_stop_preserved=True)


def run():
    registration = validate()
    require((HERE/'decision_pilot.json').is_file(), 'all scientific source results and decision must exist before final QA')
    seal_path = HERE/'seal_pilot.json'; committed(seal_path); seal = read(seal_path)
    expected_jobs = {job_id(f, a) for f in range(4) for a in ARMS}
    require(set(seal['checkpoints']) == set(seal['metadata_sha256']) == expected_jobs, 'exact20 sealed jobs')
    require(seal['all_checkpoints_complete'] and seal['formal_new_fits'] == 16 and seal['reused_B0'] == 4, 'fits16 <= budget20 and reuse4')
    require(seal['real_target_access'] is False and seal['held_source_read_before_seal'] is False, 'seal data boundaries')
    records = read(HISTORY/'17b-phase-intervention/final_evidence/MANIFEST.json')['records']
    evidence_checks, rows, checkpoint_checks = [], [], []
    for fold in range(4):
        metadata, manifest, run_path = source_metadata(fold)
        candidates = [r for r in records if r['job']['fold'] == fold]
        require(len(candidates) == 1, 'unique historical B0 per fold')
        evidence_checks.append(verify_evidence(fold, metadata, manifest, candidates[0]['checkpoint_sha256']))
        require(sha(HERE/f'evidence_f{fold}.json') == seal['evidence_sha256'][str(fold)], 'evidence record sealed before held-source')
        held_rows, jobs = verify_jobs(fold, metadata, manifest, seal, candidates[0])
        rows.extend(held_rows); checkpoint_checks.extend(jobs)
        print(f'PASS fold{fold}: evidence/logits/physical controls, five sealed models, all source and held TX metrics', flush=True)
    actual = read(HERE/'decision_pilot.json')
    expected = decide(rows)
    for key, value in expected.items():
        require(actual[key] == value, 'source decision exact recomputation: '+key)
    require(actual['new_fits'] == 16 and actual['reused_B0'] == 4 and actual['real_target_access'] is False
            and actual['ManySig_read'] is False and actual['Shen_read'] is False, 'decision scope and budget')
    access = verify_access(seal)
    original = verify_s18()
    changed = git('diff', '--name-only', '933d4a6', '--').splitlines()
    untracked = git('ls-files', '--others', '--exclude-standard').splitlines()
    require(all(p.startswith('19-tx-spectral-evidence/') for p in changed+untracked), 'S19 changes isolated from all historical experiments')
    artifacts = {str(path.relative_to(HERE)):sha(path) for path in HERE.glob('evidence_f*.json')}
    artifacts.update({str(path.relative_to(HERE)):sha(path) for path in [HERE/'decision_pilot.json', seal_path]})
    audit = dict(time=now(), status='PASS', registration_sha256=sha(HERE/'registration.json'),
                 audit_code_sha256=sha(Path(__file__)), result_sha256=artifacts,
                 evidence=evidence_checks, checkpoint_and_prediction_checks=checkpoint_checks,
                 source_decision_exact=True, decision_status=actual['status'], source_access=access, S18=original,
                 formal_new_fits=16, maximum_formal_new_fits=20, reused_B0=4,
                 raw_iq_read_by_audit=False, checkpoint_bytes_hashed_only=True, model_forward_calls_by_audit=0,
                 source_metric_scope='Every saved float32 source/held logit independently rescored; CE/margin also checked using float64 math',
                 source_iq_hash_scope='No fit_x.npy or source-cell IQ opened by audit; original guarded-loader SHA records checked against metadata and roles',
                 initialization_scope='Full initial-state hashes B3/B4 and historical backbone hashes match; all new-arm common epoch order hashes match',
                 limitations='Teacher source validation is historical; masks are circular counterfactuals; RX bootstrap is conditional and does not cover all seed/shared-fit dependence',
                 changed_paths=changed, untracked_paths=untracked)
    write(HERE/'FINAL_ACCESS_AUDIT.json', audit)
    print('PASS S19 final QA: registered history/code, conservative TX evidence, all logits/metrics,16 fits,20 sealed models and source-only access', flush=True)


if __name__ == '__main__':
    run()
