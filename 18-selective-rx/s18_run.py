"""One registered source-only D0, fixed token intervention, no fitting entry point."""
import argparse
import time
from contextlib import nullcontext
import numpy as np
import torch
from scipy.special import logsumexp
from s18_core import HERE, HISTORY, read, write, sha, now, validate, access
from s18_data import load, estimation_ids, halves
from s18_stats import fit_directions, scatters, overlap, random_basis, direction_metrics, compress, fit_rx_probe, score_rx_probe
from s18_hooks import capture, intervene, LAYERS
from s17_models import build
from metrics import rx_metrics


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


@torch.no_grad()
def forward(model, x, ids, feature_capture=True, intervention=None):
    logits = []
    context = intervene(model, 'tokens', **intervention) if intervention else nullcontext()
    collector_context = capture(model) if feature_capture else nullcontext()
    with context, collector_context as collector:
        for start in range(0, len(ids), 128):
            batch = torch.as_tensor(np.array(x[ids[start:start+128]], copy=True), device='cuda')
            logits.append(model(batch).cpu().numpy())
    return np.concatenate(logits), collector.arrays() if feature_capture else None


def scores(logits, y, rx, names):
    ce = logsumexp(logits, axis=1)-logits[np.arange(len(y)), y]
    other = logits.copy()
    other[np.arange(len(y)), y] = -np.inf
    margin = logits[np.arange(len(y)), y]-other.max(1)
    per = {}
    for r in np.unique(rx):
        ids = rx == r
        per[names[int(r)]] = dict(**rx_metrics(logits[ids], y[ids]), ce=float(ce[ids].mean()), margin=float(margin[ids].mean()))
    tx = {str(int(t)): float((logits[y == t].argmax(1) == t).mean()) for t in np.unique(y)}
    summary = {k: float(np.mean([v[k] for v in per.values()])) for k in ['acc', 'macro_f1', 'ce', 'margin']}
    summary.update(worst_rx=min(v['acc'] for v in per.values()), worst_tx=min(tx.values()), margin_sd=float(margin.std()))
    return dict(summary=summary, per_rx=per, per_tx=tx)


def delta(score, base):
    a, b = score['summary'], base['summary']
    return dict(delta_acc_pp=100*(a['acc']-b['acc']), delta_f1_pp=100*(a['macro_f1']-b['macro_f1']),
                delta_ce=a['ce']-b['ce'], delta_margin=a['margin']-b['margin'],
                normalized_delta_margin=(a['margin']-b['margin'])/max(b['margin_sd'], 1e-12),
                worst_rx_paired_pp=100*min(score['per_rx'][r]['acc']-base['per_rx'][r]['acc'] for r in score['per_rx']),
                worst_tx_paired_pp=100*min(score['per_tx'][t]-base['per_tx'][t] for t in score['per_tx']))


def leakage_probe(train_z, train_y, train_rx, train_day, val_z, val_y, val_rx, val_day):
    """Diagnostic TX/day residualization; labels never enter the network."""
    probe = fit_rx_probe(train_z, train_y, train_rx, train_day)
    scored = score_rx_probe(probe, val_z, val_y, val_rx, val_day)
    return dict(accuracy_rx_equal=scored['accuracy'], chance=scored['chance'],
                per_rx={str(int(r)): float(v) for r, v in zip(probe['rx_labels'], scored['per_rx'])},
                method='TX/day-conditioned equal-cell nearest train RX centroid; source-only diagnostic with truth labels')


def layer_statistics(z, vz, y, rx, day, vy, vrx, vday):
    fitted = fit_directions(z, y, rx, day)
    U = fitted['U']
    a, b = halves(y, rx, day)
    def basis(ids):
        return fit_directions(z[ids], y[ids], rx[ids], day[ids])['U']
    val_scatter = scatters(vz, vy, vrx, vday)
    # Evaluation fits are stability diagnostics only; never used by interventions.
    out = dict(channels=z.shape[1], rank=3, epsilon=fitted['epsilon'],
               train_cell_count_min=int(fitted['cell_counts'].min()), train_cell_count_max=int(fitted['cell_counts'].max()),
               train_metrics=direction_metrics(fitted['S_R'], fitted['S_T'], U),
               val_metrics=direction_metrics(val_scatter['S_R'], val_scatter['S_T'], U),
               stability=dict(packet_halves=overlap(basis(a), basis(b)),
                              days=overlap(basis(np.flatnonzero(day == 1)), basis(np.flatnonzero(day == 2))),
                              tx_halves=overlap(basis(np.flatnonzero(y < 5)), basis(np.flatnonzero(y >= 5))),
                              source_validation=overlap(U, fit_directions(vz, vy, vrx, vday)['U'])),
               rx_probe=leakage_probe(z, y, rx, day, vz, vy, vrx, vday),
               chance_overlap=3/z.shape[1], U=U, center=fitted['center'], U_tx=fitted['U_tx'])
    return fitted, val_scatter, out


@torch.no_grad()
def costs(model, U, center):
    """Dense arithmetic proxy plus actual synchronized batch1/128 CUDA latency."""
    mac = [0]
    handles = []
    def count(module, args, output):
        if isinstance(module, torch.nn.Conv1d):
            mac[0] += output.numel()*module.kernel_size[0]*module.in_channels/module.groups
        elif isinstance(module, torch.nn.Linear):
            mac[0] += output.numel()*module.in_features
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv1d, torch.nn.Linear)):
            handles.append(m.register_forward_hook(count))
    model(torch.zeros(1, 2, 256, device='cuda'))
    for h in handles:
        h.remove()
    dense = int(2*mac[0]+2*2*2*32*32*64)
    latency = []
    for n in [1, 128]:
        x = torch.randn(n, 2, 256, device='cuda')
        for arm in ['B0', 'static_selective']:
            ctx = intervene(model, 'tokens', U, center, alpha=.5) if arm != 'B0' else nullcontext()
            with ctx:
                for _ in range(10): model(x)
                torch.cuda.synchronize()
                elapsed = []
                for _ in range(30):
                    start = time.perf_counter()
                    model(x)
                    torch.cuda.synchronize()
                    elapsed.append(1000*(time.perf_counter()-start))
            latency.append(dict(arm=arm, batch=n, median_batch_ms=float(np.median(elapsed)), p90_batch_ms=float(np.quantile(elapsed, .9)), per_packet_ms=float(np.median(elapsed))/n))
    return dict(parameters=sum(p.numel() for p in model.parameters()), added_trainable_parameters_static=0,
                proposed_gate_parameters=3, fixed_basis_and_center_floats=64*3+64,
                dense_core_flops=dense, bottleneck_dense_flops=4*32*64*3,
                flop_convention='2 per MAC; conv/linear/attention products only; excludes FFT, norms, activations, softmax and elementwise work',
                latency=latency, device=torch.cuda.get_device_name(), torch=torch.__version__)


def run_fold(fold):
    validate()
    out = HERE/f'D0_fold{fold}.json'
    if out.exists():
        return
    if (HERE/f'attempt_fold{fold}.json').exists():
        raise RuntimeError('Interrupted attempt retained: no automatic scientific rerun')
    write(HERE/f'attempt_fold{fold}.json', dict(time=now(), status='RUNNING', fold=fold))
    pool, tr, va, meta, run, cfg = load(fold)
    ids = estimation_ids(pool, tr, fold)
    y, rx, day = [np.array(pool[k][ids]) for k in ['y', 'rx', 'day']]
    vy, vrx, vday = [np.array(pool[k][va]) for k in ['y', 'rx', 'day']]
    records = read(HISTORY/'17b-phase-intervention/final_evidence/MANIFEST.json')['records']
    matches = [r for r in records if r['job']['fold'] == fold]
    if len(matches) != 1: raise ValueError('Ambiguous historical checkpoint reference')
    expected = matches[0]['checkpoint_sha256']
    if sha(run/'model.pt') != expected:
        raise ValueError('Frozen checkpoint hash mismatch')
    access(fold=fold, kind='frozen_source_checkpoint', path=str(run/'model.pt'), sha256=expected)
    torch.manual_seed(cfg['data_seed'])
    model = build('ManyRx', cfg['data_seed']).cuda().eval()
    model.load_state_dict(torch.load(run/'model.pt', map_location='cpu', weights_only=True))
    for p in model.parameters(): p.requires_grad_(False)
    started = time.perf_counter()
    _, features = forward(model, pool['x'], ids)
    clean, val_features = forward(model, pool['x'], va)
    base = scores(clean, vy, vrx, meta['rx_names'])
    historical = read(run/'metrics.json')['source_validation']['summary']['acc_mean']
    if abs(base['summary']['acc']-historical) > 1e-10:
        raise ValueError('Frozen baseline reproduction mismatch')
    result = dict(time=now(), fold=fold, checkpoint_sha256=expected, data_seed=cfg['data_seed'], train_packets=len(ids), validation_packets=len(va),
                  train_indices_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest(), baseline=base, layers={}, interventions=[])
    primary = None
    for layer in LAYERS:
        fitted, vs, stats = layer_statistics(features[layer], val_features[layer], y, rx, day, vy, vrx, vday)
        result['layers'][layer] = stats
        if layer == 'tokens': primary = fitted, vs
    fitted, vs = primary
    bases = [('selective', fitted['U']), ('tx_dominant', fitted['U_tx'])]
    bases.extend((f'random_{i}', random_basis(64, 3, 18200+fold*10+i)) for i in range(3))
    z, vz = features['tokens'], val_features['tokens']
    for arm, U in bases:
        logits, _ = forward(model, pool['x'], va, feature_capture=False, intervention=dict(U=U, center=fitted['center'], alpha=.5))
        score = scores(logits, vy, vrx, meta['rx_names'])
        transformed = compress(vz, fitted['center'], U, alpha=.5)
        transformed_scatter = scatters(transformed, vy, vrx, vday)
        row = dict(arm=arm, alpha=.5, rank=3, score=score, **delta(score, base),
                   rx_scatter_reduction=1-float(np.trace(transformed_scatter['S_R']))/max(float(np.trace(vs['S_R'])), 1e-12),
                   tx_scatter_reduction=1-float(np.trace(transformed_scatter['S_T']))/max(float(np.trace(vs['S_T'])), 1e-12),
                   rx_probe=leakage_probe(compress(z, fitted['center'], U, alpha=.5), y, rx, day, transformed, vy, vrx, vday))
        result['interventions'].append(row)
        print('D0', fold, arm, 'delta_acc_pp', round(row['delta_acc_pp'], 4), flush=True)
    result['costs'] = costs(model, fitted['U'], fitted['center'])
    result['seconds'] = time.perf_counter()-started
    cache = HERE/'cache'; cache.mkdir(exist_ok=True)
    np.savez_compressed(cache/f'features_f{fold}.npz', **{f'train_{k}': v for k, v in features.items()},
                        **{f'val_{k}': v for k, v in val_features.items()}, y=y, rx=rx, day=day, vy=vy, vrx=vrx, vday=vday, train_indices=ids, validation_indices=va)
    result['feature_cache_sha256'] = sha(cache/f'features_f{fold}.npz')
    write(out, jsonable(result))
    write(HERE/f'attempt_fold{fold}.json', dict(time=now(), status='COMPLETE', fold=fold, seconds=result['seconds']))
    print('D0 fold', fold, 'complete', round(result['seconds'], 1), 'seconds', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--fold', type=int, choices=range(4)); args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if not torch.cuda.is_available(): raise RuntimeError('Registered CUDA environment required')
    for f in [args.fold] if args.fold is not None else range(4): run_fold(f)
