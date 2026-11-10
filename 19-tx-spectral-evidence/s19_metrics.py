"""Receiver-equal TX metrics and predeclared paired source gates."""
import numpy as np
import s19_core  # Establish frozen historical module paths for standalone QA.
from scipy.special import logsumexp
from metrics import rx_metrics


def score(logits, data, names):
    logits = np.asarray(logits)
    y, rx = np.asarray(data['y']), np.asarray(data['rx'])
    if not np.isfinite(logits).all(): raise ValueError('Nonfinite predictions')
    ce = logsumexp(logits, axis=1)-logits[np.arange(len(y)), y]
    other = logits.copy(); other[np.arange(len(y)), y] = -np.inf
    margin = logits[np.arange(len(y)), y]-other.max(1)
    per = {}
    for r in np.unique(rx):
        ids = rx == r
        per[names[int(r)]] = dict(**rx_metrics(logits[ids], y[ids]), ce=float(ce[ids].mean()), margin=float(margin[ids].mean()))
    tx = {str(t):float((logits[y == t].argmax(1) == t).mean()) for t in range(logits.shape[1]) if np.any(y == t)}
    summary = {key:float(np.mean([v[key] for v in per.values()])) for key in ['acc', 'macro_f1', 'ce', 'margin']}
    summary.update(worst_rx=min(v['acc'] for v in per.values()), worst_tx=min(tx.values()))
    return dict(summary=summary, per_rx=per, per_tx=tx)


def cluster_comparison(rows, candidate, control, draws=5000, seed=19191):
    repeated = {}; folds = []
    for f in range(4):
        left = next(r for r in rows if r['fold'] == f and r['arm'] == candidate)['held_source']
        right = next(r for r in rows if r['fold'] == f and r['arm'] == control)['held_source']
        if set(left['per_rx']) != set(right['per_rx']): raise ValueError('Unpaired source RXs')
        diff = {rx:100*(v['acc']-right['per_rx'][rx]['acc']) for rx,v in left['per_rx'].items()}
        for rx,v in diff.items(): repeated.setdefault(rx, []).append(v)
        folds.append(dict(fold=f, delta_acc_pp=100*(left['summary']['acc']-right['summary']['acc']),
                          delta_f1_pp=100*(left['summary']['macro_f1']-right['summary']['macro_f1']),
                          delta_ce=left['summary']['ce']-right['summary']['ce'],
                          worst_paired_rx_pp=min(diff.values()), per_rx_pp=diff))
    unique = {k:float(np.mean(v)) for k,v in repeated.items()}
    values = np.array(list(unique.values())); rng = np.random.default_rng(seed)
    distribution = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return dict(candidate=candidate, control=control, mean_fold_gain_pp=float(np.mean([r['delta_acc_pp'] for r in folds])),
                positive_folds=sum(r['delta_acc_pp'] > 0 for r in folds), worst_paired_rx_pp=min(r['worst_paired_rx_pp'] for r in folds),
                unique_rx=len(values), unique_rx_mean_pp=float(values.mean()), ci95_pp=np.quantile(distribution,[.025,.975]).tolist(),
                per_rx_pp=unique, folds=folds, bootstrap_draws=draws,
                interval_scope='Descriptive conditional RX-identity bootstrap; fold repetitions collapsed; shared fits and seed uncertainty not covered')


def gates():
    return dict(B4_vs_B0=dict(min_mean_pp=1., min_positive_folds=3, min_worst_paired_rx_pp=-2., ci_lower_gt=0.),
                B4_vs_B3=dict(min_mean_pp=0., min_positive_folds=3, ci_lower_gt=0.),
                B4_vs_B1_B2=dict(min_mean_pp=0., min_positive_folds=3, ci_lower_gt=0.),
                selection='Only B4 primary; no switch to another arm after held-source results')


def decide(rows):
    if len(rows) != 20 or {(r['fold'],r['arm']) for r in rows} != {(f,a) for f in range(4) for a in ['B0','B1','B2','B3','B4']}:
        raise ValueError('All 20 sealed source comparisons required')
    comparisons = {a:cluster_comparison(rows,'B4',a) for a in ['B0','B1','B2','B3']}
    checks = {}
    for control,v in comparisons.items():
        threshold = 1. if control == 'B0' else 0.
        checks[control] = dict(mean=bool(v['mean_fold_gain_pp'] >= threshold if control == 'B0' else v['mean_fold_gain_pp'] > threshold),
                               folds=bool(v['positive_folds'] >= 3), interval=bool(v['ci95_pp'][0] > 0.))
        if control == 'B0': checks[control]['worst_rx'] = bool(v['worst_paired_rx_pp'] >= -2.)
    admitted = all(all(c.values()) for c in checks.values())
    return dict(status='GO_INDEPENDENT_SOURCE_CONFIRMATION' if admitted else 'STOP_S19_SOURCE_GATE', admitted=admitted,
                gates=gates(), checks=checks, comparisons=comparisons, primary='B4',
                evidence_supervision_supported=all(checks['B3'].values()),
                no_target_access=True, more_search_allowed=False)
