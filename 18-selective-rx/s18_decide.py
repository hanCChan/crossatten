"""Registered D0 decision; fixed thresholds, receiver-level paired bootstrap."""
import numpy as np
from s18_core import HERE, read, write, now, validate


def gates():
    return dict(stability=dict(packet_halves=.5, days=.35, tx_halves=.35, source_validation=.5),
                min_val_rx_fraction=.15, min_rx_reduction_advantage_over_random=.05,
                min_tx_reduction_advantage_of_tx_control=.05,
                min_ce_advantage_over_random=.005, min_ce_advantage_over_tx_control=.01,
                max_accuracy_drop_pp=1., max_f1_drop_pp=1., max_ce_increase=.05,
                max_margin_drop_sd=.10, max_worst_paired_rx_drop_pp=2., max_worst_paired_tx_drop_pp=5.,
                stable_and_specific_folds=3, require_preservation_all_folds=True,
                require_rx_cluster_accuracy_advantage_ci_lower_gt_zero=True)


def paired_rx_interval(rows, draws=5000, seed=18301):
    """Average repeated fold appearances per RX, resample unique RX identities."""
    values = {}
    for result in rows:
        selected = next(r for r in result['interventions'] if r['arm'] == 'selective')
        random = [r for r in result['interventions'] if r['arm'].startswith('random_')]
        for rx, m in selected['score']['per_rx'].items():
            diff = 100*(m['acc']-np.mean([r['score']['per_rx'][rx]['acc'] for r in random]))
            values.setdefault(rx, []).append(float(diff))
    collapsed = {r: float(np.mean(v)) for r, v in values.items()}
    sample = np.array(list(collapsed.values()))
    rng = np.random.default_rng(seed)
    boots = sample[rng.integers(0, len(sample), size=(draws, len(sample)))].mean(1)
    return dict(mean_pp=float(sample.mean()), ci95_pp=np.quantile(boots, [.025, .975]).tolist(), independent_rx=len(sample),
                per_rx_pp=collapsed, bootstrap_draws=draws,
                scope='conditional descriptive RX-cluster interval; overlapping fold fits and one seed; no training-randomness coverage')


def decide(rows):
    if sorted(r['fold'] for r in rows) != list(range(4)):
        raise ValueError('D0 requires all four folds exactly once')
    g = gates()
    folded = []
    for result in sorted(rows, key=lambda r: r['fold']):
        s = next(r for r in result['interventions'] if r['arm'] == 'selective')
        tx = next(r for r in result['interventions'] if r['arm'] == 'tx_dominant')
        rnd = [r for r in result['interventions'] if r['arm'].startswith('random_')]
        layer = result['layers']['tokens']
        checks = {f'stable_{k}': layer['stability'][k] >= v for k, v in g['stability'].items()}
        checks['rx_support'] = layer['val_metrics']['rx_fraction'] >= g['min_val_rx_fraction']
        stable = all(checks.values())
        specific = dict(rx_scatter=s['rx_scatter_reduction']-np.mean([r['rx_scatter_reduction'] for r in rnd]) >= g['min_rx_reduction_advantage_over_random'],
                        tx_protection=tx['tx_scatter_reduction']-s['tx_scatter_reduction'] >= g['min_tx_reduction_advantage_of_tx_control'],
                        random_ce=np.mean([r['delta_ce'] for r in rnd])-s['delta_ce'] >= g['min_ce_advantage_over_random'],
                        tx_control_ce=tx['delta_ce']-s['delta_ce'] >= g['min_ce_advantage_over_tx_control'])
        preserve = dict(accuracy=s['delta_acc_pp'] >= -g['max_accuracy_drop_pp'], f1=s['delta_f1_pp'] >= -g['max_f1_drop_pp'],
                        ce=s['delta_ce'] <= g['max_ce_increase'], margin=s['normalized_delta_margin'] >= -g['max_margin_drop_sd'],
                        worst_rx=s['worst_rx_paired_pp'] >= -g['max_worst_paired_rx_drop_pp'], worst_tx=s['worst_tx_paired_pp'] >= -g['max_worst_paired_tx_drop_pp'])
        folded.append(dict(fold=result['fold'], stability_checks={k:bool(v) for k,v in checks.items()},
                           specificity_checks={k:bool(v) for k,v in specific.items()}, preservation_checks={k:bool(v) for k,v in preserve.items()},
                           stable=bool(stable), specific=bool(all(specific.values())), preserved=bool(all(preserve.values()))))
    interval = paired_rx_interval(rows)
    admitted = (all(f['preserved'] for f in folded) and sum(f['stable'] and f['specific'] for f in folded) >= g['stable_and_specific_folds']
                and interval['ci95_pp'][0] > 0)
    return dict(time=now(), status='GO_SOURCE_PILOT' if admitted else 'STOP_D0_MECHANISM_GATE', admitted=bool(admitted),
                primary_layer='tokens', rank=3, alpha=.5, gates=g, folds=folded, receiver_cluster_selective_vs_random=interval,
                new_fits=0, real_target_access=False, held_source_read=False, many_sig_read=False,
                historical_stops_preserved=True, pilot_authorized_by_d0=bool(admitted),
                interpretation='D0 tests this fixed channel-subspace proposal; no causal hardware separation or target generalization conclusion')


if __name__ == '__main__':
    validate()
    path = HERE/'decision_d0.json'
    if path.exists(): raise RuntimeError('D0 decision already sealed')
    write(path, decide([read(HERE/f'D0_fold{f}.json') for f in range(4)]))
