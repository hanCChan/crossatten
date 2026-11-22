"""Verify complete retained cohorts from saved logits, without waveform IO."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import PHASE, P, atomic_json, committed, digest, read, verify_seal


def main():
    retained = verify_seal('retained54')
    for cohort, count in [('baseline48', 54), ('native_ask6', 6)]:
        state = read(ROOT / f'EVALUATION_{cohort}.json')
        assert state['status'] == 'COMPLETE' and state['completed'] == count
        committed(ROOT / f'seal_{cohort}.json')
    expected = set(retained['authorized_prediction_jobs'])
    paths = list((ROOT / 'evaluations').glob('*/metrics.json'))
    assert {path.parent.name for path in paths} == expected and len(expected) == 60
    reference, checked, records = {}, [], []
    for path in sorted(paths):
        result = read(path)
        assert result['status'] == 'COMPLETE_REPORT_ONLY' and not result['target_adaptation']
        assert result['new_training_steps'] == 0 and len(result['rows']) == 5
        cohort = 'native_ask6' if result['job']['method'] == 'ask' else 'baseline48'
        assert result['seal_sha256'] == digest(ROOT / f'seal_{cohort}.json')
        assert result['amendment_sha256'] == digest(ROOT / 'AMENDMENT.json')
        for row in result['rows']:
            assert row['rx'] in P['folds'][row['fold']]['dev_val']
            saved = ROOT / 'predictions' / path.parent.name / (row['rx'] + '.npz')
            with np.load(saved, allow_pickle=False) as data:
                logits = data['logits'].astype(np.float64)
                labels = data['y'].astype(np.int64)
                packet_rows = data['rows']
            assert logits.shape == (2000, 10) and labels.shape == (2000,)
            assert np.isfinite(logits).all()
            np.testing.assert_array_equal(np.bincount(labels, minlength=10), np.full(10, 200))
            key = (row['fold'], row['rx'])
            if key in reference:
                np.testing.assert_array_equal(reference[key][0], labels)
                np.testing.assert_array_equal(reference[key][1], packet_rows)
            else:
                reference[key] = (labels.copy(), packet_rows.copy())
            guesses = logits.argmax(1)
            cm = np.zeros((10, 10), dtype=np.int64)
            np.add.at(cm, (labels, guesses), 1)
            np.testing.assert_array_equal(cm, row['confusion'])
            accuracy = float(np.mean(guesses == labels))
            f1 = float(np.mean(2.0 * cm.diagonal() / np.maximum(cm.sum(0) + cm.sum(1), 1)))
            maxima = logits.max(1)
            ce = float(np.mean(maxima + np.log(np.exp(logits - maxima[:, None]).sum(1)) - logits[np.arange(2000), labels]))
            assert abs(accuracy - row['acc']) < 1e-12
            assert abs(f1 - row['macro_f1']) < 1e-12
            assert abs(ce - row['ce']) < 3e-6
            checked.append({'job': path.parent.name, 'rx': row['rx'], 'sha256': digest(saved)})
            records.append(row)
    assert len(checked) == 300 and len(reference) == 10
    report = read(ROOT / 'RESULTS.json')
    for group in report['summary']:
        if group['view'] == 'strict_ib' and group['method'] == 'ask':
            assert group['status'] == 'DEFERRED_BY_USER_BUDGET' and group['fits'] == 0
            continue
        assert group['status'] == 'COMPLETE_REPORT_ONLY' and group['fits'] == 6
        view = 'native' if group['method'] == 'h0' else group['view']
        rows = [r for r in records if r['method'] == group['method'] and r['view'] == view]
        assert len(rows) == 30
        assert abs(np.mean([r['acc'] for r in rows]) - group['acc']) < 1e-12
        assert abs(np.mean([r['macro_f1'] for r in rows]) - group['macro_f1']) < 1e-12
        fits = [np.mean([r['acc'] for r in rows if r['fold'] == f and r['seed'] == s])
                for f in (0, 1) for s in (201, 202, 203)]
        assert abs(np.std(fits, ddof=1) - group['fit_sd']) < 1e-12
    assert len(report['paired_comparisons']) == 26
    for pair in report['paired_comparisons']:
        def select(method):
            view = 'native' if method == 'h0' else pair['view']
            return {(r['fold'], r['seed'], r['rx']): r[pair['metric']] for r in records
                    if r['method'] == method and r['view'] == view}
        a, b = select(pair['a']), select(pair['b'])
        assert a.keys() == b.keys() and len(a) == 30
        values = np.array([np.mean([a[k] - b[k] for k in a if k[2] == rx]) for rx in sorted({k[2] for k in a})])
        assert abs(values.mean() - pair['delta']) < 1e-12
        rng = np.random.default_rng(24016)
        bootstrap = values[rng.integers(0, 10, (10000, 10))].mean(1)
        np.testing.assert_allclose(np.quantile(bootstrap, [.025, .975]), pair['conditional_rx_cluster_ci'], atol=1e-12, rtol=0)
    events = [json.loads(line) for line in (ROOT / 'access.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(events) == 300 and len({(e['job'], e['rx']) for e in events}) == 300
    assert all(e['role'] == 'dev_report_only' and e['packets'] == 2000 for e in events)
    deferred = read(ROOT / 'AMENDMENT.json')['deferred_job_ids']
    assert all(not (PHASE / 'runs' / jid / 'config.json').exists() for jid in deferred)
    costs = read(ROOT / 'COSTS.json')
    assert costs['complete'] and len(costs['rows']) == 10
    for row in costs['rows']:
        for batch in ('1', '128'):
            assert len(row['batches'][batch]['samples']) == 15
            assert all(np.isfinite(row['batches'][batch]['samples']))
            assert min(row['batches'][batch]['samples']) > 0
    output = {'status': 'PASS', 'checked_models': 60, 'new_fits': 54, 'frozen_h0_models': 6,
              'checked_rx_cells': 300, 'packet_predictions': 600000, 'unique_rx': 10,
              'all_retained_checkpoint_config_metrics_hashes_pass': True, 'same_packet_rows_and_labels': True,
              'accuracy_f1_confusion_ce_recomputed': True, 'summary_means_fit_sd_recomputed': True,
              'paired_metric_intervals_recomputed': 26, 'all_six_native_ask_included': True,
              'strict_ask_six_not_started': True, 'complete_same_gpu_cost_rows': 10,
              'new_waveform_reads': 0, 'new_training_steps': 0,
              'verifier_sha256': digest(Path(__file__)), 'results_sha256': digest(ROOT / 'RESULTS.json'),
              'retained_seal_sha256': digest(ROOT / 'seal_retained54.json'), 'predictions': checked}
    atomic_json(ROOT / 'FINAL_INTEGRITY.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'predictions'}), flush=True)


if __name__ == '__main__':
    main()
