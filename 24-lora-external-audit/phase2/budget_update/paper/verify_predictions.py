"""Independent integrity check using saved predictions only; no waveform IO."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
state = json.loads((root / 'EVALUATION_baseline48.json').read_text(encoding='utf-8'))
assert state['status'] == 'COMPLETE' and state['completed'] == 54
reference = {}
checked = []
for path in sorted((root / 'evaluations').glob('*/metrics.json')):
    record = json.loads(path.read_text(encoding='utf-8'))
    if record['job']['method'] == 'ask':
        continue
    assert record['status'] == 'COMPLETE_REPORT_ONLY'
    for row in record['rows']:
        saved = root / 'predictions' / path.parent.name / (row['rx'] + '.npz')
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
        guesses = np.argmax(logits, axis=1)
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (labels, guesses), 1)
        np.testing.assert_array_equal(confusion, row['confusion'])
        accuracy = float(np.mean(guesses == labels))
        f1 = float(np.mean(2.0 * np.diag(confusion) / np.maximum(confusion.sum(0) + confusion.sum(1), 1)))
        maxima = logits.max(axis=1)
        ce = float(np.mean(maxima + np.log(np.exp(logits - maxima[:, None]).sum(axis=1)) - logits[np.arange(2000), labels]))
        assert abs(accuracy - row['acc']) < 1e-12
        assert abs(f1 - row['macro_f1']) < 1e-12
        assert abs(ce - row['ce']) < 3e-6
        checked.append({'job': path.parent.name, 'rx': row['rx'], 'sha256': hashlib.sha256(saved.read_bytes()).hexdigest()})
assert len(checked) == 270 and len(reference) == 10
summary = json.loads((root / 'RESULTS.json').read_text(encoding='utf-8'))
rows = [r for p in (root / 'evaluations').glob('*/metrics.json') for r in json.loads(p.read_text(encoding='utf-8'))['rows']]
for group in summary['summary']:
    if group['status'] != 'COMPLETE_REPORT_ONLY':
        continue
    members = [r for r in rows if r['method'] == group['method'] and r['view'] == ('native' if group['method'] == 'h0' else group['view'])]
    assert len(members) == 30
    assert abs(np.mean([r['acc'] for r in members]) - group['acc']) < 1e-12
    assert abs(np.mean([r['macro_f1'] for r in members]) - group['macro_f1']) < 1e-12
report = {'status': 'PASS', 'checked_models': 54, 'checked_rx_cells': 270, 'packet_predictions': 540000, 'unique_rx': 10,
          'same_labels_and_packet_rows_across_methods_views_seeds': True, 'accuracy_f1_confusion_independently_recomputed': True,
          'cross_entropy_absolute_tolerance': 3e-6, 'summary_means_recomputed': True, 'new_waveform_reads': 0, 'new_training_steps': 0,
          'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'results_sha256': hashlib.sha256((root / 'RESULTS.json').read_bytes()).hexdigest(),
          'baseline_seal_sha256': hashlib.sha256((root / 'seal_baseline48.json').read_bytes()).hexdigest(),
          'predictions': checked}
(root / 'PREDICTION_INTEGRITY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in report.items() if k != 'predictions'}, ensure_ascii=False))
