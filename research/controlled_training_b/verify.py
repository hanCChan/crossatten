"""Independent derived-JSON QA; no IQ, predictions or checkpoints are decoded."""
from __future__ import annotations
from datetime import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def close(a, b):
    if not np.allclose(a, b, rtol=0, atol=1e-12):
        raise AssertionError(f'Derived values differ: {a} / {b}')


def metric_row(row):
    cm = np.asarray(row['confusion'])
    n = int(cm.sum()); tp = np.diag(cm)
    assert n == row['packets'] and n > 0
    assert n-int(tp.sum()) == row['errors']
    close(tp.sum()/n, row['accuracy'])
    denominator = cm.sum(0)+cm.sum(1)
    f1 = np.divide(2*tp, denominator, out=np.zeros(len(tp), dtype=float), where=denominator>0)
    close(f1.mean(), row['macro_f1'])
    q = np.asarray(list(row['margin_quantiles'].values()))
    assert np.isfinite(q).all() and np.all(np.diff(q) >= 0)
    assert math.isfinite(row['ce']) and row['ce'] >= 0


def paired_row(row):
    n = row['packets']
    assert row['prediction_disagreements'] == row['corrected_errors']+row['new_errors']+row['both_wrong_changed_class']
    assert row['second_errors'] == row['first_errors']-row['corrected_errors']+row['new_errors']
    close(row['prediction_disagreements']/n, row['disagreement'])
    close((row['corrected_errors']-row['new_errors'])/n, row['accuracy_delta'])


def main():
    out = HERE/'QA_REPORT.json'
    if out.exists(): raise FileExistsError('Keep original QA')
    raw = (HERE/'ANALYSIS.json').read_bytes()
    assert gzip.decompress((HERE/'ANALYSIS.json.gz').read_bytes()) == raw
    data = json.loads(raw)
    evidence, seal, reg = (read(HERE/name) for name in ('RUN_EVIDENCE.json', 'STAGE_B_SEAL.json', 'PRE_REGISTRATION.json'))
    rows, pairs_checked, intervals = 0, 0, 0
    for run, records in evidence.items():
        history = records['HISTORY']; best, best_ep, wait = math.inf, 0, 0
        lr, scheduler_best, bad = .001, math.inf, 0
        for epoch, row in enumerate(history, 1):
            assert row['epoch'] == epoch and row['updates'] == epoch*124
            close(row['lr_used'], lr)
            ce = row['val_rx_ce']
            if ce < best: best, best_ep, wait = ce, epoch, 0
            else: wait += 1
            if ce < scheduler_best*(1-1e-4): scheduler_best, bad = ce, 0
            else: bad += 1
            if bad > 5: lr, bad = max(lr*.1, 1e-6), 0
            close(row['lr_next'], lr)
            assert row['best_epoch'] == best_ep and row['wait'] == wait
        assert wait == 10 or len(history) == 40
        assert records['COMPLETE']['selected_epoch'] == best_ep and records['COMPLETE']['epochs'] == len(history)
        assert records['COMPLETE']['active_parameters'] == 82426
        close(records['COMPLETE']['best_val_rx_ce'], best)
        assert all(x['exact'] for x in records['COMPLETE']['val_inference_replay'].values())
    for split in ('val', 'held'):
        for endpoint in ('selected', 'epoch10'):
            r = data[split][endpoint]
            summaries, pairs = r['runs'], r['paired_changes']
            for summary in summaries.values():
                total = summary['packets']
                assert total == (3551 if split == 'val' else 12000)
                group_rows = [summary['packet_weighted']]+list(summary['per_rx'].values())+summary['tx_rx']+summary['tx_rx_day']
                for row in group_rows: metric_row(row); rows += 1
                for table in ('tx_rx', 'tx_rx_day'): assert sum(x['packets'] for x in summary[table]) == total
                for metric in ('accuracy', 'macro_f1', 'ce'): close(summary['rx_equal'][metric], np.mean([x[metric] for x in summary['per_rx'].values()]))
            receivers = sorted(summaries['B0_S00']['per_rx'])
            draws = np.random.default_rng(22402).integers(0, len(receivers), size=(10000,len(receivers)))
            for metric, effect in r['composition_effects'].items():
                delta = {}
                for flow in ('S00', 'S11'):
                    delta[flow] = np.asarray([summaries[f'B1_{flow}']['per_rx'][rx][metric]-summaries[f'B0_{flow}']['per_rx'][rx][metric] for rx in receivers])
                    close(effect['streams'][flow]['estimate'], delta[flow].mean())
                    close(effect['streams'][flow]['paired_rx_bootstrap_95_interval'], np.quantile(delta[flow][draws].mean(1), [.025,.975])); intervals += 1
                mean, diff = (delta['S00']+delta['S11'])/2, delta['S11']-delta['S00']
                for label, values in (('average',mean), ('between_stream_difference',diff)):
                    close(effect[label]['estimate'], values.mean())
                    close(effect[label]['paired_rx_bootstrap_95_interval'], np.quantile(values[draws].mean(1), [.025,.975])); intervals += 1
                ci = effect['average']['paired_rx_bootstrap_95_interval']
                same = delta['S00'].mean()*delta['S11'].mean() > 0
                cross = all(abs(v.mean()) >= effect['practical_scale'] for v in (delta['S00'],delta['S11'],mean))
                assert effect['consistent_finite_effect_support'] == (same and cross and (ci[0]>0 or ci[1]<0))
            for flow, pair in pairs.items():
                for row in [pair['packet_weighted']]+list(pair['per_rx'].values())+pair['tx_rx']+pair['tx_rx_day']:
                    paired_row(row); pairs_checked += 1
                for table in ('tx_rx','tx_rx_day'): assert sum(x['packets'] for x in pair[table]) == pair['packet_weighted']['packets']
                for metric, ci in pair['paired_rx_bootstrap_95_intervals'].items():
                    values = np.asarray([pair['per_rx'][rx][metric] for rx in receivers])
                    close(ci, np.quantile(values[draws].mean(1),[.025,.975])); intervals += 1
            consistent = 0
            for row in r['local_risk_consistency']['groups']:
                key = row['rx'], row['tx_class_index']
                values, supports = [], []
                for flow in ('S00','S11'):
                    p = next(x for x in pairs[flow]['tx_rx'] if (x['rx'],x['tx_class_index']) == key)
                    values.append(-p['accuracy_delta']); supports.append(p['packets'])
                    close(row['risk_change'][flow], values[-1])
                repeated = min(supports)>=50 and values[0]*values[1]>0 and all(abs(v)>=.05-1e-12 for v in values)
                assert row['repeated_local_change'] == repeated
                consistent += repeated
            assert r['local_risk_consistency']['repeated_local_groups'] == consistent
    logs = [json.loads(x) for x in (HERE/'ACCESS_LOG.jsonl').read_text(encoding='utf-8').splitlines()]
    allowed = {i['path'] for i in list(reg['inputs'].values())+reg['held_inputs']}
    assert all(x['path'] in allowed for x in logs)
    held = [x for x in logs if x['action'].startswith('load_registered_held')]
    assert len(held)==12 and all(datetime.fromisoformat(x['time'])>=datetime.fromisoformat(seal['time']) for x in held)
    assert len(evidence)==4 and not list((HERE/'artifacts').rglob('FAILURE.json'))
    assert all(seal['pairing_checks'].values())
    report = dict(status='PASS', scope='Independent text-derived numeric checks; no IQ/weights/prediction decoding or extra fits.',
                  metric_group_rows=rows, paired_group_rows=pairs_checked, recomputed_bootstrap_intervals=intervals,
                  total_fits=4, failed_fits=0, held_payload_loads=12, target_loads=0,
                  pairing_checks_passed=len(seal['pairing_checks']), implementation_code_unchanged=True,
                  finite_level_interpretation_only=True, auxiliary_endpoint_does_not_replace_primary=True)
    for relative, digest in reg['code_sha256'].items():
        assert hashlib.sha256((HERE.parent.parent/relative).read_bytes()).hexdigest() == digest
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__': main()
