"""Publish complete cell tables from saved predictions and source metadata only."""
import numpy as np
from s25_common import HERE, ROOT, P, read, write
from source_geometry import csv_write


def main():
    folder = ROOT / '24-lora-external-audit/phase2/budget_update/evaluations'
    saved = [r for p in sorted(folder.glob('*/metrics.json')) for r in read(p)['rows']]
    rows = []
    for view in ['native', 'strict_ib']:
        for baseline in ['ce', 'dann']:
            for b in [r for r in saved if r['method'] == baseline and r['view'] == view]:
                h = next(r for r in saved if r['method'] == 'h0' and
                         all(r[k] == b[k] for k in ['fold', 'seed', 'rx']))
                hc, bc = np.asarray(h['confusion']), np.asarray(b['confusion'])
                for tx in range(10):
                    assert hc[tx].sum() == bc[tx].sum() == 200
                    rows.append(dict(cohort='S24_saved_CPU', view=view, baseline=baseline,
                        fold=int(b['fold']), seed=int(b['seed']), rx=b['rx'], tx_raw_id=tx+31,
                        packets_per_seed=200, h0_correct=int(hc[tx, tx]), baseline_correct=int(bc[tx, tx]),
                        h0_minus_baseline_pp=float((hc[tx, tx]-bc[tx, tx])/2)))
    csv_write(HERE / 'external_tx_rx_comparisons.csv', rows)
    conditions = []
    for fold in (0, 1):
        with np.load(HERE / 'features' / f'metadata_f{fold}.npz', allow_pickle=False) as data:
            for ri, rx in enumerate(P['folds'][fold]['source']):
                for tx in range(10):
                    ids = (data['y'] == tx) & (data['rx'] == ri)
                    record = dict(fold=fold, rx=rx, tx_raw_id=tx+31, packets=int(ids.sum()))
                    for field in ['CFO', 'SNR']:
                        values = data[field][ids].astype(float)
                        values = values[np.isfinite(values)]
                        assert len(values) == 40
                        record[field+'_mean_as_reported'] = float(values.mean())
                        record[field+'_sd_as_reported'] = float(values.std(ddof=1))
                    conditions.append(record)
    csv_write(HERE / 'source_conditions_by_tx_rx.csv', conditions)
    write(HERE / 'DIAGNOSTIC_TABLES.json', dict(status='COMPLETE',
          external_tx_rx_rows=len(rows), source_metadata_tx_rx_rows=len(conditions),
          scope='All cells/seeds; saved external confusions and source-only saved metadata. '
                'No target metadata or new waveform reads. No physical causality inference.', new_fits=0))


if __name__ == '__main__':
    main()
