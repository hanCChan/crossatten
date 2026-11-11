"""Decode only verified count tensors and device/date identifiers, never I/Q."""
from pathlib import Path
import ast
import io
import json
import pickle
import numpy as np
from inspect_counts_metadata import HERE, admitted_blob

class CountMetadataUnpickler(pickle.Unpickler):
    ALLOWED = {
        ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
        ('numpy', 'ndarray'): np.ndarray,
        ('numpy', 'dtype'): np.dtype,
    }
    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(f'Unregistered global: {module}.{name}') from exc

def main():
    source = admitted_blob()
    metadata = CountMetadataUnpickler(io.BytesIO(source)).load()
    expected = {'capture_date_list', 'tx_list', 'rx_list', 'mat_date', 'mat_date_eq'}
    assert type(metadata) is dict and set(metadata) == expected, sorted(metadata)
    for key, length in [('capture_date_list', 4), ('tx_list', 174), ('rx_list', 41)]:
        assert type(metadata[key]) is list and len(metadata[key]) == length
        assert all(type(value) is str for value in metadata[key])
        assert len(set(metadata[key])) == length
    matrices = {}
    for field in ('mat_date', 'mat_date_eq'):
        assert type(metadata[field]) in (list, np.ndarray)
        if isinstance(metadata[field], list):
            assert all(type(a) is np.ndarray for a in metadata[field])
        matrix = np.asarray(metadata[field])
        assert matrix.shape == (4, 41, 174), (field, matrix.shape)
        assert matrix.dtype.kind in 'biuf' and matrix.dtype.kind not in 'cO'
        assert np.isfinite(matrix).all() and (matrix >= 0).all()
        assert (matrix == np.floor(matrix)).all()
        assert (matrix < 1000000).all(), 'count sanity cap, not waveform samples'
        matrices[field] = matrix.astype(np.int64)

    # Existing historical metadata only; no compact pickle, waveform or scores.
    root = HERE.parents[1]
    manifest_path = root / '05-wisig/out/05-1-ManyRx-manifest.json'
    historical = json.loads(manifest_path.read_text(encoding='utf-8'))
    tx = historical['tx_list']
    rx = historical['official_rx_order']
    assert len(tx) == 10 and len(rx) == 32 and len(set(rx)) == 32
    # Parse the fixed ManyRx list as Python syntax without importing a data loader.
    module = ast.parse((root / 'lib/xa/wisig.py').read_text(encoding='utf-8'))
    rx_constant = next(ast.literal_eval(n.value)['ManyRx'] for n in module.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'RX' for t in n.targets))
    assert set(rx_constant) == set(rx)
    assert set(rx) <= set(metadata['rx_list']) and set(tx) <= set(metadata['tx_list'])
    extras = [r for r in metadata['rx_list'] if r not in rx]
    assert len(extras) == 9
    tx_index = [metadata['tx_list'].index(t) for t in tx]
    candidates = []
    for r in extras:
        ri = metadata['rx_list'].index(r)
        per_day = []
        for di, day in enumerate(metadata['capture_date_list']):
            raw = matrices['mat_date'][di, ri, tx_index].tolist()
            eq = matrices['mat_date_eq'][di, ri, tx_index].tolist()
            per_day.append({'date': day, 'raw_counts_tx_order': raw, 'eq_counts_tx_order': eq,
                            'raw_positive_tx_count': sum(n > 0 for n in raw), 'eq_positive_tx_count': sum(n > 0 for n in eq),
                            'raw_min_count': min(raw), 'eq_min_count': min(eq), 'raw_total': sum(raw), 'eq_total': sum(eq)})
        candidates.append({'rx': r, 'not_in_manyrx32': True, 'per_day': per_day,
                           'common_tx_id_mapping': 'VERIFIED_HISTORICAL_MANIFEST_AND_OFFICIAL_CREATION_SCRIPT',
                           'historical_exposure': 'UNKNOWN', 'confirmation_eligible': False})
    result = {'status': 'METADATA_IDENTITIES_AND_SUPPORT_VERIFIED_NO_CONFIRMATION_OPENING',
              'source': 'data_summary.pkl', 'full_rx': metadata['rx_list'], 'full_tx_count': 174,
              'manyrx32': rx, 'original_manyrx10_tx_in_model_class_order': tx,
              'capture_dates': metadata['capture_date_list'],
              'matrix_shape': [4,41,174], 'matrix_axes': ['date', 'rx', 'tx'],
              'raw_total_full': int(matrices['mat_date'].sum()), 'eq_total_full': int(matrices['mat_date_eq'].sum()),
              'candidates': candidates, 'confirmation_access': 'CLOSED'}
    (HERE / 'CANDIDATE_METADATA.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    # Auxiliary aggregate-only support for the separately registered M0 package.
    fit18 = ['7-7','7-14','8-7','8-8','8-14','13-7','13-14','14-7','18-2','18-19',
             '23-3','23-5','23-6','23-7','24-5','24-6','24-13','24-16']
    differences = []
    totals = {'raw': 0, 'eq': 0}
    for r in fit18:
        ri = metadata['rx_list'].index(r)
        for di in (0, 1):
            for ti, t in zip(tx_index, tx):
                raw = int(matrices['mat_date'][di,ri,ti])
                eq = int(matrices['mat_date_eq'][di,ri,ti])
                totals['raw'] += raw
                totals['eq'] += eq
                if raw != eq:
                    differences.append({'rx': r, 'tx': t, 'date': metadata['capture_date_list'][di], 'raw_count': raw, 'eq_count': eq})
    (HERE / 'M0_SOURCE_COUNT_SUPPORT.json').write_text(json.dumps({
        'scope': 'fit18 x dates1-2 x historical ManyRx10TX aggregate counts only',
        'fit18': fit18, 'cell_count': 360, 'unequal_count_cells': len(differences),
        'totals': totals, 'unequal_cells': differences,
        'interpretation': 'Counts do not prove same-packet mapping or subset containment. No IQ loaded.'
    }, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'extra_rx': extras, 'candidate_count': len(candidates), 'confirmation_access': 'CLOSED'}, indent=2))

if __name__ == '__main__':
    main()
