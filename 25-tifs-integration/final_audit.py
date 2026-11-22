"""Integrity checks using saved features/evidence only; no inference or training."""
import collections
import csv
import json
import platform
from pathlib import Path
import re
import subprocess
import sys
import importlib.metadata

import numpy as np
from PIL import Image
from pypdf import PdfReader

from geometry_stats import geometry
from mechanism_extension import validate_extension
from s25_common import HERE, ROOT, P, authorize_source, digest, now, read, validate, write


def csv_rows(name):
    with (HERE / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    registration = validate()
    extension = validate_extension()
    records = registration['models'] + extension['additional_models']
    assert len(records) == len({r['id'] for r in records}) == 24
    historical = read(ROOT / '23-hstem-mechanism-evidence/registration.json')['references']
    observations = read(HERE / 'MECHANISM_EXTENSION.json')
    assert observations['models'] == 24 and observations['total_source_feature_predictions'] == 48000
    assert observations['new_fits'] == observations['new_optimizer_steps'] == 0
    checkpoint_checks = []
    baselines = {}
    accuracy_differences = []
    max_geometry_error = 0.0
    for record in records:
        identity, fold = record['id'], record['fold']
        checkpoint = Path(record['checkpoint'])
        reference = historical[identity]
        for name, expected in reference['sha256'].items():
            assert digest(checkpoint.parent / name) == expected, (identity, name)
        assert digest(checkpoint) == record['checkpoint_sha256']
        path = HERE / 'features' / (identity + '.npz')
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == {'tokens', 'pre_mha', 'final', 'logits', 'y', 'rx', 'rows'}
            y, rx, rows = (data[k] for k in ['y', 'rx', 'rows'])
            assert len(y) == 2000 and set(y) == set(range(10)) and set(rx) == set(range(5))
            assert all(np.sum((y == t) & (rx == r)) == 40 for t in range(10) for r in range(5))
            for r in range(5):
                assert len(np.unique(rows[rx == r])) == 400
            if fold not in baselines:
                baselines[fold] = {k: data[k].copy() for k in ['y', 'rx', 'rows']}
            for key, expected in baselines[fold].items():
                np.testing.assert_array_equal(data[key], expected)
            for layer in ['tokens', 'pre_mha', 'final']:
                assert data[layer].shape == (2000, 64)
                actual = geometry(data[layer], y, rx)
                saved = next(r for r in observations['geometry'] if r['arm'] == record['arm'] and
                             r['fold'] == fold and r['seed'] == record['seed'] and r['layer'] == layer)
                for key, value in actual.items():
                    error = abs(value - saved[key])
                    max_geometry_error = max(max_geometry_error, error)
                    np.testing.assert_allclose(value, saved[key], rtol=1e-12, atol=1e-12)
            assert data['logits'].shape == (2000, 10) and np.isfinite(data['logits']).all()
            predictions = data['logits'].argmax(1)
            current = int(np.sum(predictions == y))
            archived = read(checkpoint.parent / 'metrics.json')['source_validation']
            original = round(archived['summary']['acc_mean'] * 2000)
            cm_changed = 0
            for ri, name in enumerate(P['folds'][fold]['source']):
                selected = rx == ri
                cm = np.bincount(y[selected] * 10 + predictions[selected], minlength=100).reshape(10, 10)
                old = np.asarray(archived['per_rx'][str(ri)]['confusion'])
                cm_changed += int(np.sum(np.abs(cm - old)))
                selected_rows = csv_rows('source_tx_rx_margins.csv')
                for t in range(10):
                    row = next(r for r in selected_rows if r['arm'] == record['arm'] and int(r['fold']) == fold
                               and int(r['seed']) == record['seed'] and r['rx'] == name and int(r['tx_raw_id']) == t + 31)
                    np.testing.assert_allclose(float(row['accuracy_pct']), cm[t, t] / 40 * 100, atol=1e-12)
            accuracy_differences.append(dict(id=identity, archived_correct=original, analysis_correct=current,
                net_correct_difference=current-original, confusion_cell_L1_difference=cm_changed))
        checkpoint_checks.append(dict(id=identity, checkpoint_sha256=digest(checkpoint), feature_sha256=digest(path),
            model_config_metrics_history_unchanged=True))
    mismatches = {r['id']: r['net_correct_difference'] for r in accuracy_differences if r['net_correct_difference']}
    assert mismatches == {'P0_f0_s202': -1, 'S1_f0_s203': 1}, mismatches
    for fold in (0, 1):
        with np.load(HERE / 'features' / f'metadata_f{fold}.npz', allow_pickle=False) as data:
            assert set(data.files) == {'CFO', 'SNR', 'y', 'rx', 'rows'}
            for key, expected in baselines[fold].items():
                np.testing.assert_array_equal(data[key], expected)
            for field in ['CFO', 'SNR']:
                assert data[field].shape == (2000,) and np.isfinite(data[field]).all()

    events = [json.loads(line) for line in (HERE / 'access.jsonl').read_text(encoding='utf-8').splitlines()]
    counts = collections.Counter(r['role'] for r in events)
    assert counts == {'source_val': 20, 'source_metadata': 10}, counts
    for event in events:
        authorize_source(event['fold'], event['rx'])
        if event['role'] == 'source_metadata':
            assert event['fields'] == ['CFO', 'SNR', 'label'] and not event['waveform_field_read']
            assert event['packets'] == 400
        else:
            assert event['feature_packets'] == 400
    table_counts = {'mechanism_geometry.csv': 72, 'source_tx_rx_margins.csv': 1200,
        'source_rx_conditions.csv': 120, 'metadata_correlations.csv': 48, 'kernel_geometry_join.csv': 54,
        'matched_tx_rx_comparisons.csv': 900, 'external_tx_rx_comparisons.csv': 1200,
        'source_conditions_by_tx_rx.csv': 100, 'long_patch_candidates.csv': 504}
    for name, expected in table_counts.items():
        assert len(csv_rows(name)) == expected, (name, expected)
    long = read(HERE / 'LONG_PATCH_AUDIT.json')
    assert long['closed_data_free_candidate_count'] == 504 and long['new_fits'] == 0
    assert all(r['partial_dispatcher_macs'] > 0 for r in long['actual_architecture_checks'].values())
    assert long['technical_correction']['original_sha256'] == digest(HERE / long['technical_correction']['original_preserved'])
    assert read(HERE / 'GO_STOP.json')['new_fit_budget'] == 0
    assert read(HERE / 'INDEPENDENT_DATA_STATUS.json')['status'] == 'NOT_AVAILABLE_NOT_RUN'
    assert len(read(HERE / 'REVIEWER_QA.json')['questions']) == 10
    assert len(read(HERE / 'EVIDENCE_MATRIX.json')['rows']) == 8

    provenance = read(HERE / 'FIGURE_PROVENANCE.json')
    assert provenance['plot_code_sha256'] == digest(HERE / 'paper_assets.py')
    for record in provenance['evidence']:
        assert digest(ROOT / record['path']) == record['sha256']
    pca = read(HERE / 'PCA_PROVENANCE.json')['rows']
    assert {(r['arm'], r['fold'], r['seed']) for r in pca} == {
        (a, f, s) for a in ['P0', 'S1', 'H1', 'H0'] for f in (0, 1) for s in (201, 202, 203)}
    for record in pca:
        assert record['feature_sha256'] == digest(HERE / 'features' / f"{record['arm']}_f{record['fold']}_s{record['seed']}.npz")
    figure_files = sorted((HERE / 'paper/figures').glob('*.pdf'))
    assert len(figure_files) == 13
    for file in figure_files:
        assert len(PdfReader(file).pages) == 1
        with Image.open(file.with_suffix('.png')) as im:
            im.verify()
    pdf = HERE / 'output/pdf/hstem_tifs_draft.pdf'
    reader = PdfReader(pdf)
    assert len(reader.pages) == 9
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)
    assert '??' not in text and '\ufffd' not in text
    assert 'Hierarchical RF Tokenization' in reader.metadata.title
    log = (HERE / 'tmp/pdfs/main.log').read_text(encoding='utf-8', errors='replace')
    assert not re.search(r'Overfull|undefined|Missing character|Fatal error|LaTeX Warning', log)
    renders = []
    for page in range(1, len(reader.pages)+1):
        path = HERE / 'tmp/pdfs/render' / f'page-{page}.png'
        assert path.stat().st_mtime >= pdf.stat().st_mtime
        with Image.open(path) as im:
            assert min(im.size) >= 900
            im.verify()
        renders.append(dict(page=page, png_sha256=digest(path)))
    visual = read(HERE / 'VISUAL_QA.json')
    assert visual['status'] == 'PASS' and visual['manuscript_sha256'] == digest(pdf)
    assert visual['reviewed_manuscript_pages'] == list(range(1, 10))
    assert visual['reviewed_pca_figures'] == [f'supp_pca_f{f}_s{s}' for f in (0, 1) for s in (201, 202, 203)]
    artifacts = []
    for path in sorted(HERE.rglob('*')):
        rel = path.relative_to(HERE)
        if not path.is_file() or any(p in {'features', 'tmp', 'local', '__pycache__'} for p in rel.parts):
            continue
        if rel.as_posix() in {'FINAL_AUDIT.json', 'STATUS.json', 'PUBLICATION.json'}:
            continue
        artifacts.append(dict(path=rel.as_posix(), bytes=path.stat().st_size, sha256=digest(path)))
    # Only S25 paths may differ from the pinned S24 base. No staging/mutation here.
    changed = subprocess.check_output(['git', 'diff', registration['base_git'], '--name-only'], cwd=ROOT, text=True).splitlines()
    assert all(p.startswith('25-tifs-integration/') for p in changed), changed
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    assert all(p.startswith('25-tifs-integration/') for p in untracked), untracked
    write(HERE / 'FINAL_AUDIT.json', dict(status='PASS', time=now(),
        checks=dict(registration_and_extension_valid=True, frozen_model_config_metrics_history_count=24,
            geometry_recomputed_from_saved_features=True, geometry_max_absolute_error=max_geometry_error,
            same_balanced_source_packets_across_arms=True, access_events=dict(counts),
            all_access_events_source_authorized=True, historical_tracked_files_unchanged=True,
            table_row_counts=table_counts, main_figures=7, pca_cohorts=6, manuscript_pages=9,
            no_TeX_overflow_undefined_or_missing_glyph_warnings=True),
        new_fits=0, new_optimizer_steps=0, new_target_waveform_reads=0,
        source_accuracy_reexecution=accuracy_differences, models=checkpoint_checks,
        source_result_sha256=digest(HERE / 'SOURCE_GEOMETRY.json'),
        manuscript_sha256=digest(pdf), renders=renders, artifacts=artifacts,
        environment=dict(python=sys.version, platform=platform.platform(),
            packages={name:importlib.metadata.version(name) for name in
                      ['numpy', 'torch', 'scipy', 'matplotlib', 'h5py', 'pypdf', 'Pillow']}),
        limits=['Source validation previously used for selection; exploratory diagnostic.',
                'Two net source accuracy changes disclosed; not bit-exact archived recovery.',
                'No independent acquisition or new blind validation.',
                'Long-Patch is audit only; no automatic new fits.']))
    write(HERE / 'STATUS.json', dict(status='COMPLETE_READONLY_EVIDENCE_AND_WORKING_DRAFT', time=now(),
        frozen_models=24, total_source_feature_predictions=48000, new_fits=0,
        automatic_new_training='STOP', independent_acquisition='NOT_AVAILABLE_NOT_RUN',
        manuscript='output/pdf/hstem_tifs_draft.pdf', main_figures=7, pca_cohorts=6,
        integrity_audit='PASS', final_audit_sha256=digest(HERE / 'FINAL_AUDIT.json')))
    print('PASS: 24 frozen models, 48000 saved source predictions, all roles/figures/tables, 9-page draft; 0 fits.')


if __name__ == '__main__':
    main()
