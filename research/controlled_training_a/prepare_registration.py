"""Freeze Stage A metadata before fitting. Never decode an IQ/NPZ/NPY payload."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
HISTORY = Path('D:/crossatten/worktrees/s17-rx-distortion/17b-phase-intervention')
S19 = Path('D:/crossatten/worktrees/s19-tx-spectral-evidence/19-tx-spectral-evidence')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    dest = HERE / 'PRE_REGISTRATION.json'
    if dest.exists():
        raise RuntimeError('Registration already exists; do not overwrite a frozen protocol')
    if (HERE / 'artifacts').exists():
        raise RuntimeError('Artifacts already exist before registration')
    cache = HISTORY / 'cache/ManyRx_f2_first_s16001'
    manifest_path = cache / 'fit_manifest.json'
    manifest = read(manifest_path)
    fit, held = manifest['fit_rx'], manifest['held_rx']
    target = read(ROOT / 'research/training_instance_attribution/CONTROL_PANEL_PLAN.json')['forbidden_target_rx']
    assert len(fit) == 18 and len(held) == 6 and len(target) == 8
    assert not (set(fit) & set(held) or (set(fit) | set(held)) & set(target))
    inputs = {}
    hashed = []
    for name in ('fit_x', 'fit_y', 'fit_rx', 'fit_day', 'train_ids', 'val_ids'):
        path = cache / (name + '.npy')
        digest = sha(path)
        assert digest == manifest['sha256'][path.name]
        inputs[name] = {'path': str(path), 'sha256': digest}
        hashed.append(str(path))
    inputs['fit_manifest'] = {'path': str(manifest_path), 'sha256': sha(manifest_path)}
    cells = {}
    held_cells = {}
    logs = [HISTORY / 'access.jsonl', S19 / 'access.jsonl']
    for log in logs:
        for line in log.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('dataset') != 'ManyRx' or row.get('rx') not in fit + held or row.get('day') not in (1, 2):
                continue
            if row.get('mode') != 'read_only' and row.get('kind') != 'held_source_cell':
                continue
            key = row['rx'] + '|' + str(row['day'])
            digest = row['sha256']
            assert key not in cells or cells[key] == digest
            cells[key] = digest
            if row['rx'] in held and row.get('kind') == 'held_source_cell':
                held_cells[key] = {k: row[k] for k in ('path', 'sha256', 'rx', 'day')}
    assert set(cells) == {r + '|' + str(d) for r in fit + held for d in (1, 2)}
    assert len(held_cells) == 12
    held_inputs = [held_cells[r + '|' + str(d)] for r in held for d in (1, 2)]
    for row in held_inputs:
        assert sha(row['path']) == row['sha256']
        hashed.append(row['path'])
    paths = set(ROOT.joinpath('lib').rglob('*.py'))
    for directory in ('13-mixer-cfo-mechanisms', '17-rx-distortion'):
        paths.update(ROOT.joinpath(directory).glob('*.py'))
    paths.add(ROOT / '09-phase-invariance/phase_features.py')
    paths.update(HERE.glob('*.py'))
    paths.update(HERE.glob('test*.py'))
    paths.update(ROOT.joinpath('tests').glob('test_controlled_training_*.py'))
    paths.add(HERE / 'PREREGISTRATION.zh-CN.md')
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    base = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert base.startswith('ebd7b20'), 'Register against the audited base before the preregistration commit'
    runs = [
        {'id': 'A00', 'initialization_seed': 22101, 'order_seed': 22201, 'stochastic_seed': 22301},
        {'id': 'A00-replay', 'initialization_seed': 22101, 'order_seed': 22201, 'stochastic_seed': 22301},
        {'id': 'A10', 'initialization_seed': 22102, 'order_seed': 22201, 'stochastic_seed': 22301},
        {'id': 'A01', 'initialization_seed': 22101, 'order_seed': 22202, 'stochastic_seed': 22301},
        {'id': 'A11', 'initialization_seed': 22102, 'order_seed': 22202, 'stochastic_seed': 22301},
    ]
    record = dict(
        schema_version=1, stage='A', base_commit=base,
        registered_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        status='PREREGISTERED_BEFORE_NEW_FIT', maximum_new_fits=5,
        run_sequence=[r['id'] for r in runs], runs=runs,
        replay_gate='BITWISE_SEMANTIC_EQUALITY_REQUIRED_BEFORE_A10_A01_A11',
        inputs=inputs, held_inputs=held_inputs,
        rx_names=manifest['rx_names'], fit_rx=fit, held_rx=held, target_rx=target,
        source_cell_sha256=cells, code_sha256=code,
        output_dir='research/controlled_training_a/artifacts',
        expected_shapes={'fit_x': [34668, 2, 256], 'fit_y': [34668], 'train_ids': [24267], 'val_ids': [10401]},
        training=dict(optimizer='Adam', lr=0.001, weight_decay=0, batch_size=128, maximum_epochs=40,
                      precision='FP32', selection='RX_equal_validation_CE_strict_less',
                      scheduler=dict(name='ReduceLROnPlateau', factor=0.1, patience=5, min_lr=0.000001),
                      early_wait=10, gradient_clipping=False, augmentation=False, dropout=False,
                      deterministic_algorithms=True, warn_only=False, cudnn_benchmark=False,
                      cudnn_deterministic=True, matmul_allow_tf32=False, cudnn_allow_tf32=False,
                      cublas_workspace_config=':4096:8', cpu_threads=2),
        endpoints={'primary': 'selected', 'auxiliary': 'epoch10', 'engineering': 'final'},
        analysis=dict(bootstrap_replicates=10000, bootstrap_seed=22401, confidence_level=0.95,
                      meaningful_accuracy=0.01, meaningful_macro_f1=0.01, meaningful_ce=0.05,
                      meaningful_disagreement=0.05, uncertainty='paired_RX_cluster_conditional_descriptive'),
        held_access_gate='ALL_FIVE_RUNS_AND_ENDPOINTS_SEALED_BEFORE_FIRST_HELD_ARRAY_LOAD',
        stage_B='SEPARATE_PREREGISTRATION_REQUIRED_NOT_STARTED',
        metadata_access_before_registration=dict(
            array_payload_loads=0, model_forward_calls=0, checkpoint_loads=0, new_fits=0,
            raw_file_hash_reads=hashed,
            historical_text_logs={str(p): sha(p) for p in logs},
            note='Byte hashing to verify registered lineage is recorded; no NPZ/NPY payload was decoded.'),
    )
    draft = HERE / 'DRAFT_REGISTRATION_BEFORE_LINE_ENDING_FIX.json'
    if draft.exists():
        record['preflight_correction'] = dict(
            retained_draft=draft.name, retained_draft_sha256=sha(draft),
            cause='Historical tracked Python files use CRLF checkout while Git stores LF; raw Git blob comparison would reject unchanged files.',
            correction='Commit checks normalize CRLF/LF only; exact file SHA256 still pins runtime bytes. A synthetic regression verifies substantive changes fail.',
            scientific_array_payload_reads=0, new_fits=0, checkpoint_loads=0,
            metadata_byte_hash_reads='Registered fit/held files were byte-hashed once for the retained draft and once for this final registration.')
    dest.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'status': record['status'], 'registration_sha256': sha(dest), 'registered_code_files': len(code), 'held_payload_reads': 0}))


if __name__ == '__main__':
    main()
