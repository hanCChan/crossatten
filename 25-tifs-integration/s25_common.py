"""Frozen-model, source-validation-only analyses; historical stages stay read-only."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S24 = Path('D:/crossatten/worktrees/s24-lora-external-audit/24-lora-external-audit')
CACHE = Path('D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/shen')
P = json.loads((ROOT / 'configs/protocol.json').read_text(encoding='utf-8'))
for path in [ROOT/'lib', ROOT/'13-mixer-cfo-mechanisms', ROOT/'15-hstem-attribution', ROOT/'21-hstem-mechanisms']:
    sys.path.insert(0, str(path))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    os.replace(temp, path)


def committed(path):
    path = Path(path)
    relative = path.resolve().relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', '-C', str(ROOT), 'show', 'HEAD:' + relative]) != path.read_bytes():
        raise PermissionError('Uncommitted analysis registration/code: ' + relative)


def validate():
    path = HERE / 'registration.json'
    committed(path)
    registration = read(path)
    assert registration['new_fit_budget'] == 0
    for relative, expected in registration['code_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise PermissionError('Frozen analysis code changed: ' + relative)
    for relative, expected in registration['evidence_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise PermissionError('Historical evidence changed: ' + relative)
    return registration


def authorize_source(fold, rx, role='source_val'):
    # Must run before np.load or any other waveform IO.
    if role != 'source_val' or fold not in (0, 1):
        raise PermissionError('Only registered source validation is authorized')
    if rx in P['blind_receivers_never_open'] or rx in P['confirm_receivers'] or rx not in P['folds'][fold]['source']:
        raise PermissionError('RX is not source for this fold: ' + str(rx))
    return CACHE / rx


def source_validation(fold):
    registration = validate()
    import numpy as np
    from xa.data import source_split
    cells = []
    for domain, rx in enumerate(P['folds'][fold]['source']):
        path = authorize_source(fold, rx)
        expected = registration['source_caches'][rx]
        x = np.load(path/'x.npy', mmap_mode='r')
        y = np.load(path/'y.npy')
        rows = np.load(path/'rows.npy')
        for array, key in [(x, 'selected_iq_float32_sha256'), (y, 'selected_labels_sha256'), (rows, 'selected_rows_sha256')]:
            if hashlib.sha256(array.tobytes()).hexdigest() != expected[key]:
                raise PermissionError('Source cache identity changed: ' + rx)
        _, ids = source_split(y, rows)
        assert len(ids) == 400
        np.testing.assert_array_equal(np.bincount(y[ids], minlength=10), np.full(10, 40))
        selected = np.ascontiguousarray(x[ids])
        labels, original_rows = y[ids], rows[ids]
        cells.append((selected, labels, np.full(400, domain, dtype=np.int64), original_rows))
        event = dict(time=now(), role='source_val', fold=fold, rx=rx, source_cache_packets_verified=2000,
                     feature_packets=400, selected_iq_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                     selected_rows_sha256=hashlib.sha256(original_rows.tobytes()).hexdigest(),
                     registration_sha256=digest(HERE/'registration.json'))
        with (HERE/'access.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(event) + '\n')
    return {name: np.concatenate([cell[index] for cell in cells]) for index, name in enumerate(['x', 'y', 'rx', 'rows'])}


def load_model(record, device='cuda'):
    import torch
    from s21_models import build
    path = Path(record['checkpoint'])
    if digest(path) != record['checkpoint_sha256']:
        raise PermissionError('Frozen checkpoint changed')
    model = build(record['arm'], record['data_seed']).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    assert sum(p.numel() for p in model.parameters()) == record['params']
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
