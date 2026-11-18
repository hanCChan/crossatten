"""Append-only output repair; original frozen analysis and controller stay unchanged."""
import argparse
import json
import sys
from pathlib import Path

from common import HERE, PHASE, ROOT, read, digest, committed, policy


def validate():
    policy()
    path = HERE / 'TECHNICAL_FIX_001.json'
    committed(path)
    fix = read(path)
    if fix['status'] != 'FROZEN_ENGINEERING_REPAIR':
        raise RuntimeError('Unfrozen technical repair')
    for relative, expected in fix['repair_code_sha256'].items():
        path = ROOT / relative
        committed(path)
        if digest(path) != expected:
            raise RuntimeError('Repair code changed: ' + relative)
    return fix


def native_json(value):
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: native_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native_json(item) for item in value]
    return value


def summary():
    validate()
    sys.stdout.reconfigure(encoding='utf-8')
    import summarize
    original_write = summarize.atomic_json
    def write(path, obj):
        original_write(path, native_json(obj))
    summarize.atomic_json = write
    summarize.main()


def resume():
    validate()
    import controller
    original_run = controller.subprocess.run
    summary_path = HERE / 'summarize.py'
    def run(command, *args, **kwargs):
        if isinstance(command, (tuple, list)) and len(command) == 3 and Path(command[2]) == summary_path:
            command = [command[0], command[1], str(Path(__file__).resolve()), '--summary']
        return original_run(command, *args, **kwargs)
    controller.subprocess.run = run
    controller.main()


def self_test():
    import numpy as np
    value = {'counts': [np.int32(6), np.int64(10)], 'number': np.float64(0.125), 'flag': np.bool_(True)}
    fixed = native_json(value)
    assert json.loads(json.dumps(fixed)) == {'counts': [6, 10], 'number': 0.125, 'flag': True}
    for item, expected in zip(fixed['counts'], (6, 10)):
        assert type(item) is int and item == expected
    print('PASS: NumPy scalar serialization preserves values; no model, metric, or waveform IO.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--summary', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.summary:
        summary()
    elif args.resume:
        resume()
    else:
        self_test()
