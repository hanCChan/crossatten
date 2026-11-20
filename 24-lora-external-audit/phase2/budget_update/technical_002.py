"""Use NUL-delimited Git path manifests without changing frozen scientific code."""
import argparse
import json
import subprocess
import tempfile
import uuid
from pathlib import Path

import common
from common import HERE, PHASE, ROOT, read, digest, committed
import technical_001


def validate():
    technical_001.validate()
    path = HERE / 'TECHNICAL_FIX_002.json'
    committed(path)
    fix = read(path)
    if fix['status'] != 'FROZEN_ENGINEERING_REPAIR':
        raise RuntimeError('Unfrozen command-length repair')
    if fix['prior_repair_sha256'] != digest(HERE / 'TECHNICAL_FIX_001.json'):
        raise RuntimeError('Prior repair changed')
    for relative, expected in fix['repair_code_sha256'].items():
        target = ROOT / relative
        committed(target)
        if digest(target) != expected:
            raise RuntimeError('Command-length repair changed: ' + relative)
    return fix


def commit_paths(root, rels, message, scratch):
    root = Path(root).resolve()
    scratch = Path(scratch).resolve()
    scratch.relative_to(root)
    scratch.mkdir(parents=True, exist_ok=True)
    rels = sorted(set(rels))
    if not rels:
        return subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    for relative in rels:
        target = (root / relative).resolve()
        target.relative_to(root)
        if '\x00' in relative:
            raise ValueError('NUL in filename')
    base = ['git', '-C', str(root), '--literal-pathspecs']
    with tempfile.NamedTemporaryFile(dir=scratch, prefix='paths_', suffix='.nul', delete=False) as manifest:
        manifest_path = Path(manifest.name)
        manifest.write(b'\x00'.join(name.encode('utf-8') for name in rels) + b'\x00')
    try:
        path_args = ['--pathspec-from-file=' + str(manifest_path), '--pathspec-file-nul']
        subprocess.run(base + ['add', *path_args], check=True)
        changed = subprocess.check_output(base + ['diff', '--cached', '--name-only', '-z'])
        changed = {name.decode('utf-8') for name in changed.split(b'\x00') if name}
        if changed.intersection(rels):
            subprocess.run(base + ['-c', 'user.name=Codex', '-c', 'user.email=codex@local.invalid',
                                  'commit', '-m', message, '--only', *path_args], check=True)
        return subprocess.check_output(base + ['rev-parse', 'HEAD'], text=True).strip()
    finally:
        manifest_path.unlink()


def safe_commit(paths, message):
    rels = [Path(path).resolve().relative_to(ROOT).as_posix() for path in paths]
    if not all(name.startswith('24-lora-external-audit/') for name in rels):
        raise PermissionError('Commit outside the authorized S24 directory')
    return commit_paths(ROOT, rels, message, PHASE / 'local')


def resume():
    validate()
    import execution
    import controller
    common.commit = safe_commit
    execution.commit = safe_commit
    controller.commit = safe_commit
    technical_001.resume()


def self_test():
    fixture = PHASE / 'local' / ('git_paths_002_' + uuid.uuid4().hex[:8])
    fixture.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(fixture)], check=True)
    base = ['git', '-C', str(fixture)]
    (fixture / 'unrelated.txt').write_text('initial\n', encoding='utf-8')
    subprocess.run(base + ['add', 'unrelated.txt'], check=True)
    subprocess.run(base + ['-c', 'user.name=Codex', '-c', 'user.email=codex@local.invalid',
                           'commit', '-q', '-m', 'fixture'], check=True)
    (fixture / 'unrelated.txt').write_text('preserve staged edit\n', encoding='utf-8')
    subprocess.run(base + ['add', 'unrelated.txt'], check=True)
    rels = [f'file_{index:04d}_' + 'x' * 56 + '.txt' for index in range(520)]
    rels += ['with spaces [literal].txt', 'unicode_\u6d4b\u8bd5.txt']
    for name in rels:
        (fixture / name).write_text('fixture\n', encoding='utf-8')
    simulated = subprocess.list2cmdline(base + ['add', '--', *rels])
    assert len(simulated) > 32767
    head = commit_paths(fixture, rels, 'long manifest fixture', fixture / 'scratch')
    actual = subprocess.check_output(base + ['diff-tree', '--no-commit-id', '--name-only', '-r', '-z', head])
    actual = {name.decode('utf-8') for name in actual.split(b'\x00') if name}
    assert actual == set(rels)
    pending = subprocess.check_output(base + ['diff', '--cached', '--name-only', '-z'])
    assert pending == b'unrelated.txt\x00'
    second = commit_paths(fixture, rels, 'no empty commit', fixture / 'scratch')
    assert head == second
    assert not list((fixture / 'scratch').glob('*.nul'))
    report = {'status': 'PASS', 'fixture': str(fixture), 'simulated_windows_command_chars': len(simulated),
              'selected_files': len(rels), 'unrelated_staged_edit_preserved': True, 'literal_unicode_paths': True,
              'empty_commit_avoided': True, 'new_waveform_reads': 0, 'new_training_steps': 0}
    (HERE / 'TECHNICAL_TEST_002.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    resume() if args.resume else self_test()
