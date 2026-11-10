"""Additive support correction, with no prior learned-feature outcomes."""
from s18_core import HERE, now, read, sha, write, validate, git


def register():
    validate()
    if (HERE/'registration_support.json').exists(): raise RuntimeError('Already registered')
    if list(HERE.glob('D0_fold*.json')) or list((HERE/'cache').glob('features*')) or list((HERE/'d0_support_matched').glob('D0_fold*.json')):
        raise RuntimeError('Learned feature outcomes already exist')
    support = read(HERE/'SUPPORT_AUDIT.json')
    for fold in range(4):
        tr = next(r for r in support['rows'] if r['fold'] == fold and r['part'] == 'train')
        va = next(r for r in support['rows'] if r['fold'] == fold and r['part'] == 'val')
        import numpy as np
        if not np.array_equal(np.array(tr['counts']) > 0, np.array(va['counts']) > 0):
            raise ValueError('Train/validation support mismatch')
    tests = read(HERE/'SUPPORT_ENGINEERING_TESTS.json')
    if tests['status'] != 'PASS': raise ValueError('Support synthetic checks failed')
    code = ['s18_sparse_stats.py', 'test_sparse_stats.py', 's18_support_run.py', 's18_support_audit.py', 's18_register_support.py']
    docs = ['SUPPORT_AMENDMENT.zh-CN.md', 'SUPPORT_AUDIT.json', 'SUPPORT_ENGINEERING_TESTS.json', 'registration.json']
    write(HERE/'registration_support.json', dict(time=now(), stage='S18-D0S', parent_registration_sha256=sha(HERE/'registration.json'),
          parent_commit=git('rev-parse', 'HEAD'), reason='Missing source cells; correction before any learned feature outcome',
          code_sha256={name:sha(HERE/name) for name in code}, design_sha256={name:sha(HERE/name) for name in docs},
          rank_alpha_controls_gates_changed=False, original_attempt_retained=True, feature_outcomes_before_amendment=0,
          real_target_access=False, held_source_read=False, new_fits=0))


if __name__ == '__main__': register()
