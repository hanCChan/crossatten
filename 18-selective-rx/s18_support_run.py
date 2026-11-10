"""Prospective D0S: correct missing-cell support BEFORE any feature outcome.

The original registered D0 remains stopped. This module changes only the
scatter/probe support estimator and output namespace; rank, alpha, controls,
classification evaluation, checkpoint, gates and source split remain fixed.
"""
import argparse
import hashlib
import numpy as np
import torch
import s18_core as core
import s18_run as engine
import s18_sparse_stats as sparse
from s18_decide import decide

HERE = core.HERE/'d0_support_matched'


def validate():
    core.validate()
    path = core.HERE/'registration_support.json'
    if core.git('show', 'HEAD:18-selective-rx/registration_support.json') != path.read_text(encoding='utf-8').strip():
        raise ValueError('Support amendment must be committed')
    reg = core.read(path)
    for name, digest in reg['code_sha256'].items():
        if core.sha(core.HERE/name) != digest: raise ValueError('Support code changed: '+name)
    for name, digest in reg['design_sha256'].items():
        if core.sha(core.HERE/name) != digest: raise ValueError('Support design changed: '+name)
    return reg


def estimation_ids(pool, ids, fold, cap=48):
    rng = np.random.default_rng(18001+fold)
    selected = []
    for t in range(10):
        for r in sorted(np.unique(pool['rx'])):
            for d in (1, 2):
                cell = ids[(pool['y'][ids] == t) & (pool['rx'][ids] == r) & (pool['day'][ids] == d)]
                if len(cell) == 0: continue
                if len(cell) < 4: raise ValueError('Insufficient observed estimation cell')
                selected.extend(rng.permutation(cell)[:cap])
    return np.array(selected, dtype=np.int64)


def configure():
    engine.HERE = HERE
    engine.validate = validate
    engine.estimation_ids = estimation_ids
    engine.fit_directions = sparse.fit_directions
    engine.scatters = sparse.scatters
    engine.fit_rx_probe = sparse.fit_rx_probe
    engine.score_rx_probe = sparse.score_rx_probe


def run():
    p = argparse.ArgumentParser(); p.add_argument('--decide', action='store_true'); p.add_argument('--fold', type=int, choices=range(4)); args = p.parse_args()
    validate(); configure()
    if args.decide:
        path = HERE/'decision_d0.json'
        if path.exists(): raise RuntimeError('D0S decision already sealed')
        decision = decide([core.read(HERE/f'D0_fold{f}.json') for f in range(4)])
        decision.update(stage='S18-D0S', support_amendment=True, original_d0_status='STOP_PRE_FEATURE_SUPPORT',
                        sparse_estimator='TX-conditional RX scatter and paired-TX common-RX PSD scatter; no imputation')
        core.write(path, decision)
    else:
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if not torch.cuda.is_available(): raise RuntimeError('Registered CUDA environment required')
        for fold in [args.fold] if args.fold is not None else range(4): engine.run_fold(fold)


if __name__ == '__main__': run()
