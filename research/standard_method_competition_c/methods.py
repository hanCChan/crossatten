"""Preregistered standard kernels; no dataset reads or scientific training entry."""
from __future__ import annotations
import copy
import random
import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn, update_bn


def make_ema(student):
    return AveragedModel(student, multi_avg_fn=get_ema_multi_avg_fn(.999), use_buffers=False)


@torch.no_grad()
def calibrated_ema_candidate(shadow, train_batches, *, role):
    """Calibrate a deployment copy, preserving all student/shadow/RNG state."""
    if role != 'fit_train':
        raise ValueError('BN calibration accepts fit_train only')
    candidate = copy.deepcopy(shadow.module).eval().requires_grad_(False)
    device = next(candidate.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    samples, batches = 0, 0
    def inputs():
        nonlocal samples, batches
        for x in train_batches:
            if not torch.is_tensor(x) or len(x) == 0:
                raise ValueError('Calibration batches must be nonempty input tensors')
            samples += len(x); batches += 1
            yield x
    try:
        with torch.random.fork_rng(devices=devices):
            update_bn(inputs(), candidate, device=device)
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
    bn = [m for m in candidate.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if bn and (batches == 0 or any(int(m.num_batches_tracked) == 0 for m in bn)):
        raise ValueError('Every registered BN module must receive training-source calibration inputs')
    candidate.eval()
    return candidate, dict(role=role, samples=samples, batches=batches, official_update_bn=True,
                           parameters_updated=False, student_or_shadow_modified=False)


class RxGroupDRO:
    """Reference-family multiplicative q updates represented in log space."""
    def __init__(self, n_groups, *, device='cpu'):
        if not isinstance(n_groups, int) or n_groups < 1:
            raise ValueError('Need a nonempty registered RX group set')
        self.n_groups = n_groups
        self.step_size = .01
        self.log_q = torch.full((n_groups,), -np.log(n_groups), dtype=torch.float64, device=device)

    def loss(self, per_example_ce, groups):
        if per_example_ce.ndim != 1 or per_example_ce.numel() == 0 or groups.shape != per_example_ce.shape:
            raise ValueError('Expected aligned nonempty CE and group vectors')
        if groups.dtype != torch.int64 or groups.device != per_example_ce.device or self.log_q.device != groups.device:
            raise ValueError('Group dtype/device mismatch')
        if not torch.isfinite(per_example_ce).all() or (per_example_ce < 0).any():
            raise ValueError('Nonfinite or negative CE')
        if (groups < 0).any() or (groups >= self.n_groups).any():
            raise ValueError('Unregistered group ID')
        membership = (groups[None, :] == torch.arange(self.n_groups, device=groups.device)[:, None]).to(per_example_ce.dtype)
        counts = membership.sum(1)
        means = (membership @ per_example_ce) / counts.clamp_min(1)
        if not torch.isfinite(means).all():
            raise ValueError('Nonfinite group CE reduction')
        with torch.no_grad():
            proposed = self.log_q + self.step_size*means.detach().to(torch.float64)
            normalized = proposed-torch.logsumexp(proposed, 0)
            q = normalized.exp()
        if not torch.isfinite(q).all():
            raise ValueError('Nonfinite GroupDRO weights')
        objective = (means*q.to(means.dtype)).sum()
        if not torch.isfinite(objective):
            raise ValueError('Nonfinite GroupDRO objective')
        self.log_q = normalized.detach().clone()
        return objective, dict(q=q.detach().clone(), group_loss=means.detach().clone(),
                               group_count=counts.detach().clone(), present=(counts > 0).detach().clone())

    def state_dict(self):
        return dict(n_groups=self.n_groups, step_size=self.step_size, log_q=self.log_q.detach().clone())

    def load_state_dict(self, state):
        if state['n_groups'] != self.n_groups or state['step_size'] != .01:
            raise ValueError('GroupDRO state schema differs')
        value = state['log_q'].to(device=self.log_q.device, dtype=torch.float64)
        if value.shape != self.log_q.shape or not torch.isfinite(value).all() or abs(float(value.exp().sum())-1) > 1e-12:
            raise ValueError('Invalid probability state')
        self.log_q = value.detach().clone()
