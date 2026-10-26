"""Reuse the S11 method, fixing eligibility without changing its estimator."""
import numpy as np
import torch
from core import ROOT, module

old = module("s12_frozen_spectral", ROOT / "11-spectral-generalization/spectral.py")
views, center_weight, apply_prior = old.views, old.center_weight, old.apply_prior
paired_model, normalize_statistics = old.paired_model, old.normalize_statistics


def estimate(values, y, rx, day, min_cell=10, dates=(1, 2)):
    if not isinstance(min_cell, int) or min_cell < 2:
        raise ValueError("sample uncertainty requires min_cell >= 2")
    if values.shape != (len(y), 5, 256) or not (len(y) == len(rx) == len(day)):
        raise ValueError("spectral samples and metadata dimensions differ")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite spectral input")
    return old.estimate(values, y, rx, day, min_cell=min_cell, dates=dates)


@torch.no_grad()
def batch_views(x):
    return torch.cat([views(torch.from_numpy(x[i:i + 256]))
                      for i in range(0, len(x), 256)]).numpy()


def fit_prior(feature, fit):
    if feature == "hard":
        return None
    if feature == "F1":
        return dict(weight=center_weight(), fit_only=True)
    if feature != "F2":
        raise ValueError(feature)
    stats = estimate(batch_views(fit["x"]), fit["y"], fit["rx"], fit["day"])
    return dict(weight=stats["weight"], c=stats["c"], eps=stats["eps"],
                included_cells=stats["included_cells"], fit_only=True)


def negative_weights(values, y, rx, day, weight, seed):
    rng = np.random.default_rng(seed)
    pseudo = y.copy()
    for name in np.unique(rx):
        for date in np.unique(day):
            ids = np.flatnonzero((rx == name) & (day == date))
            pseudo[ids] = rng.permutation(y[ids])
    return dict(uniform=np.full(256, .75, np.float32),
                shuffled=weight[rng.permutation(256)],
                tx_permuted=estimate(values, pseudo, rx, day)["weight"])
