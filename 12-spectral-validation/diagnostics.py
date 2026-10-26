"""Descriptive source mechanisms; no proxy is an identity-information veto."""
import numpy as np
import torch
from spectral import apply_prior, normalize_statistics


def scatter_diagnostics(prior, weight):
    w2 = np.asarray(weight, dtype=np.float64) ** 2
    tx = float((prior["b"] * w2).sum())
    nuisance = float(((prior["r"] + prior["d"]) * w2).sum())
    base_tx = float(prior["b"].sum())
    return dict(tx_scatter_retention=tx / base_tx if base_tx > 0 else None,
                J=tx / nuisance if nuisance > 0 else None,
                zero_nuisance=nuisance == 0,
                scope="per-bin channel-averaged scatter proxy; no fixed epsilon",
                admission_veto=False)


def geometry(fit, held, y, yh, weight, kind):
    a, b = [apply_prior(torch.from_numpy(v), torch.as_tensor(weight), kind).numpy()
            for v in (fit, held)]
    a, scale = normalize_statistics(a)
    b = b / scale[None, :, None]
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    labels = np.unique(y)
    means = np.stack([a[y == tx].mean(0) for tx in labels])
    distances = (b ** 2).sum(1, keepdims=True) + (means ** 2).sum(1) - 2 * b @ means.T
    norm = np.linalg.norm(means, axis=1, keepdims=True)
    angles = None
    if len(labels) > 1 and (norm > 0).all():
        centers = means / norm
        cosine = np.clip(centers @ centers.T, -1, 1)
        angles = float(np.arccos(cosine[np.triu_indices(len(labels), 1)]).min())
    within = float(np.mean([((a[y == tx] - means[i]) ** 2).sum(1).mean()
                            for i, tx in enumerate(labels)]))
    between = float(((means - means.mean(0)) ** 2).sum(1).mean())
    return dict(centroid_acc_pct=float((labels[distances.argmin(1)] == yh).mean() * 100),
                minimum_center_angle_rad=angles, unit_within=within, unit_between=between)
