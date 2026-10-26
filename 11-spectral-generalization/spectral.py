"""TX-discriminative RX/day-stable fixed priors; all statistics are fit-only."""
import numpy as np
import torch
from core import s09
from xa.explore_models import frequency_views
from xa.frontends import rms_normalize


def views(x):
    return s09.phase_features_from_base(frequency_views(rms_normalize(x)), "first")


def center_weight(k=32):
    if k != 32:
        raise ValueError("fixed diagnostic bandwidth is k=32, not searched")
    w = np.zeros(256, dtype=np.float32)
    w[128 - k + 1:128 + k] = 1.
    return w


def apply_prior(value, weight, kind):
    if kind == "hard":
        return value
    if value.ndim != 3 or value.shape[1:] != (5, 256) or weight.shape != (256,):
        raise ValueError("prior dimensions differ")
    out = value * weight[None, None]
    if kind == "F1":
        pair = weight * torch.nn.functional.pad(weight[:-1], (1, 0))
        out[:, 3:] = value[:, 3:] * pair[None, None]
    elif kind != "F2":
        raise ValueError(kind)
    return out


def normalize_statistics(values):
    scales = np.ones(5, dtype=np.float64)
    # Couple Cartesian Re/Im components: do not rotate or unwrap phases.
    for ids in ([0, 1], [2], [3, 4]):
        scale = np.sqrt(np.mean(np.var(values[:, ids], axis=0)))
        scales[ids] = max(scale, 1e-6)
    return values / scales[None, :, None], scales


def smooth(v):
    return np.convolve(np.pad(v, (2, 2), mode="reflect"), np.ones(5) / 5, mode="valid")


def estimate(values, y, rx, day, min_cell=10, dates=(1, 2)):
    """Equal TX/RX/day centroids, missing cells excluded, sampling noise in denominator."""
    v, scale = normalize_statistics(values.astype(np.float64))
    means, uncertainty, counts = {}, {}, {}
    for tx in np.unique(y):
        for name in np.unique(rx):
            for date in dates:
                mask = (y == tx) & (rx == name) & (day == date)
                n = int(mask.sum())
                if n >= min_cell:
                    key = (int(tx), str(name), date)
                    means[key] = v[mask].mean(0)
                    uncertainty[key] = v[mask].var(0, ddof=1) / n
                    counts[key] = n
    txmeans, rxvar, dayvar, noise = [], [], [], []
    included = {}
    for tx in np.unique(y):
        pairs = [name for name in np.unique(rx)
                 if all((int(tx), str(name), date) in means for date in dates)]
        if len(pairs) < 3:
            continue
        cells = np.stack([[means[int(tx), str(name), date] for date in dates]
                          for name in pairs])
        unc = np.stack([[uncertainty[int(tx), str(name), date] for date in dates]
                        for name in pairs])
        txmeans.append(cells.mean((0, 1)))
        rxvar.append(cells.mean(1).var(0))
        dayvar.append(cells.var(1).mean(0))
        noise.append(unc.mean((0, 1)))
        included[str(tx)] = len(pairs)
    if len(txmeans) < 3:
        raise ValueError("too few complete TX/RX/day cells for a spectral prior")
    # Aggregate channels only AFTER estimating feature-specific between/within scatter.
    b_raw = np.stack(txmeans).var(0).mean(0)
    u = np.stack(noise).mean(0).mean(0)
    b = np.maximum(b_raw - u, 0)
    r = np.stack(rxvar).mean(0).mean(0)
    d = np.stack(dayvar).mean(0).mean(0)
    eps = max(float(np.median(r + d + u)) * 1e-3, 1e-8)
    score = smooth(b / (r + d + u + eps))
    positive = score[score > 0]
    c = float(np.median(positive)) if len(positive) else 1.
    g = .5 + .5 * score / (score + c)
    if not np.isfinite(g).all():
        raise ValueError("nonfinite prior")
    return dict(weight=g.astype(np.float32), b=b, b_raw=b_raw, r=r, d=d,
                uncertainty=u, score=score, c=c, eps=eps, channel_scale=scale,
                eligible_rx_per_tx=included, included_cells=len(counts))


def split_stability(values, y, rx, day):
    # Deterministic disjoint per-cell halves, no held-RX labels or packets.
    mask = np.zeros(len(y), bool)
    for tx in np.unique(y):
        for name in np.unique(rx):
            for date in np.unique(day):
                ids = np.flatnonzero((y == tx) & (rx == name) & (day == date))
                mask[ids[::2]] = True
    a, b = [estimate(values[m], y[m], rx[m], day[m], min_cell=5) for m in (mask, ~mask)]
    return dict(correlation=float(np.corrcoef(a["weight"], b["weight"])[0, 1]),
                max_abs_weight_difference=float(np.max(np.abs(a["weight"] - b["weight"]))))


class SpectralRF(s09.PhaseRF):
    def __init__(self, classes, kind, weight):
        super().__init__(classes, "first")
        self.selection = kind
        self.register_buffer("spectral_prior", torch.as_tensor(weight, dtype=torch.float32).clone())

    def tokens(self, x):
        return self.frequency(apply_prior(views(x), self.spectral_prior, self.selection))


def paired_model(dataset, kind, seed, weight=None):
    canonical, shared_sha = s09.paired_model(dataset, "first", seed)
    if kind in ("hard", "first"):
        return canonical, shared_sha
    if weight is None:
        raise ValueError("F1/F2 model requires a fit-only registered prior")
    with torch.random.fork_rng(devices=[]):
        model = SpectralRF(10 if dataset == "ManyRx" else 6, kind, weight)
    model.load_state_dict(dict(canonical.state_dict(), spectral_prior=model.spectral_prior))
    return model, shared_sha
