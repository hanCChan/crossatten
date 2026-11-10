"""Prospective D0S support-aware source statistics; no missing-cell imputation.

S_R averages TX/date-specific RX population covariances over observed receivers.
S_T uses TX-pair differences on common RX support. This keeps every term PSD and
prevents additive receiver shifts from becoming spurious between-TX scatter.
The formulas equal the original complete-grid definitions when support is full.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from s18_stats import (_features, _labels, _orthonormal, _rank, compress,
                       direction_metrics, overlap, random_basis)


def scatters(z, y, rx, day):
    """Observed-cell balanced scatter with explicit support and common RXs.

    Return keys match s18_stats.scatters, with ``support_mask[T,R,D]`` and
    ``common_rx_counts[T,T,D]`` added. Missing cell means are NaN, never filled
    or used. Every TX/date group needs >=2 receivers; every distinct TX pair
    at each date needs >=2 common receivers, otherwise estimation fails closed.
    """
    z = _features(z)
    tx_labels, ti = _labels(y, len(z), "TX")
    rx_labels, ri = _labels(rx, len(z), "RX")
    day_labels, di = _labels(day, len(z), "day")
    nt, nr, nd = len(tx_labels), len(rx_labels), len(day_labels)
    if nt < 2 or nr < 2:
        raise ValueError("at least two TXs and two receivers are required")
    channels = z.shape[1]
    indices = (ti * nr + ri) * nd + di
    counts = np.bincount(indices, minlength=nt * nr * nd)
    sums = np.zeros((len(counts), channels), dtype=np.float64)
    np.add.at(sums, indices, z)
    means = np.full(sums.shape, np.nan)
    np.divide(sums, counts[:, None], out=means, where=counts[:, None] > 0)
    mu = means.reshape(nt, nr, nd, channels)
    counts = counts.reshape(nt, nr, nd)
    support = counts > 0
    sr, st = np.zeros((channels, channels)), np.zeros((channels, channels))
    common_counts = np.zeros((nt, nt, nd), dtype=np.int64)
    for d in range(nd):
        for t in range(nt):
            observed = support[t, :, d]
            n = int(observed.sum())
            if n < 2:
                raise ValueError("each TX/day group requires at least two observed receivers")
            cells = mu[t, observed, d]
            residual = cells-cells.mean(axis=0)
            sr += residual.T @ residual / (n * nt * nd)
            common_counts[t, t, d] = n
        for t in range(nt):
            for other in range(t+1, nt):
                common = support[t, :, d] & support[other, :, d]
                n = int(common.sum())
                if n < 2:
                    raise ValueError("each TX pair/day requires at least two common receivers")
                common_counts[t, other, d] = common_counts[other, t, d] = n
                delta = (mu[t, common, d]-mu[other, common, d]).mean(axis=0)
                # Sum of unordered pairs equals the ordered-pair sum / 2.
                st += np.outer(delta, delta)/(nt * nt * nd)
    return dict(
        S_R=(sr+sr.T)/2,
        S_T=(st+st.T)/2,
        center=mu[support].mean(axis=0),
        cell_means=mu,
        cell_counts=counts,
        tx_labels=tx_labels,
        rx_labels=rx_labels,
        day_labels=day_labels,
        support_mask=support,
        common_rx_counts=common_counts,
    )


def fit_directions(z, y, rx, day, rank=3, ridge_relative=1e-3):
    """Same registered generalized eigenspace construction on sparse scatters."""
    result = scatters(z, y, rx, day)
    sr, st = result["S_R"], result["S_T"]
    channels = sr.shape[0]
    rank = _rank(rank, channels)
    if ridge_relative <= 0 or not np.isfinite(ridge_relative):
        raise ValueError("ridge_relative must be positive and finite")
    epsilon = max(float(ridge_relative*np.trace(sr+st)/channels), 1e-12)
    values, vectors = eigh(sr, st+epsilon*np.eye(channels), check_finite=True)
    order = np.argsort(values)[::-1][:rank]
    u = _orthonormal(vectors[:, order])
    _, tx_vectors = np.linalg.eigh(st)
    u_tx = _orthonormal(tx_vectors[:, ::-1][:, :rank])
    result.update(
        U=u, U_tx=u_tx, epsilon=epsilon, rank=rank,
        generalized_eigenvalues=values[order],
        selective_metrics=direction_metrics(sr, st, u, epsilon),
        tx_metrics=direction_metrics(sr, st, u_tx, epsilon),
    )
    return result


def fit_rx_probe(z, y, rx, day):
    """Equal-observed-cell source centroid probe, with explicit fit support.

    Conditional TX/date offsets use the available source RX cells. Each RX
    centroid averages its observed TX/date residual cells equally. Evaluation
    must have the same support mask; this descriptive probe is never inference.
    """
    result = scatters(z, y, rx, day)
    mu, support = result["cell_means"], result["support_mask"]
    nt, nr, nd, channels = mu.shape
    offsets = np.empty((nt, nd, channels))
    for t in range(nt):
        for d in range(nd):
            offsets[t, d] = mu[t, support[t, :, d], d].mean(axis=0)
    residual = mu-offsets[:, None]
    centroids = np.empty((nr, channels))
    for r in range(nr):
        observed = support[:, r, :]
        if not observed.any():
            raise ValueError("every source receiver requires observed support")
        centroids[r] = residual[:, r, :][observed].mean(axis=0)
    return dict(
        offsets=offsets, centroids=centroids,
        tx_labels=result["tx_labels"], rx_labels=result["rx_labels"],
        day_labels=result["day_labels"], support_mask=support,
    )


def score_rx_probe(probe, z, y, rx, day):
    """Fixed centroid score: equal observed TX/date cells, then equal RXs."""
    z = _features(z)
    labels = {}
    for name, values in (("tx", y), ("rx", rx), ("day", day)):
        values = np.asarray(values)
        _labels(values, len(z), name)
        reference = np.asarray(probe[f"{name}_labels"])
        lookup = {label: index for index, label in enumerate(reference.tolist())}
        if any(label not in lookup for label in values.tolist()):
            raise ValueError(f"unseen {name} in fixed RX probe")
        labels[name] = np.array([lookup[label] for label in values.tolist()])
    nt, nr, nd = (len(probe[f"{name}_labels"]) for name in ("tx", "rx", "day"))
    cell_index = (labels["tx"]*nr+labels["rx"])*nd+labels["day"]
    counts = np.bincount(cell_index, minlength=nt*nr*nd)
    support = counts.reshape(nt, nr, nd) > 0
    if not np.array_equal(support, probe["support_mask"]):
        raise ValueError("evaluation support mask must equal fixed training support")
    residual = z-probe["offsets"][labels["tx"], labels["day"]]
    distance = ((residual[:, None]-probe["centroids"][None])**2).sum(axis=-1)
    correct = distance.argmin(axis=1) == labels["rx"]
    hits = np.bincount(cell_index, weights=correct, minlength=len(counts))
    cell_acc = np.full(len(counts), np.nan)
    np.divide(hits, counts, out=cell_acc, where=counts > 0)
    cell_acc = cell_acc.reshape(nt, nr, nd)
    per_rx = np.array([cell_acc[:, r, :][support[:, r, :]].mean() for r in range(nr)])
    return dict(accuracy=float(per_rx.mean()), per_rx=per_rx,
                cell_accuracy=cell_acc, support_mask=support, chance=1.0/nr)
