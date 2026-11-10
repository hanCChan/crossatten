"""Source-only, cell-balanced TX/RX feature scatter and frozen directions.

All fitted quantities come from the caller's estimation samples.  In particular,
``compress`` never reads labels or recomputes a center on evaluation samples.
Scatter is descriptive: receiver effects also include channel/synchronization
differences and are not identified receiver hardware factors.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _features(z):
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not z.shape[0] or not z.shape[1]:
        raise ValueError("features must be a nonempty [samples, channels] array")
    if not np.isfinite(z).all():
        raise ValueError("nonfinite features")
    return z


def _labels(values, n, name):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f"{name} must have one label per sample")
    if values.dtype.kind in "fc" and not np.isfinite(values).all():
        raise ValueError(f"nonfinite {name} labels")
    return np.unique(values, return_inverse=True)


def _orthonormal(columns):
    """A deterministic Euclidean basis for the supplied column space."""
    q, r = np.linalg.qr(columns, mode="reduced")
    if np.any(np.abs(np.diag(r)) <= np.finfo(float).eps * np.linalg.norm(columns)):
        raise ValueError("rank-deficient directions")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs


def _rank(rank, channels):
    if isinstance(rank, bool) or int(rank) != rank or not 1 <= rank <= channels:
        raise ValueError("rank must be an integer between 1 and channels")
    return int(rank)


def scatters(z, y, rx, day):
    """Estimate equally weighted cell means and population scatter matrices.

    Returns a dict with ``S_R[C,C]``, ``S_T[C,C]``, ``center[C]``,
    ``cell_means[T,R,D,C]``, ``cell_counts[T,R,D]``, and the sorted label
    vectors ``tx_labels``, ``rx_labels``, ``day_labels``. All cells must exist;
    each observed TX/RX/day cell has equal weight regardless of packet count.

    At least two TXs and two receivers are required to identify both scatter
    proxies. A single date is accepted for prespecified cross-date checks.
    """
    z = _features(z)
    tx_labels, ti = _labels(y, len(z), "TX")
    rx_labels, ri = _labels(rx, len(z), "RX")
    day_labels, di = _labels(day, len(z), "day")
    nt, nr, nd = len(tx_labels), len(rx_labels), len(day_labels)
    if nt < 2 or nr < 2:
        raise ValueError("at least two TXs and two receivers are required")
    indices = (ti * nr + ri) * nd + di
    counts = np.bincount(indices, minlength=nt * nr * nd)
    if np.any(counts == 0):
        raise ValueError("incomplete TX/RX/day rectangle; missing cells are forbidden")
    sums = np.zeros((len(counts), z.shape[1]), dtype=np.float64)
    np.add.at(sums, indices, z)
    mu = (sums / counts[:, None]).reshape(nt, nr, nd, z.shape[1])
    # RX residuals at fixed TX/date; population covariance, not sample ddof=1.
    rx_residual = mu - mu.mean(axis=1, keepdims=True)
    sr = np.einsum("trdc,trde->ce", rx_residual, rx_residual) / (nt * nr * nd)
    # Between TX means after integrating equally over source receivers.
    tx_means = mu.mean(axis=1)
    tx_residual = tx_means - tx_means.mean(axis=0, keepdims=True)
    st = np.einsum("tdc,tde->ce", tx_residual, tx_residual) / (nt * nd)
    return dict(
        S_R=(sr + sr.T) / 2,
        S_T=(st + st.T) / 2,
        center=mu.mean(axis=(0, 1, 2)),
        cell_means=mu,
        cell_counts=counts.reshape(nt, nr, nd),
        tx_labels=tx_labels,
        rx_labels=rx_labels,
        day_labels=day_labels,
    )


def direction_metrics(SR, ST, U, epsilon=1e-12):
    """Projected scatter, energy fractions, and RX/TX Rayleigh ratios.

    These are source-domain statistical proxies, not mutual information.
    ``rx_fraction`` and ``tx_fraction`` divide projected trace by full trace.
    A zero full trace gives fraction zero. ``rho`` uses the explicit epsilon.
    """
    sr, st, u = (np.asarray(v, dtype=np.float64) for v in (SR, ST, U))
    if sr.ndim != 2 or sr.shape[0] != sr.shape[1] or st.shape != sr.shape:
        raise ValueError("scatter matrices must be square and have equal shape")
    if u.ndim != 2 or u.shape[0] != len(sr) or not u.shape[1]:
        raise ValueError("directions must have shape [channels, rank]")
    if not all(np.isfinite(v).all() for v in (sr, st, u)):
        raise ValueError("nonfinite scatter or directions")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be positive and finite")
    r = np.maximum(np.einsum("ci,cd,di->i", u, sr, u), 0)
    t = np.maximum(np.einsum("ci,cd,di->i", u, st, u), 0)
    er, et = float(r.sum()), float(t.sum())
    tr, tt = max(float(np.trace(sr)), 0), max(float(np.trace(st)), 0)
    return dict(
        rho=r / (t + epsilon),
        rx_energy=er,
        tx_energy=et,
        rx_fraction=er / tr if tr else 0.0,
        tx_fraction=et / tt if tt else 0.0,
        aggregate_ratio=er / (et + epsilon * u.shape[1]),
    )


def fit_directions(z, y, rx, day, rank=3, ridge_relative=1e-3):
    """Fit selective and TX-control directions using estimation samples only.

    ``U`` is the Euclidean-orthonormal QR basis of the leading generalized
    eigenvectors of S_R v = lambda (S_T + epsilon I) v. ``U_tx`` contains
    the leading ordinary eigenvectors of S_T. The QR basis spans the selected
    generalized eigenspace; its individual vectors need not retain the original
    generalized eigenvector Rayleigh values. Returns ``scatters`` fields plus
    U, U_tx, epsilon, rank, generalized_eigenvalues, selective_metrics, tx_metrics.
    """
    result = scatters(z, y, rx, day)
    sr, st = result["S_R"], result["S_T"]
    channels = sr.shape[0]
    rank = _rank(rank, channels)
    if ridge_relative <= 0 or not np.isfinite(ridge_relative):
        raise ValueError("ridge_relative must be positive and finite")
    epsilon = max(float(ridge_relative * np.trace(sr + st) / channels), 1e-12)
    eigenvalues, vectors = eigh(sr, st + epsilon * np.eye(channels), check_finite=True)
    order = np.argsort(eigenvalues)[::-1][:rank]
    u = _orthonormal(vectors[:, order])
    _, tx_vectors = np.linalg.eigh(st)
    u_tx = _orthonormal(tx_vectors[:, ::-1][:, :rank])
    result.update(
        U=u,
        U_tx=u_tx,
        epsilon=epsilon,
        rank=rank,
        generalized_eigenvalues=eigenvalues[order],
        selective_metrics=direction_metrics(sr, st, u, epsilon),
        tx_metrics=direction_metrics(sr, st, u_tx, epsilon),
    )
    return result


def random_basis(channels, rank, seed):
    """Fixed-seed Haar-distributed orthonormal Euclidean control directions."""
    if isinstance(channels, bool) or int(channels) != channels or channels < 1:
        raise ValueError("channels must be a positive integer")
    channels = int(channels)
    rank = _rank(rank, channels)
    return _orthonormal(np.random.default_rng(seed).normal(size=(channels, rank)))


def overlap(U, V):
    """Mean squared principal cosine for equally ranked orthonormal bases."""
    u, v = np.asarray(U, dtype=float), np.asarray(V, dtype=float)
    if u.ndim != 2 or v.shape != u.shape or not u.shape[1]:
        raise ValueError("subspaces must have matching [channels, rank] shape")
    eye = np.eye(u.shape[1])
    if not np.allclose(u.T @ u, eye, atol=1e-7) or not np.allclose(v.T @ v, eye, atol=1e-7):
        raise ValueError("subspace bases must be orthonormal")
    return float(np.clip(np.square(u.T @ v).sum() / u.shape[1], 0, 1))


def compress(z, center, U, alpha=.5):
    """Bounded residual projection on the final feature dimension.

    Inference accepts features only. The fitted center and basis are immutable
    inputs; labels or evaluation-set statistics are neither needed nor read.
    """
    z, center, u = (np.asarray(v, dtype=np.float64) for v in (z, center, U))
    if z.ndim < 1 or center.shape != (z.shape[-1],) or u.ndim != 2 or u.shape[0] != z.shape[-1]:
        raise ValueError("feature, center, and direction shapes do not agree")
    if not all(np.isfinite(v).all() for v in (z, center, u)):
        raise ValueError("nonfinite compression input")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if not np.allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-7):
        raise ValueError("compression directions must be orthonormal")
    return z - float(alpha) * ((z - center) @ u) @ u.T


def fit_rx_probe(z, y, rx, day):
    """Fit an equal-cell RX centroid probe after source TX/date demeaning.

    This conditional descriptive probe intentionally needs TX/date labels in
    evaluation. It is not the TX classifier and must never enter inference.
    Only this fit function estimates centroids and conditioning offsets.
    """
    result = scatters(z, y, rx, day)
    mu = result["cell_means"]
    offsets = mu.mean(axis=1)
    centroids = (mu - offsets[:, None]).mean(axis=(0, 2))
    return dict(
        offsets=offsets,
        centroids=centroids,
        tx_labels=result["tx_labels"],
        rx_labels=result["rx_labels"],
        day_labels=result["day_labels"],
    )


def score_rx_probe(probe, z, y, rx, day):
    """Evaluate fixed source centroids, reporting RX-equal and cell-equal acc."""
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
    residual = z - probe["offsets"][labels["tx"], labels["day"]]
    distance = ((residual[:, None] - probe["centroids"][None]) ** 2).sum(axis=-1)
    correct = distance.argmin(axis=1) == labels["rx"]
    nt, nr, nd = (len(probe[f"{name}_labels"]) for name in ("tx", "rx", "day"))
    cell_index = (labels["tx"] * nr + labels["rx"]) * nd + labels["day"]
    counts = np.bincount(cell_index, minlength=nt * nr * nd)
    if np.any(counts == 0):
        raise ValueError("incomplete evaluation rectangle")
    cell_acc = (np.bincount(cell_index, weights=correct, minlength=len(counts)) / counts).reshape(nt, nr, nd)
    per_rx = cell_acc.mean(axis=(0, 2))
    return dict(accuracy=float(per_rx.mean()), per_rx=per_rx,
                cell_accuracy=cell_acc, chance=1.0 / nr)
