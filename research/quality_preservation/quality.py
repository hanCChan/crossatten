"""Label-free repeatability proxies; fixed packet-relative windows, no alignment."""
import numpy as np


def correlation(z, lag, start=20, stop=140):
    a = z[:, start:stop-lag]
    b = z[:, start+lag:stop]
    energy = np.sqrt(np.sum(abs(a)**2, axis=1)*np.sum(abs(b)**2, axis=1))
    return np.clip(np.divide(abs(np.sum(a.conj()*b, axis=1)), energy,
                             out=np.zeros(len(z)), where=energy > 0), 0, 1)


def proxies(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (2, 256):
        raise ValueError('Expected N x 2 x 256 real IQ')
    valid = np.isfinite(x).all(axis=(1, 2)) & (np.sum(x*x, axis=(1, 2)) > 0)
    z = np.nan_to_num(x[:, 0] + 1j*x[:, 1], nan=0, posinf=0, neginf=0)
    blocks = z[:, 20:140].reshape(-1, 6, 20)
    a, b = blocks[:, :-1], blocks[:, 1:]
    energy = np.sqrt(np.sum(abs(a)**2, axis=2)*np.sum(abs(b)**2, axis=2))
    q = np.clip(np.divide(abs(np.sum(a.conj()*b, axis=2)), energy,
                         out=np.zeros_like(energy), where=energy > 0), 0, 1)
    # CFO / constant phase rotates the correlations; taking magnitude avoids
    # interpreting that phase as poor repeatability. This is not an SNR estimate.
    cross = np.sum(a.conj()*b, axis=2)
    phase_coherence = abs(np.mean(np.divide(cross, abs(cross),
        out=np.zeros_like(cross), where=abs(cross)>0), axis=1))
    features = np.column_stack((q.mean(1), q.min(1), phase_coherence,
        correlation(z, 19), correlation(z, 21), correlation(z, 20, 160, 256)))
    features[~valid] = 0
    return features, q, valid


FEATURE_NAMES = ('q20_mean', 'q20_min', 'phase_consistency', 'lag19', 'lag21', 'tail20')


def within_cell_low(q, y, rx, day, ids):
    """Bottom 25% per TX x RX x day; stable IDs break exact ties, no selection."""
    low = np.zeros(len(q), dtype=bool)
    for r, t, d in sorted(set(zip(rx, y, day))):
        pos = np.flatnonzero((rx == r) & (y == t) & (day == d))
        if len(pos) < 8:
            raise ValueError('Insufficient support for a conditional quartile')
        order = np.lexsort((ids[pos], q[pos]))
        low[pos[order[:max(1, len(pos)//4)]]] = True
    return low


def conditional_effect(error, low, y, rx, day):
    """Average streams first, days within TX, TX within RX; RX is inference unit."""
    rows = []
    for r, t, d in sorted(set(zip(rx, y, day))):
        m = (rx == r) & (y == t) & (day == d)
        lo, hi = error[m & low].mean(), error[m & ~low].mean()
        rows.append(dict(rx=str(r), tx=int(t), day=int(d), n=int(m.sum()),
                         n_low=int((m & low).sum()), low_error=float(lo),
                         other_error=float(hi), difference_pp=float(100*(lo-hi))))
    receivers = []
    for r in np.unique(rx):
        cell = [v for v in rows if v['rx'] == r]
        effects = [np.mean([v['difference_pp'] for v in cell if v['tx'] == t])
                   for t in sorted({v['tx'] for v in cell})]
        receivers.append(dict(rx=str(r), difference_pp=float(np.mean(effects))))
    return rows, receivers


def bootstrap_rx(values, seed=27113, count=20000):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(len(v), size=(count, len(v)))].mean(1)
    return np.quantile(means, [.025, .975]).tolist()
