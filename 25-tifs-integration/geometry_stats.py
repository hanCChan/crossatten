"""Balanced descriptive scatter decomposition, with unit-normalized embeddings."""
import numpy as np


def geometry(features, tx, rx):
    features = np.asarray(features, dtype=np.float64)
    tx, rx = np.asarray(tx), np.asarray(rx)
    norm = np.linalg.norm(features, axis=1)
    if not np.isfinite(features).all() or np.any(norm <= 1e-12):
        raise ValueError('Invalid or zero embedding')
    z = features / norm[:, None]
    txs, rxs = np.unique(tx), np.unique(rx)
    counts = [int(np.sum((tx == t) & (rx == r))) for t in txs for r in rxs]
    if not counts or min(counts) == 0 or len(set(counts)) != 1:
        raise ValueError('Balanced TX x RX cells required')
    mean = z.mean(0)
    tx_centroids = {t: z[tx == t].mean(0) for t in txs}
    b_tx = float(np.mean([np.sum((tx_centroids[t] - mean)**2) for t in txs]))
    d_rx, w_cell = [], []
    for t in txs:
        for r in rxs:
            samples = z[(tx == t) & (rx == r)]
            centroid = samples.mean(0)
            d_rx.append(np.sum((centroid - tx_centroids[t])**2))
            w_cell.append(np.mean(np.sum((samples - centroid)**2, axis=1)))
    d_rx, w_cell = float(np.mean(d_rx)), float(np.mean(w_cell))
    total = float(np.mean(np.sum((z - mean)**2, axis=1)))
    residual = abs(total - b_tx - d_rx - w_cell)
    if residual > 1e-10:
        raise ArithmeticError('Scatter decomposition failed')
    return dict(between_tx=b_tx, conditional_rx=d_rx, within_tx_rx=w_cell, total=total,
                tx_to_rx_centroid_ratio=b_tx/(d_rx+1e-12),
                tx_to_within_class_ratio=b_tx/(d_rx+w_cell+1e-12),
                tx_fraction=b_tx/(total+1e-12), rx_fraction=d_rx/(total+1e-12),
                decomposition_abs_error=residual, min_embedding_norm=float(norm.min()),
                tx_classes=len(txs), source_receivers=len(rxs), packets_per_tx_rx=counts[0],
                feature_dimension=z.shape[1])
