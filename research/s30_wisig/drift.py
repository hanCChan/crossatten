"""Cross-fit descriptive RX-associated CFO using the existing source audit."""
from pathlib import Path

import numpy as np


def pool_audit(parent):
    keys = ("ids", "y", "rx", "day", "cfo_hz", "reliable")
    raw = []
    for view in ("V0", "V1"):
        path = parent / f"AUDIT_{view}.npz"
        with np.load(path, allow_pickle=False) as z:
            raw.append({k: z[k] for k in keys})
    overlap, a, b = np.intersect1d(raw[0]["ids"], raw[1]["ids"], return_indices=True)
    for k in keys:
        assert np.array_equal(raw[0][k][a], raw[1][k][b]), k
    unique = ~np.isin(raw[1]["ids"], overlap)
    data = {k: np.r_[raw[0][k], raw[1][k][unique]] for k in keys}
    return data, len(overlap)


def observed_cells(data):
    factors = np.column_stack([data[k] for k in ("y", "rx", "day")])
    counts, rows, observations = [], [], []
    for tx, rx, day in np.unique(factors, axis=0):
        valid = np.all(factors == [tx, rx, day], axis=1)
        kept = valid & data["reliable"]
        counts.append(dict(tx=int(tx), rx=int(rx), day=int(day), packets=int(valid.sum()), reliable=int(kept.sum())))
        if kept.sum() >= 8:
            rows.append([tx, rx, day])
            observations.append(float(np.median(data["cfo_hz"][kept]) / 1000))
    return np.asarray(rows), np.asarray(observations), counts


def loo(model, response):
    # The SVD computes the projection and leverage even for redundant dummies.
    u, s, _ = np.linalg.svd(model, full_matrices=False)
    basis = u[:, s > 1e-10]
    hat = (basis ** 2).sum(1)
    residual = response - basis @ (basis.T @ response)
    predictable = (1 - hat) > 1e-9
    error = np.full(len(response), np.nan)
    error[predictable] = residual[predictable] / (1 - hat[predictable])
    return error, predictable, basis.shape[1]


def model_comparison(rows, y):
    codes = dict(tx=rows[:, [0]], rx=rows[:, [1]], day=rows[:, [2]],
                 tx_rx=rows[:, [0, 1]], rx_day=rows[:, [1, 2]])
    dummies = {}
    for name, level in codes.items():
        _, inv = np.unique(level, axis=0, return_inverse=True)
        dummies[name] = np.eye(inv.max() + 1)[inv]
    designs = {"tx_day": ["tx", "day"], "add_rx": ["tx", "day", "rx"],
               "tx_rx": ["tx", "day", "rx", "tx_rx"], "rx_day": ["tx", "day", "rx", "rx_day"]}
    output, flags = {}, {}
    for name, factors in designs.items():
        design = np.column_stack([dummies[f] for f in factors])
        output[name], flags[name], _ = loo(design, y)
    both = flags["add_rx"] & flags["tx_day"]
    strict = np.logical_and.reduce(list(flags.values()))

    def summarize(names, idx):
        return dict(cells=int(idx.sum()), mae_khz={
            name: float(np.abs(output[name][idx]).mean()) for name in names})

    match = summarize(("tx_day", "add_rx"), both)
    match["mae_reduction"] = float(1 - match["mae_khz"]["add_rx"] / match["mae_khz"]["tx_day"])
    return dict(additive_comparison=match, full_common_comparison=summarize(designs, strict),
                individually_estimable={name: int(keep.sum()) for name, keep in flags.items()})


def cross_rx(rows, y, names):
    tx, rx, day = rows.T
    residual = np.full(len(y), np.nan)
    for i, (t, r, d) in enumerate(rows):
        others = (tx == t) & (rx != r) & (day == d)
        if others.sum() >= 3:
            residual[i] = y[i] - np.median(y[others])
    estimate, details = {}, []
    for r in np.unique(rx):
        for d in (1, 2):
            for half in (0, 1):
                mask = (rx == r) & (day == d) & (tx % 2 == half) & np.isfinite(residual)
                if mask.sum() >= 2:
                    value = float(np.median(residual[mask]))
                    estimate[r, d, half] = value
                    details.append(dict(rx=names[r], day=d, parity=half, supported_tx=int(mask.sum()),
                                        receiver_associated_khz=value))
    folds = []
    for half in (0, 1):
        common = [r for r in np.unique(rx) if (r, 1, half) in estimate and (r, 2, 1-half) in estimate]
        x = np.array([estimate[r, 1, half] for r in common])
        z = np.array([estimate[r, 2, 1-half] for r in common])
        error = float(np.abs(x - z).mean())
        zero = float(np.abs(z).mean())
        folds.append(dict(day1_parity=half, day2_parity=1-half, n=len(common),
                          receivers=[names[r] for r in common], pearson_r=float(np.corrcoef(x, z)[0, 1]),
                          mae_khz=error, zero_mae_khz=zero, reverse_zero_mae_khz=float(np.abs(x).mean()),
                          same_sign=int((x*z > 0).sum()), median_abs_rx_khz=float(np.median(np.abs(np.r_[x, z]))),
                          p90_abs_rx_khz=float(np.quantile(np.abs(np.r_[x, z]), .9))))
    return folds, details


def analyze(parent: Path, names):
    packets, duplicate = pool_audit(parent)
    rows, y, counts = observed_cells(packets)
    comparison = model_comparison(rows, y)
    folds, details = cross_rx(rows, y, names)
    gate = comparison["additive_comparison"]["mae_reduction"] >= .15 and all(
        f["n"] >= 6 and f["pearson_r"] >= .6 and f["mae_khz"] < f["zero_mae_khz"] for f in folds)
    return dict(unique_training_packets=len(packets["ids"]), duplicate_packets_removed=duplicate,
                coverage=float(packets["reliable"].mean()), supported_cells=len(rows),
                cells=counts, model_comparison=comparison, cross_rx=folds, effects=details,
                stable_rx_gate=gate)
