"""Recompute metrics and pairing from the stored arrays without reading IQ."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from source import C, ROOT, read, sha, write

VIEWS = ("V0", "V1")
ARCHITECTURES = ("HStem", "LongPatch")
NAMES = read(C / "PRE_REGISTRATION.json")["rx_names"]


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_access(out, required):
    views = {view["id"]: view for view in read(C / "SOURCE_PLAN.json")["views"]}
    observed = []
    for line in (out / "ACCESS.jsonl").read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event["action"] != "copy_role_iq":
            continue
        view, role = event["view"], event["role"]
        declared = views[view][role]
        assert event["rx"] == sorted(declared["rx"])
        assert event["indices_sha256"] == declared["indices_array_sha256"]
        assert event["rows"] == declared["rows"]
        observed.append((event["stage"], view, role))
    assert observed == required
    return len(observed)


def pair_summary(arr):
    """Axes are held views, fixed seeds, and held RX in that view."""
    assert arr.ndim == 3 and arr.shape[::2] == (2, 6)
    mean_seed = arr.mean(axis=1)
    rng = np.random.default_rng(24501)
    bootstrap = mean_seed[np.arange(2)[:, None, None],
                          rng.integers(0, 6, size=(2, 10_000, 6))].mean(axis=(0, 2))
    return dict(mean_pp=float(arr.mean()), view_pp=arr.mean(axis=(1, 2)).tolist(),
                view_seed_pp=arr.mean(axis=2).tolist(), positive_view_seed=int((arr.mean(2) > 0).sum()),
                rx_pp=mean_seed.tolist(), worst_rx_pp=float(mean_seed.min()),
                worst_seed_rx_pp=float(arr.min()), interval95_pp=np.quantile(bootstrap, [.025, .975]).tolist())


def audit(out):
    cells, excluded, grouping = [], {}, {}
    for view in VIEWS:
        with np.load(out / f"AUDIT_{view}.npz", allow_pickle=False) as z:
            reasons = np.stack([z["block_min_coherence"] < .7, z["block_spread_hz"] > 5000,
                                z["lag_margin"] < .05])
            assert np.array_equal(~reasons.any(0), z["reliable"])
            excluded[view] = dict(coherence=int(reasons[0].sum()), spread=int(reasons[1].sum()),
                                  neighbor=int(reasons[2].sum()), any_failure=int(reasons.any(0).sum()))
            for r, tx, day in np.unique(np.stack([z["rx"], z["y"], z["day"]], axis=1), axis=0):
                part = (z["rx"] == r) & (z["y"] == tx) & (z["day"] == day)
                reliable = part & z["reliable"]
                count = int(reliable.sum())
                cells.append(dict(view=view, rx=NAMES[r], tx=int(tx), day=int(day),
                                  rows=int(part.sum()), reliable=count,
                                  reliable_pct=100 * count / int(part.sum()),
                                  median_cfo_hz=float(np.median(z["cfo_hz"][reliable])) if count else None))
        selected = [c for c in cells if c["view"] == view and c["reliable"] >= 8]
        by_tx = {tx: float(np.median([c["median_cfo_hz"] for c in selected if c["tx"] == tx]))
                 for tx in sorted({c["tx"] for c in selected})}
        by_rxday = {f"{rx}|{day}": float(np.median(
            [c["median_cfo_hz"] - by_tx[c["tx"]] for c in selected if (c["rx"], c["day"]) == (rx, day)]))
                    for rx, day in sorted({(c["rx"], c["day"]) for c in selected})}
        grouping[view] = dict(cells_with_8_reliable=len(selected), tx_cfo_hz=by_tx,
                              rxday_tx_centered_hz=by_rxday,
                              tx_span_hz=float(np.ptp(list(by_tx.values()))),
                              tx_centered_rxday_span_hz=float(np.ptp(list(by_rxday.values()))))
    csv_write(out / "AUDIT_CELLS.csv", cells)
    return dict(excluded=excluded, grouping=grouping)


def frozen(out):
    filenames = {v: out / f"FROZEN_{v}.npz" for v in VIEWS}
    summary, rows = {}, []
    contrasts = {}
    for v, path in filenames.items():
        with np.load(path, allow_pickle=False) as z:
            y, rx, reliable = z["y"], z["rx"], z["reliable"]
            folder = ROOT / "research" / {"V0": "wisig_w4", "V1": "wisig_w5"}[v] / "artifacts"
            for seed in range(3):
                with np.load(folder / f"P_A0_S{seed}_selection.npz", allow_pickle=False) as history:
                    assert np.array_equal(history["pool_ids"], z["ids"])
                    assert np.array_equal(history["y"], y)
                    assert np.array_equal(history["logits"].argmax(1), z["native"][seed])
            conditions = [name for name in z.files if z[name].shape == (3, len(y))]
            out_view = dict(reliable_pct=float(100 * reliable.mean()), historical_native_argmax_exact=True)
            condition_hits = {key: (z[key] == y[None]) for key in conditions}
            for r in np.unique(rx):
                for acc_name, hits in condition_hits.items():
                    for selection, keep in (("all", rx == r), ("reliable", (rx == r) & reliable)):
                        if keep.any():
                            for seed, value in enumerate(100 * hits[:, keep].mean(1)):
                                rows.append(dict(view=v, rx=NAMES[r], seed=seed, condition=acc_name,
                                                 subset=selection, rows=int(keep.sum()), accuracy=float(value)))
            for key in conditions:
                out_view[key] = {subset: float(np.mean([r["accuracy"] for r in rows
                                           if r["view"] == v and r["condition"] == key and r["subset"] == subset]))
                                 for subset in ("all", "reliable")}
            oracles = {str(hz): f"cfo_{hz}_oracle" for hz in (-10000, -2500, 2500, 10000)}
            out_view["oracle_prediction_changes"] = {
                hz: int((z["native"] != z[key]).sum()) for hz, key in oracles.items()}
            assert all(np.array_equal(condition_hits["native"], condition_hits[key])
                       for key in oracles.values())
            summary[v] = out_view
            values = {key: np.stack([hits[:, rx == r].mean(1) * 100 for r in np.unique(rx)], axis=1)
                      for key, hits in condition_hits.items()}
            contrasts[v] = contrast_arrays(values)
            assert np.array_equal(z["native"][:, ~reliable], z["native_cfo"][:, ~reliable])
    csv_write(out / "FROZEN_RX_METRICS.csv", rows)
    return dict(by_view=summary, contrasts={key: pair_summary(np.stack([g[key] for g in contrasts.values()]))
                                            for key in contrasts["V0"]})


def contrast_arrays(a):
    def averaged(left, right):
        return (a[left] + a[right]) / 2

    return dict(cfo10k_loss=a["native"] - averaged("cfo_-10000", "cfo_10000"),
                cfo2p5k_loss=a["native"] - averaged("cfo_-2500", "cfo_2500"),
                native_cfo_delta=a["native_cfo"] - a["native"],
                circular9_loss=a["native"] - averaged("time_-9_circular", "time_9_circular"),
                aligned9_loss=a["native"] - averaged("time_-9_aligned", "time_9_aligned"),
                time9_matched_loss=averaged("time_-9_aligned", "time_9_aligned")
                - averaged("time_-9_shifted", "time_9_shifted"))


def packet_scores(logits, y):
    prediction = logits.argmax(1)
    centered = logits.astype(np.float64) - logits.max(1, keepdims=True)
    ce = np.log(np.exp(centered).sum(1)) - centered[np.arange(len(logits)), y]
    return prediction, ce


def group_scores(y, prediction, ce):
    counts = np.bincount(10 * y + prediction, minlength=100).reshape(10, 10)
    actual, predicted = counts.sum(1), counts.sum(0)
    f1 = 2 * counts.diagonal() / (actual + predicted).clip(min=1)
    return dict(accuracy_pct=float(100 * np.trace(counts) / len(y)),
                macro_f1_pct=float(100 * f1.mean()), ce=float(ce.mean()))


def raw_control(out):
    output = out / "raw_control"
    access_checks = check_access(output, [("fit", v, role) for v in VIEWS
                                          for role in ("fit_train", "checkpoint_val")]
                                 + [("new_prediction", v, "source_selection") for v in VIEWS])
    prediction_seal = read(output / "PREDICTION_SEAL.json")
    fit_seal = read(output / "FIT_SEAL.json")
    selected = {}
    for job in fit_seal["results"]:
        assert sha(output / job["selected_path"]) == job["selected_sha256"]
        history = job["history"]
        assert len(history) == 40 and job["updates"] == 5080
        best = min(history, key=lambda event: event["val_ce"])
        selected["|".join(map(str, job["job"][:3]))] = dict(epoch=best["epoch"],
                   validation_ce=best["val_ce"], seconds=job["seconds"])
    for view in VIEWS:
        for seed in range(2):
            h, lp = [r for r in fit_seal["results"] if r["job"][:2] == [view, seed]]
            assert h["head_sha256"] == lp["head_sha256"]
            assert [a["order_sha256"] for a in h["history"]] == [a["order_sha256"] for a in lp["history"]]

    all_accuracy, rx_labels, rows, tx_rows = {}, {}, [], []
    for view in VIEWS:
        path = output / f"RAW_{view}.npz"
        assert sha(path) == prediction_seal[view]
        with np.load(path, allow_pickle=False) as z:
            y, rx = z["y"], z["rx"]
            with np.load(out / f"FROZEN_{view}.npz", allow_pickle=False) as frozen_data:
                assert all(np.array_equal(z[k], frozen_data[k]) for k in ("ids", "y", "rx", "day"))
            receivers = np.unique(rx)
            assert len(receivers) == 6
            rx_labels[view] = [NAMES[r] for r in receivers]
            view_arrays = {arch: {} for arch in ARCHITECTURES}
            conditions = [key.removeprefix("0_HStem_") for key in z.files if key.startswith("0_HStem_")]
            for arch in ARCHITECTURES:
                for condition in conditions:
                    ar = np.empty((2, 6))
                    for seed in (0, 1):
                        pred, ce = packet_scores(z[f"{seed}_{arch}_{condition}"], y)
                        for i, r in enumerate(receivers):
                            mask = rx == r
                            scores = group_scores(y[mask], pred[mask], ce[mask])
                            ar[seed, i] = scores["accuracy_pct"]
                            rows.append(dict(view=view, rx=NAMES[r], seed=seed, architecture=arch,
                                             condition=condition, rows=int(mask.sum()), **scores))
                            if condition == "native":
                                for tx in range(10):
                                    part = mask & (y == tx)
                                    tx_rows.append(dict(view=view, rx=NAMES[r], tx=tx, seed=seed,
                                                        architecture=arch, rows=int(part.sum()),
                                                        wrong=int((pred[part] != y[part]).sum())))
                    view_arrays[arch][condition] = ar
            all_accuracy[view] = view_arrays
    assert not set(rx_labels["V0"]) & set(rx_labels["V1"])
    acc = {arch: {cond: np.stack([all_accuracy[v][arch][cond] for v in VIEWS])
                  for cond in all_accuracy["V0"][arch]} for arch in ARCHITECTURES}
    h, lp = (acc[a] for a in ARCHITECTURES)
    h_losses, lp_losses = contrast_arrays(h), contrast_arrays(lp)
    comparisons = dict(native=pair_summary(h["native"] - lp["native"]))
    comparisons.update({k: pair_summary(lp_losses[k] - h_losses[k])
                        for k in h_losses if k != "native_cfo_delta"})
    nat = comparisons["native"]
    native_gate = (nat["mean_pp"] >= 1 and min(nat["view_pp"]) > 0 and
                   nat["interval95_pp"][0] > 0 and nat["worst_rx_pp"] >= -2)
    perturbation_gates = {}
    for key in ("cfo10k_loss", "time9_matched_loss"):
        part = comparisons[key]
        lo, hi = part["interval95_pp"]
        perturbation_gates[key] = (abs(part["mean_pp"]) >= 1 and lo * hi > 0 and
                                  all(v * part["mean_pp"] > 0 for v in part["view_pp"]))
    scores = {a: {k: {"accuracy_pct": float(values.mean()),
                      "view_accuracy_pct": values.mean(axis=(1, 2)).tolist()}
                  for k, values in acc[a].items()} for a in ARCHITECTURES}
    for a in ARCHITECTURES:
        native = [r for r in rows if r["architecture"] == a and r["condition"] == "native"]
        for metric in ("macro_f1_pct", "ce"):
            scores[a]["native"][metric] = float(np.mean([r[metric] for r in native]))
    csv_write(output / "RX_METRICS.csv", rows)
    csv_write(output / "NATIVE_TX_RX.csv", tx_rows)
    return dict(scores=scores, absolute_deltas={a: {k: pair_summary(d) for k, d in contrast_arrays(acc[a]).items()}
                                              for a in ARCHITECTURES},
                comparisons=comparisons, gates=dict(native=native_gate, **perturbation_gates),
                rx_labels=rx_labels, fits=selected, total_updates=fit_seal["total_updates"],
                role_payload_access_checks=access_checks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    records = check_access(out, [("audit", v, "fit_train") for v in VIEWS]
                          + [("frozen_diagnostic", v, "source_selection") for v in VIEWS])
    latest = dict(audit=audit(out), frozen=frozen(out), role_payload_access_checks=records)
    if (out / "raw_control" / "PREDICTION_SEAL.json").is_file():
        latest["raw_control"] = raw_control(out)
    write(out / "SUMMARY.json", latest)


if __name__ == "__main__":
    main()
