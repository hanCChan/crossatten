"""Real source-IQ feature perturbations; not target-driven feature selection."""
import csv
import json
import math
import numpy as np
import torch
from training import HERE, read_cells, frequency_views, rms_normalize, phase_features_from_base, paired_model, s08, save


def transformed(x, kind):
    z = torch.complex(x[:, 0], x[:, 1])
    n = torch.arange(z.shape[-1], dtype=z.real.dtype)[None]
    if kind == "circular_shift9":
        z = torch.roll(z, 9, -1)
    elif kind == "crop_shift9":
        z = torch.nn.functional.pad(z[:, :-9], (9, 0))
    elif kind == "global_rotation":
        z = z * torch.exp(torch.tensor(1.23j))
    elif kind == "cfo10khz":
        z = z * torch.exp(2j * math.pi * 10000 * n / 25e6)
    elif kind == "smooth_channel":
        gain = torch.exp(.15 * torch.sin(2 * math.pi * n / 256) +
                         .3j * torch.sin(6 * math.pi * n / 256))
        z = torch.fft.ifft(torch.fft.fft(z) * gain)
    else:
        raise ValueError(kind)
    return torch.stack((z.real, z.imag), 1)


def fisher(features, y):
    x = features.numpy().reshape(len(y), -1).astype(np.float64)
    total = x.mean(0)
    means = {c: x[y == c].mean(0) for c in np.unique(y)}
    between = sum((y == c).sum() * np.square(mean - total).sum() for c, mean in means.items())
    within = sum(np.square(x[y == c] - mean).sum() for c, mean in means.items())
    return float(between / max(within, 1e-12))


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    rows, separation, parity = [], [], []
    for dataset in ("ManyRx", "ManySig"):
        for fold in range(4):
            from xa.wisig import receivers
            source, _ = receivers(dataset, fold)
            job = dict(dataset=dataset, fold=fold, phase="source", id=f"diagnostic_{dataset}_f{fold}")
            cells = read_cells(job, [source[0], source[-1]], (1, 2), "diagnostic")
            for rx, cell in cells.items():
                # Balanced fixed prefix per TX, never rank packets using labels or target errors.
                ids = np.concatenate([np.flatnonzero(cell["y"] == c)[:16] for c in np.unique(cell["y"])])
                x, y = torch.from_numpy(cell["x"][ids]), cell["y"][ids]
                base = frequency_views(rms_normalize(x))
                z = torch.complex(base[:, 0], base[:, 1])
                amp = z.abs()
                valid = (amp[:, 1:] >= .05) & (amp[:, :-1] >= .05)
                p = z[:, 1:] * z[:, :-1].conj()
                p = p / (amp[:, 1:] * amp[:, :-1]).clamp_min(1e-6) * valid
                strength = p.sum(-1).abs() / valid.sum(-1).clamp_min(1)
                for feature in ("first", "centered", "curvature"):
                    original = phase_features_from_base(base, feature)
                    separation.append(dict(dataset=dataset, fold=fold, rx=rx, feature=feature,
                        extra_fisher=fisher(original[:, 3:], y), base_fisher=fisher(base, y),
                        reliable_pair_fraction=float(valid.float().mean()),
                        centered_fallback_fraction=float((strength < .1).float().mean()),
                        resultant_median=float(strength.median())))
                    for kind in ("global_rotation", "circular_shift9", "crop_shift9", "cfo10khz", "smooth_channel"):
                        altered = phase_features_from_base(frequency_views(rms_normalize(transformed(x, kind))), feature)
                        extra_error = (altered[:, 3:] - original[:, 3:]).square().mean((1, 2)).sqrt()
                        full_error = (altered - original).square().mean((1, 2)).sqrt()
                        rows.append(dict(dataset=dataset, fold=fold, rx=rx, feature=feature, perturbation=kind,
                                         extra_rms_change=float(extra_error.mean()), full_rms_change=float(full_error.mean()),
                                         extra_max_change=float((altered[:, 3:] - original[:, 3:]).abs().max())))
                model, shared = paired_model(dataset, "first", 17)
                old = s08.build_model(dataset, "phase")
                old.load_state_dict(model.state_dict())
                model.eval()
                old.eval()
                with torch.no_grad():
                    delta = float((model(x) - old(x)).abs().max())
                if delta != 0:
                    raise ValueError("real IQ S08 parity failed")
                parity.append(dict(dataset=dataset, fold=fold, rx=rx, max_logit_delta=delta, shared_sha256=shared))
    for name, values in (("source_perturbations.csv", rows), ("source_separation.csv", separation),
                         ("real_iq_parity.csv", parity)):
        with (HERE / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
    save(HERE / "diagnostics.json", dict(parity_checks=len(parity), perturbation_rows=len(rows),
                                        separation_rows=len(separation), source_only=True))
    lines = ["# S09 Source-IQ Diagnostics\n\n",
             "Fixed 16 packets/TX from first/last legal source RX, days1-2. "
             "No target access. Fisher ratios are in-sample feature diagnostics, not trained "
             "classifier accuracy or independent identity preservation evidence.\n\n",
             "| Dataset | Feature | Circular extra RMS | Crop extra RMS | Channel extra RMS | Fallback fraction |\n",
             "|---|---|---:|---:|---:|---:|\n"]
    for dataset in ("ManyRx", "ManySig"):
        for feature in ("first", "centered", "curvature"):
            values = []
            for kind in ("circular_shift9", "crop_shift9", "smooth_channel"):
                values.append(np.mean([r["extra_rms_change"] for r in rows
                                       if r["dataset"] == dataset and r["feature"] == feature and r["perturbation"] == kind]))
            fallback = np.mean([r["centered_fallback_fraction"] for r in separation if r["dataset"] == dataset])
            lines.append(f"| {dataset} | {feature} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | {fallback:.4f} |\n")
    lines.append("\nThe retained Re/Im channels are still shift sensitive. Centered weak-resultant "
                 "fallback retains first-order timing sensitivity. Curvature invariance applies "
                 "to ideal circular shifts only; ordinary truncation and channel perturbations "
                 "need not cancel. Changes do not prove a dominant receiver nuisance.\n")
    (HERE / "DIAGNOSTICS.md").write_text("".join(lines))
    print(json.dumps(dict(real_iq_parity="PASS", rows=len(rows))))


if __name__ == "__main__":
    main()
