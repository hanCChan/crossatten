"""Frozen diagnostic perturbations. No fitting or target-driven choices."""
import json
import numpy as np
import pandas as pd
import torch
from stage import HERE, DATA, ROOT, RX, s09, sha, save, cells, ask_modules, receivers
from features import phase_view, paired_model
from xa.explore_models import frequency_views
from xa.frontends import rms_normalize
from xa.wisig import load_cell
from report_utils import table


def sample(x, y, count=20):
    ids = np.concatenate([np.flatnonzero(y == tx)[:count] for tx in np.unique(y)])
    return x[ids], y[ids]


def perturb(x, kind, value, seed):
    if kind == "clean":
        return x
    if kind == "awgn":
        g = torch.Generator(device=x.device).manual_seed(seed)
        noise = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
        power = x.square().sum(1).mean(-1, keepdim=True)
        return x + noise * (power / (2 * 10 ** (value / 10))).sqrt()[:, None]
    z = torch.complex(x[:, 0], x[:, 1])
    if kind == "shift":
        shift = int(value)
        shifted = torch.zeros_like(z)
        if shift > 0:
            shifted[:, shift:] = z[:, :-shift]
        else:
            shifted[:, :shift] = z[:, -shift:]
        z = shifted
    elif kind == "multipath":
        delay, gain = value
        shifted = torch.zeros_like(z)
        shifted[:, delay:] = z[:, :-delay]
        z = z + gain * shifted
    else:
        raise ValueError(kind)
    return torch.stack((z.real, z.imag), 1)


def conditions(reg):
    settings = reg["robustness"]
    return [("clean", 0)] + [("awgn", v) for v in settings["awgn_db"]] + [
        ("multipath", v) for v in settings["multipath"]] + [("shift", v) for v in settings["shifts"]]


def feature(x, kind):
    return phase_view(frequency_views(rms_normalize(x)), kind)[:, 3:].flatten(1)


def distances(v, y, rx, scale):
    v = v / scale
    centroids = [v[y == tx].mean(0) for tx in np.unique(y)]
    c = torch.stack(centroids)
    dtx = float(torch.pdist(c).square().mean() / c.shape[1])
    offsets = []
    for tx in np.unique(y):
        avg = v[y == tx].mean(0)
        for name in np.unique(rx):
            mask = (y == tx) & (rx == name)
            if mask.any():
                offsets.append(float((v[mask].mean(0) - avg).square().mean()))
    return dtx, float(np.mean(offsets))


def feature_probe(reg, device):
    output = []
    for dataset in RX:
        for fold in range(4):
            source, _ = receivers(dataset, fold)
            xs, ys, rxs = [], [], []
            source_job = dict(dataset=dataset, fold=fold, phase="physics",
                              id=f"physics-source-{dataset}-{fold}")
            cells(source_job, source, (1, 2), "diagnostic")
            for rx in source:
                for day in (1, 2):
                    x, y = sample(*load_cell(dataset, rx, day))
                    xs.append(x)
                    ys.append(y)
                    rxs.extend([rx] * len(y))
            x = torch.from_numpy(np.concatenate(xs)).to(device)
            y, names = np.concatenate(ys), np.asarray(rxs)
            # One common scalar defined using ONLY clean source hard phase.
            clean_hard = feature(x, "hard")
            scale = clean_hard.square().mean().sqrt().clamp_min(1e-6)
            base = frequency_views(rms_normalize(x))
            z = torch.complex(base[:, 0], base[:, 1])
            p = z[:, 1:] * z[:, :-1].conj()
            valid = (z[:, 1:].abs() > .05) & (z[:, :-1].abs() > .05)
            for kind, value in conditions(reg):
                xp = perturb(x, kind, value, 10101 + fold)
                bz = frequency_views(rms_normalize(xp))
                zz = torch.complex(bz[:, 0], bz[:, 1])
                pp = zz[:, 1:] * zz[:, :-1].conj()
                angle_error = torch.angle(pp * p.conj())
                phase_var = float(angle_error[valid].var()) if valid.any() else None
                for encoder in ("hard", "soft", "unweighted"):
                    v = feature(xp, encoder)
                    clean = feature(x, encoder)
                    dtx, drx = distances(v, y, names, scale)
                    output.append(dict(dataset=dataset, fold=fold, role="source_diagnostic",
                        encoder=encoder, perturbation=kind, value=json.dumps(value),
                        packets=len(y), source_scale=float(scale), phase_error_variance=phase_var,
                        feature_mse=float(((v - clean) / scale).square().mean()),
                        d_tx=dtx, d_rx=drx, d_rx_over_d_tx=drx / max(dtx, 1e-12)))
            print("Feature physics", dataset, fold, flush=True)
    return output


def accuracy_probe(reg, device):
    rows, seals = [], {}
    for phase in ("ask", "final", "ablation"):
        for path in sorted((HERE / "runs" / phase).glob("*/target_metrics.json")):
            result = json.loads((path.parent / "metrics.json").read_text())
            job = result["job"]
            weight = path.parent / "model.pt"
            if sha(weight) != result["weights_sha256"]["model.pt"]:
                raise ValueError("unsealed physics checkpoint")
            seals[str(weight.relative_to(HERE))] = sha(weight)
            if job["feature"] == "ask":
                official = ask_modules()
                conf = __import__("argparse").Namespace(candidates=[8, 16, 24, 32, 48, 64],
                    patch_size=64, embed_dim=128, mlp_ratio=2.)
                model = official.build_model(conf, 10 if job["dataset"] == "ManyRx" else 6)
            else:
                model, _ = paired_model(job["dataset"], job["feature"], 0)
            model.load_state_dict(torch.load(weight, map_location="cpu", weights_only=True))
            model.to(device).eval()
            _, targets = receivers(job["dataset"], job["fold"])
            data = cells(job, targets, (3, 4), "target", "CRD")
            for rx, cell in data.items():
                # Keep exactly the first20 packets per TX/day, with audit IDs.
                x1, y1 = sample(*load_cell(job["dataset"], rx, 3))
                x2, y2 = sample(*load_cell(job["dataset"], rx, 4))
                x = torch.from_numpy(np.concatenate([x1, x2])).to(device)
                y = torch.from_numpy(np.concatenate([y1, y2])).to(device)
                for kind, value in conditions(reg):
                    xp = perturb(x, kind, value, 10101 + job["fold"])
                    lv = s09.s08.predict(model, xp, 128)
                    rows.append(dict(dataset=job["dataset"], fold=job["fold"], seed=job["seed"],
                        model=job["feature"], rx=rx, perturbation=kind, value=json.dumps(value),
                        accuracy_pct=100 * float((lv.argmax(1) == y).float().mean()),
                        packets=len(y)))
            del model
            print("Frozen physics", job["id"], flush=True)
    return rows, seals


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    reg = json.loads((HERE / "registration.json").read_text())
    device = "cuda"
    if (HERE / "physical_evidence.json").exists():
        feature_rows = pd.read_csv(HERE / "physical_feature_probes.csv").to_dict("records")
        accuracy_rows = pd.read_csv(HERE / "physical_accuracy_probes.csv").to_dict("records")
        seals = json.loads((HERE / "physical_evidence.json").read_text())["checkpoints"]
        for path, digest in seals.items():
            if sha(HERE / path) != digest:
                raise ValueError("cached physical evidence checkpoint changed")
    else:
        feature_rows = feature_probe(reg, device)
        accuracy_rows, seals = accuracy_probe(reg, device)
    pd.DataFrame(feature_rows).to_csv(HERE / "physical_feature_probes.csv", index=False)
    pd.DataFrame(accuracy_rows).to_csv(HERE / "physical_accuracy_probes.csv", index=False)
    save(HERE / "physical_evidence.json", dict(checkpoints=seals, additional_fits=0,
        conditions=reg["robustness"], source_normalization="same clean hard source scalar RMS for all encoders",
        phase_variance="wrapped perturbation angle variance on clean reliable pairs; high-SNR approximation only",
        d_tx="mean squared pairwise TX-centroid distance / feature dimension",
        d_rx="mean squared TX-conditioned RX-centroid displacement / feature dimension",
        missing_historical_hard_weights=True))
    frame = pd.DataFrame(feature_rows)
    aggregate = frame.groupby(["encoder", "perturbation", "value"])[
        ["feature_mse", "d_tx", "d_rx", "d_rx_over_d_tx"]].mean().reset_index()
    lines = ["# Physical Robustness", "",
        "Frozen diagnostics only; zero additional fits. Conditions registered before source outcomes.",
        "Source phase encoder comparisons use identical packets and draws, common source-only RMS scaling.",
        "DTX=TX-centroid separation; DRX=TX-conditioned RX displacement, both divided by feature dimension.",
        "", table(aggregate, 6), "",
        "Accuracy curves: `physical_accuracy_probes.csv` (all surviving S10 formal checkpoints, CRD only).",
        "These are controlled distortions added to recorded IQ, not independently acquired electromagnetic environments.",
        "No claim of arbitrary-channel robustness, authentication security or causal RX independence is supported.",
        "Historical S07-S09 weights are missing. Consequently, soft-v-hard CLASSIFIER robustness cannot "
        "be supplied if the soft candidate fails its source gate. Encoder-only differences do not replace it.",
        "No target soft-model diagnostics are run after a STOP gate."]
    awgn = frame[(frame.perturbation == "awgn") & (frame.value.astype(str) == "20")]
    if not awgn.empty:
        means = awgn.groupby(["dataset", "encoder"]).feature_mse.mean()
        lines += ["", "At20dB AWGN, source normalized phase-feature MSE (hard -> soft): " +
            "; ".join(f"{d}: {means[d, 'hard']:.6f} -> {means[d, 'soft']:.6f}"
                      for d in ("ManyRx", "ManySig")) + ".",
            "Lower feature MSE is consistent with attenuation of uncertain phase, but is NOT sufficient "
            "to preserve identity discrimination: source ManyRx classification declined under the same candidate.",
            "Amplitude shrinkage itself lowers MSE; DTX and DRX/DTX are reported to avoid equating "
            "feature stability with useful RX invariance."]
    if accuracy_rows:
        scores = pd.DataFrame(accuracy_rows).groupby(
            ["dataset", "model", "perturbation", "value"]).accuracy_pct.mean().reset_index()
        lines += ["", "RX/seed-equal frozen classifier accuracy:", "", table(scores)]
    (HERE / "PHYSICAL_ROBUSTNESS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
