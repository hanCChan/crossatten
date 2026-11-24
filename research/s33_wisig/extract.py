"""Fixed physical readouts and source-only centroid probes for frozen models."""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s29_wisig"))
from experiment import DEVICE, FeatureModel, featurize, tensor, torch
from physics import characterize, complex_iq, lag_coherence
from source import ROOT, Source, read, sha, write

VIEWS = ("V0", "V1")
ROLES = ("fit_train", "checkpoint_val", "source_selection")
LAYERS = ("token_mean", "token_flat", "pooled")
PROXY_NAMES = ("cfo_abs_khz", "cfo_departure_khz", "repeat_loss", "phase_spread_khz",
               "outband_fraction", "spectral_distance", "transition_dev_samples")


def physical(x):
    stats = characterize(x)
    z = complex_iq(x)
    power = torch.fft.fftshift(torch.fft.fft(z), dim=-1).abs().square()
    coarse = power.reshape(len(x), 16, 16).sum(2)
    coarse = coarse / coarse.sum(1, keepdim=True)
    locations = tuple(range(20, 177, 4))
    coherence = torch.stack([lag_coherence(z[:, at:at+80], 20) for at in locations], 1)
    lost = (coherence[:, :-1] < .7) & (coherence[:, 1:] < .7)
    valid = (coherence[:, 0] >= .7) & lost.any(1)
    transition = 20 + 4 * lost.int().argmax(1)
    stats["transition"] = torch.where(valid, transition.float(), torch.nan)
    stats["shape"] = coarse
    return {key: a.cpu().numpy() for key, a in stats.items()}


def fit_reference(stats, data):
    """Sum the contribution of each admissible source RX equally."""
    by_tx_day = {}
    for tx in range(10):
        for day in (1, 2):
            base = (data["y"] == tx) & (data["day"] == day)
            cfo, transition, spectral = [], [], []
            for code in np.unique(data["rx"][base]):
                mask = base & (data["rx"] == code)
                good = mask & stats["reliable"]
                time = mask & np.isfinite(stats["transition"])
                if good.sum() >= 8:
                    cfo.append(np.median(stats["cfo_hz"][good]))
                if time.sum() >= 8:
                    transition.append(np.median(stats["transition"][time]))
                if mask.sum() >= 8:
                    spectral.append(stats["shape"][mask].mean(0))
            by_tx_day[tx, day] = (np.median(cfo) if cfo else np.nan,
                                   np.median(transition) if transition else np.nan,
                                   np.mean(spectral, 0) if spectral else np.full(16, np.nan))
    return by_tx_day


def deviations(stats, data, refs):
    n = len(data["y"])
    cfo, timing, spectral = (np.full(n, np.nan) for _ in range(3))
    for (tx, day), (f_ref, dt_ref, spec_ref) in refs.items():
        mask = (data["y"] == tx) & (data["day"] == day)
        rel = mask & stats["reliable"]
        cfo[rel] = np.abs(stats["cfo_hz"][rel] - f_ref) / 1000
        timing[mask] = np.abs(stats["transition"][mask] - dt_ref)
        spectral[mask] = np.linalg.norm(np.sqrt(stats["shape"][mask]) - np.sqrt(spec_ref), axis=1) / np.sqrt(2)
    return dict(cfo_abs_khz=np.where(stats["reliable"], np.abs(stats["cfo_hz"]) / 1000, np.nan),
                cfo_departure_khz=cfo, repeat_loss=1 - stats["block_min_coherence"],
                phase_spread_khz=stats["block_spread_hz"] / 1000,
                outband_fraction=1 - stats["inband_fraction"], spectral_distance=spectral,
                transition_dev_samples=timing, transition_pos_samples=stats["transition"],
                cfo_hz=stats["cfo_hz"], reliable=stats["reliable"])


def feature_pass(model, data):
    net = model.net
    arrays = {name: [] for name in (*LAYERS, "logits")}
    x = tensor(data["x"])
    for z in x.split(256):
        tokens = net.frequency(featurize(net, z))
        h = net.pre_ln(net.encoder.prepare_tokens(tokens))
        for block in net.encoder.blocks:
            h = block(h)
        pooled = net.encoder.final_norm(h).mean(1)
        normalize = torch.nn.functional.normalize
        arrays["token_flat"].append(normalize(tokens.flatten(1)))
        arrays["token_mean"].append(normalize(tokens.mean(1)))
        arrays["pooled"].append(normalize(pooled))
        arrays["logits"].append(net.classifier(pooled))
    return {k: torch.cat(values) for k, values in arrays.items()}


def centers(x, data):
    y, rx = (tensor(data[k]) for k in ("y", "rx"))
    rx_codes = np.unique(data["rx"])
    values = []
    for a in range(10):
        means = [x[(y == a) & (rx == b)].mean(0) for b in rx_codes
                 if np.count_nonzero((data["y"] == a) & (data["rx"] == b)) >= 8]
        assert len(means) >= 3
        values.append(torch.stack(means).mean(0))
    return torch.nn.functional.normalize(torch.stack(values))


def center_shift(x, data, reference, names):
    mats = torch.cdist(reference, reference)
    mats.fill_diagonal_(torch.inf)
    minimum = mats.amin(1)
    rows = []
    for rx, tx, day in np.unique(np.column_stack([data["rx"], data["y"], data["day"]]), axis=0):
        mask = (data["rx"] == rx) & (data["y"] == tx) & (data["day"] == day)
        if mask.sum() < 20:
            continue
        centroid = torch.nn.functional.normalize(x[tensor(mask)].mean(0), dim=0)
        distance = (centroid - reference[tx]).norm() / minimum[tx]
        rows.append(dict(rx=names[rx], tx=int(tx), day=int(day), rows=int(mask.sum()),
                         relative_shift=float(distance)))
    return rows


def a1_predictions(view, raw, names, files):
    folder = ROOT / "research" / {"V0": "wisig_w4", "V1": "wisig_w5"}[view]
    manifest = read(folder / "REPLAY_FILES.json")
    expected = {r["relative"]: r["sha256"] for r in manifest["files"]}
    found = {}
    for seed in (0, 1):
        p = folder / "artifacts" / f"P_A1_S{seed}_selection.npz"
        checksum = sha(p)
        path = p.relative_to(ROOT).as_posix()
        assert checksum == expected[path]
        files[path] = checksum
        with np.load(p, allow_pickle=False) as pack:
            assert np.array_equal(pack["pool_ids"], raw["ids"])
            assert np.array_equal(pack["y"], raw["y"]) and np.array_equal(pack["day"], raw["day"])
            assert np.array_equal(pack["rx"], np.asarray(names)[raw["rx"]])
            assert np.isfinite(pack["logits"]).all() and pack["logits"].shape == (len(raw["y"]), 10)
            found[f"{seed}_A1_logits"] = pack["logits"]
    return found


@torch.inference_mode()
def execute(s29, s31, out):
    if any(out.iterdir()):
        raise FileExistsError("S33 requires an empty output directory")
    old = s29 / "raw_control"
    frozen = read(old / "FIT_SEAL.json")
    preserved = read(old / "PREDICTION_SEAL.json")
    s31_result = read(s31 / "SUMMARY.json")
    assert s31_result["decision"]["status"] == "STOP_SOURCE_GATE"
    write(out / "EXECUTION.json", dict(protocol_sha256=sha(HERE / "PROTOCOL.zh-CN.md"),
                                      code_sha256=sha(Path(__file__)), s29_fit_sha256=sha(old / "FIT_SEAL.json"),
                                      s31_summary_sha256=sha(s31 / "SUMMARY.json"), neural_fits=0,
                                      optimizer_updates=0, device=torch.cuda.get_device_name()))
    source = Source(out)
    leaf_digests, a1_files, shift_rows = {}, {}, []
    for view in VIEWS:
        cached = old / f"RAW_{view}.npz"
        assert sha(cached) == preserved[view]
        data = {role: source.load(view, role, "s33_frozen") for role in ROLES}
        held, fit = data["source_selection"], data["fit_train"]
        fit_stats = physical(tensor(fit["x"]))
        held_stats = physical(tensor(held["x"]))
        fields = deviations(held_stats, held, fit_reference(fit_stats, fit))
        payload = {}
        for role in ROLES[1:]:
            payload.update({f"{role}_{k}": data[role][k] for k in ("ids", "y", "rx", "day")})
        with np.load(cached, allow_pickle=False) as raw:
            assert all(np.array_equal(held[k], raw[k]) for k in ("ids", "y", "rx", "day"))
            payload.update(a1_predictions(view, raw, source.names, a1_files))
            for job in frozen["results"]:
                v, seed, arch, policy = job["job"]
                if v != view:
                    continue
                path = old / job["selected_path"]
                assert policy == "raw" and sha(path) == job["selected_sha256"]
                model = FeatureModel(arch, 24101 + seed)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
                model.to(DEVICE).eval()
                training = feature_pass(model, fit)
                reference = {layer: centers(training[layer], fit) for layer in LAYERS}
                del training
                for role in ROLES[1:]:
                    current = feature_pass(model, data[role])
                    here = f"{seed}_{arch}_{role}"
                    logits = current["logits"].cpu().numpy()
                    assert np.isfinite(logits).all()
                    if role == "source_selection":
                        frozen_logits = raw[f"{seed}_{arch}_native"]
                        assert np.array_equal(logits.argmax(1), frozen_logits.argmax(1))
                        logits = frozen_logits
                    payload[here + "_logits"] = logits
                    for layer in LAYERS:
                        feats = current[layer]
                        payload[f"{here}_{layer}_pred"] = (feats @ reference[layer].T).argmax(1).cpu().numpy().astype(np.uint8)
                        shift_rows.extend(dict(view=view, role=role, seed=seed, arch=arch, layer=layer, **r)
                                          for r in center_shift(feats, data[role], reference[layer], source.names))
                    del current
            filename = out / f"AUDIT_{view}.npz"
            np.savez_compressed(filename, **payload, **fields)
            leaf_digests[view] = sha(filename)
        print(view, "frozen probes and physical readouts saved", flush=True)
    write(out / "CENTROID_SHIFT.json", shift_rows)
    write(out / "RECEIPT.json", dict(packets=leaf_digests, strong_reference=a1_files,
                                     shift_sha256=sha(out / "CENTROID_SHIFT.json"),
                                     source_seed_held_argmax_checks=8, base_prediction_hashes=preserved))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("s29", "s31", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    execute(args.s29, args.s31, args.output)
