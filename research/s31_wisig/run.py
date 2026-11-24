"""Fit a single online IQ CFO treatment and read frozen source endpoints."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s29_wisig"))
from experiment import (ARCHITECTURES, DEVICE, SEEDS, VIEWS, FeatureModel, build_original,
                        featurize, fit_one, predict, tensor, torch)
from physics import FS, complex_iq, rotate
from source import Source, array_sha, read, sha, write

BOUND_HZ = 2500
PROBABILITY = .5
CONDITIONS = {"native": 0, **{f"center_{sign * hz}_0": sign * hz
                            for hz in (1000, 2000, 2500, 5000) for sign in (-1, 1)}}


def center(x, hz):
    hz = torch.as_tensor(hz, dtype=x.dtype, device=x.device)
    centered_samples = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device) - 127.5
    angles = hz[..., None] * (2 * math.pi / FS) * centered_samples
    z = complex_iq(x) * torch.polar(torch.ones_like(angles), angles)
    return torch.stack((z.real, z.imag), 1)


class OnlineFeatures:
    """Called once by the unchanged S29 loop for each fit batch."""
    def __init__(self, raw, original_features, reference, seed):
        self.raw = raw
        self.original_features = original_features
        self.reference = reference
        self.rng = torch.Generator(device=DEVICE).manual_seed(seed)
        self.drawn = []

    @torch.no_grad()
    def __getitem__(self, indices):
        sampled = torch.rand((2, len(indices)), device=DEVICE, generator=self.rng)
        mask = sampled[0] < PROBABILITY
        effective_hz = torch.where(mask, (2 * sampled[1] - 1) * BOUND_HZ, 0)
        result = featurize(self.reference, center(self.raw[indices], effective_hz))
        self.drawn.append(effective_hz)
        return torch.where(mask[:, None, None], result, self.original_features[indices])

    def receipt(self):
        doses = torch.cat(self.drawn).cpu().numpy()
        selected = doses[doses != 0]
        assert np.isfinite(doses).all() and np.abs(doses).max() <= BOUND_HZ
        assert .49 < len(selected) / len(doses) < .51
        return dict(steps=len(self.drawn), packet_accesses=len(doses), nonzero_draws=len(selected),
                    active_fraction=len(selected) / len(doses), min_hz=float(selected.min()),
                    max_hz=float(selected.max()), mean_hz=float(selected.mean()),
                    draw_sha256=array_sha(doses)), doses


@torch.no_grad()
def engineering(model, device):
    cpu_rng = torch.Generator(device="cpu").manual_seed(31)
    x = torch.randn((128, 2, 256), generator=cpu_rng).to(device)
    doses = torch.linspace(-2500, 2500, len(x), device=device)
    y = center(x, doses)
    t = torch.arange(256, device=device, dtype=torch.float64)
    z = complex_iq(x).to(torch.complex128) * torch.exp(
        1j * (2 * math.pi / FS) * doses.double()[:, None] * (t - 127.5))
    diff = float((complex_iq(y) - z).abs().max())
    power = float((x.square().sum(1) - y.square().sum(1)).abs().max())
    inverse = float((center(y, -doses) - x).abs().max())
    assert diff < 1e-6 and inverse < 2e-6 and power < 8e-6
    assert torch.equal(center(x, 0), x)
    raw = featurize(model, x)
    a, b = (OnlineFeatures(x, raw, model, 31000) for _ in range(2))
    subset = torch.arange(len(x), device=device)
    ya, yb = a[subset], b[subset]
    first, second = a.drawn[0], b.drawn[0]
    assert torch.equal(first, second) and torch.equal(ya, yb)
    assert torch.equal(ya[first == 0], raw[first == 0])
    assert not torch.equal(ya[first != 0], raw[first != 0])
    return dict(complex128_max_abs=diff, inverse_max_abs=inverse, power_max_abs=power,
                untreated_features_exact=True, paired_draws_exact=True)


def register(parent, s30, out):
    s29 = parent / "raw_control"
    s29_fit = s29 / "FIT_SEAL.json"
    p29, p30 = read(s29 / "PREDICTION_SEAL.json"), read(s30 / "RECEIPT.json")
    s30_summary = read(s30 / "SUMMARY.json")
    assert s30_summary["decision"]["cfo_candidate"] and read(s30 / "QA.json")["status"] == "PASS"
    for v in VIEWS:
        assert sha(s29 / f"RAW_{v}.npz") == p29[v]
        assert sha(s30 / f"PACKETS_{v}.npz") == p30["hashes"][v]
    base = {(v, seed, arch): row for row in read(s29_fit)["results"]
            for v, seed, arch, policy in [row["job"]] if policy == "raw"}
    assert len(base) == 8
    for row in base.values():
        assert row["epochs"] == 40 and row["updates"] == 5080
        assert sha(s29 / row["selected_path"]) == row["selected_sha256"]
    write(out / "EXECUTION.json", dict(protocol_sha256=sha(HERE / "PROTOCOL.zh-CN.md"),
                                      script_sha256=sha(Path(__file__)), parent_fit_sha256=sha(s29_fit),
                                      base_prediction_hashes=p29, dose_prediction_hashes=p30["hashes"],
                                      s30_summary_sha256=sha(s30 / "SUMMARY.json"),
                                      engineering=engineering(build_original("ManyRx", 24101), DEVICE),
                                      gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                      batch=128, epochs=40, probability=PROBABILITY, limit_hz=BOUND_HZ))
    return base


def matched_history(record, prior):
    assert record["initial_sha256"] == prior["initial_sha256"]
    assert record["head_sha256"] == prior["head_sha256"]
    assert record["epochs"] == prior["epochs"] and record["updates"] == prior["updates"]
    assert [a["order_sha256"] for a in record["history"]] == [a["order_sha256"] for a in prior["history"]]


def fit(base, out):
    if (out / "FIT_SEAL.json").exists():
        raise FileExistsError("S31 fits have already been sealed")
    source, records = Source(out), []
    for index, view in enumerate(VIEWS):
        train = source.load(view, "fit_train", "fit")
        val = source.load(view, "checkpoint_val", "fit")
        raw = tensor(train["x"])
        reference = build_original("ManyRx", 24101)
        original, validation = [featurize(reference, x) for x in (raw, tensor(val["x"]))]
        for seed in SEEDS:
            pairs = []
            schedule_path = out / f"DOSES_{view}_{seed}.npz"
            draw_seed = 31201 + 100 * index + seed
            for arch in ARCHITECTURES:
                job = (view, seed, arch, "cfo25")
                folder = out / "_".join(map(str, job))
                completed = folder / "COMPLETE.json"
                audit = folder / "AUGMENTATION.json"
                if completed.exists() and audit.exists():
                    result, augmentation = read(completed), read(audit)
                else:
                    aug = OnlineFeatures(raw, original, reference, draw_seed)
                    result = fit_one(out, job, aug, validation, train, val)
                    augmentation, doses = aug.receipt()
                    assert augmentation["steps"] == result["updates"]
                    assert augmentation["packet_accesses"] == len(train["y"]) * result["epochs"]
                    if schedule_path.exists():
                        with np.load(schedule_path, allow_pickle=False) as schedule:
                            assert np.array_equal(doses, schedule["effective_hz"])
                    else:
                        np.savez_compressed(schedule_path, effective_hz=doses)
                    write(audit, augmentation)
                    print("dose audit", job, augmentation["active_fraction"], flush=True)
                matched_history(result, base[(view, seed, arch)])
                assert sha(out / result["selected_path"]) == result["selected_sha256"]
                entry = dict(**result, augmentation=augmentation)
                pairs.append(entry)
                records.append(entry)
            assert pairs[0]["augmentation"] == pairs[1]["augmentation"]
            assert pairs[0]["head_sha256"] == pairs[1]["head_sha256"]
        del raw, original, validation
    write(out / "FIT_SEAL.json", dict(results=records, total_updates=sum(r["updates"] for r in records),
                                     dose_schedule_hashes={p.name: sha(p) for p in sorted(out.glob("DOSES*.npz"))}))
    print("S31 fits sealed", len(records), flush=True)


@torch.inference_mode()
def evaluate(parent, out):
    seal = read(out / "FIT_SEAL.json")
    assert len(seal["results"]) == 8 and seal["total_updates"] == 40640
    source, hashes = Source(out), {}
    for view in VIEWS:
        data = source.load(view, "source_selection", "new_prediction")
        payload = {k: data[k] for k in ("ids", "y", "rx", "day")}
        prior = parent / "raw_control" / f"RAW_{view}.npz"
        with np.load(prior, allow_pickle=False) as base:
            assert all(np.array_equal(base[k], value) for k, value in payload.items())
        models = {}
        for row in seal["results"]:
            v, seed, arch, _ = row["job"]
            if v == view:
                path = out / row["selected_path"]
                assert sha(path) == row["selected_sha256"]
                model = FeatureModel(arch, 24101 + seed)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
                assert sum(p.numel() for p in model.parameters()) == {"HStem": 82426, "LongPatch": 82364}[arch]
                models[seed, arch] = model.to(DEVICE).eval()
        reference = next(iter(models.values())).net
        x = tensor(data["x"])
        for condition, f in CONDITIONS.items():
            # S30 uses two composed rotations. Reproduce its implementation exactly at evaluation.
            if f:
                radians = -2 * math.pi * 127.5 * (f / FS)
                i, q = rotate(x, f).unbind(1)
                c, s = math.cos(radians), math.sin(radians)
                iq = torch.stack((c * i - s * q, s * i + c * q), 1)
            else:
                iq = x
            features = featurize(reference, iq)
            for (seed, arch), model in models.items():
                logits = predict(model, features).cpu().numpy()
                assert np.isfinite(logits).all()
                payload[f"{seed}_{arch}_{condition}"] = logits
        destination = out / f"S31_{view}.npz"
        np.savez_compressed(destination, **payload)
        hashes[view] = sha(destination)
    write(out / "PREDICTION_SEAL.json", hashes)
    print("S31 predictions sealed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("s29", "s30", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(exist_ok=True)
    base = register(args.s29, args.s30, args.output)
    fit(base, args.output)
    evaluate(args.s29, args.output)


if __name__ == "__main__":
    main()
