"""S29 engine. Run prepare, fit, then evaluate in the same output directory."""

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from physics import RAD_TO_HZ, characterize, correct, estimate, matched_time_shift, rotate
from source import C, HERE, ROOT, Source, array_sha, read, sha, write

for relative in ("lib", "13-mixer-cfo-mechanisms", "17-rx-distortion"):
    sys.path.insert(0, str(ROOT / relative))
from s17_models import build as build_original

DEVICE = torch.device("cuda")
ARCHITECTURES = ("HStem", "LongPatch")
POLICIES = ("raw", "cfo")
VIEWS = ("V0", "V1")
EPOCHS = 40
BATCH = 128
SEEDS = (0, 1)
DIAGNOSTIC_CFO = (-10_000, -2_500, 2_500, 10_000)


class LongPatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Conv1d(5, 64, 32, stride=8, padding=12),
                                     nn.BatchNorm1d(64), nn.GELU())
        self.per_token = nn.Sequential(nn.Conv1d(64, 242, 1), nn.GELU(),
                                       nn.Conv1d(242, 64, 1))

    def forward(self, x):
        token = self.project(x)
        return (token + self.per_token(token)).transpose(1, 2)


class FeatureModel(nn.Module):
    def __init__(self, architecture, seed):
        super().__init__()
        self.net = build_original("ManyRx", seed)
        if architecture == "LongPatch":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + 783)
                self.net.frequency = LongPatch()

    def forward(self, features):
        net = self.net
        h = net.pre_ln(net.encoder.prepare_tokens(net.frequency(features)))
        for block in net.encoder.blocks:
            h = block(h)
        return net.classifier(net.encoder.final_norm(h).mean(1))


def tensor(x):
    return torch.as_tensor(x, device=DEVICE)


def predict(model, features):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(batch) for batch in features.split(256)])


@torch.no_grad()
def featurize(model, x):
    # WiSigModel.tokens applies RMS before calling .features, which applies RMS again.
    from xa.frontends import rms_normalize
    return torch.cat([model.features(rms_normalize(part)) for part in x.split(512)])


def relative_shift(perturbed, cfo_anchor):
    hz, reliable, *_ = estimate(perturbed)
    baseline, baseline_reliable = cfo_anchor
    angle = (hz - baseline) / RAD_TO_HZ
    delta = torch.atan2(angle.sin(), angle.cos()) * RAD_TO_HZ
    return delta, reliable & baseline_reliable


def frozen_models(view):
    folder = ROOT / "research" / {"V0": "wisig_w4", "V1": "wisig_w5"}[view]
    manifest = read(folder / "REPLAY_FILES.json")
    entries = {entry["relative"]: entry["sha256"]
               for items in manifest.values() if isinstance(items, list)
               for entry in items if isinstance(entry, dict) and "relative" in entry}
    result = []
    for seed in (0, 1, 2):
        path = folder / "artifacts" / "review_checkpoints" / f"A0_S{seed}.pt"
        checksum = sha(path)
        if checksum != entries[path.relative_to(ROOT).as_posix()]:
            raise ValueError("Frozen A0 checkpoint checksum changed")
        wrapped = FeatureModel("HStem", 24101 + seed)
        state = torch.load(path, weights_only=True, map_location="cpu")
        wrapped.net.load_state_dict(state["model"])
        result.append((wrapped.to(DEVICE).eval(), checksum))
    return result


def audit(out, source):
    summaries = {}
    for view in VIEWS:
        data = source.load(view, "fit_train", "audit")
        metrics = {name: value.cpu().numpy() for name, value in characterize(tensor(data["x"])).items()}
        np.savez_compressed(out / f"AUDIT_{view}.npz", **metrics,
                            **{k: data[k] for k in ("y", "rx", "day", "ids")})
        summaries[view] = dict(rows=len(data["y"]), coverage=float(metrics["reliable"].mean()),
                               quantiles={name: np.quantile(vals, [.1, .5, .9]).tolist()
                                          for name, vals in metrics.items() if name != "reliable"})
    write(out / "AUDIT.json", summaries)
    return summaries


def engineering():
    torch.manual_seed(29)
    short = torch.randn(12, 2, 20, device=DEVICE)
    a = short.repeat(1, 1, 13)[..., :256]
    x = rotate(a, -13_000)
    hz, reliable, *_ = estimate(x)
    assert bool(reliable.all()) and float((hz + 13_000).abs().max()) < .05
    assert torch.allclose(correct(x), a, atol=2e-6, rtol=1e-6)
    for offset in (-9, 9):
        aligned, shifted = matched_time_shift(x, offset)
        assert torch.equal(shifted.roll(-offset, -1), aligned)
        assert torch.equal(aligned.abs().sort(-1).values, shifted.abs().sort(-1).values)
    result = {}
    for architecture in ARCHITECTURES:
        m = FeatureModel(architecture, 24101).to(DEVICE)
        feat = featurize(m.net, x)
        assert m.net.frequency(feat).shape == (12, 32, 64)
        if architecture == "HStem":
            m.eval()
            with torch.no_grad():
                assert torch.equal(m(feat), m.net(x))
        m.train()
        loss = F.cross_entropy(m(feat), torch.arange(12, device=DEVICE) % 10)
        loss.backward()
        assert all(p.grad is not None and bool(p.grad.isfinite().all()) for p in m.parameters())
        result[architecture] = sum(p.numel() for p in m.parameters())
    assert result == {"HStem": 82426, "LongPatch": 82364}, result
    return result


def diagnostic(out, source, view):
    data = source.load(view, "source_selection", "frozen_diagnostic")
    x = tensor(data["x"])
    models = frozen_models(view)
    raw_hz, reliable, *_ = estimate(x)
    results, inject_stats = {}, {}
    oracle_max = 0.

    def measure(name, iq):
        features = featurize(models[0][0].net, iq)
        results[name] = np.stack([predict(m, features).argmax(dim=1).cpu().numpy()
                                  for m, _ in models]).astype(np.uint8)

    measure("native", x)
    measure("native_cfo", correct(x))
    for hz in DIAGNOSTIC_CFO:
        shifted = rotate(x, hz)
        delta, both_reliable = relative_shift(shifted, (raw_hz, reliable))
        err = (delta[both_reliable] - hz).abs().cpu().numpy()
        if len(err) == 0:
            raise ValueError("No reliable paired CFO anchors")
        oracle = rotate(shifted, -hz)
        oracle_max = max(oracle_max, float((oracle - x).abs().max()))
        inject_stats[str(hz)] = dict(rows=len(err), abs_error_p99_hz=float(np.quantile(err, .99)),
                                     abs_error_median_hz=float(np.median(err)),
                                     gate_changed=int((estimate(shifted)[1] != reliable).sum()))
        measure(f"cfo_{hz}", shifted)
        measure(f"cfo_{hz}_corrected", correct(shifted))
        measure(f"cfo_{hz}_oracle", oracle)
        measure(f"cfo_{hz}_anchored", rotate(shifted, -torch.where(both_reliable, delta, 0)))
    for offset in (-9, -3, 3, 9):
        aligned, shifted = matched_time_shift(x, offset)
        measure(f"time_{offset}_circular", x.roll(offset, -1))
        measure(f"time_{offset}_aligned", aligned)
        measure(f"time_{offset}_shifted", shifted)
    np.savez_compressed(out / f"FROZEN_{view}.npz", **results, reliable=reliable.cpu().numpy(),
                        **{k: data[k] for k in ("y", "rx", "day", "ids")})

    def acc(name):
        hit = results[name] == data["y"][None]
        return 100 * float(np.mean([hit[:, data["rx"] == r].mean() for r in np.unique(data["rx"])]))

    return dict(checkpoints_sha256=[h for _, h in models], accuracies={r: acc(r) for r in results},
                injected_10k_loss_pp=acc("native") - (acc("cfo_-10000") + acc("cfo_10000")) / 2,
                reliable_fraction=float(reliable.float().mean()), oracle_iq_max_abs=oracle_max,
                anchor_estimation=inject_stats)


def prepare(out):
    if any(out.iterdir()):
        raise FileExistsError(f"Prepare needs an empty directory: {out}")
    engineering_result = engineering()
    write(out / "EXECUTION.json", dict(protocol_sha256=sha(HERE / "PROTOCOL.zh-CN.md"),
          scripts={p.name: sha(p) for p in HERE.glob("*.py")}, parameters=engineering_result,
          torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
          cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(),
          epochs=EPOCHS, batch=BATCH, seeds=SEEDS,
          original_source_plan_sha256=sha(C / "SOURCE_PLAN.json"),
          original_source_registration_sha256=sha(C / "PRE_REGISTRATION.json")))
    source = Source(out)
    summaries = audit(out, source)
    frozen = {view: diagnostic(out, source, view) for view in VIEWS}
    write(out / "FROZEN.json", frozen)
    gates = dict(coverage=all(s["coverage"] > .5 for s in summaries.values()),
                 sensitivity=float(np.mean([d["injected_10k_loss_pp"] for d in frozen.values()])) >= 2.,
                 oracle=max(d["oracle_iq_max_abs"] for d in frozen.values()) < 2e-6,
                 relative_recovery=max(s["abs_error_p99_hz"] for d in frozen.values()
                                       for s in d["anchor_estimation"].values()) < 10.)
    write(out / "PREFLIGHT.json", dict(checks=gates, run_factorial=all(gates.values())))
    print("S29-0/1 complete", gates, flush=True)


def model_hash(model, head_only=False):
    from hashlib import sha256
    digest = sha256()
    for name, value in model.state_dict().items():
        if not head_only or not name.startswith("net.frequency."):
            digest.update(name.encode())
            digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def fit_one(out, job, train_x, val_x, train, val):
    view, seed, architecture, arm = job
    start = time.perf_counter()
    model = FeatureModel(architecture, 24101 + seed).to(DEVICE)
    initial, head = model_hash(model), model_hash(model, head_only=True)
    torch.manual_seed(24301)
    order = torch.Generator(device=DEVICE).manual_seed(24201 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.1, patience=5, min_lr=1e-6)
    y, vy, vr = (tensor(a).long() for a in (train["y"], val["y"], val["rx"]))
    groups = [vr == r for r in np.unique(val["rx"])]
    work = out / ("_".join(map(str, job)))
    work.mkdir(exist_ok=True)
    checkpoint = work / "selected.pt"
    best, history = math.inf, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        indices = torch.randperm(len(y), generator=order, device=DEVICE)
        order_sha = array_sha(train["ids"][indices.cpu().numpy()])
        total_loss = torch.zeros((), device=DEVICE)
        for batch in indices.split(BATCH):
            loss = F.cross_entropy(model(train_x[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(batch)
        lr = optimizer.param_groups[0]["lr"]
        val_logits = predict(model, val_x)
        per_sample = F.cross_entropy(val_logits, vy, reduction="none")
        ce = float(torch.stack([per_sample[group].mean() for group in groups]).mean())
        train_ce = float(total_loss / len(y))
        if not math.isfinite(ce) or not math.isfinite(train_ce):
            raise ValueError(f"Nonfinite loss in {job} epoch {epoch}")
        scheduler.step(ce)
        if ce < best:
            best = ce
            temp = checkpoint.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(), epoch=epoch, val_ce=ce), temp)
            temp.replace(checkpoint)
        history.append(dict(epoch=epoch, val_ce=ce, train_ce=train_ce, learning_rate=lr,
                            order_sha256=order_sha))
        if epoch % 5 == 0:
            print("fit", job, epoch, round(ce, 5), flush=True)
    result = dict(job=job, selected_path=str(checkpoint.relative_to(out)),
                  selected_sha256=sha(checkpoint), initial_sha256=initial, head_sha256=head,
                  epochs=EPOCHS, updates=EPOCHS * math.ceil(len(y) / BATCH),
                  seconds=time.perf_counter() - start, history=history)
    write(work / "COMPLETE.json", result)
    return result


def check_plan(out):
    execution = read(out / "EXECUTION.json")
    if execution["protocol_sha256"] != sha(HERE / "PROTOCOL.zh-CN.md"):
        raise ValueError("Protocol checksum changed")
    for name, digest in execution["scripts"].items():
        if digest != sha(HERE / name):
            raise ValueError(f"Frozen script checksum changed for {name}")
    return read(out / "PREFLIGHT.json")["run_factorial"]


def fit(out):
    if not check_plan(out):
        print("No factorial fits: preflight criteria were not met", flush=True)
        return
    if (out / "FIT_SEAL.json").exists():
        raise FileExistsError("All fitting already sealed")
    source, results = Source(out), []
    for view in VIEWS:
        train = source.load(view, "fit_train", "fit")
        val = source.load(view, "checkpoint_val", "fit")
        features = {}
        reference = build_original("ManyRx", 24101)
        for arm in POLICIES:
            mapped = [tensor(item["x"]) for item in (train, val)]
            if arm == "cfo":
                mapped = [correct(x) for x in mapped]
            features[arm] = [featurize(reference, x) for x in mapped]
        del mapped, reference
        for seed in SEEDS:
            paired = []
            for arch in ARCHITECTURES:
                for arm in POLICIES:
                    job = (view, seed, arch, arm)
                    done = out / "_".join(map(str, job)) / "COMPLETE.json"
                    result = read(done) if done.exists() else fit_one(
                        out, job, *features[arm], train, val)
                    assert sha(out / result["selected_path"]) == result["selected_sha256"]
                    paired.append(result)
            assert len({record["head_sha256"] for record in paired}) == 1
            for i in (0, 2):
                assert paired[i]["initial_sha256"] == paired[i + 1]["initial_sha256"]
            assert len({tuple(e["order_sha256"] for e in r["history"]) for r in paired}) == 1
            results.extend(paired)
        del features
    write(out / "FIT_SEAL.json", dict(results=results, total_updates=sum(r["updates"] for r in results)))
    print("S29-2 fits sealed", len(results), flush=True)


def evaluate(out):
    if not check_plan(out):
        return
    seal = read(out / "FIT_SEAL.json")
    assert len(seal["results"]) == 16
    for result in seal["results"]:
        assert result["selected_sha256"] == sha(out / result["selected_path"])
    source, predictions = Source(out), {}
    for view in VIEWS:
        data = source.load(view, "source_selection", "new_prediction")
        x = tensor(data["x"])
        # Every classified input uses exactly the policy seen in fitting.
        reference = build_original("ManyRx", 24101)
        for arm in POLICIES:
            transforms = (("native", x), ("minus10k", rotate(x, -10_000)),
                          ("plus10k", rotate(x, 10_000)))
            for perturbation, iq in transforms:
                feat = featurize(reference, correct(iq) if arm == "cfo" else iq)
                for job in seal["results"]:
                    v, seed, arch, policy = job["job"]
                    if (v, policy) != (view, arm):
                        continue
                    model = FeatureModel(arch, 24101 + seed)
                    state = torch.load(out / job["selected_path"], weights_only=True, map_location="cpu")
                    model.load_state_dict(state["model"])
                    logits = predict(model.to(DEVICE), feat).cpu().numpy()
                    assert np.isfinite(logits).all()
                    predictions[f"{seed}_{arch}_{arm}_{perturbation}"] = logits
        prediction_file = out / f"PILOT_{view}.npz"
        np.savez_compressed(prediction_file, **predictions,
                            **{k: data[k] for k in ("ids", "y", "rx", "day")})
        predictions.clear()
    write(out / "PREDICTION_SEAL.json", {v: sha(out / f"PILOT_{v}.npz") for v in VIEWS})
    print("S29-2 predictions sealed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "fit", "evaluate", "all"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = (prepare, fit, evaluate) if args.stage == "all" else (
        {"prepare": prepare, "fit": fit, "evaluate": evaluate}[args.stage],)
    for stage in stages:
        stage(out)


if __name__ == "__main__":
    main()
