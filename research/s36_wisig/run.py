"""S36 A–D: replay layers, fit only probes, and seal held source decisions."""

import argparse
import math
from pathlib import Path

import numpy as np

from heads import LOCAL, calibrate, fuse, nonlinear, select_linear
from common import (DEVICE, ROLES, VIEWS, FeatureModel, FrozenStudy, featurize,
                    read, restrict_torch, sha, tensor, torch, write)
from source import ROOT

TAPS = ("token", "prepared", "pre_ln", "q", "k", "v", "attention", "residual", "mix1", "pooled")
ABLATIONS = ("self_attention", "no_attention", "skip_block_norm")
MODELS = ("HStem", "A1")


def qualified_models(study, view, frozen_files):
    for seed, arch, model in study.models(view):
        if arch == "HStem":
            yield seed, arch, model.requires_grad_(False)
    folder = ROOT / "research" / ("wisig_w4" if view == "V0" else "wisig_w5")
    manifest = {row["relative"]: row["sha256"] for row in read(folder / "REPLAY_FILES.json")["files"]}
    for seed in (0, 1):
        checkpoint = folder / "artifacts" / "review_checkpoints" / f"A1_S{seed}_downstream.pt"
        relative = checkpoint.relative_to(ROOT).as_posix()
        checksum = sha(checkpoint)
        assert manifest[relative] == checksum
        frozen_files[relative] = checksum
        model = FeatureModel("HStem", 24101 + seed)
        model.net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
        yield seed, "A1", model.to(DEVICE).eval().requires_grad_(False)


def taps(net, spec, interventions):
    first, second = net.encoder.blocks
    assert set(dict(first.named_children())) == {"qkv", "out", "norm"}
    tokens = net.frequency(spec)
    projected = net.encoder.prepare_tokens(tokens)
    x = net.pre_ln(projected)
    q, k, v = first.qkv(x).reshape(*x.shape[:2], 3, first.heads, first.inner // first.heads).unbind(2)
    q2, k2, v2 = [p.transpose(1, 2) for p in (q, k, v)]
    weights = (q2 @ k2.transpose(-2, -1) / math.sqrt(first.inner // first.heads)).softmax(-1)
    value = (weights @ v2).transpose(1, 2).reshape_as(x)
    att = first.out(value)
    residual = x + att
    after1 = first.norm(residual)

    def pool(h):
        return net.encoder.final_norm(second(h)).mean(1)

    pooled = pool(after1)
    out = dict(token=tokens.mean(1), prepared=projected.mean(1), pre_ln=x.mean(1),
               q=q.reshape_as(x).mean(1), k=k.reshape_as(x).mean(1), v=v.reshape_as(x).mean(1),
               attention=att.mean(1), residual=residual.mean(1), mix1=after1.mean(1),
               pooled=pooled, native=net.classifier(pooled))
    if interventions:
        states = dict(self_attention=first.norm(x + first.out(v.reshape_as(x))),
                      no_attention=first.norm(x), skip_block_norm=residual)
        out.update({key: net.classifier(pool(source)) for key, source in states.items()})
    return out


@torch.no_grad()
def capture(net, features, batch, interventions=False):
    arrays = {key: [] for key in (*TAPS, "native", *(ABLATIONS if interventions else ()))}
    for x in features.split(batch):
        for key, block in taps(net, x, interventions).items():
            arrays[key].append(block)
    return {k: torch.cat(batch) for k, batch in arrays.items()}


def permutations(data, number):
    rng = np.random.default_rng(36061 + number)
    result = {}
    for role in ROLES:
        rows = data[role]
        rx, day = rows["rx"], rows["day"]
        mapping = np.arange(len(rx))
        for r, d in np.unique(np.column_stack((rx, day)), axis=0):
            index = np.flatnonzero((rx == r) & (day == d))
            mapping[index] = rng.permutation(index)
        assert np.array_equal(rx[mapping], rx) and np.array_equal(day[mapping], day)
        result[role] = tensor(mapping)
    return result


def encode(captured):
    return {key: {role: torch.nn.functional.normalize(captured[role][key]) for role in ROLES}
            for key in TAPS}


def execute(s29, s33, s34, out):
    if any(out.iterdir()):
        raise FileExistsError("S36 A-D needs an empty execution directory")
    study = FrozenStudy(s29, s33, out)
    s34_receipt = read(s34 / "RECEIPT.json")
    write(out / "EXECUTION.json", dict(protocol=sha(LOCAL / "PROTOCOL.zh-CN.md"), engine=sha(Path(__file__)),
          probes=sha(LOCAL / "heads.py"), s33_receipt=sha(s33 / "RECEIPT.json"), s34_receipt=sha(s34 / "RECEIPT.json"),
          backbone_training_fits=0, backbone_updates=0, tapes=TAPS, views=VIEWS, gpu=torch.cuda.get_device_name()))
    from experiment import build_original
    ref = build_original("ManyRx", 24101)
    saved, fits, controls, hashes = {}, [], [], {}
    for vi, view in enumerate(VIEWS):
        data, audit = study.data(view, ROLES, "s36_neural")
        held = data["source_selection"]
        s34_p = s34 / f"PRED_{view}.npz"
        assert sha(s34_p) == s34_receipt["predictions"][view]
        payload = {k: held[k] for k in ("y", "rx", "day", "ids")}
        payload["repeat_loss"] = audit["repeat_loss"]
        permutation = permutations(data, vi)
        spectra = {role: featurize(ref, tensor(data[role]["x"])) for role in ROLES}
        with np.load(s34_p, allow_pickle=False) as old_score:
            assert np.array_equal(old_score["ids"], held["ids"])
            payload["quality_quartile"] = old_score["quality_quartile"]
            for seed, arch, model in qualified_models(study, view, saved):
                number = 128 if arch == "A1" else 256
                f = {role: capture(model.net, spectra[role], number, role == "source_selection") for role in ROLES}
                prior = audit[f"{seed}_A1_logits" if arch == "A1" else f"{seed}_HStem_source_selection_logits"]
                assert np.array_equal(f["source_selection"]["native"].argmax(1).cpu().numpy(), prior.argmax(1))
                prefix = f"{seed}_{arch}_"
                payload[prefix + "native"] = prior
                for ablation in ABLATIONS:
                    payload[prefix + ablation] = f["source_selection"][ablation].argmax(1).cpu().numpy().astype(np.uint8)
                values = encode(f)
                values["shuffled"] = {role: dat[permutation[role]] for role, dat in values["token"].items()}
                original = f["checkpoint_val"]["native"], f["source_selection"]["native"]
                calibrated = calibrate(*original, data["checkpoint_val"])
                linear = {}
                identical = {}
                for key, feature in values.items():
                    val, logits, item = select_linear(feature, data)
                    linear[key] = val, logits
                    predicted = logits.argmax(1).cpu().numpy().astype(np.uint8)
                    if key in ("token", "mix1"):
                        payload[prefix + f"{key}_linear_logits"] = logits.cpu().numpy()
                        if arch == "HStem":
                            saved_s34 = old_score[f"{seed}_HStem_{key}_mean_ridge"]
                            differences = int((saved_s34 != predicted).sum())
                            assert differences <= 2, (view, seed, key, differences)
                            identical[key] = differences
                            # The historical transition uses exactly the previously sealed decisions.
                            predicted = saved_s34
                    payload[prefix + f"{key}_linear"] = predicted
                    fits.append(dict(view=view, arch=arch, seed=seed, tap=key, **item))
                for mode, key in (("early_fusion", "token"), ("deep_double", "pooled"), ("shuffle_fusion", "shuffled")):
                    logits, record = fuse(original, linear[key], data["checkpoint_val"], calibrated)
                    if record["alpha"] == 0.:
                        assert np.array_equal(logits.argmax(1).cpu().numpy(), prior.argmax(1))
                    payload[prefix + mode] = logits.cpu().numpy()
                    controls.append(dict(view=view, arch=arch, seed=seed, mode=mode, **record))
                nl, history = nonlinear({k: values[k] for k in TAPS}, data, 36001 + 100*vi + seed)
                for key, out_pred in nl.items():
                    payload[prefix + f"{key}_nonlinear"] = out_pred
                write(out / f"NONLINEAR_{view}_{seed}_{arch}.json", dict(**history, historical_linear_diff=identical))
                assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
                print(view, seed, arch, "probes and validation-only fusion complete", flush=True)
            for seed in (0, 1):
                payload[f"{seed}_LongPatch_native"] = audit[f"{seed}_LongPatch_source_selection_logits"]
        filename = out / f"PRED_{view}.npz"
        np.savez_compressed(filename, **payload)
        hashes[view] = sha(filename)
        del spectra
    write(out / "CALIBRATION.json", controls)
    write(out / "LINEAR.json", fits)
    write(out / "RECEIPT.json", dict(predictions=hashes, a1_checkpoints=saved, original_heads_replayed=8,
          calibrations=sha(out / "CALIBRATION.json"), fits=sha(out / "LINEAR.json"),
          nonlinear={p.name: sha(p) for p in sorted(out.glob("NONLINEAR_*.json"))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("s29", "s33", "s34", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    restrict_torch()
    execute(args.s29, args.s33, args.s34, args.output)
