"""Frozen S29 inference under signed CFO, time, and phase controls."""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s29_wisig"))
from experiment import DEVICE, FeatureModel, featurize, predict, tensor, torch
from physics import FS, matched_time_shift, rotate
from source import Source, read, sha, write

VIEWS = ("V0", "V1")
FREQ = (1000, 2000, 2500, 5000, 10000, 15000, 20000)
TIME = (1, 2, 4, 8, 9, 16)
ARCHES = ("HStem", "LongPatch")
MIDPOINT = 127.5


def signed(values):
    return tuple(sign * v for v in values for sign in (-1, 1))


def key(kind, f=0, d=0):
    return f"{kind}_{f}_{d}"


def phase_only(x, hz):
    angle = 2 * math.pi * MIDPOINT * (hz / FS)
    i, q = x.unbind(1)
    c, s = math.cos(angle), math.sin(angle)
    return torch.stack((c * i - s * q, s * i + c * q), 1)


def center_rotate(x, hz):
    return phase_only(rotate(x, hz), -hz)


def variants(x):
    yield key("native"), x, None, True
    for value in signed(FREQ):
        yield key("edge", value), rotate(x, value), None, False
        yield key("phase", value), phase_only(x, value), None, False
        capture = abs(value) in (5000, 10000)
        yield key("center", value), center_rotate(x, value), x if capture else None, capture
    for d in signed(TIME):
        aligned, shifted = matched_time_shift(x, d)
        capture = abs(d) in (4, 9)
        yield key("aligned", d=d), aligned, None, capture
        yield key("shifted", d=d), shifted, aligned if capture else None, capture
        yield key("cyclic", d=d), x.roll(d, -1), None, False
        if capture:
            for value in signed((2000, 5000, 10000)):
                yield key("joint", value, d), center_rotate(shifted, value), None, False
                yield key("joint_ref", value, d), center_rotate(aligned, value), None, False
    for value in signed((10000,)):
        yield key("recover", value), center_rotate(center_rotate(x, value), -value), None, False


def stages(model, feat, architecture):
    net = model.net
    stem = net.frequency
    result = {}
    if architecture == "HStem":
        z = feat
        blocks = list(stem.stem) + list(stem.tokenize)
        for j in range(0, len(blocks), 3):
            for layer in blocks[j:j + 3]:
                z = layer(z)
            if j < len(blocks) - 3:
                result[f"hstem_conv{j // 3 + 1}"] = z.transpose(1, 2)
        tokens = z.transpose(1, 2)
    else:
        tokens = stem(feat)
    h = net.pre_ln(net.encoder.prepare_tokens(tokens))
    for block in net.encoder.blocks:
        h = block(h)
    pooled = net.encoder.final_norm(h).mean(1)
    logits = net.classifier(pooled)
    result.update(token=tokens, pooled=pooled, logits=logits - logits.mean(1, keepdim=True))
    return logits, result


class LayerStats:
    def __init__(self, groups, y, receiver):
        self.groups = groups
        self.y = y
        self.receiver = receiver
        self.num_rx = int(receiver.max()) + 1
        self.counts = torch.bincount(groups, minlength=self.num_rx * 10).cpu().numpy()
        self.rows = {}
        self.margin = []
        self.margin_delta = []

    def update(self, observed, reference, logits, ref_logits, lo, hi):
        g = self.groups[lo:hi]
        rx = self.receiver[lo:hi]
        for name, values in observed.items():
            pooled = values.mean(1) if values.ndim == 3 else values
            if name not in self.rows:
                opts = dict(device=values.device, dtype=torch.float64)
                self.rows[name] = [torch.zeros((self.num_rx * 10, pooled.shape[-1]), **opts),
                                   torch.zeros(self.num_rx * 10, **opts),
                                   torch.zeros((3, self.num_rx), **opts)]
            sums, squared, drift = self.rows[name]
            sums.index_add_(0, g, pooled.double())
            squared.index_add_(0, g, pooled.square().sum(1).double())
            if reference is not None:
                base = reference[name].flatten(1)
                diff = (values.flatten(1) - base).norm(dim=1)
                scale = base.norm(dim=1)
                for target, array in zip(drift, (diff / scale.clamp_min(1e-8),
                                                diff.square(), scale.square())):
                    target.index_add_(0, rx, array.double())
        y = self.y[lo:hi]
        self.margin.append(margin(logits, y).cpu().numpy())
        if ref_logits is not None:
            self.margin_delta.append((margin(logits, y) - margin(ref_logits, y)).cpu().numpy())

    def finalize(self, names):
        rx = self.receiver.cpu().numpy()
        margin_values = np.concatenate(self.margin)
        deltas = np.concatenate(self.margin_delta) if self.margin_delta else None
        output = []
        for layer, tensors in self.rows.items():
            sums, squared, drift = [item.cpu().numpy() for item in tensors]
            for i, name in enumerate(names):
                sl = slice(i * 10, (i + 1) * 10)
                count = self.counts[sl]
                keep = count >= 20
                center = sums[sl][keep] / count[keep, None]
                between = ((center - center.mean(0)) ** 2).sum(1).mean()
                within = ((squared[sl][keep] / count[keep]) - (center ** 2).sum(1)).mean()
                assert keep.sum() >= 2 and within >= 0
                size = int((rx == i).sum())
                rel, ss_delta, ss_ref = drift[:, i]
                output.append(dict(layer=layer, rx=name, supported_tx=int(keep.sum()), rows=size,
                                   tx_separation=float(between / max(within, 1e-12)),
                                   rel_l2=float(rel / size) if deltas is not None else None,
                                   rms_l2=float(math.sqrt(ss_delta / max(ss_ref, 1e-12))) if deltas is not None else None,
                                   margin=float(margin_values[rx == i].mean()),
                                   margin_change=float(deltas[rx == i].mean()) if deltas is not None else None))
        return output


def margin(logits, y):
    best_other = logits.scatter(1, y[:, None], -torch.inf).amax(1)
    return logits.gather(1, y[:, None])[:, 0] - best_other


def capture(model, feat, ref_feat, arch, data, names):
    labels = tensor(data["y"]).long()
    which = tensor(np.searchsorted(np.unique(data["rx"]), data["rx"])).long()
    stats = LayerStats(10 * which + labels, labels, which)
    output = []
    for lo in range(0, len(feat), 256):
        hi = lo + 256
        logits, layer = stages(model, feat[lo:hi], arch)
        ref_logits, ref = (None, None) if ref_feat is None else stages(model, ref_feat[lo:hi], arch)
        stats.update(layer, ref, logits, ref_logits, lo, hi)
        output.append(logits)
    return torch.cat(output), stats.finalize(names)


def engineering(x):
    errors = {}
    for f in signed(FREQ):
        centered = center_rotate(x, f)
        errors[str(f)] = dict(decomposition_error=float((phase_only(centered, f) - rotate(x, f)).abs().max()),
                              reversible_error=float((center_rotate(centered, -f) - x).abs().max()))
    assert max(float(v) for row in errors.values() for v in row.values()) < 2e-6
    for d in signed(TIME):
        aligned, shifted = matched_time_shift(x, d)
        assert torch.equal(shifted.roll(-d, -1), aligned)
    return errors


@torch.inference_mode()
def run(parent, out):
    out.mkdir(exist_ok=True)
    seal_file = parent / "raw_control" / "FIT_SEAL.json"
    seal = read(seal_file)
    pseal = read(parent / "raw_control" / "PREDICTION_SEAL.json")
    write(out / "EXECUTION.json", dict(protocol_sha256=sha(HERE / "PROTOCOL.zh-CN.md"),
                                      code_sha256=sha(Path(__file__)), parent_fit_sha256=sha(seal_file),
                                      gradient_updates=0, fp32=True, device=torch.cuda.get_device_name()))
    source = Source(out)
    all_layers, verification = [], {}
    for view in VIEWS:
        output = out / f"PACKETS_{view}.npz"
        if output.exists():
            raise FileExistsError(output)
        data = source.load(view, "source_selection", "s30_frozen")
        models = {}
        for row in seal["results"]:
            v, seed, arch, _ = row["job"]
            if v != view:
                continue
            p = seal_file.parent / row["selected_path"]
            assert sha(p) == row["selected_sha256"]
            model = FeatureModel(arch, 24101 + seed)
            model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True)["model"])
            models[(seed, arch)] = model.to(DEVICE).eval()
        x = tensor(data["x"])
        names = [source.names[i] for i in np.unique(data["rx"])]
        verification[view] = dict(iq=engineering(x), prediction_disagreements={})
        old = seal_file.parent / f"RAW_{view}.npz"
        assert sha(old) == pseal[view]
        payload = {k: data[k] for k in ("y", "rx", "day", "ids")}
        with np.load(old, allow_pickle=False) as prior:
            assert all(np.array_equal(prior[k], data[k]) for k in payload)
            native = {}
            prototype = next(iter(models.values())).net
            for condition, iq, ref, capture_layers in variants(x):
                feats = featurize(prototype, iq)
                ref_feats = featurize(prototype, ref) if ref is not None else None
                for (seed, arch), model in models.items():
                    if capture_layers:
                        logits, rows = capture(model, feats, ref_feats, arch, data, names)
                        all_layers.extend(dict(view=view, seed=seed, arch=arch, condition=condition, **r) for r in rows)
                    else:
                        logits = predict(model, feats)
                    label = f"{seed}_{arch}_{condition}"
                    y = tensor(data["y"]).long()
                    result = logits.argmax(1).cpu().numpy().astype(np.uint8)
                    payload[label] = result
                    payload["ce_" + label] = torch.nn.functional.cross_entropy(
                        logits, y, reduction="none").cpu().numpy()
                    if condition == key("native"):
                        native[seed, arch] = result
                    if condition in (key("native"), key("edge", -10000), key("edge", 10000),
                                     key("shifted", d=-9), key("shifted", d=9)):
                        name, f, d = condition.split("_")
                        previous = {"native": "native", "edge": f"cfo_{f}",
                                    "shifted": f"time_{d}_shifted"}[name]
                        compare = prior[f"{seed}_{arch}_{previous}"].argmax(1)
                        count = int((compare != result).sum())
                        assert count == 0, (label, count)
                    if condition.startswith("recover"):
                        count = int((native[seed, arch] != result).sum())
                        verification[view]["prediction_disagreements"][label] = count
                if len(payload) % 100 == 4:
                    print(view, condition, flush=True)
        np.savez_compressed(output, **payload)
        print(view, "predictions saved", flush=True)
    write(out / "LAYER.json", all_layers)
    write(out / "RECEIPT.json", dict(hashes={v: sha(out / f"PACKETS_{v}.npz") for v in VIEWS},
                                     layer_sha256=sha(out / "LAYER.json"), verification=verification))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("s29", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    run(args.s29, args.output)
