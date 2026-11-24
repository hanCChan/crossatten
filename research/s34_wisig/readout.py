"""Frozen matched-width readouts, controlled positional bases, and token geometry."""

import argparse
import math
from pathlib import Path

import numpy as np

from common import (DEVICE, HERE, LAYERS, ROLES, VIEWS, FrozenStudy, centers, featurize,
                    restrict_torch, sha, tensor, torch, write)
from physics import characterize

WIDTH = 64
KINDS = ("mean", "mean_std", "region2", "region4", "flat", "dct4", "random4",
         "shuffle_region4", "shuffle_dct4")
REG = (0.1, .01, .0001)
norm = torch.nn.functional.normalize


def orthogonal(matrix):
    q, r = np.linalg.qr(matrix)
    return q * np.where(np.diag(r) < 0, -1, 1)


class Bases:
    def __init__(self):
        n = 32
        dct = np.cos(np.pi * (np.arange(n) + .5)[None] * np.arange(4)[:, None] / n)
        dct[0] /= np.sqrt(n)
        dct[1:] *= np.sqrt(2 / n)
        rng = np.random.default_rng(34061)
        random = orthogonal(np.column_stack([np.ones(n), rng.normal(size=(n, 3))])).T
        assert np.allclose(dct[0], random[0])
        region = {}
        for k in (2, 4):
            bin_means = np.repeat(np.eye(k), n // k, axis=1) / np.sqrt(n // k)
            rotation = np.cos(np.pi * (np.arange(k) + .5)[None] * np.arange(k)[:, None] / k)
            rotation[0] /= np.sqrt(k)
            rotation[1:] *= np.sqrt(2 / k)
            region[k] = rotation @ bin_means
            assert np.allclose(region[k][0], dct[0])
        for basis in (region[2], region[4], random, dct):
            assert np.allclose(basis @ basis.T, np.eye(len(basis)), atol=1e-12)
        self.mode = {k: tensor(v.astype(np.float32)) for k, v in
                     dict(dct4=dct, random4=random, region2=region[2], region4=region[4]).items()}
        rng = np.random.default_rng(34051)
        self.shuffle = tensor(np.argsort(rng.random((34668, n)), axis=1))
        self.projectors = {}

    def project(self, x):
        dim = x.shape[-1]
        if dim != WIDTH:
            if dim not in self.projectors:
                generator = np.random.default_rng(34001 + dim)
                self.projectors[dim] = tensor(orthogonal(generator.normal(size=(dim, WIDTH))).astype(np.float32))
            x = x @ self.projectors[dim]
        return norm(x)

    def encode(self, values, kind, ids):
        if kind == "input" or kind == "flat":
            x = values.flatten(1)
        elif kind == "pooled":
            x = values
        elif kind == "mean":
            x = values.mean(1)
        elif kind == "mean_std":
            x = torch.cat((values.mean(1), values.std(1, correction=0)), 1)
        else:
            mode = kind.removeprefix("shuffle_")
            if mode != kind:
                permutation = self.shuffle[tensor(ids).long()]
                values = values.gather(1, permutation[:, :, None].expand_as(values))
            x = torch.einsum("kt,btd->bkd", self.mode[mode], values).flatten(1)
        return self.project(x)


def stages(net, spectral):
    blocks = list(net.encoder.blocks)
    assert len(blocks) == 2
    tok = net.frequency(spectral)
    h = net.pre_ln(net.encoder.prepare_tokens(tok))
    first = blocks[0](h)
    last = blocks[1](first)
    z = net.encoder.final_norm(last)
    pooled = z.mean(1)
    return dict(token=tok, mix1=first, mix2=last, pre_pool=z, pooled=pooled,
                logits=net.classifier(pooled))


def capture(net, features):
    output = {k: [] for k in (*LAYERS, "pooled", "logits")}
    for part in features.split(256):
        for key, values in stages(net, part).items():
            output[key].append(values)
    return {k: torch.cat(v) for k, v in output.items()}


def rx_scores(pred, role):
    return float(np.mean([(pred[role["rx"] == r] == role["y"][role["rx"] == r]).mean()
                          for r in np.unique(role["rx"])]))


def fit_probe(encoded, data):
    assert all(v.shape[-1] == WIDTH for v in encoded.values())
    fit, val, held = (encoded[role] for role in ROLES)
    train = data["fit_train"]
    y = train["y"]
    r = train["rx"]
    counts = np.bincount(r)
    w = tensor(np.float32(1 / len(set(r)) / counts[r]))
    targets = torch.nn.functional.one_hot(tensor(y).long(), 10).float()
    xbar, ybar = w @ fit, w @ targets
    xf, yf = fit - xbar, targets - ybar
    cov = (xf.T * w) @ xf
    rhs = (xf.T * w) @ yf
    eye = torch.eye(WIDTH, device=DEVICE)
    scale = cov.diag().mean().clamp_min(1e-8)
    best_score = -1.
    val_scores = []
    for factor in REG:
        weights = torch.linalg.solve(cov + factor * scale * eye, rhs)
        val_p = ((val - xbar) @ weights + ybar).argmax(1).cpu().numpy()
        score = rx_scores(val_p, data["checkpoint_val"])
        val_scores.append(score)
        if score > best_score:
            best_score, selected, chosen, val_pred = score, factor, weights, val_p
    ridge = ((held - xbar) @ chosen + ybar).argmax(1)
    prototypes = centers(fit, train)
    return dict(ridge=ridge.cpu().numpy().astype(np.uint8),
                prototype=(held @ prototypes.T).argmax(1).cpu().numpy().astype(np.uint8),
                selection=dict(factor=selected, checkpoint_val_scores=val_scores))


def geometry(tokens, data):
    tokens = norm(tokens, dim=-1)
    y, rx = data["y"], data["rx"]
    within, tx_centers = [], []
    for t in range(10):
        cells = [tokens[tensor((y == t) & (rx == r))].mean(0) for r in np.unique(rx)
                 if ((y == t) & (rx == r)).sum() >= 8]
        assert len(cells) >= 2
        cells = torch.stack(cells)
        tx_center = cells.mean(0)
        tx_centers.append(tx_center)
        within.append((cells - tx_center).square().sum(-1).mean(0))
    tx_centers = torch.stack(tx_centers)
    indices = torch.triu_indices(10, 10, offset=1, device=DEVICE)
    between = (tx_centers[indices[0]] - tx_centers[indices[1]]).square().sum(-1).mean(0)
    return (between / torch.stack(within).mean(0).clamp_min(1e-12)).cpu().numpy().tolist()


@torch.inference_mode()
def execute(s29, s33, out):
    if any(out.iterdir()):
        raise FileExistsError("S34-A requires an empty output directory")
    source = FrozenStudy(s29, s33, out)
    write(out / "EXECUTION.json", dict(protocol=sha(HERE / "PROTOCOL.zh-CN.md"),
        engine=sha(Path(__file__)), common=sha(HERE / "common.py"), field_width=WIDTH,
        s29_fit=sha(source.s29 / "FIT_SEAL.json"), s33_receipt=sha(s33 / "RECEIPT.json"),
        backbone_trainings=0, optimizer_updates=0, selection_factors=REG))
    bases = Bases()
    record, hashes, shifts = [], {}, []
    from experiment import build_original
    fmodel = build_original("ManyRx", 24101)
    for v in VIEWS:
        data, audit = source.data(v, ROLES, "s34_readout")
        held = data["source_selection"]
        fit_rep = characterize(tensor(data["fit_train"]["x"]))
        cuts = np.quantile(1 - fit_rep["block_min_coherence"].cpu().numpy(), [.25,.5,.75])
        quality = np.searchsorted(cuts, audit["repeat_loss"], side="right").astype(np.uint8)
        payload = dict(y=held["y"], rx=held["rx"], day=held["day"], ids=held["ids"],
                       quality_quartile=quality, quality_cuts=cuts)
        spectra = {role: featurize(fmodel, tensor(data[role]["x"])).transpose(1, 2).contiguous()
                   for role in ROLES}
        input_probe = fit_probe({role: bases.encode(spectra[role], "input", data[role]["ids"])
                                 for role in ROLES}, data)
        for method in ("ridge", "prototype"):
            payload[f"input_{method}"] = input_probe[method]
        record.append(dict(view=v, seed=None, arch=None, layer="input", readout="input", **input_probe["selection"]))
        for seed, architecture, model in source.models(v):
            captured = {role: capture(model.net, spectra[role].transpose(1, 2).contiguous()) for role in ROLES}
            prefix = f"{seed}_{architecture}"
            old = audit[f"{prefix}_source_selection_logits"].argmax(1)
            native = captured["source_selection"]["logits"].argmax(1).cpu().numpy()
            assert np.array_equal(native, old)
            payload[prefix + "_head"] = native.astype(np.uint8)
            payload[f"{seed}_A1_head"] = audit[f"{seed}_A1_logits"].argmax(1).astype(np.uint8)
            for layer in (*LAYERS, "pooled"):
                if layer in LAYERS:
                    for role in ("fit_train", "source_selection"):
                        shifts.append(dict(view=v, seed=seed, arch=architecture, layer=layer, role=role,
                                           ratio=geometry(captured[role][layer], data[role])))
                for kind in KINDS if layer in LAYERS else ("pooled",):
                    encoded = {role: bases.encode(captured[role][layer], kind, data[role]["ids"])
                               for role in ROLES}
                    item = fit_probe(encoded, data)
                    for method in ("ridge", "prototype"):
                        payload[f"{prefix}_{layer}_{kind}_{method}"] = item[method]
                    record.append(dict(view=v, seed=seed, arch=architecture, layer=layer, readout=kind,
                                       **item["selection"]))
            del captured
            print(v, seed, architecture, "frozen readouts complete", flush=True)
        assert np.all((quality >= 0) & (quality <= 3))
        path = out / f"PRED_{v}.npz"
        np.savez_compressed(path, **payload)
        hashes[v] = sha(path)
        del spectra
    write(out / "PROBES.json", record)
    write(out / "POSITION_GEOMETRY.json", shifts)
    write(out / "RECEIPT.json", dict(predictions=hashes, probe_choices=sha(out / "PROBES.json"),
                                     position_geometry=sha(out / "POSITION_GEOMETRY.json")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("s29", "s33", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    restrict_torch()
    execute(args.s29, args.s33, args.output)
