"""Shared-waveform and residual transfer on the sealed six-block decomposition."""

import argparse
from pathlib import Path

import numpy as np

from heads import LOCAL, select_linear
from common import ROLES, VIEWS, FrozenStudy, centers, read, restrict_torch, sha, tensor, torch, write
from preamble import decompose
from extract import fit_reference, physical
from readout import orthogonal

FEATURES = ("raw", "shared", "residual", "joint")


def rows_for(role, subset, label):
    return dict(y=label[subset], rx=role["rx"][subset])


def probe_rx(data, x, output, name):
    f, v = (data[k] for k in ROLES[:2])
    admissible = np.unique(f["rx"])
    assert np.array_equal(np.unique(v["rx"]), admissible)
    classes = len(admissible)
    for parity in (0, 1):
        fit = f["y"] % 2 == parity
        val = v["y"] % 2 != parity
        fit_rx = np.searchsorted(admissible, f["rx"])
        val_rx = np.searchsorted(admissible, v["rx"])
        assert set(fit_rx[fit]) == set(val_rx[val]) == set(range(classes))
        subset = {ROLES[0]: x[ROLES[0]][fit], ROLES[1]: x[ROLES[1]][val],
                  ROLES[2]: x[ROLES[1]][val]}
        roles = {ROLES[0]: rows_for(f, fit, fit_rx), ROLES[1]: rows_for(v, val, val_rx)}
        _, p, _ = select_linear(subset, roles, classes=classes, factors=(.01,))
        prefix = f"rx_{name}_{parity}"
        output[prefix+"_y"] = val_rx[val]
        output[prefix+"_pred"] = p.argmax(1).cpu().numpy().astype(np.uint8)


@torch.inference_mode()
def execute(s29, s33, s34, out):
    if any(out.iterdir()):
        raise FileExistsError("S36-E needs an empty execution directory")
    study = FrozenStudy(s29, s33, out)
    preamble_receipt = read(s34 / "RECEIPT.json")
    write(out / "EXECUTION.json", dict(engine=sha(Path(__file__)), protocol=sha(LOCAL / "PROTOCOL.zh-CN.md"),
         s34_receipt=sha(s34 / "RECEIPT.json"), dimension=120, backbone_fits=0, optimizer_updates=0,
         raw_eq="STOP_NO_VERIFIABLE_LOCAL_PAIRING"))
    projection = tensor(orthogonal(np.random.default_rng(36120).normal(size=(240,120))).astype(np.float32))
    hashes, records = {}, []
    norm = torch.nn.functional.normalize
    for view in VIEWS:
        roles, _ = study.data(view, ROLES, "s36_preamble")
        train, held = roles["fit_train"], roles["source_selection"]
        profile = fit_reference(physical(tensor(train["x"])), train)
        extracted = {}
        for role, data in roles.items():
            _, f = decompose(tensor(data["x"]), profile, data["y"], data["day"], features=True)
            f["joint"] = norm(torch.cat((f["shared"], f["residual"]), 1) @ projection)
            extracted[role] = f
        payload = {k: held[k] for k in ("y", "day", "rx", "ids")}
        prior = s34 / f"PRED_{view}.npz"
        assert sha(prior) == preamble_receipt["predictions"][view]
        with np.load(prior, allow_pickle=False) as sealed:
            assert all(np.array_equal(held[k], sealed[k]) for k in payload)
            for method in FEATURES:
                fs = {role: f[method] for role, f in extracted.items()}
                assert all(x.shape[1] == 120 for x in fs.values())
                proto = centers(fs["fit_train"], train)
                pred = (fs["source_selection"] @ proto.T).argmax(1).cpu().numpy().astype(np.uint8)
                if method != "joint":
                    assert np.array_equal(pred, sealed[method+"_prototype"])
                payload[f"{method}_prototype"] = pred
                _, ridge, selected = select_linear(fs, roles)
                payload[f"{method}_ridge"] = ridge.argmax(1).cpu().numpy().astype(np.uint8)
                records.append(dict(view=view, representation=method, **selected))
                day_labels = {role: dict(y=rows["day"] - 1, rx=rows["rx"]) for role, rows in roles.items()}
                _, scores, _ = select_linear(fs, day_labels, classes=2, factors=(.01,))
                payload[f"{method}_day"] = scores.argmax(1).cpu().numpy().astype(np.uint8)
                probe_rx(roles, fs, payload, method)
        file = out / f"PRED_{view}.npz"
        np.savez_compressed(file, **payload)
        hashes[view] = sha(file)
        print(view, "TX/RX/day transfer complete", flush=True)
    write(out / "LINEAR.json", records)
    write(out / "RECEIPT.json", dict(predictions=hashes, selected_readouts=sha(out / "LINEAR.json"),
                                     s34_original_centroid_parity_checks=6))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("s29", "s33", "s34", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    restrict_torch()
    execute(a.s29, a.s33, a.s34, a.output)
