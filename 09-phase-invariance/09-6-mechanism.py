"""Source-held perturbation probes of frozen trained models, not model selection."""
import csv
import json
import numpy as np
import torch
from training import HERE, read_cells, paired_model, phase_features_from_base, frequency_views, rms_normalize, save
import importlib.util

spec = importlib.util.spec_from_file_location("diagnostics", HERE / "09-0-diagnostics.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def logits(model, views):
    h = model.pre_ln(model.encoder.prepare_tokens(model.frequency(views)))
    for block in model.encoder.blocks:
        h = block(h)
    return model.classifier(model.encoder.final_norm(h).mean(1))


def main():
    torch.set_num_threads(2)
    rows = []
    jobs = json.loads((HERE / "jobs_source.json").read_text())
    for job in jobs:
        out = HERE / "runs/source" / job["id"]
        config = json.loads((out / "config.json").read_text())
        probe = dict(job, id="mechanism_" + job["id"])
        cells = read_cells(probe, config["roles"]["held_rx"], (1, 2), "diagnostic")
        model, _ = paired_model(job["dataset"], job["feature"], 0)
        model.load_state_dict(torch.load(out / "model_raw_held.pt", map_location="cpu", weights_only=True))
        model.eval()
        for rx, cell in cells.items():
            ids = np.concatenate([np.flatnonzero(cell["y"] == c)[:16] for c in np.unique(cell["y"])])
            x, y = torch.from_numpy(cell["x"][ids]), torch.from_numpy(cell["y"][ids])
            base = frequency_views(rms_normalize(x))
            clean = phase_features_from_base(base, job["feature"])
            with torch.no_grad():
                clean_logits = logits(model, clean)
                clean_acc = float((clean_logits.argmax(1) == y).float().mean())
                for kind in ("circular_shift9", "crop_shift9", "cfo10khz", "smooth_channel"):
                    altered = phase_features_from_base(frequency_views(rms_normalize(diag.transformed(x, kind))),
                                                       job["feature"])
                    for scope in ("all_channels", "extra_only"):
                        views = altered if scope == "all_channels" else torch.cat((clean[:, :3], altered[:, 3:]), 1)
                        changed = logits(model, views)
                        acc = float((changed.argmax(1) == y).float().mean())
                        rows.append(dict(dataset=job["dataset"], fold=job["fold"], group=job["group"],
                            seed=job["seed"], feature=job["feature"], rx=rx, perturbation=kind, scope=scope,
                            clean_acc_pct=100 * clean_acc, changed_acc_pct=100 * acc, delta_pp=100 * (acc - clean_acc),
                            prediction_disagreement_pct=100 * float((changed.argmax(1) != clean_logits.argmax(1)).float().mean())))
    with (HERE / "trained_source_perturbations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# S09 Trained Source Mechanism Probes\n\n",
        "Fixed16 packets/TX, held-source RX only, raw minimum held-CE weights. "
        "These are post-training diagnostics, not independent validation or selection. "
        "Extra-only intervention keeps original Re/Im/log magnitude unchanged and changes "
        "only engineered channels. All-channel perturbation changes the full observed packet.\n\n",
        "| Dataset | Feature | Perturbation | Scope | Acc delta pp | Prediction disagreement % |\n",
        "|---|---|---|---|---:|---:|\n"]
    for dataset in ("ManyRx", "ManySig"):
        for feature in ("first", "centered", "curvature"):
            for kind in ("circular_shift9", "crop_shift9", "cfo10khz", "smooth_channel"):
                for scope in ("all_channels", "extra_only"):
                    local = [r for r in rows if (r["dataset"], r["feature"], r["perturbation"], r["scope"]) ==
                             (dataset, feature, kind, scope)]
                    lines.append(f"| {dataset} | {feature} | {kind} | {scope} | "
                        f"{np.mean([r['delta_pp'] for r in local]):+.2f} | "
                        f"{np.mean([r['prediction_disagreement_pct'] for r in local]):.2f} |\n")
    lines.append("\nFull-classifier shift invariance is not implied by invariant engineered channels. "
                 "Synthetic perturbations are not measured receiver time offsets; they cannot "
                 "establish that timing causes the remaining WiSig error.\n")
    (HERE / "MECHANISM_PROBES.md").write_text("".join(lines))
    save(HERE / "mechanism_probe_status.json", dict(rows=len(rows), fits=len(jobs), target_access=False))
    print(json.dumps(dict(rows=len(rows), source_fits=len(jobs))))


if __name__ == "__main__":
    main()
