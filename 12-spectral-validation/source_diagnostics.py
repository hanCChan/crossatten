"""Frozen fit-only spectral diagnostics and negative controls, no weight audits."""
import json
import numpy as np
import torch
from core import HERE, partition, save, check_manifest, require_committed
from diagnostics import geometry, scatter_diagnostics
from register import DATASETS, job
from spectral import batch_views, estimate, center_weight, negative_weights


def main():
    torch.set_num_threads(2)
    require_committed(HERE / "registration.json")
    reg = json.loads((HERE / "registration.json").read_text())
    check_manifest(reg["code_manifest"])
    path = HERE / "source_diagnostics.json"
    if path.exists():
        raise ValueError("source diagnostics already exist")
    rows, priors = [], []
    for dataset in DATASETS:
        for fold in range(4):
            j = job("diagnostic", dataset, fold, "F2")
            splits, held, roles, seals = partition(j)
            fit = splits["train"]
            value = batch_views(fit["x"])
            prior = estimate(value, fit["y"], fit["rx"], fit["day"])
            weight = prior["weight"]
            controls = dict(hard=np.ones(256, np.float32), F1=center_weight(), F2=weight,
                **negative_weights(value, fit["y"], fit["rx"], fit["day"], weight, 12026))
            priors.append(dict(dataset=dataset, fold=fold, group="first", seed=8101,
                split_sha256=seals, roles=roles, fit_packets=len(fit["y"]),
                weight=weight.tolist(), c=prior["c"], eps=prior["eps"],
                included_cells=prior["included_cells"]))
            for rx, cell in held.items():
                hv = batch_views(cell["x"])
                for kind, w in controls.items():
                    rows.append(dict(dataset=dataset, fold=fold, rx=rx, kind=kind,
                        packets=len(cell["y"]),
                        **geometry(value, hv, fit["y"], cell["y"], w,
                                   "F1" if kind == "F1" else "F2"),
                        **scatter_diagnostics(prior, w)))
            print("S12 source diagnostics", dataset, fold, flush=True)
    save(path, dict(priors=priors, rows=rows, no_target_reads=True,
                    descriptive_only=True, candidate_admission_veto=False))


if __name__ == "__main__":
    main()
