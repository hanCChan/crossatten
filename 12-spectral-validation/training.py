"""Frozen S09 optimizer and pairing, with full-fit priors before smoke sampling."""
import torch
from core import HERE, ROOT, module, partition, smoke_subset, validate, save, s09
from spectral import fit_prior, paired_model


def load_arrays(job):
    splits, held, roles, seals = partition(job)
    prior = fit_prior(job["feature"], splits["train"])
    if prior is not None:
        prior = dict(prior, split_sha256=dict(seals),
                     fit_packets=len(splits["train"]["y"]))
    if job["phase"] == "smoke":
        splits, held = smoke_subset(splits, held, job["seed"])
        seals = dict(full_fit=seals,
                     smoke_arrays={key: {k: s09.array_sha(v) for k, v in cell.items()}
                                   for key, cell in splits.items()},
                     smoke_held_arrays={rx: {k: s09.array_sha(v) for k, v in cell.items()}
                                        for rx, cell in held.items()})
    return splits, held, roles, seals, prior


def adapter(job):
    engine = module("s12_fit_engine", ROOT / "09-phase-invariance/training.py")
    engine.HERE, engine.validate = HERE, validate
    prior = {}

    def load_data(j):
        splits, held, roles, seals, stats = load_arrays(j)
        if stats is not None:
            prior["weight"] = stats["weight"]
            save(HERE / "runs" / j["phase"] / j["id"] / "spectral_prior.json",
                 dict(stats, weight=stats["weight"].tolist(),
                      network_train_packets=len(splits["train"]["y"])))
        tr, va = [{k: torch.from_numpy(v).cuda() for k, v in splits[key].items()
                   if k in ("x", "y")} for key in ("train", "val")]
        held = s09.s08.on_device({rx: {k: v for k, v in c.items() if k in ("x", "y")}
                                 for rx, c in held.items()})
        return tr, va, held, roles, seals

    engine.load_data = load_data
    engine.paired_model = lambda dataset, feature, seed: paired_model(
        dataset, feature, seed, prior.get("weight"))
    return engine


def fit(job):
    validate(job)
    adapter(job).fit(job)
