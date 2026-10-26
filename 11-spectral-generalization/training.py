"""Frozen S09 recipe with stage-owned partitions and fit-only spectral buffers."""
import json
import numpy as np
import torch
from core import HERE, ROOT, s09, module, validate, read, partition, save, sha
from spectral import estimate, views, center_weight, paired_model


def adapter(job):
    engine = module("s11_fit_engine", ROOT / "09-phase-invariance/training.py")
    engine.HERE = HERE
    engine.validate = validate
    prior = {}

    def load_data(j):
        splits, held, roles, seals = partition(j, smoke=j["phase"] == "smoke")
        fit = splits["train"]
        if j["feature"] == "F2":
            with torch.no_grad():
                value = torch.cat([views(torch.from_numpy(fit["x"][i:i+256]))
                                   for i in range(0,len(fit["y"]),256)]).numpy()
            stats = estimate(value, fit["y"], fit["rx"], fit["day"],
                             min_cell=1 if j["phase"] == "smoke" else 10,
                             dates=(1,) if j["phase"] == "rx_day" else (1, 2))
            prior["weight"] = stats["weight"]
            p = HERE / "runs" / j["phase"] / j["id"] / "spectral_prior.json"
            save(p,dict(weight=prior["weight"].tolist(), c=stats["c"], eps=stats["eps"],
                        split_sha256=seals, fit_packets=len(fit["y"]), fit_only=True))
        elif j["feature"] == "F1":
            prior["weight"] = center_weight()
        tr, va = [{k:torch.from_numpy(v).cuda() for k,v in splits[key].items() if k in ("x","y")}
                  for key in ("train","val")]
        held = s09.s08.on_device({rx:{k:v for k,v in c.items() if k in ("x","y")}
                                 for rx,c in held.items()})
        return tr,va,held,roles,seals

    engine.load_data = load_data
    engine.paired_model = lambda dataset,feature,seed: paired_model(dataset,feature,seed,prior.get("weight"))
    engine.read_cells = lambda j,names,days,role,task=None: {
        rx:{k:v for k,v in c.items() if k in ("x","y")}
        for rx,c in read(j,names,days,role,task).items()}
    return engine


def fit(job):
    validate(job)
    adapter(job).fit(job)


def evaluate(job):
    validate(job)
    out = HERE / "runs" / job["phase"] / job["id"]
    m = json.loads((out / "metrics.json").read_text())
    if sha(out / "model.pt") != m["weights_sha256"]["model.pt"]:
        raise ValueError("checkpoint changed")
    state = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    model,_ = paired_model(job["dataset"],job["feature"],0,state.get("spectral_prior"))
    model.load_state_dict(state)
    model.cuda().eval()
    _,target = __import__("core").receivers(job["dataset"],job["fold"])
    result={}
    for task,days in (("CR",(1,2)),("CRD",(3,4))):
        cells = read(job,target,days,"target",task)
        data = s09.s08.on_device({rx:{k:v for k,v in c.items() if k in ("x","y")} for rx,c in cells.items()})
        result[task]=s09.s08.evaluate(model,data,10 if job["dataset"]=="ManyRx" else 6,out,task)
    save(out / "target_metrics.json",dict(job=job,selected=result,
        predictions_sha256={p.name:sha(p) for p in out.glob("preds_*.npz")}))
