"""Source-only BN audit of two frozen S09-config refits, never target tuning."""
import copy
import json
import numpy as np
import pandas as pd
import torch
from core import HERE, ROOT, s09, partition, sha, save, once, manifest, require_committed
from spectral import paired_model


def scatter(v, y, rx):
    # One clean source RMS scale, shared across TX/RX distances at this layer.
    scale = max(float(np.sqrt((v**2).mean())), 1e-6)
    v = v / scale
    classes = np.unique(y)
    center = {tx: v[y == tx].mean(0) for tx in classes}
    between = np.mean([np.mean((center[a]-center[b])**2)
                       for i,a in enumerate(classes) for b in classes[i+1:]])
    shifts = [np.mean((v[(y==tx)&(rx==name)].mean(0)-center[tx])**2)
              for tx in classes for name in np.unique(rx) if ((y==tx)&(rx==name)).any()]
    return float(between),float(np.mean(shifts))


def probe(model, x, y, names, roles, dataset):
    layers={name:layer for name,layer in model.named_modules()
            if isinstance(layer,torch.nn.BatchNorm1d)}
    captured={name:dict(mu=[],var=[],out=[]) for name in layers}
    handles=[]
    for name,layer in layers.items():
        def hook(obj,inputs,output,key=name):
            z=inputs[0].detach()
            captured[key]["mu"].append(z.mean(-1).cpu().numpy())
            captured[key]["var"].append(z.var(-1,unbiased=False).cpu().numpy())
            captured[key]["out"].append(output.detach().mean(-1).cpu().numpy())
        handles.append(layer.register_forward_hook(hook))
    with torch.no_grad():
        for i in range(0,len(y),128):
            model(x[i:i+128].cuda())
    for handle in handles:
        handle.remove()
    rows=[]
    for name,layer in layers.items():
        values={k:np.concatenate(v) for k,v in captured[name].items()}
        txdist,rxdist=scatter(values["out"],y,names)
        mu=layer.running_mean.detach().cpu().numpy()
        var=layer.running_var.detach().cpu().numpy()
        for rx in np.unique(names):
            ids=names==rx
            rxmu=values["mu"][ids].mean(0)
            # Total activation variance, not only each packet's within-window variance.
            rxvar=(values["var"][ids]+values["mu"][ids]**2).mean(0)-rxmu**2
            rows.append(dict(dataset=dataset,layer=name,rx=rx,
                role="fit" if rx in roles["fit_rx"] else "held",packets=int(ids.sum()),
                normalized_mean_shift=float(np.sqrt(np.mean((rxmu-mu)**2/(var+layer.eps)))),
                absolute_log_variance_ratio=float(np.mean(np.abs(np.log(
                    np.maximum(rxvar,layer.eps)/(var+layer.eps))))),
                source_normalized_d_tx=txdist,source_normalized_d_rx=rxdist,
                source_d_rx_over_d_tx=rxdist/max(txdist,1e-12)))
    return rows


def main():
    torch.set_num_threads(2)
    require_committed(HERE/"bn_protocol.json")
    protocol=json.loads((HERE/"bn_protocol.json").read_text())
    rows,effects,replay=[],[],[]
    evidence={}
    for path in sorted((HERE/"runs/restore").glob("*/metrics.json")):
        require_committed(path)
        result=json.loads(path.read_text())
        j=result["job"]
        config=json.loads((path.parent/"config.json").read_text())
        weight=path.parent/"model_raw_packet.pt"
        if sha(weight)!=result["weights_sha256"][weight.name]:
            raise ValueError("BN checkpoint seal mismatch")
        splits,held,roles,seals=partition(j)
        if seals!=config["split_sha256"]:
            raise ValueError("BN data split differs")
        model,_=paired_model(j["dataset"],"hard",0)
        model.load_state_dict(torch.load(weight,map_location="cpu",weights_only=True))
        model.cuda().eval()
        fit=splits["train"]
        xs=[fit["x"]]
        ys=[fit["y"]]
        names=[fit["rx"]]
        for rx,c in held.items():
            xs.append(c["x"]);ys.append(c["y"]);names.append(np.full(len(c["y"]),rx))
        x=torch.from_numpy(np.concatenate(xs))
        y=np.concatenate(ys)
        rxnames=np.concatenate(names)
        rows.extend(probe(model,x,y,rxnames,roles,j["dataset"]))
        data=s09.s08.on_device({rx:{k:v for k,v in c.items() if k in ("x","y")} for rx,c in held.items()})
        before=s09.s08.evaluate(model,data,10 if j["dataset"]=="ManyRx" else 6)
        calibrated=copy.deepcopy(model)
        s09.s08.recalibrate_bn(calibrated,torch.from_numpy(fit["x"]).cuda(),128)
        after=s09.s08.evaluate(calibrated,data,10 if j["dataset"]=="ManyRx" else 6)
        effects.append(dict(dataset=j["dataset"],fold=j["fold"],group=j["group"],seed=j["seed"],
            before_pct=100*before["summary"]["acc_mean"],after_fit_only_recalibration_pct=100*after["summary"]["acc_mean"],
            delta_pp=100*(after["summary"]["acc_mean"]-before["summary"]["acc_mean"]),
            before_ce=before["summary"]["ce_mean"],after_ce=after["summary"]["ce_mean"],
            no_target_adaptation=True,no_fit_or_model_selection=True))
        old=ROOT/"09-phase-invariance/runs/source"/f"{j['dataset']}__f0__last__s8101__first__source"
        oldconfig=json.loads((old/"config.json").read_text())
        historical=json.loads((old/"metrics.json").read_text())
        fields=["params","recipe","data_seed","roles","split_sha256","shared_initialization_sha256",
                "train_packets","val_packets","held_packets","ema","torch","gpu"]
        if any(config[k]!=oldconfig[k] for k in fields):
            raise ValueError("S09 matched refit config differs")
        for name,hashvalue in result["weights_sha256"].items():
            replay.append(dict(dataset=j["dataset"],weight=name,new_sha256=hashvalue,
                historical_sha256=historical["weights_sha256"][name],
                checkpoint_bytes_match=hashvalue==historical["weights_sha256"][name]))
        evidence[str(path.relative_to(ROOT))]=sha(path)
        evidence[str((path.parent/"config.json").relative_to(ROOT))]=sha(path.parent/"config.json")
        print("BN source audit",j["dataset"],effects[-1],flush=True)
        del model,calibrated
    if len(effects)!=2:
        raise ValueError("both registered refits required")
    pd.DataFrame(rows).to_csv(HERE/"source_bn_layer_rx.csv",index=False)
    pd.DataFrame(effects).to_csv(HERE/"source_bn_recalibration.csv",index=False)
    pd.DataFrame(replay).to_csv(HERE/"checkpoint_refit_replay.csv",index=False)
    # Two fold0 audits are not sufficient to authorize a32-fold-wide normalization claim.
    supported=len({e["fold"] for e in effects})>=protocol["minimum_distinct_folds"] and all(
        e["delta_pp"]>=protocol["minimum_recalibration_gain_pp"] for e in effects)
    once(HERE/"decision_bn.json",dict(admitted=supported,jobs=[],
        evidence=manifest([HERE/"source_bn_layer_rx.csv",HERE/"source_bn_recalibration.csv",
                           HERE/"checkpoint_refit_replay.csv"]),verdict="NOT_ADMITTED",
        coverage="two dataset-specific fold0/last/seed8101 source refits only",
        no_target_data=True,limitations="moment drift is descriptive; fit-only recalibration is not BN->GN efficacy; "
                                      "GN requires independently preregistered broader source evidence"))
    save(HERE/"bn_evidence.json",dict(protocol=protocol,checkpoint_metrics=evidence,
        effects=effects,registered_refits=2,additional_training=0,
        original_weights_missing=True,refit_checkpoint_byte_matches=sum(r["checkpoint_bytes_match"] for r in replay)))


if __name__=="__main__":
    main()
