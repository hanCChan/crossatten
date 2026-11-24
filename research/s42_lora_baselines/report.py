"""Score the sealed cohort on matched historical Shen rows and reference predictions."""

import argparse
from collections import defaultdict
import csv
import io
from pathlib import Path
import statistics

import numpy as np
import torch
from torch.nn import functional as F

from fit import Data, ROOT, SEEDS, atomic, complex_cuda, config, git_file, logits_for, read, sha
from siamese import SpecSiamese, spectrum


REF = ROOT / "24-lora-external-audit/phase2/budget_update"
PREDICTIONS = Path("D:/crossatten/worktrees/s24-lora-external-audit") / REF.relative_to(ROOT) / "predictions"
HISTORICAL = ("h0", "ce", "dann")


def check(out):
    from hashlib import sha256

    identity = config()
    plan = read(out/"PLAN.json")
    seal = read(out/"FIT_SEAL.json")
    assert seal["plan"] == sha(out/"PLAN.json") and plan["files"] == identity
    assert {(row["fold"], row["seed"]) for row in seal["fits"]} == {
        (f,s) for f in (0,1) for s in SEEDS}
    assert len(seal["fits"]) == 6
    for row in seal["fits"]:
        directory = out/f"f{row['fold']}_s{row['seed']}"
        assert row == read(directory/"DONE.json") and row["binding"] == identity
        assert row["status"] == "COMPLETE" and row["weights"] == sha(directory/"selected.pt")
        assert row["history"] == sha(directory/"HISTORY.json")
        h = read(directory/"HISTORY.json")
        assert row["updates"] == 250*len(h) and row["epochs"] == len(h)
        assert h[row["selected_epoch"]-1]["source_ce"] == min(x["source_ce"] for x in h)
    assert sha256(git_file("research/s42_lora_baselines/report.py")).hexdigest() == sha(Path(__file__))
    assert (out/"ACCESS.jsonl").exists()
    return seal


def score(logits, y):
    predicted = logits.argmax(1)
    matrix = np.bincount(10*y+predicted, minlength=100).reshape(10,10)
    f1 = 2*np.diag(matrix)/np.maximum(matrix.sum(0)+matrix.sum(1),1)
    loss = F.cross_entropy(torch.from_numpy(logits),torch.from_numpy(y)).item()
    return dict(acc=float((predicted==y).mean()), macro_f1=float(f1.mean()), ce=loss,
                confusion=matrix.tolist())


def reference(method, fold, seed, rx, integrity, ids, y):
    key = f"{method}__native__f{fold}__s{seed}"
    path = PREDICTIONS/key/(rx+".npz")
    assert sha(path) == integrity[key,rx], path
    with np.load(path,allow_pickle=False) as file:
        assert np.array_equal(file["rows"],ids) and np.array_equal(file["y"],y)
        return file["logits"]


def summary(rows):
    rx_values, fit_values = defaultdict(list),defaultdict(list)
    for row in rows:
        rx_values[row["rx"]].append(row["acc"])
        fit_values[row["fold"],row["seed"]].append(row["acc"])
    assert len(rx_values)==10 and all(len(values)==3 for values in rx_values.values())
    assert len(fit_values)==6 and all(len(values)==5 for values in fit_values.values())
    means = {r:statistics.mean(vals) for r,vals in rx_values.items()}
    worst = min(means,key=means.get)
    return dict(acc=statistics.mean(means.values()),
                macro_f1=statistics.mean(r["macro_f1"] for r in rows),
                ce=statistics.mean(r["ce"] for r in rows),worst_rx=worst,worst_rx_acc=means[worst],
                fit_sd=statistics.stdev(statistics.mean(v) for v in fit_values.values()),
                receiver_accuracy=means)


def paired(rows, target):
    packets = {(r["fold"],r["seed"],r["rx"],r["method"]):r for r in rows}
    by_rx,by_fit,by_fold = defaultdict(list),defaultdict(list),defaultdict(list)
    for (f,s,r,method),row in packets.items():
        if method=="ma":
            diff = row["acc"]-packets[f,s,r,target]["acc"]
            by_rx[r].append(diff)
            by_fit[f,s].append(diff)
            by_fold[f].append(diff)
    by_rx = {r:statistics.mean(v)*100 for r,v in by_rx.items()}
    sizes = np.asarray([by_rx[rx] for rx in sorted(by_rx)])
    rng = np.random.default_rng(24016)
    interval = np.quantile(rng.choice(sizes,(10000,10),replace=True).mean(1),[.025,.975])
    worst = min(by_rx,key=by_rx.get)
    return dict(mean_pp=float(sizes.mean()),conditional_interval95_pp=interval.tolist(),
                positive_fit_pairs=sum(statistics.mean(v)>0 for v in by_fit.values()),
                folds_pp=[100*statistics.mean(by_fold[f]) for f in (0,1)],
                worst_rx=worst,worst_rx_delta_pp=by_rx[worst],per_rx_pp=by_rx)


@torch.inference_mode()
def cost(model):
    generator = torch.Generator(device="cuda").manual_seed(42017)
    rates={}
    for batch in (1,128):
        z = torch.view_as_complex(torch.randn((batch,8192,2),device="cuda",generator=generator))
        for _ in range(3):
            model(spectrum(z))
        torch.cuda.synchronize()
        duration=[]
        for _ in range(15):
            a,b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            model(spectrum(z))
            b.record()
            b.synchronize()
            duration.append(a.elapsed_time(b))
        rates[str(batch)] = dict(p50_ms=float(np.median(duration)),
                                 p95_ms=float(np.quantile(duration,.95)),sample_ms=duration)
    rates["128"]["packets_per_second"] = 128000/rates["128"]["p50_ms"]
    return dict(gpu=torch.cuda.get_device_name(),batches=rates,dtype="FP32 TF32 off",
                workload="synthetic resident IQ to logits; 3 warmups, 15 repeats; S42 session")


@torch.inference_mode()
def run(out):
    seal=check(out)
    if (out/"RESULTS.json").exists():
        raise FileExistsError("Score cohort already reported")
    proof=read(REF/"PREDICTION_INTEGRITY.json")
    integrity={(v["job"],v["rx"]):v["sha256"] for v in proof["predictions"]}
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=Data(out)
    rows=[]
    p=out/"predictions"
    p.mkdir(exist_ok=True)
    pred_seal={}
    for f in (0,1):
        models={}
        for s in SEEDS:
            model=SpecSiamese().cuda().eval().to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(out/f"f{f}_s{s}"/"selected.pt",map_location="cpu",
                                             weights_only=True))
            models[s]=model
        for rx in data.views[f]["dev_val"]:
            iq,y,ids=data.load(f,rx,"dev_val")
            features=torch.cat([spectrum(z) for z in complex_cuda(iq).split(128)])
            for s,model in models.items():
                logits=logits_for(model,features).cpu().numpy()
                assert logits.shape==(2000,10) and np.isfinite(logits).all()
                path=p/f"f{f}_s{s}_{rx}.npz"
                np.savez_compressed(path,logits=logits,rows=ids,y=y)
                pred_seal[path.name]=sha(path)
                for arm in ("ma",*HISTORICAL):
                    v=logits if arm=="ma" else reference(arm,f,s,rx,integrity,ids,y)
                    rows.append(dict(fold=f,seed=s,rx=rx,method=arm,**score(v,y)))
            print(f"matched rows verified: fold {f} {rx}",flush=True)
    assert len(rows)==120 and len(pred_seal)==30
    old=read(REF/"RESULTS.json")
    values={m:summary([r for r in rows if r["method"]==m]) for m in ("ma",*HISTORICAL)}
    for method in HISTORICAL:
        expected=next(r for r in old["summary"] if r["method"]==method and r["view"]=="native")
        for key in ("acc","macro_f1","fit_sd","worst_rx_acc"):
            assert np.isclose(values[method][key],expected[key],atol=1e-10,rtol=0.),(method,key)
    result=dict(protocol="P-H5x160 Native",historical_dev=True,strict_ib="NOT_RUN",
                methods=values,paired_ma_minus={m:paired(rows,m) for m in HISTORICAL},
                fit_seal_sha256=sha(out/"FIT_SEAL.json"),predictions=pred_seal,
                reference_prediction_integrity_sha256=sha(REF/"PREDICTION_INTEGRITY.json"),
                selected_source=seal["fits"],cost=cost(model))
    stream=io.StringIO(newline="")
    keys=["fold","seed","rx","method","acc","macro_f1","ce"]
    writer=csv.DictWriter(stream,fieldnames=keys,extrasaction="ignore",lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    (out/"RX_METRICS.csv").write_text(stream.getvalue(),encoding="utf-8",newline="\n")
    atomic(out/"RESULTS.json",result)
    print(values["ma"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    run(parser.parse_args().output)
