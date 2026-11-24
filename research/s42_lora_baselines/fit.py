"""Six source-selected Shen fits and an atomic checkpoint seal."""

import argparse
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from siamese import Channel, SpecSiamese, pair_loss, spectrum
from xa.data import source_split


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = Path("D:/crossatten/stage_1/13-mixer-cfo-mechanisms/cache/shen")
FILES = ("PROTOCOL.zh-CN.md", "fit.py", "siamese.py")
SEEDS = (201, 202, 203)
DEVICE = "cuda"
MAX_EPOCHS = 160
PAIR_BATCH = 32


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as instance:
        return file_digest(instance, "sha256").hexdigest()


def atomic(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def save(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def git_file(relative):
    return subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)


def config():
    bound = {}
    for name in FILES:
        file = HERE / name
        assert git_file(file.relative_to(ROOT).as_posix()) == file.read_bytes(), name
        bound[name] = sha(file)
    for name in ("configs/protocol.json", "26-long-patch-control/registration.json",
                 "lib/xa/data.py", "lib/xa/models.py"):
        file = ROOT / name
        assert git_file(name) == file.read_bytes().replace(b"\r\n", b"\n"), name
        bound[name] = sha(file)
    return bound


def packet_digest(x):
    return sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


class Data:
    def __init__(self, output):
        self.output = output
        p = read(ROOT / "configs/protocol.json")
        self.views = p["folds"]
        self.closed = set(p["confirm_receivers"] + p["blind_receivers_never_open"])
        self.manifest = read(ROOT / "26-long-patch-control/registration.json")["source_caches"]

    def load(self, fold, receiver, role):
        if role not in ("source", "dev_val"):
            raise PermissionError(role)
        if receiver in self.closed or receiver not in self.views[fold][role]:
            raise PermissionError((fold, receiver, role))
        if role == "dev_val" and not (self.output / "FIT_SEAL.json").is_file():
            raise PermissionError("Report calls require the fit seal")
        path = CACHE / receiver
        x = np.load(path / "x.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(path / "y.npy", allow_pickle=False)
        ids = np.load(path / "rows.npy", allow_pickle=False)
        assert x.shape == (2000,2,8192) and x.dtype == np.float32
        assert y.shape == ids.shape == (2000,) and len(set(ids)) == 2000
        assert np.array_equal(np.bincount(y), np.full(10,200))
        for values, key in zip((x,y,ids), ("iq_float32","labels","rows")):
            expected = self.manifest[receiver][f"selected_{key}_sha256"]
            assert packet_digest(values) == expected, (receiver,key)
        with (self.output / "ACCESS.jsonl").open("a", encoding="utf-8", newline="\n") as log:
            log.write(json.dumps(dict(role=role, fold=fold, rx=receiver, rows=len(y),
                                      rows_sha256=packet_digest(ids))) + "\n")
        return x,y,ids

    def source(self, fold):
        train, val = [], []
        for r, name in enumerate(self.views[fold]["source"]):
            x,y,ids = self.load(fold,name,"source")
            for rows, split in zip((train,val), source_split(y,ids)):
                rows.append((x[split], y[split], np.full(len(split), r)))
        batches = [tuple(np.concatenate(col) for col in zip(*part)) for part in (train,val)]
        assert len(batches[0][1]) == 8000 and len(batches[1][1]) == 2000
        return batches


def complex_cuda(x):
    x = torch.tensor(x, device=DEVICE)
    return torch.complex(x[:,0],x[:,1])


@torch.inference_mode()
def logits_for(model, features):
    model.eval()
    return torch.cat([model(s)[0] for s in features.split(128)])


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}


def fit(data, val_features, val_y, folder, binding, fold, seed):
    torch.manual_seed(seed + fold*10000)
    folder.mkdir(exist_ok=True)
    if (folder/"DONE.json").exists():
        result = read(folder/"DONE.json")
        assert result["binding"] == binding and result["weights"] == sha(folder/"selected.pt")
        return result
    x,y,rx = data
    channels = Channel(DEVICE)
    iq = complex_cuda(x)
    labels = torch.tensor(y,device=DEVICE,dtype=torch.long)
    receivers = torch.tensor(rx,device=DEVICE)
    pools = torch.as_tensor(
        np.array([[np.flatnonzero((rx==r)&(y==t)) for t in range(10)] for r in range(5)]),
        device=DEVICE)
    assert pools.shape == (5,10,160)
    model = SpecSiamese().to(DEVICE, memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003, fused=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=9, threshold=0.)
    generator = torch.Generator(device=DEVICE).manual_seed(seed + fold*10000 + 9000)
    history = []
    best, best_epoch, best_state = float("inf"), 0, None
    state_path = folder/"LAST.pt"
    if state_path.exists():
        state = torch.load(state_path, weights_only=True, map_location="cpu")
        assert state["binding"] == binding
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        history = state["history"]
        best,best_epoch,best_state = (state[k] for k in ("best","best_epoch","best_state"))
        del state
    clock = time.perf_counter()
    previous_seconds = history[-1]["fit_seconds"] if history else 0.
    for epoch in range(len(history)+1,MAX_EPOCHS+1):
        if epoch-1-best_epoch>=30 and best_epoch:
            break
        model.train()
        anchors = torch.randperm(len(y),device=DEVICE,generator=generator)
        target_rx = (receivers[anchors]+torch.randint(1,5,(len(anchors),),
                     generator=generator,device=DEVICE))%5
        choices = torch.randint(160,(len(anchors),),device=DEVICE,generator=generator)
        mates = pools[target_rx,labels[anchors],choices]
        assert torch.equal(labels[anchors],labels[mates]) and bool((receivers[anchors]!=receivers[mates]).all())
        loss_sum = torch.zeros((),device=DEVICE)
        for a,b in zip(anchors.split(PAIR_BATCH),mates.split(PAIR_BATCH)):
            indices = torch.cat((a,b))
            inputs = spectrum(channels(iq[indices],generator))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE,dtype=torch.bfloat16):
                logits,z = model(inputs)
            loss = pair_loss(logits,z,labels[indices])
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
        logits = logits_for(model,val_features)
        ce = float(F.cross_entropy(logits,val_y))
        acc = float((logits.argmax(1)==val_y).float().mean())
        train = float(loss_sum/(len(y)//PAIR_BATCH))
        if not np.isfinite((ce,acc,train)).all():
            raise FloatingPointError((fold,seed,epoch,ce,acc,train))
        scheduler.step(ce)
        if ce < best:
            best,best_epoch,best_state = ce,epoch,cpu_state(model)
        history.append(dict(epoch=epoch,source_ce=ce,source_accuracy=acc,train_loss=train,
                            lr=optimizer.param_groups[0]["lr"],
                            fit_seconds=previous_seconds+time.perf_counter()-clock))
        save(state_path,dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(),generator=generator.get_state(),
                             history=history,best=best,best_epoch=best_epoch,best_state=best_state,
                             binding=binding))
        atomic(folder/"HISTORY.json",history)
        if epoch==1 or epoch%5==0:
            print(f"f{fold} s{seed} e{epoch:03d} source acc={acc:.4f} ce={ce:.5f} "
                  f"best={best_epoch:03d} wall={history[-1]['fit_seconds']:.1f}s",flush=True)
    save(folder/"selected.pt",best_state)
    model.load_state_dict(best_state)
    selected = logits_for(model,val_features)
    assert abs(float(F.cross_entropy(selected,val_y))-best) < 2e-6
    report = dict(fold=fold,seed=seed,binding=binding,status="COMPLETE",
                  epochs=len(history),updates=len(history)*(len(y)//PAIR_BATCH),
                  selected_epoch=best_epoch,source_ce=best,
                  source_accuracy=history[best_epoch-1]["source_accuracy"],
                  fit_seconds=history[-1]["fit_seconds"],
                  params=sum(p.numel() for p in model.parameters()),
                  peak_gpu_allocated=torch.cuda.max_memory_allocated(),
                  weights=sha(folder/"selected.pt"),history=sha(folder/"HISTORY.json"))
    atomic(folder/"DONE.json",report)
    print(f"f{fold} s{seed} complete",report["source_accuracy"],flush=True)
    return report


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    binding = config()
    plan = dict(files=binding,protocol="P-H5x160",jobs=[f"f{f}_s{s}" for f in (0,1) for s in SEEDS])
    if (out/"PLAN.json").exists():
        assert read(out/"PLAN.json") == plan
    else:
        atomic(out/"PLAN.json",plan)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Data(out)
    completed = []
    for f in (0,1):
        source, validation = data.source(f)
        val, y, _ = validation
        val_features = torch.cat([spectrum(z) for z in complex_cuda(val).split(128)])
        val_y = torch.tensor(y,device=DEVICE,dtype=torch.long)
        for s in SEEDS:
            torch.cuda.reset_peak_memory_stats()
            completed.append(fit(source,val_features,val_y,out/f"f{f}_s{s}",binding,f,s))
    assert len(completed) == 6
    atomic(out/"FIT_SEAL.json",dict(plan=sha(out/"PLAN.json"),fits=completed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    run(parser.parse_args().output)
