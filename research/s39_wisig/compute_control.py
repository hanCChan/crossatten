"""Train the single Target-Extra equal-exposure control and seal held predictions."""

import argparse
import importlib.util
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"s29_wisig"))
from experiment import DEVICE, tensor, torch
from source import C, ROOT, Source, array_sha, read, sha, write
from eligibility import VIEWS, assets

spec = importlib.util.spec_from_file_location("s39_c_engine",C/"engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

EPOCHS=40
BATCH=192
EXPOSURES=20736
FIRST_VAL=6912


def old_record(home,token):
    result=[r for r in read(home/"REPLAY_FILES.json")["files"]
            if r["source"].replace("\\","/").endswith(token)]
    assert len(result)==1,(home,token)
    file=ROOT/result[0]["relative"]
    assert sha(file)==result[0]["sha256"]
    return file


def rows_by_view(source,view):
    r={}
    ref=source.registration
    keys=engine.A.make_keys(source.y,source.rx,source.day,source.names,ref["source_cell_sha256"])
    for label,role in (("train","fit_train"),("val","checkpoint_val")):
        part=source.load(view,role,"fit")
        r[label]=dict(x=part["x"],y=part["y"].astype(np.int64),rx=part["rx"].astype(np.int64),
            day=part["day"].astype(np.int64),keys=keys[part["ids"]],pool_ids=part["ids"].astype(np.int64))
    return r


def balanced_exposure(size,seed,epoch):
    rng=np.random.default_rng(41301+seed+1009*epoch)
    indices=np.r_[rng.permutation(size),rng.permutation(size)[:EXPOSURES-size]]
    assert size<EXPOSURES<2*size and len(indices)==EXPOSURES
    return rng.permutation(indices)


def torch_data(x):
    return {k:tensor(x[k]) for k in ("x","y","rx")}


def phase_one(view,seed,train,validation,path):
    path.mkdir()
    engine.A.setup(24101+seed)
    model=engine.A.build_model(24101+seed)
    base=model.classifier.weight.detach().cpu().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41501+seed)
        model.classifier=torch.nn.Linear(64,10).to(DEVICE)
    assert not torch.equal(model.classifier.weight.detach().cpu(),base)
    torch.manual_seed(41601+seed)
    np.random.seed(41601+seed)
    random.seed(41601+seed)
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.1,patience=5,min_lr=1e-6)
    shadow=engine.M.make_ema(model)
    tx=torch_data(train)
    rng=np.random.default_rng(39001+int(view[-1]))
    matched_val=np.sort(rng.choice(len(validation["y"]),FIRST_VAL,replace=False))
    sample={k:v[matched_val] for k,v in validation.items()}
    va=torch_data(sample)
    np.save(path/"validation_ids.npy",sample["pool_ids"])
    bn_x=tx["x"][tensor(balanced_exposure(len(train["y"]),seed,0))]
    outcomes=[]
    best=math.inf
    updates=0
    start=time.perf_counter()
    for epoch in range(1,EPOCHS+1):
        model.train()
        order=balanced_exposure(len(train["y"]),seed,epoch)
        visits=np.bincount(order,minlength=len(train["y"]))
        assert visits.min()==1 and visits.max()==2
        total=torch.zeros((),device=DEVICE)
        for ids in tensor(order).split(BATCH):
            out=model(tx["x"][ids])
            ce=torch.nn.functional.cross_entropy(out,tx["y"][ids])
            opt.zero_grad(set_to_none=True)
            ce.backward()
            opt.step()
            shadow.update_parameters(model)
            updates+=1
            total+=ce.detach()
        raw=tensor(engine.predict_tensor(model,va["x"],BATCH))
        val_ce=engine.rx_ce(raw,va["y"],va["rx"])
        candidate, cal=engine.M.calibrated_ema_candidate(shadow,bn_x.split(BATCH),role="fit_train")
        assert cal["samples"]==EXPOSURES
        picked=engine.predict_tensor(candidate,va["x"],BATCH)
        ema_ce=engine.rx_ce(tensor(picked),va["y"],va["rx"])
        sch.step(val_ce)
        train_ce=float(total/(EXPOSURES//BATCH))
        assert all(math.isfinite(v) for v in (train_ce,val_ce,ema_ce))
        if ema_ce<best:
            best=ema_ce
            torch.save(dict(model=engine.cpu_tree(candidate.state_dict()),epoch=epoch,validation_ce=best),
                path/"selected.pt")
            np.save(path/"selected_val_logits.npy",picked)
        outcomes.append(dict(epoch=epoch,updates=updates,train_ce=train_ce,raw_validation=val_ce,
                 ema_validation=ema_ce,visit_sha256=array_sha(train["pool_ids"][order]),
                 distinct_packets=int((visits>0).sum()),exposures=len(order),lr_next=opt.param_groups[0]["lr"]))
        write(path/"HISTORY.json",outcomes)
        if epoch%5==0:
            print(view,seed,"phase1",epoch,round(ema_ce,5),flush=True)
    assert updates==4320
    selected=torch.load(path/"selected.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(selected["model"])
    assert np.array_equal(engine.predict_tensor(model,va["x"],BATCH),np.load(path/"selected_val_logits.npy"))
    execution=dict(view=view,seed=seed,updates=updates,observations=EXPOSURES*EPOCHS,
                   batch=BATCH,epochs=EPOCHS,unique_fit_packets=len(train["y"]),val_rows=len(matched_val),
                   selected_epoch=selected["epoch"],selected_sha=sha(path/"selected.pt"),
                   seconds=time.perf_counter()-start,replicated_validation=True)
    write(path/"COMPLETE.json",execution)
    return execution


def downstream(view,seed,train,val,path,first,old,names):
    path.mkdir()
    engine.A.setup(24101+seed)
    model=engine.A.build_model(24101+seed)
    base={k:p.detach().clone() for k,p in model.classifier.state_dict().items()}
    state=torch.load(first/"selected.pt",map_location="cpu",weights_only=True)["model"]
    missing,unknown=model.load_state_dict({k:v for k,v in state.items() if not k.startswith("classifier.")},strict=False)
    assert not unknown and set(missing)=={"classifier.weight","classifier.bias"}
    assert all(torch.equal(v,base[k]) for k,v in model.classifier.state_dict().items())
    recipe=read(old/"DATA_PLAN.json")["A0"][str(seed)]["recipe"].copy()
    recipe["id"]=f"TargetExtra_{view}_{seed}_downstream"
    assert recipe["batch"]==128 and recipe["epochs"]==40 and recipe["max_gradient_updates"]==5080
    details=engine.fit_loop(recipe,train,val,model,DEVICE,path,names)
    match=read(old_record(old,f"/A1_S{seed}_downstream/HISTORY.json"))
    current=read(path/"HISTORY.json")
    order_key="canonical_order_sha256"
    assert [r[order_key] for r in match]==[r[order_key] for r in current]
    details.update(view=view,seed=seed,identical_downstream_order_to_a1=True,
                   pretrain_sha256=sha(first/"selected.pt"))
    write(path/"COMPLETE.json",details)
    return details


def execute(root,out):
    if any(out.iterdir()):
        raise FileExistsError("S39 D requires an empty directory")
    assert read(root/"A.json")["decision"]=="STOP_COMPACT_IDENTITY_UNAVAILABLE"
    assert not read(root/"B.json")["passes"] and read(root/"C.json")["qualified"]
    write(out/"EXECUTION.json",dict(protocol=sha(HERE/"PROTOCOL.zh-CN.md"),engine=sha(Path(__file__)),
             audit_hashes={k:sha(root/(k+".json")) for k in "ABC"},candidate="Target-Extra",
             budget=dict(fits=8,updates=37600), c_fit_loop=sha(C/"engine.py")))
    source=Source(out)
    fits=[]
    for view,folder in VIEWS.items():
        cohort=rows_by_view(source,view)
        for seed in (0,1):
            known=assets(view,seed)
            assert set(known["A1"]["rx"]).isdisjoint(source.names[i] for i in set(cohort["train"]["rx"]))
            work=out/f"{view}_{seed}"
            work.mkdir()
            first=phase_one(view,seed,cohort["train"],cohort["val"],work/"first")
            down=downstream(view,seed,cohort["train"],cohort["val"],work/"down",work/"first",
                            HERE.parent/folder,source.names)
            fits.append(dict(view=view,seed=seed,first=first,down=down,
                weights_sha=sha(work/"down/C1_selected.pt")))
            write(out/"PROGRESS.json",fits)
    assert sum(r["first"]["updates"]+r["down"]["updates"] for r in fits)==37600
    write(out/"FIT_SEAL.json",dict(records=fits,fits=8,updates=37600))
    evaluate(out,source)


def evaluate(out,source):
    seal=read(out/"FIT_SEAL.json")
    result={}
    for view in VIEWS:
        data=source.load(view,"source_selection","new_prediction")
        payload={k:data[k] for k in ("y","rx","day","ids")}
        for r in seal["records"]:
            if r["view"]!=view:
                continue
            seed=r["seed"]
            file=out/f"{view}_{seed}/down/C1_selected.pt"
            assert sha(file)==r["weights_sha"]
            engine.A.setup(24101+seed)
            model=engine.A.build_model(24101+seed)
            model.load_state_dict(torch.load(file,map_location="cpu",weights_only=True)["model"])
            cohort=assets(view,seed)
            for method,data_old in cohort.items():
                for name,src in (("y","y"),("day","day"),("ids","pool_ids")):
                    assert np.array_equal(data[name],data_old[src])
                assert np.array_equal(np.asarray(source.names)[data["rx"]],data_old["rx"])
                payload[f"{seed}_{method}"]=data_old["logits"]
            payload[f"{seed}_extra"]=engine.predict_tensor(model,tensor(data["x"]),128)
        dest=out/f"PRED_{view}.npz"
        np.savez_compressed(dest,**payload)
        result[view]=sha(dest)
    write(out/"PREDICTION_SEAL.json",result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    d=a.output/"D"
    d.mkdir(exist_ok=True)
    execute(a.output,d)
