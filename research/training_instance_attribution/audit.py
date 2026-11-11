"""Retrospective training-factor audit. Reads logs and registered labels, never IQ/models.

The optional control panels are deterministic design artifacts, not training runs.
"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PREVIOUS=ROOT/"research/classification_failure_diagnostics"
HISTORY=Path("D:/crossatten/worktrees/s17-rx-distortion/17b-phase-intervention")
S20=Path("D:/crossatten/worktrees/s20-asknet-source-comparison/20-asknet-source-comparison")
records=[]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):
    (HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
def read(p):
    p=Path(p);records.append(dict(path=str(p),sha256=sha(p),kind="text_metadata"))
    return json.loads(p.read_text(encoding="utf-8"))
def array_sha(a): return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def actual_seed(f):
    return int(hashlib.sha256(f"S17-WiSig|ManyRx|{f}|first|18101".encode()).hexdigest()[:8],16)


def read_history(path):
    records.append(dict(path=str(path),sha256=sha(path),kind="read_only_training_csv"))
    with path.open(encoding="utf-8",newline="") as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for k,v in row.items():
            if k=="epoch": row[k]=int(v)
            elif k!="order_sha256": row[k]=float(v)
    assert [r["epoch"] for r in rows]==list(range(1,len(rows)+1))
    assert all(math.isfinite(r["lr"]) and r["lr"]>0 for r in rows)
    return rows


def trace(arm,fold,cfg,metrics,rows,ntrain):
    field="source_val_ce" if arm=="B0" else "val_packet_ce"
    best=min(rows,key=lambda r:r[field])
    assert len(rows)==metrics["epochs"] and best["epoch"]==metrics["best_epoch"]
    init=cfg["initial_state_sha256"]
    assert len(init)==64 and cfg["data_seed"]==actual_seed(fold)
    # Historical B0 logs AFTER scheduler.step; ASK logs INSIDE train_epoch, before validation.
    used=([1e-3]+[r["lr"] for r in rows[:-1]]) if arm=="B0" else [r["lr"] for r in rows]
    changes=[dict(epoch=i+1,lr=value) for i,value in enumerate(used) if i==0 or value!=used[i-1]]
    batches=math.ceil(ntrain/128)
    result=dict(fold=fold,arm=arm,macro_seed=18101,actual_seed=cfg["data_seed"],split_seed=16001,
                initial_state_sha256=init,train_packets=ntrain,batch_size=128,batches_per_epoch=batches,
                epoch_budget=40 if arm=="B0" else 200,epochs_executed=len(rows),selected_epoch=best["epoch"],
                selected_checkpoint_sha256=metrics["model_sha256"],initial_weight_snapshot_retained=False,
                lr_log_semantics="post_scheduler_next_epoch" if arm=="B0" else "current_training_epoch",
                lr_used_by_epoch=used,lr_used_changes=changes,lr_at_selected_epoch=used[best["epoch"]-1],
                selected_updates=batches*best["epoch"],executed_updates=batches*len(rows),
                selected_sample_exposures=ntrain*best["epoch"],executed_sample_exposures=ntrain*len(rows),
                selection="RX-equal validation CE" if arm=="B0" else "packet-equal validation CE",
                scheduler="Adam1e-3 wd0; plateau factor0.1/patience5/min1e-6; earlywait10",
                recorded_epoch_order_hashes=[r["order_sha256"] for r in rows] if arm=="ASK" else None,
                order_hash_missing=(arm=="B0"),historical_order_replayed=False,
                train_transform="none: baseline M0 has no family" if arm=="B0" else "official model-internal stochastic cutoff; no extra input augmentation",
                metric_trace=rows)
    if arm=="ASK":
        rxbest=min(rows,key=lambda r:r["val_rx_ce"])
        assert rxbest["epoch"]==metrics["diagnostic_rx_ce_best_epoch"]
        result["packet_vs_rx_selection_same_epoch"]=best["epoch"]==rxbest["epoch"]
        assert all(len(r["order_sha256"])==64 for r in rows)
    else:
        result["packet_vs_rx_selection_same_epoch"]=None
        result["packet_weighted_ce_trace_missing"]=True
    return result


def pool_keys(meta,names,cell_hashes):
    y,rx,day=(meta[k] for k in ("fit_y","fit_rx","fit_day"))
    assert y.shape==rx.shape==day.shape and set(day)=={1,2}
    counters={};keys=[];labels={}
    for t,r,d in zip(y,rx,day):
        cell=(names[int(r)],int(d));ordinal=counters.get(cell,0);counters[cell]=ordinal+1
        key=(cell[0],cell[1],cell_hashes[cell],ordinal)
        keys.append(key);labels[key]=int(t)
    return keys,labels


def overlap(a,b):
    pa,pb=set(a["keys"]),set(b["keys"])
    shared=pa&pb
    assert all(a["labels"][k]==b["labels"][k] for k in shared), "Source row-label mismatch"
    ta,tb={a["keys"][i] for i in a["train_ids"]},{b["keys"][i] for i in b["train_ids"]}
    va,vb=pa-ta,pb-tb
    tt=len(ta&tb);tv=len(ta&vb);vt=len(va&tb);vv=len(va&vb)
    assert tt+tv+vt+vv==len(shared)
    common_rx=sorted(set(a["fit_rx"])&set(b["fit_rx"]))
    return dict(folds=[a["fold"],b["fold"]],common_fit_rx=common_rx,common_fit_rx_count=len(common_rx),
                rx_jaccard=len(common_rx)/len(set(a["fit_rx"])|set(b["fit_rx"])),
                common_pool_rows=len(shared),both_train=tt,a_train_b_val=tv,a_val_b_train=vt,both_val=vv,
                membership_disagreement_fraction=(tv+vt)/len(shared) if shared else None,
                global_train_row_jaccard=tt/len(ta|tb),
                common_rx_train_jaccard=tt/len((ta|tb)&shared) if shared else None,
                integer_train_index_vectors_equal=np.array_equal(a["train_ids"],b["train_ids"]),
                integer_validation_index_vectors_equal=np.array_equal(a["val_ids"],b["val_ids"]),
                identity_basis="RX/day cell SHA + deterministic ordinal within that cell; no cross-RX pairing or independent physical packet IDs")


def panels(pool):
    names=sorted(pool["fit_rx"]);core=names[:6];extensions=[names[6:12],names[12:18]]
    keys=pool["keys"];y=pool["fit_y"];day=pool["fit_day"]
    rx=np.array([k[0] for k in keys]);train=pool["train_ids"];val=pool["val_ids"]
    common=[int(i) for i in train if rx[i] in core]
    common.sort(key=lambda i:keys[i])
    assert all(any(y[i]==t and day[i]==d for i in common) for t in range(10) for d in (1,2))
    validation=np.asarray(sorted([int(i) for i in val if rx[i] in core],key=lambda i:keys[i]),dtype=np.int32)
    caps=[];selected=[[],[]]
    def ordering(i): return hashlib.sha256(("source-panel-v1|"+repr(keys[i])).encode()).hexdigest()
    for t in range(10):
        for d in (1,2):
            available=[[int(i) for i in train if rx[i] in ext and y[i]==t and day[i]==d] for ext in extensions]
            cap=min(map(len,available));caps.append(dict(tx=t,day=d,extension_counts=list(map(len,available)),retained_per_extension=cap))
            for j in range(2): selected[j].extend(sorted(available[j],key=ordering)[:cap])
    out=[]
    arrays=[]
    for j in range(2):
        ext=sorted(selected[j],key=lambda i:(int(y[i]),int(day[i]),keys[i]))
        ids=np.array(common+ext,dtype=np.int64);arrays.append(ids)
        assert not set(ids)&set(validation) and len(np.unique(ids))==len(ids)
        assert set(rx[ids])<=set(pool["fit_rx"])
        out.append(dict(arm=f"composition_{j}",shared_core_rx=core,extension_rx=extensions[j],
                        train_packets=len(ids),fixed_core_train_packets=len(common),extension_packets=len(ext),
                        validation_packets=len(validation),train_indices_dtype="int64",validation_indices_dtype="int32",train_indices_sha256=array_sha(ids),
                        validation_indices_sha256=array_sha(validation)))
    assert np.array_equal(y[arrays[0]],y[arrays[1]]) and np.array_equal(day[arrays[0]],day[arrays[1]])
    return dict(status="DESIGN_ONLY_NOT_EXECUTED",anchor_fold=2,legal_fit_universe=pool["fit_rx"],
                selection_rule="lexicographic first6 shared core; remaining12 split6/6; extension class/day caps=min(counts); SHA256 rank source keys; fixed core first",
                arms=out,extension_caps=caps,train_class_and_day_sequences_equal=True,
                common_core_row_positions_identical=True,validation_core_identical=True,
                held_source_rx=pool["held_rx"],forbidden_target_rx=pool["forbidden_target_rx"],
                new_iq_access_or_model_training=False)


def main():
    assert not (HERE/"AUDIT.json").exists(), "Keep original retrospective audit"
    previous=read(PREVIOUS/"INPUT_REGISTRY.json")
    source_log=HISTORY/"access.jsonl"
    records.append(dict(path=str(source_log),sha256=sha(source_log),kind="historical_source_access_metadata"))
    cell_hashes={}
    for line in source_log.read_text(encoding="utf-8").splitlines():
        r=json.loads(line)
        if r.get("dataset")=="ManyRx" and r.get("day") in (1,2) and r.get("mode")=="read_only":
            key=(r["rx"],r["day"])
            if key in cell_hashes: assert cell_hashes[key]==r["sha256"]
            cell_hashes[key]=r["sha256"]
    whitelist=[r for r in previous["arrays"] if r["kind"]=="source_labels_or_split_indices"]
    assert len(whitelist)==20 and all(r["name"] in {"fit_y","fit_rx","fit_day","train_ids","val_ids"} for r in whitelist)
    save("ACCESS_SCOPE.json",dict(status="RETROSPECTIVE_READ_ONLY",base_commit="b3035c9db8cc2e65488e8b606ff68e102ae006d3",client_date="2026-11-11",
                                  label_index_whitelist=whitelist,prediction_arrays_to_open=0,iq_arrays_to_open=0,checkpoint_loads=0,
                                  note="Whitelist recorded before label-array payloads; CSV/log metadata may already have been inspected. Not an efficacy preregistration."))
    metadata={}
    for item in whitelist:
        assert sha(item["path"])==item["sha256"]
        a=np.load(item["path"],allow_pickle=False)
        assert a.ndim==1 and a.dtype.kind in "iu"
        metadata.setdefault(item["fold"],{})[item["name"]]=a
        records.append(dict(path=item["path"],sha256=item["sha256"],kind="registered_source_labels_or_indices",rows=len(a)))
    original=read(HISTORY/"final_evidence/MANIFEST.json")
    s20_manifest=read(S20/"MANIFEST.json")["files_sha256"]
    runs=[];pools=[]
    for f in range(4):
        meta=metadata[f];role=previous["folds"][f]
        for part in ("train","val"):
            assert array_sha(meta[part+"_ids"])==role[part+"_indices_array_sha256"]
        assert np.array_equal(np.sort(np.r_[meta["train_ids"],meta["val_ids"]]),np.arange(len(meta["fit_y"])))
        keys,labels=pool_keys(meta,role["rx_names"],cell_hashes)
        pools.append({**role,**{k:v for k,v in meta.items() if k!="fit_rx"},
                      "fit_rx_codes":meta["fit_rx"],"keys":keys,"labels":labels})
        old=HISTORY/"runs"/f"pilot__ManyRx__M0__f{f}__first__s18101"
        ask=S20/"runs"/f"ASK__f{f}__first__s18101"
        record=next(r for r in original["records"] if r["job"]["fold"]==f)
        for arm,path in (("B0",old),("ASK",ask)):
            cfg=read(path/"config.json");metrics=read(path/"metrics.json")
            rows=read_history(path/"history.csv")
            for name in ("config.json","metrics.json","history.csv"):
                expected=record["files_sha256"][name] if arm=="B0" else s20_manifest[(path/name).relative_to(S20).as_posix()]
                assert sha(path/name)==expected, "Historical log/config changed"
            if arm=="B0": assert metrics["model_sha256"]==record["checkpoint_sha256"]
            else:
                assert cfg["train_indices_sha256"]==role["train_indices_array_sha256"]
                assert cfg["val_indices_sha256"]==role["val_indices_array_sha256"]
                assert cfg["fit_rx"]==role["fit_rx"]
            runs.append(trace(arm,f,cfg,metrics,rows,len(meta["train_ids"])))
    comparisons=[overlap(a,b) for a,b in itertools.combinations(pools,2)]
    common_all=sorted(set.intersection(*(set(p["fit_rx"]) for p in pools[1:])))
    union_all=sorted(set.union(*(set(p["fit_rx"]) for p in pools[1:])))
    save("RUN_FACTORS.json",runs)
    save("CONTROL_PANEL_PLAN.json",panels(pools[2]))
    save("AUDIT.json",dict(status="CONFOUNDS_VERIFIED_CAUSAL_ATTRIBUTION_UNRESOLVED",runs=len(runs),
                           distinct_actual_seeds=len({r["actual_seed"] for r in runs}),
                           initialization_hashes_within_B0_distinct=len({r["initial_state_sha256"] for r in runs if r["arm"]=="B0"}),
                           initialization_hashes_within_ASK_distinct=len({r["initial_state_sha256"] for r in runs if r["arm"]=="ASK"}),
                           overlap=comparisons,folds123_shared_fit_rx=common_all,folds123_union_fit_rx=union_all,
                           source_cell_hashes_matched=len(cell_hashes),source_rows_reconstructed_without_iq=True,
                           cause_importance_ranked=False,variance_components_identified=False,
                           label_index_arrays_opened=20,raw_iq_reads=0,checkpoint_loads=0,model_forward_calls=0,
                           prediction_arrays_opened=0,new_model_fits=0,historical_STOP_changed=False,
                           row_identity_limitation="Deterministic cache lineage only; no immutable packet IDs or cross-RX physical matching"))
    save("ACCESS_LOG.json",dict(records=records,scope="Own read-only audit actions; not OS-wide instrumentation",
                               new_iq_reads=0,prediction_arrays_opened=0,checkpoint_loads=0,new_model_fits=0,
                               unique_label_index_paths=20,successful_audit_label_opens=20,
                               prior_failed_attempt_label_opens=40))
    print(json.dumps(dict(status="CONFOUNDS_VERIFIED_CAUSAL_ATTRIBUTION_UNRESOLVED",runs=8,label_arrays=20,new_fits=0)))


if __name__=="__main__": main()
