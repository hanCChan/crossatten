"""Recompute audit decision from sealed metrics and metadata; no IQ/model reads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify():
    reg=read(HERE/"AUDIT_REGISTRATION.json")
    result=read(HERE/"AUDIT_RESULT.json")
    checks={}

    def require(name,condition):
        checks[name]=bool(condition)
        if not condition:
            raise ValueError("Audit QA failed: "+name)

    require("registration_sha256",result["audit_registration_sha256"]==sha(HERE/"AUDIT_REGISTRATION.json"))
    require("metric_sha256",result["packet_metrics_sha256"]==sha(HERE/"PACKET_METRICS.npz"))
    require("access_log_sha256",result["access_log_sha256"]==sha(HERE/"ACCESS_LOG.json"))
    for relative,digest in reg["code_sha256"].items():
        require("frozen_code:"+relative,sha(ROOT/relative)==digest)
    require("parent_registration_sha256",sha(ROOT/"research/standard_method_competition_c/PRE_REGISTRATION.json")==reg["source_registration_sha256"])
    plan_path=ROOT/"research/standard_method_competition_c/SOURCE_PLAN.json"
    require("parent_source_plan_sha256",sha(plan_path)==reg["source_plan_sha256"])
    plan=read(plan_path)
    require("common_fit_RX",set(reg["audit_rx"])==set.intersection(*(set(v["fit_rx"]) for v in plan["views"])))
    require("selection_RX_excluded",not set(reg["audit_rx"]) & set.union(*(set(v["source_selection_rx"]) for v in plan["views"])))
    relative=(HERE/"AUDIT_REGISTRATION.json").relative_to(ROOT).as_posix()
    prereg_commit=subprocess.check_output(["git","-C",str(ROOT),"log","-1","--format=%H","--",relative],text=True).strip()
    blob=subprocess.check_output(["git","-C",str(ROOT),"show",prereg_commit+":"+relative])
    require("committed_registration",blob==(HERE/"AUDIT_REGISTRATION.json").read_bytes())
    prereg_time=subprocess.check_output(["git","-C",str(ROOT),"show","-s","--format=%cI",prereg_commit],text=True).strip()
    attempt=read(HERE/"AUDIT_ATTEMPT.json")
    from datetime import datetime
    require("commit_precedes_IQ_attempt",datetime.fromisoformat(prereg_time)<=datetime.fromisoformat(attempt["started_at"]))

    metadata={}
    for name in ("fit_y","fit_rx","fit_day","train_ids","val_ids"):
        entry=reg["input_whitelist"][name]
        p=Path(entry["path"])
        require("metadata_sha256:"+name,sha(p)==entry["sha256"])
        metadata[name]=np.load(p,allow_pickle=False)
    y,rx,day=(metadata["fit_"+k] for k in ("y","rx","day"))
    train=metadata["train_ids"]
    with np.load(HERE/"PACKET_METRICS.npz",allow_pickle=False) as f:
        m={k:f[k] for k in f.files}
    ids=m["pool_ids"]
    expected=[]
    for receiver in reg["audit_rx"]:
        code=reg["rx_names"].index(receiver)
        for tx in range(10):
            for date in (1,2):
                cell=train[(rx[train]==code)&(y[train]==tx)&(day[train]==date)]
                expected.extend(sorted(map(int,cell),key=lambda i:hashlib.sha256(f"periodic-feasibility-v1|{i}".encode()).digest())[:reg["maximum_packets_per_cell"]])
    require("exact_bounded_stratified_sample",np.array_equal(ids,np.asarray(expected,dtype=np.int64)))
    require("unique_original_train_rows",len(np.unique(ids))==len(ids) and np.isin(ids,train).all() and not np.isin(ids,metadata["val_ids"]).any())
    require("row_hash",hashlib.sha256(ids.tobytes()).hexdigest()==result["indices_sha256"])
    require("canonical_labels",all(np.array_equal(m[k],metadata["fit_"+k][ids]) for k in ("y","rx","day")))
    require("finite_metrics",all(np.isfinite(v).all() for v in m.values()))
    require("offsets_and_shapes",m["adjacent_min_by_offset"].shape==m["neighbor_margin_by_offset"].shape==m["rho20_by_offset"].shape==(len(ids),20))
    coherence_fail=m["adjacent_min_by_offset"].min(1)<reg["min_adjacent_coherence"]
    margin_fail=m["neighbor_margin_by_offset"].min(1)<reg["min_neighbor_lag_margin"]
    passes=m["valid"] & ~coherence_fail & ~margin_fail
    require("all_packet_decisions_recomputed",np.array_equal(passes,m["passed"]))
    require("all_packets_in_denominator",result["packets"]==len(ids) and result["passed_packets"]==int(passes.sum()) and result["pass_fraction"]==float(passes.mean()))
    for record in result["rx_day"]:
        mask=(m["rx"]==reg["rx_names"].index(record["rx"]))&(m["day"]==record["day"])
        require("RX_day:"+record["rx"]+":"+str(record["day"]),record["packets"]==int(mask.sum()) and record["pass_fraction"]==float(passes[mask].mean()))
    for record in result["cells"]:
        mask=(m["rx"]==reg["rx_names"].index(record["rx"]))&(m["day"]==record["day"])&(m["y"]==record["tx"])
        require("cell:"+record["rx"]+":"+str(record["tx"])+":"+str(record["day"]),record["rows"]==int(mask.sum()) and record["pass_fraction"]==float(passes[mask].mean()))
    decision_checks=dict(overall_coverage=float(passes.mean())>=reg["min_packet_pass_fraction"],
        every_rx_day_coverage=all(r["pass_fraction"]>=reg["min_rx_day_pass_fraction"] for r in result["rx_day"]),
        present_cell_support=bool(result["cells"]) and all(c["rows"]>=reg["minimum_packets_per_present_cell"] for c in result["cells"]),
        all_rx_days_present=all(r["packets"]>0 for r in result["rx_day"]))
    require("engineering_gates_recomputed",decision_checks==result["checks"])
    expected_status="PASS_PERIODIC_COORDINATES_REQUIRE_SEPARATE_TRAINING_REGISTRATION" if all(decision_checks.values()) else "STOP_FIXED_PERIODIC_WINDOW_NOT_RELIABLE"
    require("stop_decision_recomputed",result["status"]==expected_status)
    log=read(HERE/"ACCESS_LOG.json")
    allowed_paths={entry["path"] for entry in reg["input_whitelist"].values()}
    require("access_paths_whitelisted",all(row["path"] in allowed_paths for row in log))
    iq_actions=[row for row in log if row["action"].startswith("mmap_")]
    require("one_bounded_IQ_copy",len(iq_actions)==1 and iq_actions[0]["rows"]==len(ids) and iq_actions[0]["indices_sha256"]==result["indices_sha256"])
    require("no_gradient_fits",result["gradient_fits"]==reg["gradient_fits"]==0)
    value=dict(status="PASS",checks=checks,check_count=len(checks),preregistration_commit=prereg_commit,
        IQ_read_by_verifier=False, classifier_performance_evaluated=False,
        audit_result_sha256=sha(HERE/"AUDIT_RESULT.json"),
        failure_counts=dict(coherence=int(coherence_fail.sum()),neighbor_margin=int(margin_fail.sum()),
            both=int((coherence_fail&margin_fail).sum()),either=int((~passes).sum())),
        failed_rx_days=[r for r in result["rx_day"] if r["pass_fraction"]<reg["min_rx_day_pass_fraction"]],
        interpretation="Finite source feasibility sample and engineering gate; not a test of classifier gain or a rejection of all periodic architectures.")
    (HERE/"VERIFICATION.json").write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:value[k] for k in ("status","check_count","failure_counts","failed_rx_days")},ensure_ascii=False))
    return value


if __name__=="__main__":verify()
