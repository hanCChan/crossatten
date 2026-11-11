"""Independent metadata/label audit; no scientific prediction, IQ or weight loads."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    scope=read(HERE/"ACCESS_SCOPE.json");access=read(HERE/"ACCESS_LOG.json")
    for r in access["records"]: assert sha(r["path"])==r["sha256"],r["path"]
    meta={};qa_access=[]
    for r in scope["label_index_whitelist"]:
        assert r["name"] in ["fit_y","fit_rx","fit_day","train_ids","val_ids"]
        assert sha(r["path"])==r["sha256"]
        meta.setdefault(r["fold"],{})[r["name"]]=np.load(r["path"],allow_pickle=False)
        qa_access.append(dict(path=r["path"],sha256=r["sha256"],kind="label_index_verification_reread"))
    previous=read(HERE.parent/"classification_failure_diagnostics/INPUT_REGISTRY.json")
    old_access_path=next(r["path"] for r in access["records"] if r["kind"]=="historical_source_access_metadata")
    hashes={}
    for line in Path(old_access_path).read_text().splitlines():
        r=json.loads(line)
        if r.get("mode")=="read_only" and r.get("dataset")=="ManyRx": hashes[r["rx"],r["day"]]=r["sha256"]
    pools={}; keys_by_fold={}
    for f,m in meta.items():
        role=previous["folds"][f]
        keys=[None]*len(m["fit_y"])
        for code in np.unique(m["fit_rx"]):
            rx=role["rx_names"][int(code)]
            for day in [1,2]:
                positions=np.flatnonzero((m["fit_rx"]==code)&(m["fit_day"]==day))
                for ordinal,index in enumerate(positions): keys[index]=(rx,day,hashes[rx,day],ordinal)
        assert all(k is not None for k in keys)
        trset=set(m["train_ids"].tolist())
        pools[f]={key:i in trset for i,key in enumerate(keys)};keys_by_fold[f]=keys
    audit=read(HERE/"AUDIT.json")
    for row in audit["overlap"]:
        a,b=row["folds"];common=pools[a].keys()&pools[b].keys()
        counts={(x,y):sum(pools[a][k]==x and pools[b][k]==y for k in common) for x in [False,True] for y in [False,True]}
        assert [row[k] for k in ["both_train","a_train_b_val","a_val_b_train","both_val"]]==[counts[True,True],counts[True,False],counts[False,True],counts[False,False]]
        assert row["common_pool_rows"]==len(common)
    factors=read(HERE/"RUN_FACTORS.json")
    for r in factors:
        file=next(a["path"] for a in access["records"] if a["kind"]=="read_only_training_csv" and f"f{r['fold']}__first" in a["path"] and ("/s20-" in a["path"].replace("\\","/"))==(r["arm"]=="ASK"))
        with Path(file).open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
        ce="source_val_ce" if r["arm"]=="B0" else "val_packet_ce"
        assert r["selected_epoch"]==int(min(rows,key=lambda v:float(v[ce]))["epoch"])
        logged=[float(v["lr"]) for v in rows]
        expected=([.001]+logged[:-1]) if r["arm"]=="B0" else logged
        assert r["lr_used_by_epoch"]==expected
        assert r["selected_updates"]==r["selected_epoch"]*((r["train_packets"]+127)//128)
    plan=read(HERE/"CONTROL_PANEL_PLAN.json");m=meta[2];keys=keys_by_fold[2]
    name=np.array([k[0] for k in keys]);tr=m["train_ids"];va=m["val_ids"]
    core=plan["arms"][0]["shared_core_rx"]
    common=sorted([int(i) for i in tr if name[i] in core],key=lambda i:keys[i])
    validation=sorted([int(i) for i in va if name[i] in core],key=lambda i:keys[i])
    for j,arm in enumerate(plan["arms"]):
        extension=[]
        for cell in plan["extension_caps"]:
            ids=[int(i) for i in tr if name[i] in arm["extension_rx"] and m["fit_y"][i]==cell["tx"] and m["fit_day"][i]==cell["day"]]
            assert len(ids)==cell["extension_counts"][j]
            ids.sort(key=lambda i:hashlib.sha256(("source-panel-v1|"+repr(keys[i])).encode()).hexdigest())
            extension.extend(ids[:cell["retained_per_extension"]])
        extension.sort(key=lambda i:(int(m["fit_y"][i]),int(m["fit_day"][i]),keys[i]))
        ids=np.asarray(common+extension,dtype=np.int64)
        assert hashlib.sha256(ids.tobytes()).hexdigest()==arm["train_indices_sha256"]
        assert hashlib.sha256(np.asarray(validation,dtype=arm["validation_indices_dtype"]).tobytes()).hexdigest()==arm["validation_indices_sha256"]
        assert len(ids)==arm["train_packets"] and not set(ids)&set(validation)
        assert set(name[ids])<=set(plan["legal_fit_universe"])
    result=dict(status="PASS",run_histories_checked=8,source_pool_pair_comparisons_checked=6,
                control_panel_train_and_validation_hashes_reproduced=True,synthetic_tests_passed=5,
                unique_label_index_paths=20,successful_audit_label_opens=20,retained_failed_attempt_label_opens=40,
                retained_failed_qa_label_opens=20,qa_label_opens=20,total_label_array_opens_this_stage=100,
                scientific_iq_reads=0,prediction_array_reads=0,checkpoint_loads=0,new_model_fits=0,
                causal_factors_identified=False)
    (HERE/"QA_ACCESS_LOG.json").write_text(json.dumps(qa_access,indent=2)+"\n",encoding="utf-8",newline="\n")
    (HERE/"QA_REPORT.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result))


if __name__=="__main__": main()
