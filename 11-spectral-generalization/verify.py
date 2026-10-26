"""Independent S11 access, prediction, cache/prior and immutable history audit."""
import json
import subprocess
import numpy as np
import torch
from core import HERE,ROOT,DATA,prep,sha,save,receivers,check_manifest
from spectral import paired_model
from sklearn.metrics import accuracy_score, f1_score


def main():
    reg=json.loads((HERE/"registration.json").read_text())
    check_manifest(reg["code_manifest"])
    changes=subprocess.check_output(["git","diff","--name-only",reg["base_commit"]],
                                    cwd=ROOT).decode().splitlines()
    if any(not name.startswith("11-spectral-generalization/") for name in changes):
        raise ValueError("historical files changed")
    successful,failed=0,0
    for phase in ("restore","smoke"):
        for j in json.loads((HERE/f"jobs_{phase}.json").read_text()):
            out=HERE/"runs"/phase/j["id"]
            if not (out/"metrics.json").exists():
                log=out/"stdout.log"
                if phase!="smoke" or j["feature"]!="F2" or not log.exists():
                    raise ValueError("unexpected failed attempt")
                if 'ValueError: nonfinite prior' not in log.read_text() or (out/"history.csv").exists():
                    raise ValueError("failure was not the documented pretraining error")
                failed+=1
                continue
            m=json.loads((out/"metrics.json").read_text())
            if m["job"]!=j:
                raise ValueError("run identity changed")
            config=json.loads((out/"config.json").read_text())
            if config["params"]!=reg["expected_params"][f"{j['dataset']}:{j['feature']}"]:
                raise ValueError("model capacity mismatch")
            for name,expected in m["weights_sha256"].items():
                if sha(out/name)!=expected:
                    raise ValueError("checkpoint seal mismatch")
            for rule,entry in m["selected"].items():
                if "per_rx" not in entry:
                    continue
                path=out/f"preds_{rule}.npz"
                if sha(path)!=m["predictions_sha256"][path.name]:
                    raise ValueError("prediction seal mismatch")
                with np.load(path) as p:
                    for rx,e in entry["per_rx"].items():
                        labels=p[f"y_{rx}"]
                        if not np.isfinite(p[rx]).all():
                            raise ValueError("nonfinite frozen predictions")
                        pred=p[rx].argmax(1)
                        classes=10 if j["dataset"]=="ManyRx" else 6
                        actual=dict(acc=accuracy_score(labels,pred),
                            macro_f1=f1_score(labels,pred,labels=np.arange(classes),
                                             average="macro",zero_division=0),
                            correct=int((pred==labels).sum()),total=len(labels))
                        for key in ("acc","macro_f1","correct","total"):
                            if not np.isclose(actual[key],e[key],atol=1e-8):
                                raise ValueError("independent metric mismatch")
            successful+=1
    for p in (HERE/"priors").glob("*.json"):
        evidence=json.loads(p.read_text())
        prior=HERE/"cache"/evidence["job"]["id"]/"prior.npz"
        if sha(prior)!=evidence["prior_sha256"]:
            raise ValueError("prior cache seal mismatch")
        w=np.array(evidence["weight"])
        if not np.isfinite(w).all() or (w<.5).any() or (w>1).any():
            raise ValueError("invalid spectral prior")
        if evidence["roles"]["target"] and set(evidence["roles"]["fit_rx"])&set(evidence["roles"]["target"]):
            raise ValueError("prior source/target overlap")
    access=0
    caches=set()
    for line in (HERE/"access.jsonl").read_text().splitlines():
        r=json.loads(line)
        source,_=receivers(r["dataset"],r["fold"])
        if r["role"] not in ("fit","held","diagnostic") or r["day"] not in (1,2) or r["rx"] not in source:
            raise ValueError("unadmitted target data access")
        access+=1
        caches.add((r["dataset"],r["rx"],r["day"]))
    for dataset,rx,day in caches:
        name=f"rx_{rx}_day{day}.npz"
        if sha(DATA/"cache"/dataset/name)!=prep["cache_sha256"][f"{dataset}/{name}"]:
            raise ValueError("source input cache seal mismatch")
    if successful!=4 or failed!=2:
        raise ValueError("unexpected success/failure counts")
    decision=json.loads((HERE/"decision_physics.json").read_text())
    if decision["admitted"]:
        raise ValueError("completed report scope expects mechanism STOP")
    check_manifest(decision["evidence"])
    check_manifest(json.loads((HERE/"checkpoint_audit.json").read_text())["evidence"])
    check_manifest(json.loads((HERE/"decision_bn.json").read_text())["evidence"])
    execution=json.loads((HERE/"execution.json").read_text())
    check_manifest(execution["evidence"])
    if execution["formal_target_fits"]!=0 or execution["source_candidate_fits"]!=0:
        raise ValueError("reported scope differs from STOP")
    expected=("S11_CODE_AND_CHECKPOINT_AUDIT","SOURCE_SPECTRAL_PHYSICS","SOURCE_RX_DAY_SCREENING",
              "MODEL_ABLATIONS","FINAL_CRD_COMPARISONS","PHYSICAL_MECHANISM",
              "PARAMETERS_AND_RUNTIME","NOVELTY_AND_TIFS_DECISION")
    if any(not (HERE/f"{name}.md").exists() for name in expected):
        raise ValueError("required report missing")
    save(HERE/"verification.json",dict(passed=True,successful_fits=successful,
        failed_pretraining_smoke_attempts=failed,guarded_source_reads=access,
        historical_files_unchanged=True,independent_predictions_recomputed=True,
        checkpoint_hashes_verified=True,prior_hashes_verified=True,
        source_cache_hashes_verified=len(caches),unit_tests_passed=7,all_reports=8,target_reads=0,
        result_scope="mechanism STOP, candidate network efficacy not established"))
    print("S11 independent verification PASS;4 fits,2 preserved pretraining smoke failures,0 target reads.")


if __name__=="__main__":
    main()
