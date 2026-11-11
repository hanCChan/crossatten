"""Independent float64 rescoring of registered predictions; no IQ/models."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PREFIX=HERE.relative_to(ROOT).as_posix()+"/"


def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b): assert np.allclose(a,b,rtol=1e-10,atol=1e-10), (a,b)


def main():
    registry=read(HERE/"INPUT_REGISTRY.json")
    protocol=read(HERE/"G0_PROTOCOL.json")
    assert sha(HERE/"INPUT_REGISTRY.json")==protocol["registry_sha256"]
    for name,h in protocol["code_sha256"].items(): assert sha(HERE/name)==h
    for path,h in registry["metadata_sha256"].items(): assert sha(path)==h
    access=read(HERE/"ACCESS_LOG.json")["records"]
    assert len(access)==36 and {r["path"] for r in access}=={r["path"] for r in registry["arrays"]}
    loaded={};qa_access=[];groups_checked=0;original_metrics_checked=0
    for item in registry["arrays"]:
        if item["kind"]!="sealed_predictions": continue
        path=Path(item["path"]);assert sha(path)==item["sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files)=={"logits","y","rx","day"}
            a={k:z[k] for k in z.files}
        z=a["logits"].astype(np.float64);y=a["y"];n=len(y)
        maximum=z.max(1)
        ce=maximum+np.log(np.exp(z-maximum[:,None]).sum(1))-z[np.arange(n),y]
        pred=z.argmax(1);err=pred!=y
        wrong=z.copy();wrong[np.arange(n),y]=-np.inf
        neg=wrong.argmax(1);margin=z[np.arange(n),y]-wrong.max(1)
        scale=np.sqrt(np.mean((z-z.mean(1,keepdims=True))**2,axis=1))
        norm=np.divide(margin,scale,out=np.zeros_like(margin),where=scale>0)
        prob=np.exp(z-maximum[:,None]);prob/=prob.sum(1,keepdims=True)
        conf=prob.max(1)
        names=np.asarray(item["rx_names"])[a["rx"]]
        assert set(names)==set(item["allowed_rx"]) and set(a["day"])=={1,2}
        output=read(HERE/f"groups/{item['arm']}_f{item['fold']}_{item['role']}.json")
        summary=output["summary"]; packet=summary["packet"]
        assert packet["n"]==n and packet["errors"]==int(err.sum())
        close(packet["ce"],ce.mean());close(packet["margin_mean"],margin.mean())
        close(packet["normalized_margin_mean"],norm.mean())
        close(packet["margin_quantiles"],np.quantile(margin,[.05,.25,.5,.75,.95]))
        for row in output["per_tx_rx_day"]:
            mask=(names==row["rx"])&(y==row["tx"])&(a["day"]==row["day"])
            count=int(mask.sum());assert count==row["n"]
            assert row["errors"]==int(err[mask].sum())
            if count:
                close(row["ce"],ce[mask].mean());close(row["margin_mean"],margin[mask].mean())
                close(row["normalized_margin_mean"],norm[mask].mean())
                close(row["margin_quantiles"],np.quantile(margin[mask],[.05,.25,.5,.75,.95]))
                close(row["normalized_margin_quantiles"],np.quantile(norm[mask],[.05,.25,.5,.75,.95]))
                assert row["high_confidence_errors"]==int((err[mask]&(conf[mask]>=.9)).sum())
            else:
                assert row["accuracy"] is None and row["ce"] is None
            assert row["hardest_negative_counts"]==np.bincount(neg[mask],minlength=10).tolist()
            assert row["predicted_counts"]==np.bincount(pred[mask],minlength=10).tolist()
            groups_checked+=1
        for row in output["per_rx"]:
            mask=names==row["rx"]
            cm=np.bincount(y[mask]*10+pred[mask],minlength=100).reshape(10,10)
            assert row["confusion"]==cm.tolist()==item["expected_metrics"]["per_rx"][row["rx"]]["confusion"]
            original_metrics_checked+=1
        if item["role"]=="fit_validation":
            assert sum(r["training_support"] for r in output["per_tx_rx_day"])==registry["folds"][item["fold"]]["train_packets"]
        loaded[item["fold"],item["arm"],item["role"]]=dict(y=y,rx=names,day=a["day"],pred=pred,error=err)
        qa_access.append(dict(path=str(path),sha256=item["sha256"],kind="registered_prediction_reread",fold=item["fold"],arm=item["arm"],role=item["role"]))
    pair_groups=0
    for f in range(4):
        for role in ("fit_validation","held_source"):
            a,b=loaded[f,"B0",role],loaded[f,"ASK",role]
            for k in ("y","rx","day"): assert np.array_equal(a[k],b[k])
            pair=read(HERE/f"pairs/f{f}_{role}.json")
            for row in pair["per_tx_rx_day"]:
                mask=(a["rx"]==row["rx"])&(a["y"]==row["tx"])&(a["day"]==row["day"])
                assert int(mask.sum())==row["n"]
                if row["n"]:
                    ae,be=a["error"][mask],b["error"][mask]
                    assert row["both_wrong"]==int((ae&be).sum())
                    assert row["b0_wrong_ask_correct"]==int((ae&~be).sum())
                    assert row["b0_correct_ask_wrong"]==int((~ae&be).sum())
                    close(row["delta_accuracy_ask_minus_b0"],(ae.astype(int)-be).mean())
                pair_groups+=1
    repeated=read(HERE/"REPEATED_INPUT_FOLDS.json")
    for row in repeated["rows"]:
        errors=[]
        for f in row["folds"]:
            d=loaded[f,row["arm"],"held_source"]
            mask=d["rx"]==row["rx"]
            if row["tx"] is not None: mask&=d["y"]==row["tx"]
            errors.append(d["error"][mask])
        stack=np.stack(errors)
        assert stack.shape[1]==row["n"]
        assert row["all_folds_wrong"]==int(stack.all(0).sum())
        assert row["some_but_not_all_folds_wrong"]==int((stack.any(0)&~stack.all(0)).sum())
    changed=subprocess.check_output(["git","-C",str(ROOT),"diff","--name-only",registry["base_commit"]]).decode().splitlines()
    assert all(p.startswith(PREFIX) for p in changed), "Historical tree modified"
    result=dict(status="PASS",prediction_files_independently_rescored=16,tx_rx_day_groups_checked=groups_checked,
                paired_groups_checked=pair_groups,archived_rx_confusions_exactly_reproduced=original_metrics_checked,
                registered_code_unchanged=True,historical_tree_diff_isolated=True,
                original_array_accesses=36,qa_prediction_rereads=16,unique_registered_array_paths=36,
                iq_reads=0,checkpoint_loads=0,model_forward_calls=0,new_classifier_fits=0,
                scope="Independent numerical checks; does not establish causal RX mechanisms, new blind evidence or certified robustness")
    (HERE/"QA_ACCESS_LOG.json").write_text(json.dumps(qa_access,indent=2)+"\n",encoding="utf-8",newline="\n")
    (HERE/"QA_REPORT.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result))


if __name__=="__main__": main()
