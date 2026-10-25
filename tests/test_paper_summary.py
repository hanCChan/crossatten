"""Aggregation and paired decisions use synthetic records, not target data."""
import importlib.util
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_summary", ROOT / "06-paper-validation/06-4-summary.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def record(fold, seed, accuracy):
    cm = [[accuracy * 1000, (1 - accuracy) * 1000], [(1 - accuracy) * 1000, accuracy * 1000]]
    per = {"rx": dict(confusion=cm, acc=accuracy, macro_f1=accuracy)}
    scores = dict(acc_mean=accuracy, packet_acc=accuracy, f1_mean=accuracy)
    return dict(fold=fold, seed=seed, params=100, train_seconds=10,
                selected={"CRD": dict(summary=scores, per_rx=per)})


def main():
    base = [record(f, s, .80 + f * .01) for f in range(4) for s in (2023, 2024, 2025)]
    improved = [record(f, s, .82 + f * .01) for f in range(4) for s in (2023, 2024, 2025)]
    summary = report.aggregate(base, "CRD")
    assert abs(summary["metrics"]["acc_mean"]["mean"] - 81.5) < 1e-10
    assert summary["seed_sd"] == 0 and summary["fold_sd"] > 0
    assert abs(summary["per_fold_rx_f1_mean"]["0:rx"] - 80.0) < 1e-10
    assert summary["per_fold_rx_seed_sd"]["0:rx"] == 0
    assert summary["per_fold_rx_f1_seed_sd"]["0:rx"] == 0
    assert report.paired(improved, base, "CRD", 12)["status"] == "improves"
    assert report.paired(base, improved, "CRD", 12)["status"] == "harms"
    assert report.paired(base, base, "CRD", 12)["status"] == "unclear"
    assert report.paired(improved[:-1], base, "CRD", 12)["status"] == "incomplete"
    assert abs(report.pooled_f1(base[0]["selected"]["CRD"]["per_rx"]) - .80) < 1e-10
    assert report.paper_route({"complete": False}, {}, {})["route"] == "pending"
    comparisons, groups = {}, {}
    for dataset in ("ManyRx", "ManySig"):
        for right in ("W0", "C0", "CP", "CF"):
            comparisons[f"{dataset}:W1-{right}:CRD"] = {"status": "improves"}
        for method in ("W1", "IB-RFF", "WaveMLP", "FAN_source_selected"):
            groups[f"{dataset}:{method}:CRD"] = {"metrics": {"packet_acc": {"mean": 90}}}
    for right in ("S_cnnonly", "S_cnnparam", "S_cnnflops"):
        comparisons[f"Shen:S_hstem-{right}:dev_val"] = {"status": "improves"}
    assert report.paper_route({"complete": True}, comparisons, groups)["route"] == "cross_LoRa_WiFi_validation_supported"
    comparisons["ManyRx:W1-W0:CRD"]["status"] = "unclear"
    assert report.paper_route({"complete": True}, comparisons, groups)["route"] == "narrow_to_LoRa_specific_evidence"
    comparisons["Shen:S_hstem-S_cnnonly:dev_val"]["status"] = "unclear"
    assert report.paper_route({"complete": True}, comparisons, groups)["route"] == "engineering_or_limited_method_evidence"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        saved = root / "saved"
        saved.mkdir()
        logits = np.array([[2, 1], [1, 1], [0, 2]], dtype=np.float16)
        labels = np.array([0, 1, 0], dtype=np.int64)
        np.savez_compressed(saved / "preds_CRD.npz", rx=logits, y_rx=labels)
        prediction_result = {"selected": {"CRD": {"per_rx": {
            "rx": {"total": 3, "correct": 2, "confusion": [[1, 1], [0, 1]]}
        }}}}
        checked = report.audit_predictions(saved / "model.pt", prediction_result, 2)
        assert checked == dict(packets=3, tied_packets=1, absolute_correct_count_delta=1)
        damaged = copy.deepcopy(prediction_result)
        damaged["selected"]["CRD"]["per_rx"]["rx"]["confusion"] = [[0, 2], [0, 1]]
        try:
            report.audit_predictions(saved / "model.pt", damaged, 2)
        except AssertionError:
            pass
        else:
            raise AssertionError("untied prediction corruption was accepted")
        np.savez_compressed(saved / "preds_CRD.npz", rx=logits, y_rx=np.array([0, 2, 0]))
        try:
            report.audit_predictions(saved / "model.pt", prediction_result, 2)
        except AssertionError:
            pass
        else:
            raise AssertionError("out-of-range prediction labels were accepted")
        init = root / "init"
        init.mkdir()
        (init / "metrics.json").write_text(json.dumps(dict(fold=0, seed=2023, train_seconds=20)))
        timed = dict(stage="final", run_id="synthetic", fold=0, seed=2023, train_seconds=10)
        configs = {("final", "synthetic"): {"init": {"path": str(init / "model.pt")}}}
        cost = report.training_cost([timed], configs)
        assert cost["initialization_seconds"]["mean"] == 20 and cost["total_seconds"]["mean"] == 30
        configs[("final", "synthetic")]["init"] = None
        assert report.training_cost([timed], configs)["total_seconds"]["mean"] == 10
        report.ROOT = root
        (root / "configs").mkdir()
        (root / "configs/protocol.json").write_text("{}")
        run = root / "run"
        run.mkdir()
        config = dict(arm={"name": "synthetic", "expected_params": 1, "num_classes": 10},
                      fold=0, seed=101, stage="explore", smoke=False,
                      registration_sha256="synthetic", roles={"dev_val": ["rx"]},
                      protocol_sha256=hashlib.sha256(b"{}").hexdigest(),
                      recipe={"epochs": 2, "sched": "warmcos"})
        per = {"rx": {"confusion": [[200 if i == j else 0 for j in range(10)] for i in range(10)],
                      "correct": 2000, "total": 2000, "acc": 1.0, "macro_f1": 1.0, "ce": .1}}
        result = dict(arm="synthetic", params=1, fold=0, seed=101, stage="explore",
                      dataset="Shen", stop_epoch=2, best_val_epoch=2, source_val_ce=.1,
                      epochs_budget=2, selected={"dev_val": {"per_rx": per,
                      "summary": {"acc_mean": 1.0, "f1_mean": 1.0, "ce_mean": .1,
                                  "packet_acc": 1.0, "worst_acc": 1.0}}})
        (run / "config.json").write_text(json.dumps(config))
        (run / "history.csv").write_text("epoch,lr,train_loss,val_ce\n1,.01,.3,.2\n2,.01,.2,.1\n")
        (run / "model.pt").touch()
        (run / "preds_dev_val.npz").touch()
        path = run / "metrics.json"
        path.write_text(json.dumps(result))
        report.audit(path, "synthetic")
        for mutation in ("checkpoint", "confusion", "denominator", "budget"):
            damaged = copy.deepcopy(result)
            if mutation == "checkpoint":
                damaged["best_val_epoch"] = 1
            elif mutation == "confusion":
                damaged["selected"]["dev_val"]["per_rx"]["rx"]["confusion"][0][0] = 199
            elif mutation == "denominator":
                damaged["selected"]["dev_val"]["per_rx"]["rx"]["total"] = 1999
            else:
                damaged["epochs_budget"] = 3
            path.write_text(json.dumps(damaged))
            try:
                report.audit(path, "synthetic")
            except AssertionError:
                pass
            else:
                raise AssertionError(f"corruption was accepted: {mutation}")
    print("PASS paired completeness, source checkpoints, prediction/confusion audit, corruption rejection, pooled F1, seed/fold variation")


if __name__ == "__main__":
    main()
