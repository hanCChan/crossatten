import importlib.util
import json
import numpy as np
import pytest
from training import HERE


spec = importlib.util.spec_from_file_location("report", HERE / "09-4-report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_cluster_interval_constant_and_sign():
    mean, lo, hi = report.bootstrap([2., 2., 2., 2.])
    assert mean == lo == hi == 2
    mean, lo, hi = report.bootstrap([-1, 0, 1])
    assert mean == 0 and lo <= mean <= hi


def test_incomplete_target_cannot_generate_report(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "HERE", tmp_path)
    (tmp_path / "decision_final.json").write_text(json.dumps(dict(
        target_jobs=[dict(phase="attribution", id="missing")], selected={})))
    with pytest.raises(ValueError, match="incomplete"):
        report.target_report()
    assert not (tmp_path / "S09_FINAL_RESULTS.md").exists()


def test_complete_grid_reports_factorial_and_capacity(tmp_path, monkeypatch):
    root = tmp_path / "root"
    here = root / "09-phase-invariance"
    here.mkdir(parents=True)
    monkeypatch.setattr(report, "HERE", here)
    monkeypatch.setattr(report, "ROOT", root)
    baseline = root / "06-paper-validation/summary/paper"
    baseline.mkdir(parents=True)
    (baseline / "receiver_metrics.csv").write_text(
        "dataset,model,task,fold,rx,seed_mean_accuracy_pct\n"
        "ManySig,IB-RFF,CRD,0,r0,90\n"
        "ManySig,IB-RFF,CRD,1,r1,90\n"
        "ManySig,IB-RFF,CRD,2,r2,90\n"
        "ManySig,IB-RFF,CRD,3,r3,90\n")
    jobs = []
    for fold in range(4):
        for seed in (9101, 9102, 9103):
            for feature, policy, acc in (
                    ("first", "fixed", .95), ("null", "fixed", .9),
                    ("frequency", "fixed", .9), ("first", "packet", .92),
                    ("null", "packet", .89)):
                job = dict(dataset="ManySig", phase="attribution", feature=feature,
                           policy=policy, seed=seed, fold=fold, id=f"{fold}_{seed}_{feature}_{policy}")
                jobs.append(job)
                out = here / "runs/attribution" / job["id"]
                out.mkdir(parents=True)
                metric = dict(acc=acc, macro_f1=acc, ce=.1, correct=int(acc * 100), total=100)
                (out / "target_metrics.json").write_text(json.dumps(dict(selected={
                    t: dict(per_rx={f"r{fold}": metric}) for t in ("CR", "CRD")})))
    (here / "decision_final.json").write_text(json.dumps(dict(target_jobs=jobs, selected={})))
    report.target_report()
    import csv
    rows = list(csv.DictReader((here / "paired_comparisons.csv").open()))
    phase = next(r for r in rows if r["task"] == "CRD" and r["contrast"] == "first:fixed-null:fixed")
    assert float(phase["delta_pp"]) == pytest.approx(5)
    interaction = list(csv.DictReader((here / "factorial_interaction.csv").open()))
    assert float(interaction[0]["interaction_pp"]) == pytest.approx(2)
    assert (here / "final_fold_metrics.csv").exists()
    assert (here / "report_evidence.json").exists()
