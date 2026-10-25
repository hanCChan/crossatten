#!/usr/bin/env python3
"""Read-only acceptance checks for the completed, bounded Stage-07 artifacts."""
import csv
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "lib"))
from xa.paper_train import metrics, require_committed, sha
from xa.wisig import receivers

PHASES = ("features", "replicate", "training", "training_replicate", "combine", "final", "smoke")


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def wifi_access(jobs, log_path, freeze_time):
    allowed, expected = {}, set()
    for job in jobs:
        source, target = receivers(job["dataset"], job["fold"])
        held_count = 6 if job["dataset"] == "ManyRx" else 2
        held = [] if job["phase"] == "final" else source[-held_count:]
        fit = source if job["phase"] == "final" else source[:-held_count]
        events = {(rx, day, "fit_source") for rx in fit for day in (1, 2)}
        events |= {(rx, day, "held_source") for rx in held for day in (1, 2)}
        if job["phase"] == "final":
            events |= {(rx, day, "target_report") for rx in target for day in (1, 2, 3, 4)}
        allowed[job["id"]] = (job, events)
        expected |= {(job["id"], *event) for event in events}
    observed, reads = Counter(), 0
    for line in log_path.read_text().splitlines():
        row = json.loads(line)
        if row["run"] not in allowed:
            raise ValueError("data access has an unregistered run")
        job, events = allowed[row["run"]]
        event = (row["rx"], row["day"], row["purpose"])
        if row["dataset"] != job["dataset"] or row["fold"] != job["fold"] or event not in events:
            raise ValueError("data access violates registered RX/day/role")
        if row["purpose"] == "target_report":
            timestamp = datetime.fromisoformat(row["t"]).replace(tzinfo=ZoneInfo("Asia/Shanghai"))
            if timestamp < freeze_time:
                raise ValueError("target read predates the committed final freeze")
        observed[(row["run"], *event)] += 1
        reads += 1
    if not expected <= observed.keys():
        raise ValueError("missing registered data-access evidence")
    return dict(reads=reads, distinct_events=len(observed),
                duplicate_events=sum(n - 1 for n in observed.values()),
                note="Duplicates from retained attempts are allowed; every role must remain legal.")


def osu_fits(audit):
    reg_path, manifest_path = HERE / "osu_registration.json", HERE / "osu_data_manifest.json"
    reg, manifest = json.loads(reg_path.read_text()), json.loads(manifest_path.read_text())
    jobs = json.loads((HERE / "jobs_osu.json").read_text())
    expected = {(source, method, seed) for source in ("RX1", "RX2")
                for method in reg["methods"] for seed in reg["seeds"]}
    if len(jobs) != 24 or {(j["source"], j["method"], j["seed"]) for j in jobs} != expected:
        raise ValueError("missing or duplicate OSU formal cell")
    results, legal_reads, expected_reads = {}, {}, set()
    for job in jobs:
        out = HERE / "runs/osu" / job["id"]
        config = json.loads((out / "config.json").read_text())
        result = json.loads((out / "metrics.json").read_text())
        if job["source"] == job["target"] or config["job"] != job or result["job"] != job:
            raise ValueError("OSU job/receiver identity mismatch")
        if result["config"] != config or result["smoke"] or config["smoke"]:
            raise ValueError("OSU config or formal/smoke mismatch")
        if config["registration_sha256"] != sha(reg_path) or config["manifest_sha256"] != sha(manifest_path):
            raise ValueError("OSU frozen inputs changed")
        if config["trainer_sha256"] != sha(HERE / "07-6-osu-train.py") or config["recipe"] != reg["recipe"]:
            raise ValueError("OSU frozen trainer changed")
        if sha(out / "model.pt") != result["weights_sha256"]:
            raise ValueError("OSU weights changed")
        history = list(csv.DictReader((out / "history.csv").open()))
        if len(history) != reg["recipe"]["epochs"] or \
                [int(r["epoch"]) for r in history] != list(range(1, len(history) + 1)):
            raise ValueError("OSU incomplete epoch budget")
        if any(not np.isfinite(float(r[k])) for r in history for k in
               ("train_ce", "train_acc", "val_ce", "val_acc", "lr")):
            raise ValueError("OSU nonfinite history")
        selected = int(np.argmin([float(r["val_ce"]) for r in history])) + 1
        if result["selected_epoch"] != selected or \
                not np.isclose(result["source_val_ce"], float(history[selected - 1]["val_ce"])):
            raise ValueError("OSU checkpoint is not minimum source validation CE")
        target_records = sorted((r for r in manifest["records"] if r["rx"] == job["target"]),
                                key=lambda r: r["tx"])
        labels, starts = [], []
        for entry in target_records:
            labels.append(np.full(entry["counts"]["target"], entry["tx"] - 1, dtype=np.int64))
            with np.load(HERE / "cache/osu" / f"{job['target']}_TX{entry['tx']}.npz") as cached:
                starts.append(cached["target_start"])
        with np.load(out / "preds_target.npz") as archive:
            logits, y = archive["logits"], archive["y"]
            if logits.dtype != np.float32 or logits.shape != (len(y), 25) or not np.isfinite(logits).all():
                raise ValueError("OSU invalid float32 prediction shape")
            if not np.array_equal(y, np.concatenate(labels)) or \
                    not np.array_equal(archive["start"], np.concatenate(starts)):
                raise ValueError("OSU target labels/crop identities differ from manifest")
            actual = metrics(logits, y, 25)
            cm = np.asarray(actual["confusion"])
            actual["tx_equal_acc"] = float(np.mean(cm.diagonal() / cm.sum(1)))
            audit.same_metrics(actual, result["target"])
        expected_params = dict(patch_hstu=581111, hstem_hstu=143367,
                               hstem_cnn=101129, cis_rank16=660249)
        if result["params"] != expected_params[job["method"]] or result["params"] != config["params"]:
            raise ValueError("OSU parameter count differs")
        events = {}
        for rx, split in ((job["source"], "train"), (job["source"], "val"), (job["target"], "target")):
            for entry in (r for r in manifest["records"] if r["rx"] == rx):
                event = (rx, entry["tx"], split)
                events[event] = entry["counts"][split]
                expected_reads.add((job["id"], *event))
        legal_reads[job["id"]] = events
        legal_reads[f"posthoc:{job['id']}"] = {
            (r["rx"], r["tx"], "target"): r["counts"]["target"]
            for r in manifest["records"] if r["rx"] == job["source"]}
        for split, key in (("train", "source_train"), ("val", "source_val")):
            if config[key] != sum(n for (rx, tx, s), n in events.items() if s == split):
                raise ValueError("OSU source counts differ from manifest")
        results[job["id"]] = dict(metrics_sha256=sha(out / "metrics.json"),
                                 predictions_sha256=sha(out / "preds_target.npz"))
    check_osu_access(legal_reads, expected_reads, HERE / "osu_access.jsonl")
    return results


def check_osu_access(legal_reads, expected_reads, path):
    seen = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        event = (row["rx"], row["tx"], row["split"])
        if row["run"] not in legal_reads or legal_reads[row["run"]].get(event) != row["count"]:
            raise ValueError("OSU data access violates frozen source/target roles")
        seen.add((row["run"], *event))
    if not expected_reads <= seen:
        raise ValueError("missing OSU data access evidence")


def main():
    audit = module("s07_acceptance_wifi", "07-7-audit.py")
    jobs = []
    for phase in PHASES:
        require_committed(HERE / f"jobs_{phase}.json")
        audit.audit_wifi(phase)
        jobs.extend(json.loads((HERE / f"jobs_{phase}.json").read_text()))
    formal = [j for j in jobs if j["phase"] != "smoke"]
    if len(formal) > json.loads((HERE / "registration.json").read_text())["max_wifi_fits"]:
        raise ValueError("WiSig formal cap exceeded")
    freeze = subprocess.check_output(
        ["git", "log", "-1", "--format=%cI", "--", "07-wifi-osu/decision_final.json"],
        cwd=ROOT, text=True).strip()
    accesses = wifi_access(jobs, HERE / "access.jsonl", datetime.fromisoformat(freeze))
    osu = osu_fits(audit)
    evidence = json.loads((HERE / "wifi_final_report_evidence.json").read_text())
    if evidence["final_decision_sha256"] != sha(HERE / "decision_final.json"):
        raise ValueError("final report lineage changed")
    for name, expected in evidence["files"].items():
        if sha(HERE / name) != expected:
            raise ValueError("final report file changed")
    costs = json.loads((HERE / "costs.json").read_text())
    if costs["final_decision_sha256"] != sha(HERE / "decision_final.json"):
        raise ValueError("cost graph lineage changed")
    baseline = json.loads((HERE / "registration.json").read_text())["baseline"]
    old_names = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", baseline], cwd=ROOT, text=True).splitlines())
    changed = subprocess.check_output(["git", "diff", "--name-only", baseline],
                                      cwd=ROOT, text=True).splitlines()
    if set(changed) & (old_names - {".gitignore"}):
        raise ValueError("historical tracked baseline files changed")
    tracked = subprocess.check_output(["git", "ls-files", "07-wifi-osu"],
                                      cwd=ROOT, text=True).splitlines()
    if any(Path(name).suffix in (".pt", ".pth") for name in tracked):
        raise ValueError("new model weights are tracked")
    acceptance = dict(
        status="PASS", timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        baseline=baseline, final_freeze_committed_at=freeze, formal_wifi_fits=len(formal),
        wifi_access=accesses, osu_formal_fits=osu,
        files={name: sha(HERE / name) for name in (
            "registration.json", "decision_final.json", "access.jsonl", "osu_access.jsonl",
            "osu_data_manifest.json", "osu_cache_audit.json", "costs.json",
            "wifi_final_report_evidence.json")},
        note="No new fits, target-based decisions, merge, push or weight publication.")
    (HERE / "acceptance.json").write_text(json.dumps(acceptance, indent=1) + "\n")
    print(json.dumps(dict(status="PASS", formal_wifi_fits=len(formal), osu_fits=len(osu),
                          historical_files="unchanged", weights="not tracked")), flush=True)


if __name__ == "__main__":
    main()
