#!/usr/bin/env python3
"""Final S08 acceptance: lineage, source roles, temporal freeze and preservation."""
import csv
import importlib.util
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from training import HERE, ROOT, RULES, require_committed, sha, source_groups
from xa.wisig import receivers


def check_access(jobs, path, freeze=None):
    allowed, expected = {}, set()
    for job in jobs:
        source, target = receivers(job["dataset"], job["fold"])
        if job["phase"] == "final":
            fit, held = source, []
        else:
            fit, held = source_groups(job["dataset"], job["fold"], job["group"])
        events = {(rx, day, "fit") for rx in fit for day in (1, 2)}
        events |= {(rx, day, "held") for rx in held for day in (1, 2)}
        if job["phase"] == "final":
            events |= {(rx, day, "target") for rx in target for day in (1, 2, 3, 4)}
        allowed[job["id"]] = (job, events)
        expected |= {(job["id"], *event) for event in events}
    observed = Counter()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["role"] == "diagnostic":
            source, _ = receivers(row["dataset"], row["fold"])
            if row["rx"] not in source or row["day"] not in (1, 2):
                raise ValueError("diagnostic crossed source roles")
            continue
        if row["run"] not in allowed:
            raise ValueError("unregistered data access")
        job, events = allowed[row["run"]]
        event = (row["rx"], row["day"], row["role"])
        if row["dataset"] != job["dataset"] or row["fold"] != job["fold"] or event not in events:
            raise ValueError("RX/date/role mismatch")
        if row["role"] == "target":
            if freeze is None or datetime.fromisoformat(row["t"]) < freeze:
                raise ValueError("target read before committed final freeze")
        observed[(row["run"], *event)] += 1
    if not expected <= observed.keys():
        raise ValueError("missing access evidence")
    return dict(distinct_events=len(observed), repeated_events=sum(v - 1 for v in observed.values()))


def main():
    spec = importlib.util.spec_from_file_location("s08_acceptance_audit", HERE / "08-4-audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    registration = json.loads((HERE / "registration.json").read_text())
    decision = json.loads((HERE / "decision_final.json").read_text())
    require_committed(HERE / "decision_final.json")
    phases = ["smoke", "screen"]
    phases += [p for p in ("phase", "manyrx") if (HERE / f"jobs_{p}.json").exists()]
    if decision["jobs"]:
        phases.append("final")
    jobs = [job for phase in phases for job in audit.audit_phase(phase)]
    formal = [j for j in jobs if j["phase"] != "smoke"]
    if len(formal) > registration["max_formal_fits"]:
        raise ValueError("formal cap exceeded")
    freeze = subprocess.check_output(
        ["git", "log", "-1", "--format=%cI", "--", "08-wisig-training/decision_final.json"],
        cwd=ROOT, text=True).strip()
    access = check_access(jobs, HERE / "access.jsonl", datetime.fromisoformat(freeze))
    from xa.wisig import load_cell
    for job in jobs:
        out = HERE / "runs" / job["phase"] / job["id"]
        config = json.loads((out / "config.json").read_text())
        result = json.loads((out / "metrics.json").read_text())
        source, target = receivers(job["dataset"], job["fold"])
        expected_fit, expected_held = (source, []) if job["phase"] == "final" else source_groups(
            job["dataset"], job["fold"], job["group"])
        if config["roles"] != dict(source=source, fit_rx=expected_fit, held_rx=expected_held,
                                   target=target, source_days=[1, 2]):
            raise ValueError("recorded receiver groups differ")
        if config["ema"]["average_buffers"] or config["ema"]["half_life_epochs"] != 1:
            raise ValueError("EMA policy differs")
        hist = list(csv.DictReader((out / "history.csv").open()))
        if any(not np.isfinite(float(v)) for row in hist for v in row.values()):
            raise ValueError("nonfinite history")
        if job["phase"] == "final" and "fixed_epochs" in job and len(hist) != job["fixed_epochs"]:
            raise ValueError("source-frozen refit epoch changed")
        if job["condition"] == "official":
            if "awgn" in config["recipe"]:
                raise ValueError("official control changed augmentation")
        elif config["recipe"]["awgn"] != [10, 50]:
            raise ValueError("registered noise range changed")
        for filename in result["predictions_sha256"]:
            label = filename.removeprefix("preds_").removesuffix(".npz")
            days = (3, 4) if label == "CRD" else (1, 2)
            with np.load(out / filename) as archive:
                for rx in (k for k in archive.files if not k.startswith("y_")):
                    expected = np.concatenate([load_cell(job["dataset"], rx, day)[1] for day in days])
                    if job["phase"] == "smoke":
                        expected = expected[:64]
                    if not np.array_equal(archive[f"y_{rx}"], expected):
                        raise ValueError("prediction labels differ from frozen cache order")
    baseline = registration["baseline"]
    previous = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", baseline], cwd=ROOT, text=True).splitlines())
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", baseline], cwd=ROOT, text=True).splitlines())
    if changed & (previous - {".gitignore"}):
        raise ValueError("historical tracked evidence/code changed")
    tracked = subprocess.check_output(["git", "ls-files", "08-wisig-training"], cwd=ROOT, text=True).splitlines()
    if any(Path(p).suffix in (".pt", ".pth") for p in tracked):
        raise ValueError("new weights are tracked")
    report_path = HERE / "final_report_evidence.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["decision_sha256"] != sha(HERE / "decision_final.json"):
            raise ValueError("report decision lineage differs")
        for name, value in report["files"].items():
            if sha(HERE / name) != value:
                raise ValueError("changed report artifact")
    acceptance = dict(
        status="PASS", timestamp=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        baseline=baseline, phases=phases, formal_fits=len(formal), smoke_fits=len(jobs) - len(formal),
        final_freeze_committed_at=freeze, access=access, historical_files="unchanged", weights="untracked",
        files={p.name: sha(p) for p in (HERE / "registration.json", HERE / "decision_final.json",
                                      HERE / "access.jsonl", HERE / "costs.json")},
        note="No new target-driven search, automatic merge, push or weight publication.")
    (HERE / "acceptance.json").write_text(json.dumps(acceptance, indent=1) + "\n")
    print(json.dumps(dict(status="PASS", formal_fits=len(formal), historical_files="unchanged")), flush=True)


if __name__ == "__main__":
    main()
