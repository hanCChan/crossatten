"""Read-only acceptance: all fits, paired configs, frozen code and access roles."""
import argparse
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from training import HERE, ROOT, BASELINE, sha, validate, save
from xa.wisig import receivers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    reg = json.loads((HERE / "registration.json").read_text())
    phases = ["smoke", "attribution", "source"] + ([] if args.source_only else ["final"])
    configs, jobs_by_id, fits = defaultdict(list), {}, 0
    for phase in phases:
        jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
        if len(jobs) > reg["caps"][phase]:
            raise ValueError("phase cap exceeded")
        for job in jobs:
            validate(job)
            jobs_by_id[(phase, job["id"])] = job
            out = HERE / "runs" / phase / job["id"]
            result = json.loads((out / "metrics.json").read_text())
            config = json.loads((out / "config.json").read_text())
            if result["job"] != job or config["job"] != job or result["config_sha256"] != sha(out / "config.json"):
                raise ValueError("run/config identity changed")
            for name, expected in result["weights_sha256"].items():
                if sha(out / name) != expected:
                    raise ValueError("checkpoint hash changed")
            for name, expected in result["predictions_sha256"].items():
                if sha(out / name) != expected:
                    raise ValueError("source predictions changed")
            if "epochs" in job and job["policy"] != "packet" and result["stop_epoch"] != job["epochs"]:
                raise ValueError("fixed-budget run stopped early")
            if phase == "source" and any(k in result["selected"] for k in ("CR", "CRD")):
                raise ValueError("source fit has target output")
            key = (phase, job["dataset"], job["fold"], job.get("group", "all"), job["seed"])
            configs[key].append(config)
            fits += phase != "smoke"
    for group in configs.values():
        if len({r["shared_initialization_sha256"] for r in group}) != 1 or \
           len({json.dumps(r["split_sha256"], sort_keys=True) for r in group}) != 1:
            raise ValueError("strict pairing failed")
    if fits > reg["max_formal_fits"]:
        raise ValueError("formal budget exceeded")
    events = [json.loads(line) for line in (HERE / "access.jsonl").read_text().splitlines()]
    observed = set()
    freeze = None
    if not args.source_only:
        commit_time = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI", "--", "09-phase-invariance/decision_final.json"], cwd=ROOT).decode().strip()
        freeze = datetime.fromisoformat(commit_time)
    for event in events:
        source, target = receivers(event["dataset"], event["fold"])
        if event["role"] == "target":
            if args.source_only or datetime.fromisoformat(event["t"]) < freeze:
                raise ValueError("target access preceded committed decision")
            if event["rx"] not in target or (event["task"], event["day"]) not in (
                    ("CR", 1), ("CR", 2), ("CRD", 3), ("CRD", 4)):
                raise ValueError("invalid target role/date")
        elif event["rx"] not in source or event["day"] not in (1, 2):
            raise ValueError("illegal source access")
        observed.add((event["phase"], event["run"], event["role"], event["rx"], event["day"]))
    for (phase, run), job in jobs_by_id.items():
        config = json.loads((HERE / "runs" / phase / run / "config.json").read_text())
        for role, names in (("fit", config["roles"]["fit_rx"]), ("held", config["roles"]["held_rx"])):
            for rx in names:
                for day in (1, 2):
                    if (phase, run, role, rx, day) not in observed:
                        raise ValueError("missing dataset read event")
    if not args.source_only:
        decision = json.loads((HERE / "decision_final.json").read_text())
        for job in decision["target_jobs"]:
            out = HERE / "runs" / job["phase"] / job["id"]
            result = json.loads((out / "target_metrics.json").read_text())
            if result["decision_sha256"] != sha(HERE / "decision_final.json"):
                raise ValueError("target decision changed")
            for name, expected in result["predictions_sha256"].items():
                if sha(out / name) != expected:
                    raise ValueError("target predictions changed")
            _, target = receivers(job["dataset"], job["fold"])
            for rx in target:
                for day in (1, 2, 3, 4):
                    if (job["phase"], job["id"], "target", rx, day) not in observed:
                        raise ValueError("missing final target read event")
    changed = subprocess.check_output(["git", "diff", "--name-only", BASELINE], cwd=ROOT).decode().splitlines()
    if any(not (name.startswith("09-phase-invariance/") or name == ".gitignore") for name in changed):
        raise ValueError("historical tracked files changed")
    tracked = subprocess.check_output(["git", "ls-files", "09-phase-invariance"], cwd=ROOT).decode().splitlines()
    if any(name.endswith((".pt", ".pth")) for name in tracked):
        raise ValueError("weights staged/tracked")
    result = dict(status="PASS", source_only=args.source_only, formal_fits=fits, smoke_fits=5,
                  paired_groups=len(configs), access_events=len(events), historical_files="unchanged",
                  decision_committed_at=freeze.isoformat() if freeze else None,
                  registration_sha256=sha(HERE / "registration.json"), access_sha256=sha(HERE / "access.jsonl"))
    save(HERE / ("source_acceptance.json" if args.source_only else "acceptance.json"), result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
