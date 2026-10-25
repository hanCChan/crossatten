#!/usr/bin/env python3
"""Execute only the registered adaptive sequence; commit each source freeze."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PHASES = ["features", "replicate", "training", "training_replicate", "combine", "final"]


def call(script, *args):
    subprocess.run([sys.executable, str(HERE / script), *args], cwd=ROOT, check=True)


def commit_phase(phase, next_phase=None):
    paths = [f"07-wifi-osu/runs/{phase}", "07-wifi-osu/access.jsonl"]
    if next_phase:
        paths.extend(f"07-wifi-osu/{name}" for name in
                     (f"decision_{next_phase}.json", f"jobs_{next_phase}.json", f"source_{phase}.csv"))
    present = [p for p in paths if (ROOT / p).exists()]
    subprocess.run(["git", "add", *present], cwd=ROOT, check=True)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode
    if changed:
        subprocess.run(["git", "commit", "-m", f"Audit S07 {phase} and freeze {next_phase or 'results'}"],
                       cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-phase", choices=PHASES, default="features")
    parser.add_argument("--per-gpu", type=int, default=16)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "exp/s07-wifi-osu":
        raise SystemExit("unexpected branch")
    for index in range(PHASES.index(args.from_phase), len(PHASES)):
        phase = PHASES[index]
        call("07-2-launch.py", "--jobs", str(HERE / f"jobs_{phase}.json"),
             "--per-gpu", str(args.per_gpu), "--gpus", args.gpus)
        call("07-7-audit.py", phase)
        if phase == "final":
            commit_phase(phase)
            call("07-10-final-report.py")
            break
        call("07-3-source-select.py", phase)
        next_phase = PHASES[index + 1]
        commit_phase(phase, next_phase)
        call("07-9-report.py", "features" if phase in ("features", "replicate") else "training")
    print(json.dumps(dict(status="registered WiSig sequence completed",
                          note="no automatic merge, push or new-weight publication")), flush=True)


if __name__ == "__main__":
    main()
