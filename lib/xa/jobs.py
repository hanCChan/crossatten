"""Job lists for a stage directory (shared by stage >= 02).

make_jobs(stage_dir, stage): all arms of <stage_dir>/arms.json x folds x seeds of the protocol for that stage.
explore: dev_val only; confirm: needs <stage_dir>/decision_explore.json; smoke: fold 0, seed 0, 2 short epochs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .guard import load_protocol


def make_jobs(stage_dir: Path, stage: str, longest_first=("shen_cis",)) -> Path:
    P = load_protocol()
    arms_file = stage_dir / "arms.json"
    arms = json.loads(arms_file.read_text())["arms"]
    decision = stage_dir / "decision_explore.json"
    if stage == "confirm" and not decision.is_file():
        raise SystemExit("freeze decision_explore.json first")
    seeds = {"explore": P["seeds"]["exploration"], "confirm": P["seeds"]["confirmation"], "smoke": [0]}[stage]
    folds = [0] if stage == "smoke" else [0, 1]
    order = sorted(arms, key=lambda a: (a["family"] not in longest_first, a.get("recipe") != "long_cos_200"))
    jobs = []
    for seed in seeds:
        for fold in folds:
            for a in order:
                j = dict(arms=str(arms_file), arm=a["name"], fold=fold, seed=seed, stage=stage,
                         out=str(stage_dir / "runs" / stage / f"{a['name']}__f{fold}__s{seed}"))
                if stage == "confirm":
                    j["decision_file"] = str(decision)
                jobs.append(j)
    out = stage_dir / "runs" / f"jobs_{stage}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"{len(jobs)} jobs -> {out}")
    return out
