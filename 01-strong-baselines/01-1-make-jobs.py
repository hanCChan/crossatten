#!/usr/bin/env python3
"""01-1  Write the job list for a stage.   python 01-1-make-jobs.py explore|confirm|smoke
explore : all arms x 2 folds x exploration seeds (dev_val only)
confirm : all arms x 2 folds x confirmation seeds (dev_val + confirm RX); requires the frozen decision file
smoke   : all arms, fold 0, seed 0, 2 short epochs (pipeline check only, never reported)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DECISION = HERE / "decision_explore.json"


def main():
    stage = sys.argv[1]
    P = json.loads((ROOT / "configs/protocol.json").read_text())
    arms = json.loads((HERE / "01-arms.json").read_text())["arms"]
    if stage == "confirm" and not DECISION.is_file():
        raise SystemExit("freeze decision_explore.json first (01-3-aggregate.py explore --freeze)")
    seeds = {"explore": P["seeds"]["exploration"], "confirm": P["seeds"]["confirmation"], "smoke": [0]}[stage]
    folds = [0] if stage == "smoke" else [0, 1]
    # longest jobs first (Shen flatten heads, early-stopped up to 500 epochs), then the rest interleaved
    order = sorted(arms, key=lambda a: (a["family"] != "shen_cis", a.get("head") != "flatten"))
    jobs = []
    for seed in seeds:
        for fold in folds:
            for a in order:
                j = dict(arms=str(HERE / "01-arms.json"), arm=a["name"], fold=fold, seed=seed, stage=stage,
                         out=str(HERE / "runs" / stage / f"{a['name']}__f{fold}__s{seed}"))
                if stage == "confirm":
                    j["decision_file"] = str(DECISION)
                jobs.append(j)
    out = HERE / "runs" / f"jobs_{stage}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"{len(jobs)} jobs -> {out}")


if __name__ == "__main__":
    main()
