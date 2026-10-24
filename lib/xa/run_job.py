"""CLI for one fit:  python -m xa.run_job --arms ARMS.json --arm NAME --fold F --seed S --stage explore|confirm|smoke --out DIR"""
import argparse
import json
from pathlib import Path

from .train import fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--stage", choices=["explore", "confirm", "smoke"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decision-file", default=None)
    a = ap.parse_args()
    arms = {x["name"]: x for x in json.loads(Path(a.arms).read_text())["arms"]}
    res = fit(arms[a.arm], a.fold, a.seed, "explore" if a.stage == "smoke" else a.stage, Path(a.out),
              decision_file=a.decision_file, smoke=(a.stage == "smoke"))
    s = res["selected"]["dev_val"]["summary"]
    print(json.dumps(dict(run=res["run_id"], dev_val_acc=round(s["acc_mean"], 4), worst=s["worst_rx"],
                          best_epoch=res["best_val_epoch"], stop=res["stop_epoch"], sec=res["train_seconds"])))


if __name__ == "__main__":
    main()
