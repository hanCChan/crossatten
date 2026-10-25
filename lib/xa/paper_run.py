"""One registered paper-validation fit, compatible with xa.launch."""
import argparse
import json
from pathlib import Path

from .paper_train import fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--decision-file")
    args = parser.parse_args()
    arms = {a["name"]: a for a in json.loads(Path(args.arms).read_text())["arms"]}
    result = fit(arms[args.arm], args.fold, args.seed, args.stage, Path(args.out),
                 args.decision_file, smoke=args.stage == "smoke")
    print(json.dumps({k: v for k, v in result.items() if k not in ("selected", "source_rx_val")}), flush=True)


if __name__ == "__main__":
    main()
