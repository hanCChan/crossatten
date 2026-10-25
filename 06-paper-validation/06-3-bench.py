#!/usr/bin/env python3
"""Cost all registered architectures without concurrent training."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.bench import benchmark


if __name__ == "__main__":
    all_arms = json.loads((HERE / "arms.json").read_text())["arms"]
    for length in (256, 8192):
        arms, seen = [], set()
        for arm in all_arms:
            if arm["length"] != length:
                continue
            structure = {k: v for k, v in arm.items()
                         if k not in ("name", "recipe", "alpha", "init_arm", "expected_params")}
            if structure.get("method") == "IB-RFF":
                structure["method"] = "FAN"
            key = json.dumps(structure, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            arms.append(arm)
        benchmark(arms, HERE / "summary" / f"cost_{length}", length=length)
