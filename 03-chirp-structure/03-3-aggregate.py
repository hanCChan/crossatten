#!/usr/bin/env python3
"""Aggregate complete stage-03 runs, then freeze dev-only decisions."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.aggregate import aggregate
from xa.guard import load_protocol

PAIRS = [
    ("P1", "s03_preln", "s03_base", "Pre-LN engineering control"),
    ("P1", "s03_preln_d4", "s03_preln", "depth 4 after Pre-LN"),
    ("P2", "s03_resid", "s03_preln", "unit-RMS complex residual channels"),
    ("P2", "s03_nofft", "s03_preln", "remove whole-packet FFT channel"),
    ("P2", "s03_symstack", "s03_nofft", "equal-parameter symbol-aligned tokenizer"),
    ("ref_structure", "s03_symstack", "s03_preln", "system reference (also removes FFT)"),
    ("P3", "s03_ifreq", "s03_preln", "instantaneous-frequency deviation channel"),
    ("P4", "s03_hstem", "s03_preln", "hierarchical tokenizer"),
    ("ref_determinism", "s03_base", "s02_ep200", "bit-deterministic baseline rerun"),
]


def main():
    stage = sys.argv[1]
    arms = json.loads((HERE / "arms.json").read_text())["arms"]
    seeds = load_protocol()["seeds"]["exploration" if stage == "explore" else "confirmation"]
    expected = {f"{a['name']}__f{f}__s{s}" for a in arms for f in (0, 1) for s in seeds}
    actual = {p.parent.name for p in (HERE / "runs" / stage).glob("*/metrics.json")}
    if actual != expected:
        raise SystemExit(f"incomplete/unexpected runs; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    ref = [str(HERE.parent / f"02-single-factor-screen/runs/{stage}/s02_ep200__*/metrics.json")]
    rows, pairs, sha = aggregate(HERE, stage, PAIRS, ref)
    print((HERE / "summary" / f"{stage}_tables.md").read_text()[:6000])
    if "--freeze" not in sys.argv:
        return
    verdicts = {p["b"]: p["verdict"] for p in pairs if p["role"] == "dev_val" and not p["q"].startswith("ref")}
    dec = dict(stage=stage, rules="README.md", run_table_sha256=sha, verdicts_dev_val=verdicts,
               confirm_role="report only; repeatedly inspected; never used for decisions")
    if stage == "explore":
        dec["confirmation"] = "ALL eight registered arms; seeds 201,202,203 x folds 0,1"
    else:
        dec["eligible_factors"] = sorted(k for k, v in verdicts.items() if v == "improves")
        dec["next_round"] = "STOP; user decides combinations and stage 04"
    (HERE / f"decision_{stage}.json").write_text(json.dumps(dec, indent=1) + "\n")
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
