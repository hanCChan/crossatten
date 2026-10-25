#!/usr/bin/env python3
"""02-3  Aggregate stage 02.   python 02-3-aggregate.py explore [--freeze] | confirm [--freeze]

Reference runs from stage 01 with the same (fold, seed) data streams are included: cis_flat (for the rank-16 control)
and xa_trueib_cos_awgn (determinism check of s02_base).
explore --freeze : records verdicts + run-table SHA in decision_explore.json (no selection: all arms are confirmed)
confirm --freeze : applies the pre-registered carry-forward rule (README section 4) -> decision_confirm.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.aggregate import aggregate  # noqa: E402

BASE = "s02_base"
PAIRS = [
    ("B", "s02_meanchirp", BASE, "coherent 8-symbol mean chirp (drops residual/transient)"),
    ("B", "s02_combfft", BASE, "FFT view: comb bins only"),
    ("B", "s02_dechirp", BASE, "dechirped input"),
    ("A", "s02_rawoob_full", BASE, "+ raw OOB keys/values, all |f|>62.5 kHz"),
    ("A", "s02_rawoob_near", BASE, "+ raw OOB keys/values, 62.5-200 kHz"),
    ("A", "s02_rawoob_nearL", BASE, "+ raw OOB keys/values, left near band only"),
    ("A", "s02_cisoob", BASE, "+ CIS-quotient OOB keys/values"),
    ("C", "s02_depth4", BASE, "4 mixer blocks"),
    ("C", "s02_tfsplit", BASE, "time/frequency split tokens (strict IB)"),
    ("D", "s02_ep200", BASE, "200-epoch horizon"),
    ("D", "s02_awgn10", BASE, "AWGN U[10,50) instead of U[0,50)"),
    ("eff", "s02_cis_rank16", "cis_flat", "rank-16 head vs Flatten-Dense (stage-01 fits, same data streams)"),
    ("ref", BASE, "xa_trueib_cos_awgn", "determinism check: re-run of stage-01 arm (expected 0.00)"),
    ("ref", BASE, "s02_cis_rank16", "SYSTEM: CrossAttn-TrueIB vs rank-16 CIS"),
]
FACTORS = [p for p in PAIRS if p[0] in ("A", "B", "C", "D")]


def main():
    stage = sys.argv[1]
    ref = [str(HERE.parent / f"01-strong-baselines/runs/{stage}/{a}__f*/metrics.json") for a in ("cis_flat", "xa_trueib_cos_awgn")]
    arm_rows, pair_rows, sha = aggregate(HERE, stage, PAIRS, ref)
    print((HERE / "summary" / f"{stage}_tables.md").read_text()[:6000])
    if "--freeze" not in sys.argv:
        return
    dv = {(p["b"], p["a"]): p for p in pair_rows if p["role"] == "dev_val"}
    verdicts = {b: dv[(b, a)]["verdict"] for _, b, a, _ in FACTORS if (b, a) in dv}
    if stage == "explore":
        dec = dict(rules="README.md section 4", stage="explore", run_table_sha256=sha, verdicts_dev_val=verdicts,
                   confirmation="ALL registered arms, seeds 201-203, dev_val + confirm RX (no selection)")
        (HERE / "decision_explore.json").write_text(json.dumps(dec, indent=1))
    else:
        carry = sorted(b for b, v in verdicts.items() if v == "improves")
        oob_arms = [p[1] for p in FACTORS if p[0] == "A"]
        eff = dv.get(("s02_cis_rank16", "cis_flat"))
        dec = dict(rules="README.md section 4 (decisions use dev_val only; confirm RX are reported, never used)",
                   stage="confirm", run_table_sha256=sha, verdicts_dev_val=verdicts, carry_forward_to_stage03=carry,
                   oob_increment_supported=any(verdicts.get(a) == "improves" for a in oob_arms),
                   cis_rank16_no_material_loss=(eff is not None and eff["delta_mean"] >= -1.0 and eff["worst_rx_delta"] >= -2.0))
        (HERE / "decision_confirm.json").write_text(json.dumps(dec, indent=1))
    print(json.dumps(dec, indent=1))


if __name__ == "__main__":
    main()
