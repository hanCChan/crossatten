#!/usr/bin/env python3
"""Build a complete round-1 results document from structured, already-frozen artifacts."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def table(rows, columns):
    return ("| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n" +
            "".join("| " + " | ".join(str(r.get(k, "")) for k in columns) + " |\n" for r in rows))


def receiver_tables(registered):
    runs = [json.loads(p.read_text()) for p in (HERE / "runs/confirm").glob("*/metrics.json")]
    sections = []
    for role in ("dev_val", "confirm"):
        keys = sorted({(r["fold"], rx) for r in runs for rx in r["selected"][role]["summary"]["per_rx_acc"]})
        columns = ["arm"] + [f"f{fold}:{rx}" for fold, rx in keys]
        rows = []
        for arm in sorted(registered):
            row = {"arm": arm}
            for fold, rx in keys:
                values = [100 * r["selected"][role]["summary"]["per_rx_acc"][rx]
                          for r in runs if r["arm"] == arm and r["fold"] == fold and
                          rx in r["selected"][role]["summary"]["per_rx_acc"]]
                row[f"f{fold}:{rx}"] = f"{np.mean(values):.2f}" if values else ""
            rows.append(row)
        sections.append(f"\n### {role}\n\n" + table(rows, columns))
    return "".join(sections)


def token_scale_summary(summary):
    path = summary / "token_scale.json"
    if not path.exists():
        return ""
    records = json.loads(path.read_text())
    if len(records["rows"]) != 32 or records["selection_use"] != "none":
        raise ValueError("token-scale diagnostic must include all 32 exploration fits")
    rows = []
    for arm in sorted({r["arm"] for r in records["rows"]}):
        trained = [r["trained"] for r in records["rows"] if r["arm"] == arm]
        rms = [r["token_rms"] for r in trained]
        saturation = [100 * r["score_saturated_frac"] for r in trained]
        rows.append(dict(arm=arm, trained_rms_range=f"{min(rms):.3f}-{max(rms):.3f}",
                         saturated_pct_range=f"{min(saturation):.2f}-{max(saturation):.2f}"))
    return (
        "## Post-Hoc Token Scale\n\n"
        "A CPU-only diagnostic compares initialization and selected exploration "
        "weights on balanced source-validation packets, without target data or "
        "decision changes. All arms, both folds, and both exploration seeds are "
        "reported in [token_scale.md](summary/token_scale.md). This diagnostic "
        "was added after the exploration freeze and is not a preregistered "
        "selection criterion.\n\n"
        + table(rows, ["arm", "trained_rms_range", "saturated_pct_range"])
        + "\nPre-LN controls token amplitude but does not eliminate large learned "
        "q/k scores or sigmoid saturation. These measurements do not establish "
        "a causal explanation of historical depth4 failures.\n\n"
    )


def main():
    subprocess.run([sys.executable, str(HERE / "03-9-verify-results.py")], check=True)
    summary = HERE / "summary"
    arms = csv_rows(summary / "confirm_arms.csv")
    pairs = csv_rows(summary / "confirm_pairs.csv")
    registered = {a["name"] for a in json.loads((HERE / "arms.json").read_text())["arms"]}
    arms = [a for a in arms if a["arm"] in registered]
    if len(arms) != 8 or any(int(a["n_fits"]) != 6 for a in arms):
        raise SystemExit("need all eight arms x six confirmation fits")
    decision = json.loads((HERE / "decision_confirm.json").read_text())
    bench = json.loads((summary / "bench.json").read_text())
    mechanisms = json.loads((summary / "mechanisms.json").read_text())
    if len(mechanisms["rows"]) != 432:
        raise SystemExit("need all 432 registered per-checkpoint/RX/intervention results")
    confirmed_runs = [json.loads(path.read_text())
                      for path in (HERE / "runs/confirm").glob("*/metrics.json")]
    peak_memory = max(run["gpu_peak_mem_gb"] for run in confirmed_runs)
    status = {stage: json.loads((HERE / "runs" / f"jobs_{stage}.jsonl.status.json").read_text())
              for stage in ("smoke", "explore", "confirm")}
    if any(v["failed"] or v["done"] + v["skipped"] != v["total"] for v in status.values()):
        raise SystemExit("launcher failures/incomplete runs must be resolved and separately documented")
    verdicts = decision["verdicts_dev_val"]
    explore_decision = json.loads((HERE / "decision_explore.json").read_text())
    explore_arms = [row for row in csv_rows(summary / "explore_arms.csv") if row["arm"] in registered]
    structural = ["s03_resid", "s03_symstack", "s03_ifreq"]
    eligible = decision["eligible_factors"]
    text = [
        "# S03 Round 1: Complete Results\n\n",
        "Protocol P-H5x160, all eight preregistered arms, 200 epochs with strict IB + AWGN. "
        "32 exploration fits and 48 new-seed confirmation fits. No blind RX read; "
        "only source validation CE selected checkpoints. Confirm RX are report-only "
        "and have been repeatedly inspected.\n\n",
        "## Decision\n\n",
        f"Confirmation dev_val verdicts: `{json.dumps(verdicts, sort_keys=True)}`.\n\n",
        f"Factors eligible for a user-approved next round: `{eligible}`. "
        "Eligibility is not authorization to combine factors; stop after this report.\n\n",
        "Structural hypothesis verdicts: " + ", ".join(f"{name}={verdicts[name]}" for name in structural) + ". "
        "Symstack's clean single-factor comparison is against nofft; symstack vs preln "
        "also removes FFT and is only a system reference. "
        "Pre-LN is an engineering change, not claimed as the method contribution.\n\n",
        "## Confirmation Accuracy\n\n",
        "RX-equal mean and fit SD (%), selected checkpoint; all six fits per arm. "
        "Fit SD includes fold differences; within-fold seed SD is reported separately. "
        "Seeds are training repeats, not additional independent receivers.\n\n",
        table(arms, ["arm", "n_fits", "params", "dev_val_acc_mean", "dev_val_acc_sd",
                     "dev_val_within_fold_seed_sd", "dev_val_worst_rx_seedmean",
                     "dev_val_f1_mean", "confirm_acc_mean", "confirm_acc_sd",
                     "confirm_worst_rx_seedmean", "src_val_acc", "best_epoch_median"]),
        "\n## Exploration Versus Confirmation\n\n",
        "All four exploration fits per arm, dev_val only; none were used to remove "
        "arms from confirmation. Compare against the six new-seed fits above.\n\n",
        table(explore_arms, ["arm", "n_fits", "dev_val_acc_mean", "dev_val_acc_sd",
                            "dev_val_within_fold_seed_sd", "dev_val_worst_rx_seedmean"]),
        "\n" + table([dict(arm=name, exploration=explore_decision["verdicts_dev_val"][name],
                          confirmation=verdicts[name]) for name in sorted(verdicts)],
                     ["arm", "exploration", "confirmation"]),
        "\n## Paired Comparisons\n\n",
        "Delta is B minus A, pp. Decisions below use dev_val only. "
        "Confirm comparisons and every per-RX mean are in "
        "[confirm_tables.md](summary/confirm_tables.md).\n\n",
        table([p for p in pairs if p["role"] == "dev_val"],
              ["q", "b", "a", "n_pairs", "delta_mean", "delta_min", "delta_max",
               "pos_pairs", "worst_rx_delta", "verdict"]),
        "\n## Per-Receiver Accuracy\n\n",
        "Seed-mean accuracy (%) within each fold; all three confirmation seeds. "
        "Confirm receivers repeat across folds and are report-only.\n",
        receiver_tables(registered),
        "\n## Cost\n\n",
        f"{bench['env']['gpu']}, PyTorch {bench['env']['torch']}; random weights. "
        "Input is raw IQ already on device; preprocessing and all model execution "
        "are included in end-to-end timing. Convolution/matrix FLOPs are torch-counted; "
        "FFT and elementwise operations are not fully counted. Benchmark ran before "
        "formal training, without competing GPU experiments.\n\n",
        table(bench["arms"], ["arm", "params", "mflops_per_packet", "gpu_b1_ms_p50",
                             "gpu_b1_ms_p95", "gpu_b512_pkts_per_s", "cpu4t_b1_ms_p50",
                             "cpu4t_b1_ms_p95"]),
        f"\nObserved maximum allocated GPU memory in confirmation training/evaluation: "
        f"{peak_memory:.2f} GB per fit (includes saved-checkpoint batch500 evaluation). "
        "This exceeds the specification's 3.1-GB estimate; no tasks were omitted "
        "because of it. Training itself remains batch64.\n\n",
        "\nFewer parameters do not necessarily mean less compute: hstem removes "
        "most tokenizer weights but executes extra convolutions at higher resolution. "
        "Single-packet GPU times include host dispatch variability; use distributions, "
        "not isolated p50 differences, for efficiency claims.\n\n",
        token_scale_summary(summary),
        "## Mechanisms\n\n",
        (summary / "mechanisms.md").read_text().replace("# Confirmation interventions\n", ""),
        "\nInterventions use each of the six confirmed checkpoints, without fitting or "
        "selection. Unchanged and intervened evaluation share batch64 and identical "
        "RX packets. Packet-shuffle within a batch is a limited diagnostic if packets "
        "share identity; zero-residual and symbol misalignment are also reported. "
        "A performance drop is evidence that the representation is used, not proof "
        "that the model uses only that representation or that the signal has a "
        "unique causal interpretation. No second-round architecture is selected here.\n\n",
        "## Verification And Failures\n\n",
        "Legacy core/model2/bit-regression tests passed before training. New tests verify "
        "exact legacy weights/logits, equal-parameter symstack/nofft shared initialization, "
        "finite gradients, unit-RMS residual/reconstruction/periodic-zero behavior, "
        "instantaneous-frequency reference, and symbol-local token receptive field.\n\n",
        "Saved-result checks passed for every smoke, exploration, and confirmation "
        "fit: arm/recipe, protocol hash, seed, parameters, full epoch budget, "
        "source-CE-minimum checkpoint, per-RX confusion denominators, macro-F1, "
        "RX-equal summaries, finite weights, and frozen table hashes. See "
        "[verification.json](summary/verification.json).\n\n",
        table([dict(stage=k, total=v["total"], skipped=v["skipped"], completed=v["done"],
                    failures=len(v["failed"])) for k, v in status.items()],
              ["stage", "total", "skipped", "completed", "failures"]),
        "\nSmoke numbers are execution checks, not scientific accuracy results. "
        "Baseline rerun determinism is recorded as the s03_base vs s02_ep200 reference "
        "comparison, with selected weight and prediction equality checked separately. "
        "See exploration/confirmation tables and individual config/history/metrics "
        "for all seeds, folds, receiver scores, budgets, and checkpoint selection.\n\n",
        "## Limitations And Stop\n\n",
        "Exploration is not confirmation; report all negative and unclear factors. "
        "Only confirmation dev_val determines eligibility. Confirm RX have been reused "
        "many times and cannot support an independent generalization claim. "
        "Neither SOTA-on-Shen training nor WiSig fair multi-seed retraining was launched. "
        "WiSig preparation and released-weight validation are reported separately in "
        "[05-wisig/RESULTS.md](../05-wisig/RESULTS.md), including the 25-Msps audit "
        "and incomplete LTF crop. Literature and numerical-scale diagnostics do not "
        "establish novelty or prove the cause of historical depth4 instability.\n\n",
        "Round 1 is complete. User decides whether to proceed with combinations, "
        "stage04, and WiSig training. On 2026-10-25 the user's latest request "
        "authorized organizing and publishing existing results to GitHub, "
        "overriding the earlier no-push instruction without authorizing more "
        "training. See the repository-wide [RESULTS.md](../RESULTS.md). "
        "S03 model.pt files remain local and are not included in Git or the "
        "earlier stage00-02 checkpoint release.\n",
    ]
    (HERE / "RESULTS.md").write_text("".join(text))
    print(f"wrote {HERE / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
