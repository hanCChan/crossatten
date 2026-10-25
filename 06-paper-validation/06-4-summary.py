#!/usr/bin/env python3
"""Audit artifacts and report the complete registered matrix, including failures."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def pooled_f1(per_rx):
    cm = sum(np.asarray(m["confusion"]) for m in per_rx.values())
    tp = cm.diagonal().astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    return float(np.mean(2 * precision * recall / np.maximum(precision + recall, 1e-12)))


def audit(path, registration_sha):
    import hashlib
    result = json.loads(path.read_text())
    config = json.loads((path.parent / "config.json").read_text())
    with (path.parent / "history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    arm = config["arm"]
    for key in ("fold", "seed", "stage"):
        assert result[key] == config[key], (path, key)
    assert result["arm"] == arm["name"] and result["params"] == arm["expected_params"], path
    assert not config["smoke"] and config["registration_sha256"] == registration_sha, path
    assert config["protocol_sha256"] == hashlib.sha256((ROOT / "configs/protocol.json").read_bytes()).hexdigest(), path
    assert (path.parent / "model.pt").is_file(), path
    if arm.get("init_arm"):
        assert config["init"] is not None and Path(config["init"]["path"]).is_file(), path
        assert hashlib.sha256(Path(config["init"]["path"]).read_bytes()).hexdigest() == config["init"]["sha256"], path
    if result["stage"] in ("final", "confirm"):
        decision_path = Path(config["decision_file"])
        decision = json.loads(decision_path.read_text())
        assert hashlib.sha256(decision_path.read_bytes()).hexdigest() == config["decision_sha256"], path
        assert decision["registration_sha256"] == registration_sha, path
        assert [result["arm"], result["fold"], result["seed"]] in decision["allowed_final_jobs"], path
    assert len(history) == result["stop_epoch"], path
    minimum = min(history, key=lambda row: float(row["val_ce"]))
    assert int(minimum["epoch"]) == result["best_val_epoch"], path
    assert abs(float(minimum["val_ce"]) - result["source_val_ce"]) < 1e-8, path
    assert all(np.isfinite(float(row[k])) for row in history for k in row), path
    budget = config["recipe"]["epochs"]
    assert result["epochs_budget"] == budget, path
    if config["recipe"]["sched"] == "warmcos":
        assert len(history) == budget, path
    elif len(history) < budget:
        stop = config["recipe"]["early_stop"]
        assert len(history) - result["best_val_epoch"] >= stop, path
        assert min(float(row["val_ce"]) for row in history[-stop:]) >= result["source_val_ce"], path
    expected = ({"CR", "CRD"} if result["stage"] == "final" else
                {"dev_val", "confirm"} if result["stage"] == "confirm" else
                {"dev_val"} if result["stage"] == "explore" else set())
    assert set(result["selected"]) == expected, path
    if result["stage"] == "source":
        assert "source_rx_val" in result, path
    for label, data in list(result["selected"].items()) + (
            [("source_rx_val", result["source_rx_val"])] if "source_rx_val" in result else []):
        per = data["per_rx"]
        expected_rx = (config["roles"]["internal_val_rx"] if label == "source_rx_val" else
                       config["roles"]["target"] if label in ("CR", "CRD") else config["roles"][label])
        assert set(per) == set(expected_rx), (path, label)
        manifest = (json.loads((ROOT / f"05-wisig/out/05-1-{result['dataset']}-manifest.json").read_text())
                    if result["dataset"] != "Shen" else None)
        for rx, scores in per.items():
            cm = np.asarray(scores["confusion"])
            assert cm.shape == (arm["num_classes"], arm["num_classes"]), (path, rx)
            assert cm.sum() == scores["total"] and cm.trace() == scores["correct"], (path, rx)
            assert abs(scores["acc"] - cm.trace() / cm.sum()) < 1e-12, (path, rx)
            assert abs(scores["macro_f1"] - pooled_f1({rx: scores})) < 1e-12, (path, rx)
            days = (3, 4) if label == "CRD" else (1, 2)
            expected_total = (sum(sum(manifest["counts"][rx][str(day)]) for day in days) if manifest else 2000)
            assert scores["total"] == expected_total, (path, rx, expected_total)
        summary = data["summary"]
        for metric, field in (("acc_mean", "acc"), ("f1_mean", "macro_f1"), ("ce_mean", "ce")):
            assert abs(summary[metric] - np.mean([m[field] for m in per.values()])) < 1e-12, path
        assert abs(summary["packet_acc"] - sum(m["correct"] for m in per.values()) / sum(m["total"] for m in per.values())) < 1e-12, path
        assert abs(summary["worst_acc"] - min(m["acc"] for m in per.values())) < 1e-12, path
        assert (path.parent / f"preds_{label}.npz").is_file(), path
    return result, config


def moments(values):
    return dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def aggregate(records, label):
    seeds, folds, rx = defaultdict(list), defaultdict(list), defaultdict(list)
    values = defaultdict(list)
    for result in records:
        entry = result["selected"][label]
        scores = entry["summary"]
        for key in ("acc_mean", "packet_acc", "f1_mean"):
            values[key].append(scores[key] * 100)
        values["packet_macro_f1"].append(pooled_f1(entry["per_rx"]) * 100)
        seeds[result["seed"]].append(scores["acc_mean"] * 100)
        folds[result["fold"]].append(scores["acc_mean"] * 100)
        for name, score in entry["per_rx"].items():
            rx[(result["fold"], name)].append(score["acc"] * 100)
    return dict(n=len(records), metrics={k: moments(v) for k, v in values.items()},
                seed_means={str(k): float(np.mean(v)) for k, v in seeds.items()},
                seed_sd=moments([np.mean(v) for v in seeds.values()])["sd"],
                fold_means={str(k): float(np.mean(v)) for k, v in folds.items()},
                fold_sd=moments([np.mean(v) for v in folds.values()])["sd"],
                worst_fold_rx_mean=min(float(np.mean(v)) for v in rx.values()),
                per_fold_rx_mean={f"{f}:{r}": float(np.mean(v)) for (f, r), v in rx.items()},
                params=sorted({r["params"] for r in records}),
                train_seconds=moments([r["train_seconds"] for r in records]))


def paired(left, right, label, expected):
    a = {(r["fold"], r["seed"]): r for r in left}
    b = {(r["fold"], r["seed"]): r for r in right}
    keys = sorted(set(a) & set(b))
    if len(keys) != expected or set(a) != set(b):
        return dict(status="incomplete", n=len(keys), expected=expected)
    deltas = [100 * (a[k]["selected"][label]["summary"]["acc_mean"] -
                    b[k]["selected"][label]["summary"]["acc_mean"]) for k in keys]
    per_rx = defaultdict(list)
    for k in keys:
        pa, pb = a[k]["selected"][label]["per_rx"], b[k]["selected"][label]["per_rx"]
        assert pa.keys() == pb.keys()
        for rx in pa:
            per_rx[(k[0], rx)].append(100 * (pa[rx]["acc"] - pb[rx]["acc"]))
    worst_drop = min(float(np.mean(v)) for v in per_rx.values())
    mean = float(np.mean(deltas))
    positive, negative = sum(v > 0 for v in deltas), sum(v < 0 for v in deltas)
    needed = 10 if expected == 12 else 5
    status = ("improves" if mean >= 1 and positive >= needed and worst_drop >= -2 else
              "harms" if mean <= -1 and negative >= needed else "unclear")
    return dict(status=status, n=len(keys), delta_pp=moments(deltas), positive=positive,
                negative=negative, worst_fold_rx_delta_pp=worst_drop,
                pairs=[dict(fold=k[0], seed=k[1], delta_pp=v) for k, v in zip(keys, deltas)])


def collect():
    import hashlib
    registration = json.loads((HERE / "arms.json").read_text())
    digest = hashlib.sha256((HERE / "arms.json").read_bytes()).hexdigest()
    results, configs, problems = [], {}, []
    for stage in ("source", "explore", "final", "confirm"):
        for path in sorted((HERE / "runs" / stage).glob("*/metrics.json")):
            try:
                result, config = audit(path, digest)
                results.append(result)
                configs[(stage, result["run_id"])] = config
            except (AssertionError, ValueError, KeyError, OSError) as error:
                problems.append(dict(path=str(path.relative_to(HERE)), error=str(error)))
    expected = {}
    for stage, wifi, folds, seeds in (("source", True, range(4), [2023]),
                                     ("explore", False, range(2), [101, 102])):
        expected[stage] = [[a["name"], f, s] for a in registration["arms"]
                           if bool(a.get("dataset")) == wifi for f in folds for s in seeds]
    for stage, filename in (("final", "decision_wifi.json"), ("confirm", "decision_shen.json")):
        path = HERE / filename
        expected[stage] = json.loads(path.read_text())["allowed_final_jobs"] if path.exists() else []
    missing, extra = {}, {}
    for stage, jobs in expected.items():
        done = {(r["arm"], r["fold"], r["seed"]) for r in results if r["stage"] == stage}
        missing[stage] = [j for j in jobs if tuple(j) not in done]
        extra[stage] = [list(j) for j in done if list(j) not in jobs]
    failures = []
    for path in sorted((HERE / "runs").glob("*.status.json")):
        for row in json.loads(path.read_text()).get("failed", []):
            failures.append(dict(status_file=str(path.relative_to(HERE)), **row))
    for path in sorted((HERE / "runs").glob("*/*/stdout.log")):
        if "Traceback (most recent call last)" in path.read_text(errors="replace"):
            failures.append(dict(log=str(path.relative_to(HERE))))
    for path in sorted((HERE / "runs").glob("*/*/attempt_*")):
        failures.append(dict(attempt=str(path.relative_to(HERE)),
                             disposition="retained incomplete attempt before same-config retry"))
    checks = dict(validated=len(results), problems=problems, missing=missing,
                  extra=extra, failures=failures,
                  complete=not problems and not any(missing.values()) and not any(extra.values())
                  and bool(expected["final"]) and bool(expected["confirm"]))
    return registration, results, configs, checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--check-weights", action="store_true")
    args = parser.parse_args()
    registration, results, configs, checks = collect()
    if args.check_weights:
        import torch
        torch.set_num_threads(2)
        for result in results:
            path = HERE / "runs" / result["stage"] / result["run_id"] / "model.pt"
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not all(torch.isfinite(tensor).all() for tensor in state.values()):
                checks["problems"].append(dict(path=str(path.relative_to(HERE)), error="nonfinite saved checkpoint"))
            for label in result["selected"]:
                with np.load(path.parent / f"preds_{label}.npz") as predictions:
                    for rx, scores in result["selected"][label]["per_rx"].items():
                        logits, y = predictions[rx], predictions[f"y_{rx}"]
                        assert logits.shape == (scores["total"], configs[(result["stage"], result["run_id"])]["arm"]["num_classes"])
                        assert y.shape == (scores["total"],) and np.isfinite(logits).all()
        checks["saved_weights_checked"] = len(results)
        checks["complete"] = checks["complete"] and not checks["problems"]
    groups, arms = defaultdict(list), {a["name"]: a for a in registration["arms"]}
    wifi_path = HERE / "decision_wifi.json"
    wifi = json.loads(wifi_path.read_text()) if wifi_path.exists() else None
    for result in results:
        arm = arms[result["arm"]]
        if result["stage"] == "final":
            model = arm["method"] if arm["method"] != "FAN" else f"FAN_{arm['recipe']}"
            groups[(result["dataset"], model)].append(result)
            if arm["method"] == "FAN" and wifi["choices"][f"{result['dataset']}:{result['fold']}"]["fan"] == arm["name"]:
                groups[(result["dataset"], "FAN_source_selected")].append(result)
        elif result["stage"] == "confirm":
            groups[("Shen", arm["name"])].append(result)
    summary, comparisons = {}, {}
    costs = {}
    for length in (256, 8192):
        for row in json.loads((HERE / f"summary/cost_{length}/bench.json").read_text())["arms"]:
            representative = arms[row["arm"]]
            signature = {k: v for k, v in representative.items() if k not in ("name", "recipe", "alpha", "init_arm", "method")}
            for name, arm in arms.items():
                candidate = {k: v for k, v in arm.items() if k not in ("name", "recipe", "alpha", "init_arm", "method")}
                if signature == candidate:
                    costs[name] = row
    lines = ["# Paper Validation Results", "", "Evidence date: 2026-10-25. Registration: `bc30820`.",
             "This report includes only finished, audited fits. Smoke results are not accuracy evidence.",
             f"**Status: {'complete' if checks['complete'] else 'in progress'}**; audited fits: {checks['validated']}.",
             "", "## Completion", "", "| Stage | Finished | Missing |", "|---|---:|---:|"]
    for stage in ("source", "explore", "final", "confirm"):
        lines.append(f"| {stage} | {sum(r['stage'] == stage for r in results)} | {len(checks['missing'][stage])} |")
    lines += ["", f"Artifact audit issues: {len(checks['problems'])}; recorded failure/interruption entries: {len(checks['failures'])}.",
              "A failed execution is not discarded; a subsequent successful retry may complete its registered fit.",
              "", "## Final Comparisons", "",
              "All values are percent. RX accuracy SD across fits combines fold and seed variation.",
              "Seed SD is computed from fold-averaged seed results; folds and packets are not extra independent RX.",
              "Incomplete rows are provisional and cannot establish improvement.",
              "", "| Dataset | Model | Task | Fits | RX Acc | Packet Acc | RX F1 | Seed SD | Fold SD | Worst RX | Params |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (dataset, model), records in sorted(groups.items()):
        for label in ("CR", "CRD") if dataset != "Shen" else ("dev_val", "confirm"):
            row = aggregate(records, label)
            summary[f"{dataset}:{model}:{label}"] = row
            m = row["metrics"]
            lines.append(f"| {dataset} | {model} | {label} | {row['n']} | "
                         f"{m['acc_mean']['mean']:.2f} +/- {m['acc_mean']['sd']:.2f} | "
                         f"{m['packet_acc']['mean']:.2f} | {m['f1_mean']['mean']:.2f} | "
                         f"{row['seed_sd']:.2f} | {row['fold_sd']:.2f} | "
                         f"{row['worst_fold_rx_mean']:.2f} | {','.join(map(str, row['params']))} |")
    lines += ["", "## Registered Ablations", "",
              "| Dataset | Task | Pair | Fits | Delta pp | Positive | Worst RX Delta | Decision |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for dataset in ("ManyRx", "ManySig", "Shen"):
        pairs = ([("W1", "W0"), ("W2", "W1"), ("W1", "C0"), ("W1", "CP"), ("W1", "CF")]
                 if dataset != "Shen" else
                 [("S_hstem", n) for n in ("S_base", "S_cnnonly", "S_cnnparam", "S_cnnflops")])
        for left, right in pairs:
            for label in ("CR", "CRD") if dataset != "Shen" else ("dev_val", "confirm"):
                row = paired(groups[(dataset, left)], groups[(dataset, right)], label, 12 if dataset != "Shen" else 6)
                comparisons[f"{dataset}:{left}-{right}:{label}"] = row
                delta = f"{row['delta_pp']['mean']:+.2f}" if "delta_pp" in row else "pending"
                worst = f"{row['worst_fold_rx_delta_pp']:+.2f}" if "delta_pp" in row else "pending"
                lines.append(f"| {dataset} | {label} | {left}-{right} | {row['n']} | {delta} | "
                             f"{row.get('positive', '-')}/{row['n']} | {worst} | {row['status']} |")
    lines += ["", "## Measured Cost", "",
              "Measurements precede formal training on an otherwise idle A100. Random weights, FP32, device IQ to logits.",
              "Batch latency excludes host transfer. Repeated timing samples measure system variability, not accuracy seeds.",
              "", "| Architecture | Params | MFLOPs/packet | GPU b1 p50 ms | p95 ms | b512 packets/s | CPU4t p50 ms |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for length in (256, 8192):
        for row in json.loads((HERE / f"summary/cost_{length}/bench.json").read_text())["arms"]:
            lines.append(f"| {row['arm']} | {row['params']} | {row['mflops_per_packet']} | "
                         f"{row['gpu_b1_ms_p50']} | {row['gpu_b1_ms_p95']} | "
                         f"{row['gpu_b512_pkts_per_s']} | {row['cpu4t_b1_ms_p50']} |")
    lines += ["", "## Interpretation Limits", "",
              "- CRD is the primary WiSig task. W0/W1/W2 remain separate; no target-based winner selection.",
              "- Final configuration choices use only source RX days 1-2; source evidence and decisions are committed before target access.",
              "- Shen confirm RX were inspected repeatedly and are report-only, not independent blind-test evidence.",
              "- Published weights in `05-wisig/RESULTS.md` are historical references, not fair multiseed retraining.",
              "- HSTU attribution requires W1 vs C0 and matched CNN controls; parameter efficiency does not imply faster inference.",
              "- Cost timings use raw device-resident IQ to logits, not host transfer; FFT is timed but incompletely counted in FLOPs.",
              "- FAN upstream CUDA median issues a nondeterminism warning; fixed seeds do not guarantee bit-identical upstream training.",
              "- No new Loss, feature branch, architecture or target-selected hyperparameter is added after this registration.",
              "", "Full per-fit metrics, configurations, source choices and failures: `summary/report.json`; costs: `summary/cost_*/bench.md`.",
              "The bounded experiment matrix must finish before choosing a general RF, LoRa-only, or engineering-only paper route.", ""]
    output = dict(registration=registration, checks=checks, groups=summary, costs_by_arm=costs,
                  comparisons=comparisons, results=results,
                  configs={f"{stage}:{run}": c for (stage, run), c in configs.items()})
    (HERE / "summary").mkdir(exist_ok=True)
    (HERE / "summary/report.json").write_text(json.dumps(output, indent=1) + "\n")
    (HERE / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({k: v for k, v in checks.items() if k not in ("missing", "failures", "extra")}))
    if args.require_complete and not checks["complete"]:
        raise SystemExit("registered matrix is not yet complete")


if __name__ == "__main__":
    main()
