"""Independent prediction re-score, access/caps/seal checks and historical byte audit."""
import json
import subprocess
import numpy as np
from stage import HERE, ROOT, ASK_DIR, sha, save, receivers, require_committed
from xa.paper_train import metrics


def main():
    reg = json.loads((HERE / "registration.json").read_text())
    decision = json.loads((HERE / "decision_final.json").read_text())
    count, targets = 0, 0
    for phase in ("source", "ask", "final", "ablation", "smoke"):
        jobs_path = HERE / f"jobs_{phase}.json"
        if not jobs_path.exists():
            continue
        jobs = json.loads(jobs_path.read_text())
        if len(jobs) > reg["caps"][phase]:
            raise ValueError("cap violation")
        for job in jobs:
            out = HERE / "runs" / phase / job["id"]
            result = json.loads((out / "metrics.json").read_text())
            if result["job"] != job:
                raise ValueError("run identity mismatch")
            for name, expected in result["weights_sha256"].items():
                if sha(out / name) != expected:
                    raise ValueError("checkpoint hash mismatch")
            target = out / "target_metrics.json"
            if phase in ("ask", "final", "ablation"):
                require_committed(out / "metrics.json")
                result = json.loads(target.read_text())
                targets += 1
            for label, entry in result["selected"].items():
                if "per_rx" not in entry:
                    continue
                path = out / f"preds_{label}.npz"
                if sha(path) != result["predictions_sha256"][path.name]:
                    raise ValueError("prediction hash mismatch")
                with np.load(path) as p:
                    for rx, recorded in entry["per_rx"].items():
                        actual = metrics(p[rx].astype(np.float32), p[f"y_{rx}"],
                                         10 if job["dataset"] == "ManyRx" else 6)
                        for k in ("acc", "macro_f1", "correct", "total"):
                            if not np.isclose(actual[k], recorded[k], atol=1e-8):
                                raise ValueError(f"prediction re-score differs: {k}")
            count += phase != "smoke"
    if count > reg["total_formal_cap"]:
        raise ValueError("formal cap violation")
    reads = 0
    for line in (HERE / "access.jsonl").read_text().splitlines():
        row = json.loads(line)
        source, target = receivers(row["dataset"], row["fold"])
        if row["role"] in ("fit", "held", "diagnostic"):
            if row["rx"] not in source or row["day"] not in (1, 2):
                raise ValueError("source access leaked target")
        elif row["role"] == "target":
            if row["rx"] not in target or row["phase"] not in ("ask", "final", "ablation"):
                raise ValueError("illegal target access")
            if row["phase"] != "ask" and not decision["admitted"]:
                raise ValueError("STOP candidate accessed target")
        else:
            raise ValueError("unclassified access")
        reads += 1
    changed = subprocess.check_output(["git", "diff", "--name-only", reg["base_commit"]],
                                      cwd=ROOT).decode().splitlines()
    if any(not name.startswith("10-tifs-validation/") for name in changed):
        raise ValueError("historical stage/shared file was changed")
    for path, expected in reg["code_manifest"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("registered source changed")
    for path, expected in reg["upstream_files"].items():
        if sha(ASK_DIR / path) != expected:
            raise ValueError("official model changed")
    save(HERE / "verification.json", dict(passed=True, formal_fits=count,
        scored_target_fits=targets, guarded_reads=reads, predictions_recomputed=True,
        checkpoint_hashes_verified=True, historical_files_unchanged=True,
        no_target_tuning=True, verdict=decision["verdict"]))
    print("Independent verification PASS", count, "formal fits;", targets, "target fits.")


if __name__ == "__main__":
    main()
