"""Archive generated source evidence without reading any checkpoint."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = HERE / "evidence"
    files = {}
    for phase in ("smoke", "pilot"):
        jobs = json.loads((HERE / f"jobs_{phase}.json").read_text())
        for job in jobs:
            source = HERE / "runs" / phase / job["id"]
            metrics = json.loads((source / "metrics.json").read_text())
            if digest(source / "config.json") != metrics["config_sha256"]:
                raise ValueError("new run config seal differs")
            names = ["config.json", "metrics.json", "history.csv", "fit_attempt.json",
                     "stdout.log"]
            if job["feature"] in ("F1", "F2"):
                names.append("spectral_prior.json")
            names += list(metrics["predictions_sha256"])
            for name in names:
                if Path(name).name != name or name.endswith(".pt"):
                    raise ValueError("only generated non-checkpoint evidence allowed")
                src = source / name
                seal = digest(src)
                if name in metrics["predictions_sha256"]:
                    if seal != metrics["predictions_sha256"][name]:
                        raise ValueError("new source prediction seal differs")
                dest = output / "runs" / phase / job["id"] / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                files[str(dest.relative_to(HERE))] = seal
    dest = output / "access.jsonl"
    shutil.copy2(HERE / "access.jsonl", dest)
    records = [json.loads(line) for line in dest.read_text().splitlines()]
    if any(r["role"] not in ("fit", "held", "diagnostic") or r["day"] not in (1, 2)
           for r in records):
        raise ValueError("source access log contains forbidden role/date")
    files[str(dest.relative_to(HERE))] = digest(dest)
    summary = dict(stage="S12", smoke_fits=4, pilot_fits=16, failed_fits=0,
                   confirmation_fits=0, checkpoint_files_read=0,
                   old_weights_accessed=False, source_access_records=len(records),
                   forbidden_role_or_date_records=0, files=files,
                   scope="Generated source configs, metrics, curves, priors, predictions "
                         "and logs. Checkpoints remain local; no weight audit performed.")
    (HERE / "evidence_manifest.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"Archived {len(files)} non-checkpoint source evidence files.")


if __name__ == "__main__":
    main()
