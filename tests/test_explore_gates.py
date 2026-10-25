"""No real data reads: malformed jobs and source/target access are rejected."""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from xa.explore_train import source_cells, validate_job
from xa.wisig import receivers


def reject(call):
    try:
        call()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("invalid selection/access accepted")


def main():
    source, target = receivers("ManySig", 0)
    reject(lambda: source_cells("ManySig", target, (1, 2), "fit_source", 0, "test"))
    reject(lambda: source_cells("ManySig", source, (3, 4), "held_source", 0, "test"))
    reject(lambda: source_cells("ManySig", target, (1, 2), "target_report", 0, "test"))
    job = dict(dataset="ManySig", fold=0, seed=3101, phase="features",
               frontend="legacy", recipe="unified", id="ManySig_legacy__f0__s3101")
    with patch("xa.explore_train.require_committed"):
        validate_job(job)
        reject(lambda: validate_job(dict(job, augmentation="unregistered")))
        reject(lambda: validate_job(dict(job, epochs=1)))
        reject(lambda: validate_job(dict(job, seed=0)))
        reject(lambda: validate_job(dict(job, phase="final")))
    print("PASS source RX/day restrictions and exact committed job whitelist")


if __name__ == "__main__":
    main()
