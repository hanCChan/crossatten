"""Paper exports preserve audited numbers, separated tasks, and source evidence."""
import csv
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_export", ROOT / "06-paper-validation/06-7-export.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def main():
    points = [(200, 200), (200, 204), (201, 210)]
    positions = export.badge_positions(points, (100, 100, 400, 400))
    assert points == [(200, 200), (200, 204), (201, 210)]
    assert all(math.dist(a, b) >= 24 for i, a in enumerate(positions) for b in positions[i + 1:])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        groups = {
            "ManyRx:W1:CRD": {
                "n": 12, "metrics": {name: {"mean": 83.5, "sd": 1.2}
                                    for name in ("acc_mean", "packet_acc", "f1_mean", "packet_macro_f1")},
                "seed_sd": 0.3, "fold_sd": 2.1, "worst_fold_rx_mean": 65.0,
                "params": [91128], "training_cost": {
                    "total_seconds": {"mean": 100.0}, "initialization_seconds": {"mean": 0.0}},
                "per_fold_rx_mean": {"0:rx-a": 83.5},
            }
        }
        groups["ManyRx:W1:CR"] = groups["ManyRx:W1:CRD"]
        report = {
            "generated_at": "2026-10-25T20:00:00+08:00", "checks": {"complete": False, "problems": [], "validated": 12},
            "groups": groups, "comparisons": {"ManyRx:W1-W0:CRD": {"n": 11, "status": "incomplete"}},
            "registration": {"arms": [{"name": "W1_test", "method": "W1"}]},
            "results": [{"stage": "final", "dataset": "ManyRx", "arm": "W1_test"}],
            "costs_by_arm": {"W1_test": {"arm": "W1_test", "mflops_per_packet": 10.8}},
        }
        path = root / "report.json"
        path.write_text(json.dumps(report))
        out = root / "exports"
        with patch("sys.argv", ["export", "--report", str(path), "--out", str(out)]):
            export.main()
        with (out / "main_metrics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert {row["task"] for row in rows} == {"CR", "CRD"}
        row = next(row for row in rows if row["task"] == "CRD")
        assert row["task"] == "CRD" and float(row["rx_accuracy_pct"]) == 83.5
        assert row["parameters"] == "91128" and int(row["fits"]) == 12
        assert b"\r" not in (out / "main_metrics.csv").read_bytes()
        with (out / "registered_ablations.csv").open() as stream:
            assert next(csv.DictReader(stream))["status"] == "incomplete"
        evidence = json.loads((out / "evidence.json").read_text())
        assert not evidence["complete"] and len(evidence["files"]) == 6
        assert "进行中" in (out / "CONCLUSION.md").read_text()
        assert Image.open(out / "accuracy_cost.png").size == (1680, 840)
        with patch("sys.argv", ["export", "--report", str(path), "--require-complete"]):
            try:
                export.main()
            except SystemExit as error:
                assert str(error) == "registered matrix is incomplete"
            else:
                raise AssertionError("incomplete export passed completion gate")
    print("PASS paper CSV values, task separation, incomplete gate, image dimensions and evidence hashes")


if __name__ == "__main__":
    main()
