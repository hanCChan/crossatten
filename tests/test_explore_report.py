"""Synthetic complete-grid reporting; no real packet or target access."""
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s07_report_test", ROOT / "07-wifi-osu/07-10-final-report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
from xa.wisig import receivers


class ReportTest(unittest.TestCase):
    def test_complete_grid_identical_pipelines_and_receiver_bootstrap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            here = root / "07-wifi-osu"
            here.mkdir()
            (here / "07-7-audit.py").write_text("def audit_wifi(phase):\n    assert phase == 'final'\n")
            baseline_path = root / "06-paper-validation/summary/paper/receiver_metrics.csv"
            baseline_path.parent.mkdir(parents=True)
            baseline_rows, jobs = [], []
            for dataset in ("ManyRx", "ManySig"):
                for fold in range(4):
                    _, targets = receivers(dataset, fold)
                    for rx in targets:
                        for baseline in ("W1", "IB-RFF", "WaveMLP", "FAN_source_selected"):
                            baseline_rows.append(dict(dataset=dataset, rx=rx, model=baseline, task="CRD",
                                                      seed_mean_accuracy_pct=80))
                    for pipeline in ("source_frontend", "source_training", "source_combination"):
                        for seed in (4101, 4102, 4103):
                            job = dict(dataset=dataset, fold=fold, pipeline=pipeline,
                                       factor=pipeline, seed=seed, recipe="unified",
                                       frontend="frequency", phase="final",
                                       id=f"{dataset}_{pipeline}_f{fold}_s{seed}")
                            jobs.append(job)
                            out = here / "runs/final" / job["id"]
                            out.mkdir(parents=True)
                            selected = {}
                            for task, accuracy in (("CR", .9), ("CRD", .85)):
                                selected[task] = dict(
                                    summary=dict(acc_mean=accuracy, packet_acc=accuracy, f1_mean=accuracy),
                                    per_rx={rx: dict(acc=accuracy, macro_f1=accuracy,
                                                     correct=int(100*accuracy), total=100) for rx in targets})
                            (out / "metrics.json").write_text(json.dumps(
                                dict(job=job, params=90420, seconds=1, selected=selected)))
            with baseline_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(baseline_rows[0]))
                writer.writeheader()
                writer.writerows(baseline_rows)
            (here / "decision_final.json").write_text(json.dumps(dict(jobs=jobs)))
            (here / "registration.json").write_text(json.dumps(dict(baseline="synthetic-base")))
            with patch.object(report, "HERE", here), patch.object(report, "ROOT", root), \
                    patch.object(report, "require_committed"), \
                    patch.object(report.subprocess, "check_output", return_value=baseline_path.read_bytes()), \
                    redirect_stdout(io.StringIO()):
                report.main()
                with (here / "wifi_final_metrics.csv").open() as stream:
                    metrics = list(csv.DictReader(stream))
                with (here / "wifi_comparison_intervals.csv").open() as stream:
                    intervals = list(csv.DictReader(stream))
                evidence = json.loads((here / "wifi_final_report_evidence.json").read_text())
                self.assertEqual(len(metrics), 12)
                self.assertEqual(len(intervals), 24)
                self.assertEqual(len(evidence["duplicate_pipelines"]), 6)
                self.assertTrue(all(float(r["seed_sd_pp"]) == 0 for r in metrics))
                for row in intervals:
                    self.assertAlmostEqual(float(row["delta_pp"]), 5)
                    self.assertAlmostEqual(float(row["ci95_low_pp"]), 5)
                    self.assertAlmostEqual(float(row["ci95_high_pp"]), 5)
                    self.assertEqual(int(row["receiver_count"]), 32 if row["dataset"] == "ManyRx" else 12)
                incomplete = dict(jobs=jobs[:-1])
                (here / "decision_final.json").write_text(json.dumps(incomplete))
                with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                    report.main()


if __name__ == "__main__":
    unittest.main()
