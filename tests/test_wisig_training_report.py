"""Synthetic S08 report grid and complete-only behavior; no real target reads."""
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08-wisig-training"))
import training
from xa.wisig import receivers

spec = importlib.util.spec_from_file_location("s08_report_test", ROOT / "08-wisig-training/08-5-report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ReportTest(unittest.TestCase):
    def test_empty_or_incomplete_final_never_reports_target_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            here = Path(temp)
            with patch.object(report, "HERE", here):
                self.assertIsNone(report.final_report())
                (here / "decision_final.json").write_text(json.dumps(dict(
                    jobs=[], admitted={"ManySig": False})))
                self.assertEqual(report.final_report()["registered"], 0)
                (here / "decision_final.json").write_text(json.dumps(dict(
                    jobs=[dict(id="unfinished")], admitted={"ManySig": True})))
                self.assertEqual(report.final_report()["completed"], 0)
            self.assertFalse((here / "S08_WIFI_FINAL_RESULTS.md").exists())

    def test_complete_paired_fold_seed_comparison_and_disclosure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            here = root / "08-wisig-training"
            here.mkdir()
            (here / "08-4-audit.py").write_text("def audit_phase(phase):\n    assert phase == 'final'\n")
            jobs, old_rows, s07_rows = [], [], []
            for fold in range(4):
                _, targets = receivers("ManySig", fold)
                for rx in targets:
                    for baseline in ("IB-RFF", "WaveMLP", "FAN_source_selected", "W1"):
                        old_rows.append(dict(dataset="ManySig", fold=fold, rx=rx,
                                             model=baseline, task="CRD", seed_mean_accuracy_pct=80))
                    for seed in (4101, 4102, 4103):
                        s07_rows.append(dict(dataset="ManySig", fold=fold, seed=seed, rx=rx,
                                             pipeline="source_training", task="CRD", accuracy_pct=80))
                for seed in (7101, 7102, 7103):
                    for variant in ("candidate", "control"):
                        job = dict(dataset="ManySig", fold=fold, seed=seed, phase="final",
                                   condition="official", feature="phase" if variant == "candidate" else "frequency",
                                   rule="raw_held" if variant == "candidate" else "raw_packet",
                                   variant=variant)
                        if variant == "candidate":
                            job["fixed_epochs"] = 5
                        job["id"] = training.job_id(job)
                        jobs.append(job)
                        out = here / "runs/final" / job["id"]
                        out.mkdir(parents=True)
                        accuracy = .85 if variant == "candidate" else .8
                        selected = dict(source_packet=dict(epoch=5, acc=.98, ce=.1))
                        for task in ("CR", "CRD"):
                            selected[task] = dict(
                                summary=dict(acc_mean=accuracy),
                                per_rx={rx: dict(acc=accuracy, macro_f1=accuracy,
                                                 correct=round(accuracy * 100), total=100) for rx in targets})
                        (out / "metrics.json").write_text(json.dumps(dict(
                            job=job, stop_epoch=5, params=90868, seconds=1, selected=selected)))
            write_csv(root / "06-paper-validation/summary/paper/receiver_metrics.csv", old_rows)
            write_csv(root / "07-wifi-osu/wifi_final_receiver_metrics.csv", s07_rows)
            (here / "decision_final.json").write_text(json.dumps(dict(
                jobs=jobs, admitted={"ManySig": True, "ManyRx": False})))
            with patch.object(report, "HERE", here), patch.object(report, "ROOT", root):
                result = report.final_report()
            self.assertEqual(result["completed"], 24)
            self.assertEqual(len(result["summary"]), 4)
            self.assertEqual(len(result["intervals"]), 6)
            self.assertTrue(all(abs(r["delta_pp"] - 5) < 1e-8 for r in result["intervals"]))
            self.assertEqual([r["baseline"] for r in result["intervals"] if r["seeds_paired"]],
                             ["S08-paired-control"])
            with (here / "final_fold_metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 16)
            with (here / "final_seed_metrics.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 12)
            with (here / "paired_seed_deltas.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(float(r["delta_pp"]) == 5 for r in rows))
            text = (here / "S08_WIFI_FINAL_RESULTS.md").read_text()
            for phrase in ("No target-ranked pipeline", "no new formal target", "selection optimism",
                           "ManyRx supported only2/4", "93.57", "448", "Frozen Source Decisions"):
                self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
