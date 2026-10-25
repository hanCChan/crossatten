"""Exercise every adaptive transition with synthetic source-only evidence."""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "s07_workflow_test", ROOT / "07-wifi-osu/07-3-source-select.py"
)
select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select)


def evidence(job, acc, epoch=0):
    return dict(
        job=job,
        result=dict(held_best_epoch=epoch),
        acc=acc,
        ce=1 - acc,
        entry=dict(per_rx=dict(A=dict(acc=acc), B=dict(acc=acc))),
        evidence=f"runs/{job['phase']}/{job['id']}/metrics.json",
        sha256="synthetic-source-evidence",
    )


class WorkflowTest(unittest.TestCase):
    def exercise(self, mode):
        phases = ("features", "replicate", "training", "training_replicate", "combine")
        records = {}
        with tempfile.TemporaryDirectory() as temp:
            here = Path(temp)
            registration = json.loads((ROOT / "07-wifi-osu/registration.json").read_text())
            (here / "registration.json").write_text(json.dumps(registration))
            first = []
            for dataset in ("ManyRx", "ManySig"):
                for fold in range(4):
                    for frontend in ("frequency", "split", "legacy", "time"):
                        acc = dict(frequency=.80, split=.75, legacy=.60, time=.50)[frontend]
                        if mode == "gate" and frontend == "split":
                            acc = .90
                        job = dict(dataset=dataset, fold=fold, seed=3101,
                                   phase="features", frontend=frontend, recipe="unified")
                        job["id"] = select.job_id(job)
                        first.append(evidence(job, acc))
            records["features"] = first
            with patch.object(select, "HERE", here), \
                    patch.object(select, "require_committed"), \
                    patch.object(select, "sha", return_value="synthetic-sha"), \
                    patch.object(select, "load_phase", side_effect=lambda p: records[p]), \
                    redirect_stdout(io.StringIO()):
                for phase in phases:
                    with patch.object(sys, "argv", ["source-select", phase]):
                        select.main()
                    following = phases[phases.index(phase) + 1] if phase != "combine" else "final"
                    jobs = json.loads((here / f"jobs_{following}.json").read_text())
                    self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
                    self.assertLessEqual(len(jobs), registration["phase_caps"][following])
                    if following == "final":
                        final = json.loads((here / "decision_final.json").read_text())
                        break
                    rows = []
                    for job in jobs:
                        factor = job.get("factor", job["frontend"])
                        if following == "replicate":
                            original = next(r for r in first if
                                            all(r["job"][k] == job[k]
                                                for k in ("dataset", "fold", "frontend")))
                            acc = original["acc"]
                        elif mode in ("control", "gate"):
                            acc = .80 if factor in ("unified", "single_frozen") else .78
                            if mode == "gate" and factor == "single_gate":
                                acc = .82
                        else:
                            acc = dict(unified=.80, held_epoch=.85, phase=.84, balanced=.825,
                                       official=.82, shift=.79, cfo=.78, channel=.77,
                                       mild_noise=.79, single_frozen=.85, two_factors=.87)[factor]
                            if mode == "official" and factor in ("official", "single_frozen"):
                                acc = .86
                        epoch = 0
                        if job.get("checkpoint") == "held_epoch":
                            epoch = dict(training=40, training_replicate=55, combine=50)[following]
                        rows.append(evidence(job, acc, epoch))
                    records[following] = rows
                # A second invocation cannot silently rewrite a frozen decision.
                with patch.object(sys, "argv", ["source-select", "combine"]):
                    select.main()
                changed = dict(final, registration_sha256="different")
                with self.assertRaisesRegex(ValueError, "refusing changed frozen evidence"):
                    select.write_once(here / "decision_final.json", changed)
        return records, final

    def test_supported_factors_and_source_epoch_transfer(self):
        records, final = self.exercise("improve")
        self.assertEqual(len(records["replicate"]), 24)
        self.assertEqual(len(records["training"]), 72)
        self.assertEqual(len(records["training_replicate"]), 24)
        self.assertEqual(len(records["combine"]), 16)
        self.assertEqual(len(final["jobs"]), 72)
        for job in final["jobs"]:
            self.assertEqual(job["frontend"], "frequency")
            self.assertIn(job["seed"], (4101, 4102, 4103))
            self.assertNotIn("held_epoch", (job.get("checkpoint"),))
            if job["pipeline"] == "source_training":
                self.assertEqual(job["epochs"], 50)
                self.assertEqual(job["checkpoint"], "last")
            elif job["pipeline"] == "source_combination":
                self.assertEqual(job["epochs"], 50)
                self.assertEqual(job["augmentation"], "phase")
        self.assertTrue(all("runs/final/" not in p for p in final["evidence"]))

    def test_no_supported_factors_retains_and_reports_control(self):
        records, final = self.exercise("control")
        self.assertEqual(len(records["combine"]), 8)
        self.assertTrue(all(not d["supported_factors"] for d in final["decisions"].values()))
        for job in final["jobs"]:
            self.assertNotIn("augmentation", job)
            self.assertNotIn("epochs", job)
        self.assertEqual({j["pipeline"] for j in final["jobs"]},
                         {"source_frontend", "source_training", "source_combination"})

    def test_gate_is_only_available_for_source_selected_split(self):
        records, final = self.exercise("gate")
        self.assertEqual(len(records["combine"]), 16)
        self.assertTrue(all(j["frontend"] == "split" for j in final["jobs"]))
        for job in final["jobs"]:
            if job["pipeline"] == "source_combination":
                self.assertEqual(job["fusion"], "gate")
            else:
                self.assertNotIn("fusion", job)

    def test_official_recipe_is_not_combined_with_unified_factors(self):
        records, final = self.exercise("official")
        self.assertEqual(len(records["combine"]), 8)
        self.assertTrue(all(r["job"]["recipe"] == "official" for r in records["combine"]))
        for job in final["jobs"]:
            if job["pipeline"] != "source_frontend":
                self.assertEqual(job["recipe"], "official")
                self.assertNotIn("checkpoint", job)

    def test_held_epoch_budget_requires_identical_fusion_graph(self):
        job = dict(frontend="split", recipe="unified", checkpoint="held_epoch")
        self.assertTrue(select.held_budget_matches(job, dict(job, fusion="pooled_concat")))
        self.assertFalse(select.held_budget_matches(job, dict(job, fusion="gate")))
        self.assertTrue(select.held_budget_matches(dict(job, fusion="gate"),
                                                   dict(job, fusion="gate")))
        self.assertFalse(select.held_budget_matches(job, dict(job, augmentation="phase")))


if __name__ == "__main__":
    unittest.main()
