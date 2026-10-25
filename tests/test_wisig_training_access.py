"""Synthetic Stage08 data-role and freeze checks; never reads real WiSig IQ."""
import importlib.util
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08-wisig-training"))
import training
from xa.wisig import receivers

spec = importlib.util.spec_from_file_location("s08_access_test", ROOT / "08-wisig-training/08-7-verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class AccessTest(unittest.TestCase):
    def events(self, job):
        source, target = receivers(job["dataset"], job["fold"])
        fit, held = (source, []) if job["phase"] == "final" else training.source_groups(
            job["dataset"], job["fold"], job["group"])
        tuples = [(rx, day, "fit") for rx in fit for day in (1, 2)]
        tuples += [(rx, day, "held") for rx in held for day in (1, 2)]
        if job["phase"] == "final":
            tuples += [(rx, day, "target") for rx in target for day in (1, 2, 3, 4)]
        return [dict(t="2026-10-26T03:00:00+08:00", run=job["id"], dataset=job["dataset"],
                     fold=job["fold"], rx=rx, day=day, role=role) for rx, day, role in tuples]

    def test_roles_missing_and_final_freeze(self):
        source_job = dict(id="source", dataset="ManySig", fold=0, phase="screen", group="first")
        final_job = dict(id="final", dataset="ManySig", fold=0, phase="final")
        rows = self.events(source_job) + self.events(final_job)
        freeze = datetime.fromisoformat("2026-10-26T02:00:00+08:00")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "access.jsonl"
            def write(values):
                path.write_text("".join(json.dumps(r) + "\n" for r in values))
            write(rows)
            result = verify.check_access([source_job, final_job], path, freeze)
            self.assertEqual(result["repeated_events"], 0)
            write(rows[:-1])
            with self.assertRaisesRegex(ValueError, "missing"):
                verify.check_access([source_job, final_job], path, freeze)
            write(rows)
            with self.assertRaisesRegex(ValueError, "before committed"):
                verify.check_access([source_job, final_job], path,
                                    datetime.fromisoformat("2026-10-26T04:00:00+08:00"))
            corrupted = [dict(r) for r in rows]
            corrupted[0]["day"] = 3
            write(corrupted)
            with self.assertRaisesRegex(ValueError, "role mismatch"):
                verify.check_access([source_job, final_job], path, freeze)


if __name__ == "__main__":
    unittest.main()
