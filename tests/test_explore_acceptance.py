"""Synthetic role and freeze-time checks without real benchmark data reads."""
import importlib.util
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("s07_acceptance_test", ROOT / "07-wifi-osu/07-17-verify.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class AccessTest(unittest.TestCase):
    def records(self, job):
        source, target = verify.receivers(job["dataset"], job["fold"])
        held = [] if job["phase"] == "final" else source[-2:]
        fit = source if not held else source[:-2]
        events = [(rx, day, "fit_source") for rx in fit for day in (1, 2)]
        events += [(rx, day, "held_source") for rx in held for day in (1, 2)]
        if job["phase"] == "final":
            events += [(rx, day, "target_report") for rx in target for day in (1, 2, 3, 4)]
        return [dict(run=job["id"], dataset=job["dataset"], fold=job["fold"],
                     rx=rx, day=day, purpose=purpose, t="2026-10-26T01:00:00")
                for rx, day, purpose in events]

    def check(self, jobs, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "access.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
            return verify.wifi_access(jobs, path, datetime.fromisoformat("2026-10-26T00:30:00+08:00"))

    def test_legal_reads_and_retained_duplicates(self):
        job = dict(dataset="ManySig", fold=0, phase="final", id="synthetic")
        rows = self.records(job)
        result = self.check([job], rows + [rows[0]])
        self.assertEqual(result["reads"], len(rows) + 1)
        self.assertEqual(result["duplicate_events"], 1)

    def test_source_target_and_day_leaks_rejected(self):
        job = dict(dataset="ManySig", fold=0, phase="training", id="synthetic")
        rows = self.records(job)
        for changed in (dict(rx=verify.receivers("ManySig", 0)[1][0]),
                        dict(day=3), dict(purpose="target_report"), dict(run="unregistered")):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.check([job], rows + [dict(rows[0], **changed)])

    def test_missing_access_and_early_target_rejected(self):
        job = dict(dataset="ManySig", fold=0, phase="final", id="synthetic")
        rows = self.records(job)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.check([job], rows[:-1])
        early = [dict(r, t="2026-10-26T00:00:00") if r["purpose"] == "target_report" else r for r in rows]
        with self.assertRaisesRegex(ValueError, "predates"):
            self.check([job], early)


if __name__ == "__main__":
    unittest.main()
