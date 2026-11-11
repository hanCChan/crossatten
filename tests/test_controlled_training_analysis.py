"""Synthetic scientific checks; no registered IQ, held data, or weights are read."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("controlled_analysis", ROOT / "research/controlled_training_a/analyze.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def predictions(predictions=(0, 1, 0, 1), y=(0, 0, 1, 1), rx=("r1", "r1", "r2", "r2")):
    logits = np.full((len(y), 2), -1., dtype=np.float32)
    logits[np.arange(len(y)), predictions] = 1.
    return dict(logits=logits, y=np.array(y), rx=np.array(rx), day=np.ones(len(y), dtype=int),
                keys=np.array([f"{rx[i]}|1|sha|{i}" for i in range(len(y))]))


class AnalysisTest(unittest.TestCase):
    def test_same_accuracy_can_hide_corrected_and_new_errors(self):
        first = predictions((0, 0, 0, 0))
        second = predictions((1, 0, 1, 0))
        result = analysis.compare_predictions(first, second)
        row = result["packet_weighted"]
        self.assertEqual(row["accuracy_delta"], 0.)
        self.assertEqual(row["prediction_disagreements"], 2)
        self.assertEqual(row["corrected_errors"], 1)
        self.assertEqual(row["new_errors"], 1)
        self.assertEqual(row["disagreement"], .5)
        self.assertEqual(row["prediction_disagreements"], row["corrected_errors"] + row["new_errors"] + row["both_wrong_changed_class"])

    def test_full_input_keys_pair_predictions_even_after_permutation(self):
        first = predictions()
        permutation = np.array([3, 1, 0, 2])
        second = {key: value[permutation] for key, value in first.items()}
        paired = analysis.compare_predictions(first, second)
        self.assertEqual(paired["packet_weighted"]["disagreement"], 0.)
        self.assertEqual(paired["packet_weighted"]["ce_delta"], 0.)
        second["keys"][0] = "unrelated|1|sha|3"
        with self.assertRaisesRegex(ValueError, "same canonical inputs"):
            analysis.compare_predictions(first, second)

    def test_same_cache_position_different_rx_is_never_a_pair(self):
        first = predictions()
        changed = {key: value.copy() for key, value in first.items()}
        changed["rx"][0] = "r2"
        with self.assertRaisesRegex(ValueError, "conflicting rx"):
            analysis.compare_predictions(first, changed)
        changed = {key: value.copy() for key, value in first.items()}
        changed["keys"][0] = changed["keys"][1]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            analysis.validate_predictions(changed)

    def test_stable_ce_and_raw_margin_use_true_and_hardest_negative(self):
        logits = np.array([[10000., 9999., 9998.], [-10001., -10000., -10002.]])
        quantities = analysis.packet_quantities(logits, np.array([0, 0]))
        self.assertTrue(np.isfinite(quantities["ce"]).all())
        np.testing.assert_allclose(quantities["ce"], [np.log(1 + np.exp(-1) + np.exp(-2)), 1 + np.log(1 + np.exp(-1) + np.exp(-2))])
        np.testing.assert_array_equal(quantities["hardest_negative"], [1, 1])
        np.testing.assert_array_equal(quantities["margin"], [1., -1.])

    def test_rx_equal_metric_does_not_weight_large_rx_more(self):
        data = predictions((0, 0, 0, 0), y=(0, 0, 0, 1), rx=("r1", "r1", "r1", "r2"))
        result = analysis.summarize_predictions(data)
        self.assertEqual(result["packet_weighted"]["accuracy"], .75)
        self.assertEqual(result["rx_equal"]["accuracy"], .5)
        self.assertEqual(result["per_rx"]["r1"]["macro_f1"], .5)
        self.assertEqual(len(result["tx_rx_day"]), 2)

    def test_factorial_signs_interaction_and_paired_bootstrap(self):
        contrasts = analysis.factorial_contrasts(dict(A00=1., A10=3., A01=4., A11=10.))
        self.assertEqual(contrasts, dict(init=4., order=5., interaction=4.))
        base = predictions()
        # Equal predictions in every cell produce exactly zero effects for every
        # RX bootstrap draw, rather than uncertainty from independently sampled arms.
        result = analysis.analyze_endpoint({run: base for run in analysis.RUNS}, replicates=100, seed=4)
        for values in result["factor_contrasts"].values():
            for entry in values.values():
                self.assertEqual(entry["estimate"], 0.)
                self.assertEqual(entry["paired_rx_bootstrap_95_interval"], [0., 0.])
        self.assertEqual(len(result["paired_changes"]), 5)
        with self.assertRaisesRegex(ValueError, "replay is QC"):
            analysis.analyze_endpoint({**{run: base for run in analysis.RUNS}, "A00-replay": base})
        with self.assertRaisesRegex(ValueError, "Report both"):
            analysis.make_analysis(dict(selected={run: base for run in analysis.RUNS}))

    def test_analysis_verifies_all_training_and_prediction_bytes_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            (stage / "PRE_REGISTRATION.json").write_text("{}")
            registration_sha = analysis.file_sha256(stage / "PRE_REGISTRATION.json")
            seal = dict(replay_pass=True, registration_sha256=registration_sha, runs={})
            files = {}
            for run in (*analysis.RUNS, "A00-replay"):
                out = stage / "artifacts" / run
                out.mkdir(parents=True)
                (out / "selected.pt").write_bytes(b"synthetic state")
                (out / "COMPLETE.json").write_text(json.dumps(dict(files_sha256={"selected.pt": analysis.file_sha256(out / "selected.pt")})))
                seal["runs"][run] = dict(complete_sha256=analysis.file_sha256(out / "COMPLETE.json"))
                for endpoint in analysis.ENDPOINTS:
                    for split in ("val", "held"):
                        name = f"{endpoint}_{split}.npz"
                        (out / name).write_bytes(b"unopened synthetic prediction bytes")
                        files[f"{run}/{name}"] = analysis.file_sha256(out / name)
            (stage / "STAGE_A_SEAL.json").write_text(json.dumps(seal))
            manifest = dict(stage_a_seal_sha256=analysis.file_sha256(stage / "STAGE_A_SEAL.json"), registration_sha256=registration_sha, files=files)
            (stage / "EVALUATION_MANIFEST.json").write_text(json.dumps(manifest))
            self.assertEqual(len(analysis.verified_prediction_files(stage)), 20)
            changed = stage / "artifacts/A00/selected_held.npz"
            changed.write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "altered prediction"):
                analysis.verified_prediction_files(stage)
            changed.write_bytes(b"unopened synthetic prediction bytes")
            (stage / "artifacts/A00/selected.pt").write_bytes(b"altered")
            with self.assertRaisesRegex(ValueError, "training artifact changed"):
                analysis.verified_prediction_files(stage)

    def test_explicit_packet_diagnostics_keep_true_negative_margin_and_identity(self):
        data = predictions()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "diagnostics.npz"
            record = analysis.export_packet_diagnostics(path, data)
            self.assertEqual(record["sha256"], analysis.file_sha256(path))
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["keys"], data["keys"])
                np.testing.assert_array_equal(saved["y"], data["y"])
                np.testing.assert_array_equal(saved["margin"], [2., -2., -2., 2.])
                self.assertEqual(set(saved.files), {"y", "rx", "day", "keys", "prediction", "hardest_negative", "margin", "ce"})


if __name__ == "__main__":
    unittest.main()
