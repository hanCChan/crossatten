"""Synthetic controls: no scientific arrays, CUDA work, or checkpoint reads."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import runner


def registration():
    seeds = runner.EXPECTED_RUNS
    return {
        "schema_version": 1,
        "stage": "A",
        "maximum_new_fits": 5,
        "run_sequence": list(seeds),
        "runs": [dict(id=name, initialization_seed=s[0], order_seed=s[1], stochastic_seed=s[2])
                 for name, s in seeds.items()],
        "fit_rx": ["7-7", "7-14", "8-7", "8-8", "8-14", "13-7", "13-14", "14-7",
                   "18-2", "18-19", "23-3", "23-5", "23-6", "23-7", "24-5", "24-6", "24-13", "24-16"],
        "held_rx": ["1-1", "1-19", "1-20", "2-1", "2-19", "3-19"],
        "target_rx": ["19-1", "19-2", "19-19", "19-20", "20-1", "20-19", "20-20", "23-1"],
    }


class SemanticEvidenceTests(unittest.TestCase):
    def test_mapping_order_is_not_numerical_state(self):
        first = {"model": {"weight": torch.tensor([1.0, 2.0])}, "epoch": 10}
        second = {"epoch": 10, "model": {"weight": torch.tensor([1.0, 2.0])}}
        self.assertEqual(runner.semantic_hash(first), runner.semantic_hash(second))
        self.assertTrue(runner.compare_objects(first, second)["exact"])

    def test_hash_captures_tensor_dtype_shape_and_buffer_updates(self):
        base = {"bn.running_mean": torch.zeros(2), "bn.num_batches_tracked": torch.tensor(0)}
        changed = copy.deepcopy(base)
        changed["bn.running_mean"][0] = 0.125
        self.assertNotEqual(runner.semantic_hash(base), runner.semantic_hash(changed))
        self.assertNotEqual(runner.semantic_hash(torch.ones(2)), runner.semantic_hash(torch.ones(2).double()))
        self.assertNotEqual(runner.semantic_hash(torch.ones(2)), runner.semantic_hash(torch.ones(1, 2)))

    def test_exact_comparison_does_not_use_accuracy_or_numerical_tolerance(self):
        first = {"model": {"weight": torch.tensor([1.0, 2.0], dtype=torch.float64)}}
        second = {"model": {"weight": torch.tensor([1.0, 2.0 + 1e-9], dtype=torch.float64)}}
        result = runner.compare_objects(first, second)
        self.assertFalse(result["exact"])
        self.assertGreater(result["max_abs_numeric_difference"], 0)
        self.assertIn("root/model/weight", result["differing_paths"])

    def test_signed_zero_bytes_remain_distinct_for_bitwise_gate(self):
        positive = np.array([0.0], dtype=np.float32)
        negative = np.array([-0.0], dtype=np.float32)
        self.assertNotEqual(runner.semantic_hash(positive), runner.semantic_hash(negative))
        result = runner.compare_objects(positive, negative)
        self.assertFalse(result["exact"])
        self.assertEqual(result["max_abs_numeric_difference"], 0)

    def test_shape_or_dtype_change_is_not_zero_difference_success(self):
        array = np.array([1, 2], dtype=np.int32)
        for other in (array.astype(np.int64), array.reshape(1, 2)):
            result = runner.compare_objects(array, other)
            self.assertFalse(result["exact"])
            self.assertGreater(result["differing_path_count"], 0)

    def test_key_uses_rx_day_cell_lineage_and_cell_ordinal(self):
        keys = runner.make_keys(np.array([0, 1, 0]), np.array([0, 0, 1]),
                                np.array([1, 1, 1]), ["RX-A", "RX-B"],
                                {"RX-A|1": "a" * 64, "RX-B|1": "b" * 64})
        self.assertEqual(len(set(keys)), 3)
        self.assertTrue(keys[0].endswith("|0"))
        self.assertTrue(keys[1].endswith("|1"))
        self.assertTrue(keys[2].endswith("|0"))
        self.assertNotEqual(keys[0], keys[2])
        changed_lineage = runner.make_keys(np.array([0]), np.array([0]), np.array([1]),
                                           ["RX-A"], {"RX-A|1": "c" * 64})
        self.assertNotEqual(keys[0], changed_lineage[0])


class ScientificGateTests(unittest.TestCase):
    def test_committed_text_allows_checkout_line_endings_but_rejects_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.py"
            source.write_bytes(b"a = 1\r\nb = 2\r\n")
            with patch.object(runner, "ROOT", root), \
                    patch.object(runner.subprocess, "check_output", return_value=b"a = 1\nb = 2\n"):
                runner.require_committed(source)
                source.write_bytes(b"a = 9\r\nb = 2\r\n")
                with self.assertRaises(ValueError):
                    runner.require_committed(source)

    def test_registered_factor_levels_and_budget_are_fixed(self):
        runner.validate_runs(registration())
        changed = registration()
        changed["runs"][2]["stochastic_seed"] += 1
        with self.assertRaises(ValueError):
            runner.validate_runs(changed)
        changed = registration()
        changed["runs"][2]["id"] = "A00"
        with self.assertRaises(ValueError):
            runner.validate_runs(changed)

    def test_receiver_role_overlap_rejected(self):
        changed = registration()
        changed["target_rx"][0] = changed["fit_rx"][0]
        with self.assertRaises(ValueError):
            runner.validate_runs(changed)

    def test_failed_replay_gate_stops_before_data_or_artifact_access(self):
        with patch.object(runner, "read", return_value={"status": "FAIL_REPLAY"}), \
                patch.object(runner, "file_sha") as checksum, \
                patch.object(runner, "artifact_root") as artifacts:
            with self.assertRaises(ValueError):
                runner.check_gate(registration(), Path("synthetic-registration.json"))
            checksum.assert_not_called()
            artifacts.assert_not_called()

    def test_later_factor_fit_cannot_read_data_before_gate(self):
        with patch.object(runner, "check_gate", side_effect=ValueError("replay missing")), \
                patch.object(runner, "load_fit") as load, \
                patch.object(runner, "setup") as setup, \
                patch.object(runner, "artifact_root") as artifacts:
            with self.assertRaisesRegex(ValueError, "replay missing"):
                runner.train(registration(), Path("synthetic-registration.json"), "A10")
            load.assert_not_called()
            setup.assert_not_called()
            artifacts.assert_not_called()


if __name__ == "__main__":
    unittest.main()
