"""Synthetic Stage C engineering QA: no IQ, trained weights or scientific logits."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage_c_analysis", ROOT / "research/standard_method_competition_c/analyze.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def predictions(view="V0", correct=True, rows_per_rx=3):
    receivers = [f"{view}r{i}" for i in range(6)]
    rx = np.repeat(receivers, rows_per_rx)
    y = np.tile(np.arange(rows_per_rx) % 3, 6).astype(np.int64)
    predicted = y if correct else (y + 1) % 10
    logits = np.full((len(y), 10), -2., dtype=np.float32)
    logits[np.arange(len(y)), predicted] = 2.
    return dict(logits=logits, y=y, rx=rx, day=np.ones(len(y), dtype=np.int32),
                keys=np.array([f"{r}|1|synthetic|{i}" for i, r in enumerate(rx)]), pool_ids=np.arange(len(y), dtype=np.int64))


def grid():
    return {e: {(a, wd, v, s): predictions(v) for a in analysis.ARMS for wd in analysis.WEIGHT_DECAYS
                for v in analysis.VIEWS for s in analysis.STREAMS} for e in analysis.ENDPOINTS}


def passing_comparison():
    return dict(rx_equal=dict(accuracy_delta=.01), paired_rx_bootstrap_95_intervals=dict(accuracy_delta=[.001, .02]),
        per_view={v: dict(accuracy_delta=.01) for v in analysis.VIEWS},
        per_stream={s: dict(accuracy_delta=.01 if s != "S2" else 0.) for s in analysis.STREAMS},
        worst_paired_rx=dict(accuracy_delta=-.02),
        tx_rx_stream_mean=[dict(packets_per_stream=50, risk_increase=.05), dict(packets_per_stream=49, risk_increase=.9)])


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def sealed_fixture(stage):
    # Only metadata is reused from the frozen registration; all payload bytes
    # below are explicit toy strings, never model weights or observations.
    jobs = json.loads((analysis.HERE / "JOBS.json").read_text(encoding="utf-8"))
    plan = json.loads((analysis.HERE / "SOURCE_PLAN.json").read_text(encoding="utf-8"))
    write_json(stage / "JOBS.json", jobs)
    write_json(stage / "SOURCE_PLAN.json", plan)
    registration = dict(artifact_sha256={name: analysis.file_sha256(stage / name) for name in ("JOBS.json", "SOURCE_PLAN.json")})
    write_json(stage / "PRE_REGISTRATION.json", registration)
    write_json(stage / "EXECUTION_REGISTRATION.json", dict(synthetic=True))
    hashes = dict(registration_sha256=analysis.file_sha256(stage / "PRE_REGISTRATION.json"),
                  execution_sha256=analysis.file_sha256(stage / "EXECUTION_REGISTRATION.json"))
    seal = dict(status="SEALED_STAGE_C", runs={}, **hashes)
    files = {}
    for job in jobs:
        directory = stage / "artifacts" / job["id"]
        directory.mkdir(parents=True)
        checkpoints = {}
        training_files = {}
        for arm in job["readouts"]:
            for endpoint in analysis.ENDPOINTS:
                name = f"{arm}_{endpoint}.pt"
                (directory / name).write_bytes(b"synthetic checkpoint marker, not torch data")
                checkpoints[name] = analysis.file_sha256(directory / name)
                training_files[name] = checkpoints[name]
                valname = f"{arm}_{endpoint}_val.npz"
                (directory / valname).write_bytes(b"synthetic validation marker")
                training_files[valname] = analysis.file_sha256(directory / valname)
                name = f"{arm}_{endpoint}_selection.npz"
                (directory / name).write_bytes(b"synthetic prediction marker, not numpy data")
                files[f"{job['id']}/{name}"] = analysis.file_sha256(directory / name)
        write_json(directory / "COMPLETE.json", dict(files_sha256=training_files))
        seal["runs"][job["id"]] = dict(complete_sha256=analysis.file_sha256(directory / "COMPLETE.json"), checkpoints=checkpoints)
    write_json(stage / "STAGE_C_SEAL.json", seal)
    write_json(stage / "EVALUATION_MANIFEST.json", dict(files=files, seal_sha256=analysis.file_sha256(stage / "STAGE_C_SEAL.json"), **hashes))
    return jobs


class StageCAnalysisTest(unittest.TestCase):
    def test_exact_factorial_grid_and_family_readouts(self):
        jobs = json.loads((analysis.HERE / "JOBS.json").read_text(encoding="utf-8"))
        plan = json.loads((analysis.HERE / "SOURCE_PLAN.json").read_text(encoding="utf-8"))
        analysis.verify_design(jobs, plan)
        self.assertEqual(len(analysis.expected_prediction_files(jobs)), 108)
        bad = copy.deepcopy(jobs)
        bad[0]["readouts"] = ["C0"]
        with self.assertRaisesRegex(ValueError, "shared ERM"):
            analysis.verify_design(bad, plan)
        with self.assertRaisesRegex(ValueError, "24-fit"):
            analysis.verify_design(jobs[:-1], plan)
        bad = copy.deepcopy(jobs)
        bad[0]["order_seed"] += 1
        with self.assertRaisesRegex(ValueError, "random stream"):
            analysis.verify_design(bad, plan)

    def test_all_endpoints_all_wd_are_required_and_identical_pairs_cancel(self):
        data = grid()
        result = analysis.analyze_predictions(data, replicates=100, seed=7)
        self.assertEqual(result["weight_decay_selection"]["chosen"], {a: 0. for a in analysis.ARMS})
        self.assertEqual(result["decision"]["priority"], [])
        self.assertEqual(result["gradient_fit_count"], 24)
        self.assertEqual(result["arm_readout_candidates"], 36)
        self.assertEqual(result["uncertainty"]["receivers"], [f"V{v}r{i}" for v in range(2) for i in range(6)])
        for endpoint in analysis.ENDPOINTS:
            self.assertEqual(len(result["endpoints"][endpoint]["candidates"]), 36)
            self.assertEqual(len(result["endpoints"][endpoint]["same_weight_decay_comparisons"]), 4)
            for pair in result["endpoints"][endpoint]["same_weight_decay_comparisons"].values():
                self.assertEqual(pair["paired_rx_bootstrap_95_intervals"]["accuracy_delta"], [0., 0.])
        with self.assertRaisesRegex(ValueError, "all three endpoints"):
            analysis.analyze_predictions({"selected": data["selected"]})
        with self.assertRaisesRegex(ValueError, "all three endpoints"):
            incomplete = copy.deepcopy(data)
            incomplete["epoch10"].pop(next(iter(incomplete["epoch10"])))
            analysis.analyze_predictions(incomplete)

    def test_rx_metrics_are_stream_then_receiver_equal_not_packet_weighted(self):
        summaries = {}
        for view in analysis.VIEWS:
            for stream_index, stream in enumerate(analysis.STREAMS):
                rows = {f"{view}r{i}": dict(accuracy=(i + stream_index) / 10, macro_f1=.2, ce=.4, packets=1000 if i == 0 else 1) for i in range(6)}
                groups = [dict(rx=rx, tx_class_index=0, packets=1, accuracy=row["accuracy"], macro_f1=.2, ce=.4) for rx, row in rows.items()]
                summaries[(view, stream)] = dict(per_rx=rows, rx_equal=analysis._mean_rows(list(rows.values()), analysis.METRICS), tx_rx=groups)
        aggregate = analysis.aggregate_candidate(summaries)
        self.assertAlmostEqual(aggregate["rx_equal"]["accuracy"], .35)
        self.assertAlmostEqual(aggregate["per_rx"]["V0r0"]["accuracy"], .1)
        self.assertAlmostEqual(aggregate["per_stream"]["S0"]["accuracy"], .25)

    def test_promotion_all_gates_jointly_and_registered_boundary_directions(self):
        comparison = passing_comparison()
        result = analysis.promotion_check(comparison)
        self.assertTrue(result["passes"])
        self.assertEqual(result["supported_tx_rx_groups"], 1)
        mutations = [
            lambda d: d["rx_equal"].update(accuracy_delta=.009999),
            lambda d: d["paired_rx_bootstrap_95_intervals"].update(accuracy_delta=[0., .02]),
            lambda d: d["per_view"]["V0"].update(accuracy_delta=0.),
            lambda d: d["per_stream"]["S1"].update(accuracy_delta=0.),
            lambda d: d["worst_paired_rx"].update(accuracy_delta=-.020001),
            lambda d: d["tx_rx_stream_mean"][0].update(risk_increase=.050001),
        ]
        for mutation in mutations:
            changed = copy.deepcopy(comparison)
            mutation(changed)
            self.assertFalse(analysis.promotion_check(changed)["passes"])
        self.assertTrue(analysis.promotion_check(comparison)["passes"])

    def test_weight_decay_uses_ce_and_smaller_exact_tie_not_accuracy(self):
        candidates = {(a, wd): dict(rx_equal=dict(ce=.8, accuracy=1. if wd == 0 else 0.)) for a in analysis.ARMS for wd in analysis.WEIGHT_DECAYS}
        candidates[("C1", .001)]["rx_equal"]["ce"] = .7
        self.assertEqual(analysis.choose_weight_decays(candidates), dict(C0=0., C1=.001, C2=0.))
        promotions = {a: dict(passes=True) for a in ("C1", "C2")}
        selected = {a: dict(rx_equal=dict(ce=.8)) for a in analysis.ARMS}
        self.assertEqual(analysis.promotion_priority(promotions, selected), ["C1", "C2"])
        selected["C2"]["rx_equal"]["ce"] = .7
        self.assertEqual(analysis.promotion_priority(promotions, selected), ["C2", "C1"])
        promotions["C2"]["passes"] = False
        self.assertEqual(analysis.promotion_priority(promotions, selected), ["C1"])

    def test_fixed_ten_class_macro_f1_zero_support_pairs_and_margin(self):
        data = predictions(rows_per_rx=1)
        summary = analysis._summary(data)
        self.assertAlmostEqual(summary["rx_equal"]["macro_f1"], .1)
        self.assertEqual(len(summary["zero_support_tx_rx"]), 54)
        self.assertEqual(len(summary["class_pair_counts"]), 600)
        self.assertEqual(summary["per_rx"]["V0r0"]["margin_quantiles"]["0.5"], 4.)
        row = next(r for r in summary["class_pair_counts"] if r["rx"] == "V0r0" and r["tx_class_index"] == 0 and r["competitor_class_index"] == 1)
        self.assertEqual(row["hardest_negative_count"], 1)

    def test_same_accuracy_can_hide_error_correction_new_errors_and_class_changes(self):
        first = predictions(rows_per_rx=3)
        second = copy.deepcopy(first)
        for data, preds in ((first, [1, 1, 3]), (second, [0, 2, 4])):
            data["logits"][:] = -2
            data["logits"][np.arange(18), np.tile(preds, 6)] = 2
        a = {(v, s): dict(first, rx=np.char.replace(first["rx"], "V0", v),
                              keys=np.char.replace(first["keys"], "V0", v)) for v in analysis.VIEWS for s in analysis.STREAMS}
        b = {(v, s): dict(second, rx=np.char.replace(second["rx"], "V0", v),
                               keys=np.char.replace(second["keys"], "V0", v)) for v in analysis.VIEWS for s in analysis.STREAMS}
        pair = analysis.paired_comparison(a, b, np.zeros((20, 12), dtype=int))
        self.assertEqual(pair["rx_equal"]["accuracy_delta"], 0.)
        self.assertEqual(pair["rx_equal"]["disagreement"], 1.)
        row = pair["cells"]["V0/S0"]["packet_weighted"]
        self.assertEqual(row["corrected_errors"], 6)
        self.assertEqual(row["new_errors"], 6)
        self.assertEqual(row["both_wrong_changed_class"], 6)

    def test_pairing_uses_full_keys_and_source_pool_identity(self):
        first = predictions()
        permutation = np.arange(18)[::-1]
        second = {key: value[permutation] for key, value in first.items()}
        analysis._align(first, second)
        second["pool_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "pool row identity"):
            analysis._align(first, second)

    def test_pool_role_validation_rejects_forbidden_days_receivers_and_row_changes(self):
        data = predictions()
        role = dict(rows=len(data["y"]), dtype="int64", indices_array_sha256=hashlib.sha256(data["pool_ids"].tobytes()).hexdigest(),
                    tx_rx_day_support=[dict(rx=rx, tx=tx, day=1, rows=int(((data["rx"] == rx) & (data["y"] == tx)).sum())) for rx in sorted(set(data["rx"])) for tx in range(10)])
        view = dict(source_selection=role, source_selection_rx=sorted(set(data["rx"])))
        reg = dict(forbidden_historical_held_rx=["forbidden"], forbidden_target_rx=["target"],
                   source_cell_sha256={f"{rx}|1": "synthetic" for rx in set(data["rx"])})
        validated = analysis.validate_source_predictions(data, view, reg)
        self.assertEqual(len(validated["y"]), 18)
        changed = copy.deepcopy(data)
        changed["day"][0] = 3
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            analysis.validate_source_predictions(changed, view, reg)
        changed = copy.deepcopy(data)
        changed["pool_ids"][0] = 1000
        with self.assertRaisesRegex(ValueError, "row digest"):
            analysis.validate_source_predictions(changed, view, reg)
        changed = copy.deepcopy(data)
        changed["y"][0] = 9
        with self.assertRaisesRegex(ValueError, "support differs"):
            analysis.validate_source_predictions(changed, view, reg)
        reg["forbidden_historical_held_rx"] = [str(data["rx"][0])]
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            analysis.validate_source_predictions(data, view, reg)

    def test_all_seals_verified_without_any_payload_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = Path(temporary)
            jobs = sealed_fixture(stage)
            with mock.patch.object(np, "load", side_effect=AssertionError("No decode may occur during verification")):
                paths = analysis.verified_prediction_files(stage)
                self.assertEqual(len(paths), 108)
                # Corrupt the very last artifact: complete inventory must be checked.
                last = paths[sorted(paths)[-1]]
                last.write_bytes(b"changed synthetic payload")
                with self.assertRaisesRegex(ValueError, "altered prediction"):
                    analysis.verified_prediction_files(stage)

    def test_checkpoint_completion_registration_and_inventory_mutations_fail(self):
        mutations = ("checkpoint", "completion", "registration", "inventory")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                stage = Path(temporary)
                jobs = sealed_fixture(stage)
                if mutation == "checkpoint":
                    (stage / "artifacts" / jobs[-1]["id"] / "C2_final40.pt").write_bytes(b"changed synthetic checkpoint")
                elif mutation == "completion":
                    (stage / "artifacts" / jobs[-1]["id"] / "COMPLETE.json").write_text("{}")
                elif mutation == "registration":
                    (stage / "PRE_REGISTRATION.json").write_text("{}")
                else:
                    manifest = analysis.read_json(stage / "EVALUATION_MANIFEST.json")
                    manifest["files"].pop(next(iter(manifest["files"])))
                    write_json(stage / "EVALUATION_MANIFEST.json", manifest)
                with self.assertRaises((ValueError, KeyError)):
                    analysis.verified_prediction_files(stage)

    def test_source_views_cannot_overlap_or_mismatch_canonical_inputs(self):
        data = grid()
        bad = copy.deepcopy(data)
        bad["selected"][("C1", 0., "V0", "S0")]["keys"][0] = "different|input|key|0"
        with self.assertRaisesRegex(ValueError, "same canonical inputs"):
            analysis.analyze_predictions(bad, replicates=10)
        for endpoint in analysis.ENDPOINTS:
            for key in data[endpoint]:
                if key[2] == "V1":
                    data[endpoint][key] = predictions("V0")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            analysis.analyze_predictions(data, replicates=10)

    def test_stream_mean_risk_support_never_multiplied_by_three(self):
        first, second = {}, {}
        for v in analysis.VIEWS:
            for s in analysis.STREAMS:
                first[(v, s)] = predictions(v, rows_per_rx=49)
                second[(v, s)] = predictions(v, correct=False, rows_per_rx=49)
        pair = analysis.paired_comparison(first, second, np.zeros((10, 12), dtype=int))
        # Each TX has only16/17 observations per RX, not 48/51 independent observations.
        self.assertIsNone(pair["worst_supported_tx_rx_risk_increase"])
        self.assertTrue(all(r["packets_per_stream"] < 50 for r in pair["tx_rx_stream_mean"]))


if __name__ == "__main__":
    unittest.main()
