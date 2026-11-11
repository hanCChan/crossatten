"""Synthetic tests for independent JSON-only C execution/result checks."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


qa = module("independent_c_qa", ROOT / "research/standard_method_competition_c/independent_QA.py")
analysis = module("registered_c_analysis_for_synthetic_fixture_only", ROOT / "research/standard_method_competition_c/analyze.py")


def predictions(view):
    rx = np.repeat([f"{view}r{i}" for i in range(6)], 3)
    y = np.tile(np.arange(3), 6)
    logits = np.full((18, 10), -2., dtype=np.float32)
    logits[np.arange(18), y] = 2.
    return dict(logits=logits, y=y, rx=rx, day=np.ones(18, dtype=int),
                keys=np.array([f"{r}|1|synthetic|{i}" for i, r in enumerate(rx)]), pool_ids=np.arange(18, dtype=np.int64))


def toy_plan():
    result = []
    for view in qa.VIEWS:
        data = predictions(view)
        receivers = sorted(set(data["rx"]))
        support = [dict(rx=rx, tx=tx, day=1, rows=int(((data["rx"] == rx) & (data["y"] == tx)).sum())) for rx in receivers for tx in range(10)]
        result.append(dict(id=view, source_selection_rx=receivers, source_selection=dict(rows=18, tx_rx_day_support=support)))
    return dict(views=result)


def synthetic_training():
    # Frozen job/source metadata, with invented scalar traces and digest strings.
    stage = ROOT / "research/standard_method_competition_c"
    jobs = json.loads((stage / "JOBS.json").read_text(encoding="utf-8"))
    plan = json.loads((stage / "SOURCE_PLAN.json").read_text(encoding="utf-8"))
    views = {v["id"]: v for v in plan["views"]}
    evidence = {}
    for job in jobs:
        view = views[job["view"]]
        arms, ema = job["readouts"], job["family"] == "ERM_EMA"
        config = dict(job=job, view=job["view"], registration_sha256="reg", execution_sha256="exec", train_rows=view["fit_train"]["rows"],
            checkpoint_val_rows=view["checkpoint_val"]["rows"], allowed_training_RX=view["fit_rx"], forbidden_training_RX=view["source_selection_rx"],
            source_selection_IQ_read=False, train_keys_sha256="t"*64, val_keys_sha256="v"*64)
        initial = {f: "i"*64 for f in qa.INITIAL_FIELDS}
        history = []
        lr_path = qa.independent_lr_path([1.]*40)
        for epoch in range(1, 41):
            row = dict(epoch=epoch, updates=127*epoch, val_rx_ce={a: 1./(epoch+1) if a == "C1" else 1. for a in arms},
                best_epochs={a: epoch if a == "C1" else 1 for a in arms}, lr_used=lr_path[epoch-1][0], lr_next=lr_path[epoch-1][1],
                calibration_preserved_training_state=True, train_ce=1., train_objective=.3, train_accuracy=.5,
                **{f: str(epoch).zfill(64) for f in qa.ORDER_FIELDS})
            if not ema:
                row.update(q_min=1./12, q_max=1./12)
            history.append(row)
        checkpoints = {f"{a}_{e}.pt": "c"*64 for a in arms for e in qa.ENDPOINTS}
        complete = dict(status="COMPLETE", job=job, registration_sha256="reg", execution_sha256="exec", epochs=40, updates=5080, active_parameters=82426,
            best_epochs=history[-1]["best_epochs"], best_val_rx_ce=history[-1]["val_rx_ce"], checkpoints=checkpoints, files_sha256=checkpoints,
            val_inference_replay={f"{a}_{e}":dict(exact=True) for a in arms for e in qa.ENDPOINTS},
            costs=dict(BN_calibration_passes=40 if ema else 0, BN_calibration_samples=40*config["train_rows"] if ema else 0,
                       EMA_updates=5080 if ema else 0, training_with_ema_seconds=1.))
        evidence[job["id"]] = dict(CONFIG=config, INITIAL=initial, HISTORY=history, COMPLETE=complete)
    return evidence, jobs, plan


class IndependentCQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        grid = {e: {(a, wd, v, s): predictions(v) for a in qa.ARMS for wd in qa.WDS for v in qa.VIEWS for s in qa.STREAMS} for e in qa.ENDPOINTS}
        cls.result = analysis.analyze_predictions(grid)
        cls.plan = toy_plan()

    def test_complete_synthetic_analysis_passes_without_importing_execution(self):
        result = qa.verify_analysis(self.result, self.plan)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["paired_comparisons"], 18)
        self.assertEqual(result["independently_recomputed_bootstrap_replicates"], 10000)

    def test_lr_patience_relative_threshold_and_minimum_independent_simulation(self):
        path = qa.independent_lr_path([1.]*40)
        self.assertEqual(path[0], (.001, .001))
        self.assertEqual(path[5], (.001, .001))
        self.assertEqual(path[6], (.001, .0001))
        self.assertAlmostEqual(path[12][1], .00001)
        self.assertAlmostEqual(path[18][1], .000001)
        self.assertAlmostEqual(path[-1][1], .000001)
        # A strict checkpoint improvement can still be below scheduler's
        # relative significance threshold; this must not reset its bad counter.
        path = qa.independent_lr_path([1.] + [.99995]*6)
        self.assertEqual(path[6][1], .0001)

    def test_complete24_training_ties_lr_ema_counts_and_shared_order(self):
        evidence, jobs, plan = synthetic_training()
        result = qa.verify_training(evidence, jobs, plan, "reg", "exec")
        self.assertEqual(result["totals"]["gradient_updates"], 121920)
        self.assertEqual(result["totals"]["BN_calibration_passes"], 480)
        self.assertEqual(result["totals"]["BN_calibration_samples"], 7780560)
        evidence[jobs[0]["id"]]["HISTORY"][1]["best_epochs"]["C0"] = 2
        with self.assertRaisesRegex(ValueError, "earliest-tie"):
            qa.verify_training(evidence, jobs, plan, "reg", "exec")

    def test_training_violations_are_detected(self):
        mutations = (
            lambda e, j: e[j[0]["id"]]["COMPLETE"].update(updates=5079),
            lambda e, j: e[j[0]["id"]]["COMPLETE"]["costs"].update(BN_calibration_samples=1),
            lambda e, j: e[j[0]["id"]]["HISTORY"][6].update(lr_next=.001),
            lambda e, j: e[j[1]["id"]]["HISTORY"][0].update(canonical_order_sha256="x"*64),
            lambda e, j: e[j[1]["id"]]["INITIAL"].update(model_sha256="x"*64),
        )
        for mutation in mutations:
            evidence, jobs, plan = synthetic_training()
            mutation(evidence, jobs)
            with self.assertRaises(ValueError):
                qa.verify_training(evidence, jobs, plan, "reg", "exec")

    def test_confusion_macro_f1_class_pair_and_margin_violations(self):
        original = self.result["endpoints"]["selected"]["candidates"]["C0/wd=0/V0/S0"]
        mutations = (
            lambda d: d["packet_weighted"].update(errors=1),
            lambda d: d["per_rx"]["V0r0"].update(macro_f1=1.),
            lambda d: d["packet_weighted"]["margin_quantiles"].update({"0.01": 5.}),
            lambda d: d["class_pair_counts"][0].update(hardest_negative_count=1),
            lambda d: d["tx_rx_day"][0].update(packets=9),
        )
        for mutation in mutations:
            changed = copy.deepcopy(original)
            mutation(changed)
            with self.assertRaises(ValueError):
                qa.verify_candidate(changed, self.plan["views"][0])

    def test_reported_wd_bootstrap_and_promotion_tampering_fail(self):
        for kind in ("wd", "bootstrap", "promotion"):
            changed = copy.deepcopy(self.result)
            if kind == "wd":
                changed["weight_decay_selection"]["chosen"]["C0"] = .001
            elif kind == "bootstrap":
                changed["endpoints"]["selected"]["same_weight_decay_comparisons"]["C1-C0/wd=0"]["paired_rx_bootstrap_95_intervals"]["accuracy_delta"] = [-.1, .1]
            else:
                changed["decision"]["promotions"]["C1"]["passes"] = True
            with self.assertRaises(ValueError):
                qa.verify_analysis(changed, self.plan)

    def test_promotion_boundaries_and_support_threshold(self):
        comparison = dict(rx_equal=dict(accuracy_delta=.01), paired_rx_bootstrap_95_intervals=dict(accuracy_delta=[.001, .02]),
            per_view={v:dict(accuracy_delta=.01) for v in qa.VIEWS}, per_stream={s:dict(accuracy_delta=.01 if s != "S2" else 0.) for s in qa.STREAMS},
            worst_paired_rx=dict(accuracy_delta=-.02), tx_rx_stream_mean=[dict(packets_per_stream=50,risk_increase=.05), dict(packets_per_stream=49,risk_increase=.99)])
        self.assertTrue(qa.independent_promotion(comparison)["passes"])
        comparison["paired_rx_bootstrap_95_intervals"]["accuracy_delta"][0] = 0.
        self.assertFalse(qa.independent_promotion(comparison)["passes"])
        comparison["paired_rx_bootstrap_95_intervals"]["accuracy_delta"][0] = .001
        comparison["tx_rx_stream_mean"][0]["risk_increase"] = .05001
        self.assertFalse(qa.independent_promotion(comparison)["passes"])

    def test_metadata_reader_refuses_scientific_payload_extensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            for suffix in (".npy", ".npz", ".pt", ".pkl"):
                path = Path(temporary) / ("synthetic" + suffix)
                with self.assertRaisesRegex(ValueError, "only JSON"):
                    qa.read_json(path)
                with self.assertRaisesRegex(ValueError, "No scientific payload"):
                    qa.text_sha(path)


if __name__ == "__main__":
    unittest.main()
