"""Selection is determined by source evidence, not stored Shen dev_val scores."""
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_workflow", ROOT / "06-paper-validation/06-2-workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def main():
    registered = json.loads((ROOT / "06-paper-validation/arms.json").read_text())
    with tempfile.TemporaryDirectory() as directory:
        workflow.HERE = Path(directory)
        workflow.REG = workflow.HERE / "arms.json"
        workflow.REG.write_text(json.dumps(registered))
        for arm in registered["arms"]:
            wifi = bool(arm.get("dataset"))
            stage = "source" if wifi else "explore"
            for fold in range(4 if wifi else 2):
                for seed in ([2023] if wifi else [101, 102]):
                    path = workflow.HERE / "runs" / stage / f"{arm['name']}__f{fold}__s{seed}"
                    path.mkdir(parents=True)
                    ce = .1 if arm.get("input_fmax") else .5
                    if not wifi:
                        # Deliberately make target/dev accuracy favor the worse source config.
                        selected = {"dev_val": {"summary": {"acc_mean": 0.99 if ce == .5 else .10}}}
                    else:
                        selected = {}
                    result = dict(arm=arm["name"], fold=fold, seed=seed, stage=stage,
                                  params=arm["expected_params"], source_val_ce=ce, selected=selected,
                                  source_rx_val={"summary": {"acc_mean": .9 if arm["recipe"] == "unified" else .8}})
                    config = dict(arm=arm, smoke=False, registration_sha256=workflow.sha(workflow.REG))
                    (path / "metrics.json").write_text(json.dumps(result))
                    (path / "config.json").write_text(json.dumps(config))
        workflow.freeze_wifi(registered["arms"])
        decision = json.loads((workflow.HERE / "decision_wifi.json").read_text())
        assert len(decision["allowed_final_jobs"]) == 240
        assert len({tuple(j) for j in decision["allowed_final_jobs"]}) == 240
        assert all(c["shared_rf_recipe"] == "unified" for c in decision["choices"].values())
        assert all(c["ib"].endswith("_unified") for c in decision["choices"].values())
        assert len(decision["source_result_hashes"]) == 176
        workflow.freeze_shen(registered["arms"])
        decision = json.loads((workflow.HERE / "decision_shen.json").read_text())
        assert len(decision["source_result_hashes"]) == 100
        assert all("_ib_" in name for name in decision["choices"].values())
        assert decision["confirm_role"] == "report only; repeatedly inspected"
    print("PASS common RF recipe, fixed 240 WiSig fits, independent source-only Shen selection")


if __name__ == "__main__":
    main()
