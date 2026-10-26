"""Conditional formal/ablation job preparation, never new candidate exploration."""
import argparse
import json
from stage import HERE, ROOT, s09, once, sha, require_committed, identity
from register import job
from xa.train import stable_seed


def reuse_manysig():
    decision = json.loads((HERE / "decision_final.json").read_text())
    require_committed(HERE / "decision_final.json")
    if not decision["admitted"]:
        raise ValueError("STOP candidate cannot prepare formal evaluations")
    evidence = {}
    for fold in range(4):
        for seed in (9101, 9102, 9103):
            j = dict(dataset="ManySig", fold=fold, seed=seed, feature="first",
                     phase="attribution", epochs=200, policy="packet")
            j["id"] = identity(j)
            folder = ROOT / "09-phase-invariance/runs/attribution" / j["id"]
            config = json.loads((folder / "config.json").read_text())
            old = config["job"]
            for key in ("dataset", "fold", "seed", "feature", "policy", "epochs"):
                if old.get(key) != j[key]:
                    raise ValueError("S09 ManySig formal control incompatible")
            expected_seed = stable_seed("S09-paired", "ManySig", fold, "all", seed)
            if config["data_seed"] != expected_seed or config["recipe"] != s09.RECIPES["official"]:
                raise ValueError("formal control recipe/RNG mismatch")
            for name in ("config.json", "metrics.json", "history.csv",
                         "preds_CR.npz", "preds_CRD.npz", "target_metrics.json"):
                path = folder / name
                require_committed(path)
                evidence[str(path.relative_to(ROOT))] = sha(path)
    once(HERE / "manysig_reuse.json", dict(compatible=True, evidence=evidence,
        caveat="historical checkpoint missing; config/splits/shared-init and stored predictions verified",
        subsequent_fit_check="compare each S10 soft config to S09 hard after final fitting"))


def ablations():
    decision = json.loads((HERE / "decision_final.json").read_text())
    require_committed(HERE / "decision_final.json")
    if not decision["admitted"]:
        raise ValueError("STOP candidate cannot start ablations")
    import pandas as pd
    scores = pd.read_csv(HERE / "final_comparisons.csv")
    effectiveness = {}
    for d in ("ManyRx", "ManySig"):
        local = scores[(scores.dataset == d) & (scores.task == "CRD")].set_index("model")
        ref = "S10_hard" if d == "ManyRx" else "S09_hard_packet"
        effectiveness[d] = float(local.loc["S10_soft", "accuracy_pct"] - local.loc[ref, "accuracy_pct"])
    admitted = all(v > 0 for v in effectiveness.values())
    jobs = [job("ablation", d, f, seed, feature) for d in ("ManyRx", "ManySig")
            for f in range(4) for seed in (9101, 9102, 9103)
            for feature in ("no_hstu", "unweighted")] if admitted else []
    # The attachment's24-fit cap is absolute: choose one fixed seed per fold per arm
    # only if the resulting preregistered job space was already specified.
    if jobs:
        raise ValueError("24-fit cap cannot cover both arms x2datasets x4folds x3seeds; "
                         "preregistration did not resolve this inconsistency. Do not silently expand.")
    once(HERE / "ablation_decision.json", dict(admitted=admitted, efficacy=effectiveness,
        jobs=jobs, cap=24, no_new_training=True))
    once(HERE / "jobs_ablation.json", jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()
    (ablations if args.ablations else reuse_manysig)()
