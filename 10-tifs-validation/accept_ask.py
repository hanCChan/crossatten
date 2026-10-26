"""Accept only inspected official model/loader/training path, not target performance."""
import json
import re
from stage import HERE, ROOT, ASK_DIR, ASK_SHA, once, sha, ask_modules, RX


def main():
    official = ask_modules()
    evidence = {}
    for dataset in ("ManyRx", "ManySig"):
        path = HERE / "runs/smoke" / f"{dataset}__f0__first__s0__ask__packet/metrics.json"
        result = json.loads(path.read_text())
        if result["stop_epoch"] != 2 or not result["selected"]["source_packet"]["ce"] > 0:
            raise ValueError("smoke fit not finite/complete")
        evidence[str(path.relative_to(ROOT))] = sha(path)
    prep = json.loads((HERE / "preparation.json").read_text())
    parity = all(v["max_abs_error"] <= 1e-6 for v in prep["official_parity"].values())
    folds_match = all(official.split_receivers(len(RX[d]), 4, f)[1] ==
                      list(range(f * len(RX[d]) // 4, (f + 1) * len(RX[d]) // 4))
                      for d in RX for f in range(4))
    logs = {}
    for path in (ASK_DIR / "WiSig/logs").rglob("*.txt"):
        matches = re.findall(r"test_acc\s*:\s*([0-9.]+)", path.read_text())
        if len(matches) != 1:
            raise ValueError("ambiguous official log")
        dataset = "ManySig" if "ManySig" in path.name else "ManyRx"
        fold = int(re.search(r"round(\d)", path.name)[1])
        logs[f"{dataset}:f{fold}"] = dict(packet_acc_pct=100 * float(matches[0]),
                                        file=str(path.relative_to(ASK_DIR)), sha256=sha(path))
    accepted = parity and folds_match and len(logs) == 8
    once(HERE / "ask_acceptance.json", dict(accepted=accepted, upstream_commit=ASK_SHA,
        loader_parity=parity, folds_match=folds_match, smoke_evidence=evidence,
        upstream_logs=logs, no_target_threshold_for_acceptance=True,
        pretrained_weights="not supplied locally; Baidu share requires manual external access",
        acceptance_scope="official code path, data parity and finite source-only smoke; "
                         "not verification of published checkpoints or paper numerical reproduction",
        deviations=["PyTorch2.5/CUDA12.4/A100 vs published1.11/CUDA11.3/3080Ti",
                    "cached float32 input matches official float64 normalize then cast exactly"]))
    print("ASKNet official-path accepted:", accepted)


if __name__ == "__main__":
    main()
