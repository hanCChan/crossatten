"""Aggregation and paired decisions use synthetic records, not target data."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_summary", ROOT / "06-paper-validation/06-4-summary.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def record(fold, seed, accuracy):
    cm = [[accuracy * 1000, (1 - accuracy) * 1000], [(1 - accuracy) * 1000, accuracy * 1000]]
    per = {"rx": dict(confusion=cm, acc=accuracy, macro_f1=accuracy)}
    scores = dict(acc_mean=accuracy, packet_acc=accuracy, f1_mean=accuracy)
    return dict(fold=fold, seed=seed, params=100, train_seconds=10,
                selected={"CRD": dict(summary=scores, per_rx=per)})


def main():
    base = [record(f, s, .80 + f * .01) for f in range(4) for s in (2023, 2024, 2025)]
    improved = [record(f, s, .82 + f * .01) for f in range(4) for s in (2023, 2024, 2025)]
    summary = report.aggregate(base, "CRD")
    assert abs(summary["metrics"]["acc_mean"]["mean"] - 81.5) < 1e-10
    assert summary["seed_sd"] == 0 and summary["fold_sd"] > 0
    assert report.paired(improved, base, "CRD", 12)["status"] == "improves"
    assert report.paired(base, improved, "CRD", 12)["status"] == "harms"
    assert report.paired(base, base, "CRD", 12)["status"] == "unclear"
    assert report.paired(improved[:-1], base, "CRD", 12)["status"] == "incomplete"
    assert abs(report.pooled_f1(base[0]["selected"]["CRD"]["per_rx"]) - .80) < 1e-10
    print("PASS paired completeness, decision thresholds, pooled F1, separated seed/fold variation")


if __name__ == "__main__":
    main()
