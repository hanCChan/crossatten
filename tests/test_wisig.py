"""Synthetic checks of WiSig RX identity, normalization, and official count parsing."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from xa.wisig import RX, log_counts, power_normalize, receivers, released_log_result


def main():
    for dataset in RX:
        targets = []
        for fold in range(4):
            tr, te = receivers(dataset, fold)
            assert not set(tr) & set(te)
            assert set(tr) | set(te) == set(RX[dataset])
            targets.extend(te)
            assert min(log_counts(dataset, fold)) > 0
        assert targets == RX[dataset]
    assert log_counts("ManyRx", 0) == (32527, 13941)
    log_root = ROOT / "external/IB-RFF/logs"
    assert released_log_result(log_root / "stage1/ManyRx_pre0_a0.0_round0.log") == ("CR", 90.66)
    exp, acc = released_log_result(log_root / "stage2/ManyRx_CRD_pre1_a0.01_round0.log")
    assert exp == "CRD" and abs(acc - 91.15) < 1e-6
    rng = np.random.default_rng(4)
    x = power_normalize(rng.normal(size=(4, 2, 256)))
    assert np.allclose((x ** 2).sum(1).mean(-1), 1)
    assert power_normalize(np.empty((0, 2, 256))).shape == (0, 2, 256)
    try:
        power_normalize(np.zeros((1, 2, 256)))
    except ValueError:
        pass
    else:
        raise AssertionError("zero-power packet accepted")
    print("PASS WiSig synthetic invariants")


if __name__ == "__main__":
    main()
