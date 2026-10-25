"""Synthetic checks of WiSig RX identity, normalization, and official count parsing."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from xa.wisig import RX, log_counts, power_normalize, receivers


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
