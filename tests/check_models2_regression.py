"""Regression check: the stage-03 extensions of lib/xa/models2.py (preln / oob_null, default off) must leave every
stage-02 arm bit-identical (same initial weights for the same seed, same outputs) to the pre-registered version
committed in 048bbb7.   python tests/check_models2_regression.py"""
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "tests"))
import xa.models  # noqa: E402,F401
import xa.models2 as NEW  # noqa: E402
from test_core import chirp_packet, to_t  # noqa: E402

ARMS = [dict(family="ib_variant"), dict(family="ib_variant", combfft=True), dict(family="ib_variant", meanchirp=True),
        dict(family="ib_variant", dechirp=True), dict(family="ib_variant", depth=4),
        dict(family="ib_rawoob", oob_band=[62500, 1e9]), dict(family="ib_rawoob", oob_band=[62500, 200000]),
        dict(family="ib_rawoob", oob_band=[62500, 100000], oob_side="L"), dict(family="ib_cisoob"), dict(family="ib_tfsplit")]


def main(ref="048bbb7"):
    src = subprocess.check_output(["git", "show", f"{ref}:lib/xa/models2.py"], cwd=ROOT)
    tmp = Path(tempfile.mkdtemp()) / "models2_old.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("xa.models2_old", tmp)
    OLD = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(OLD)
    x = to_t(chirp_packet(3, snr_db=20))
    bad = 0
    for a in ARMS:
        torch.manual_seed(11)
        mo = OLD.build_pipeline2(a).eval()
        torch.manual_seed(11)
        mn = NEW.build_pipeline2(a).eval()
        so, sn = mo.state_dict(), mn.state_dict()
        same_w = so.keys() == sn.keys() and all(torch.equal(so[k], sn[k]) for k in so)
        with torch.no_grad():
            same_y = torch.equal(mo(x), mn(x))
        bad += not (same_w and same_y)
        print(f"{str(a):75s} weights_equal={same_w} outputs_equal={same_y}")
    print("REGRESSION OK" if bad == 0 else f"REGRESSION FAILED ({bad})")
    return bad


if __name__ == "__main__":
    sys.exit(main())
