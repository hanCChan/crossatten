"""Synthetic tests of pre-training gates; never load data or initialize CUDA."""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))
from xa.paper_train import read_wifi, sha, validate_fit
from xa.wisig import receivers


def denied(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("invalid access was accepted")


def main():
    registered = json.loads((ROOT / "06-paper-validation/arms.json").read_text())
    arm = registered["arms"][0]
    source, target = receivers("ManyRx", 0)
    denied(lambda: read_wifi("ManyRx", [target[0]], (1, 2), "source", "synthetic", 0))
    denied(lambda: read_wifi("ManyRx", [source[0]], (3, 4), "source", "synthetic", 0))
    denied(lambda: read_wifi("ManyRx", [source[0]], (1, 2), "unknown", "synthetic", 0))
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        reg = root / "arms.json"
        reg.write_text(json.dumps(registered))
        out = root / "runs/source" / f"{arm['name']}__f0__s2023"
        validate_fit(arm, 0, 2023, "source", out)
        denied(lambda: validate_fit(arm, 0, 2024, "source", out))
        denied(lambda: validate_fit(arm, 0, 2023, "unknown", out))
        denied(lambda: validate_fit(dict(arm, width=33), 0, 2023, "source", out))
        denied(lambda: validate_fit(arm, 0, 2023, "explore", out))
        final = root / "runs/final" / f"{arm['name']}__f0__s2023"
        denied(lambda: validate_fit(arm, 0, 2023, "final", final))
        evidence = root / "source.json"
        evidence.write_text("{}")
        decision_path = root / "decision_wifi.json"
        decision = dict(registration_sha256=sha(reg), allowed_final_jobs=[[arm["name"], 0, 2023]],
                        source_result_hashes={"source.json": sha(evidence)})
        decision_path.write_text(json.dumps(decision))
        with patch("xa.paper_train.require_committed") as committed:
            validate_fit(arm, 0, 2023, "final", final, decision_path)
            assert committed.call_count == 3
            evidence.write_text('{"changed":true}')
            denied(lambda: validate_fit(arm, 0, 2023, "final", final, decision_path))
            evidence.write_text("{}")
            decision["allowed_final_jobs"] = []
            decision_path.write_text(json.dumps(decision))
            denied(lambda: validate_fit(arm, 0, 2023, "final", final, decision_path))
    print("PASS source RX/day access, registered identities, final whitelist, frozen hashes, commit gates")


if __name__ == "__main__":
    main()
