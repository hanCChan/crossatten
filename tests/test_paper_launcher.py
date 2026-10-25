"""IB dependency readiness is metadata-only and requires completed matching fits."""
import importlib.util
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_launcher", ROOT / "06-paper-validation/06-6-launch.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def main():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        job = dict(out=str(root / "IB__f1__s2024"), fold=1, seed=2024)
        arm = dict(init_arm="FAN_unified")
        assert launcher.initialization_ready(job, {})
        assert not launcher.initialization_ready(job, arm)
        other = root / "FAN_unified__f0__s2024"
        other.mkdir()
        (other / "metrics.json").touch()
        (other / "model.pt").touch()
        assert not launcher.initialization_ready(job, arm)
        correct = root / "FAN_unified__f1__s2024"
        correct.mkdir()
        (correct / "model.pt").touch()
        assert not launcher.initialization_ready(job, arm)
        (correct / "metrics.json").touch()
        assert launcher.initialization_ready(job, arm)
        (correct / "model.pt").unlink()
        assert not launcher.initialization_ready(job, arm)
        jobs = root / "jobs.jsonl"
        jobs.write_text("")
        status = root / "assist.status.json"
        args = ["launch", "--jobs", str(jobs), "--gpus", "0", "--per-gpu", "1",
                "--status-file", str(status)]
        with patch("sys.argv", args), patch.object(launcher.time, "sleep"):
            launcher.main()
        assert json.loads(status.read_text())["complete"] == 0
        assert not Path(str(jobs) + ".status.json").exists()
    print("PASS matching IB fold/seed/recipe needs both completed metrics and weights")


if __name__ == "__main__":
    main()
