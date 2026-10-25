"""Checkpoint releases exclude private inputs and enforce immutable evidence."""
import hashlib
import importlib.util
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validation_release", ROOT / "artifacts/publish_validation.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rejected(call, text):
    try:
        call()
    except SystemExit as error:
        assert text in str(error), error
    else:
        raise AssertionError(f"missing release guard: {text}")


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        report_path = root / "06-paper-validation/summary/report.json"
        report_path.parent.mkdir(parents=True)
        (root / "artifacts").mkdir()
        report = {"checks": {"complete": False, "problems": [], "validated": 2,
                             "saved_weights_checked": 2}}
        report_path.write_text(json.dumps(report))
        out = root / "release"
        with patch.object(release, "ROOT", root):
            rejected(lambda: release.package(out), "incomplete")
            assert not out.exists()
            report["checks"]["complete"] = True
            report["checks"]["saved_weights_checked"] = 1
            report_path.write_text(json.dumps(report))
            rejected(lambda: release.package(out), "saved-weight audit")
            report["checks"]["saved_weights_checked"] = 2
            report_path.write_text(json.dumps(report))
            paths = []
            for stage in ("03-chirp-structure", "06-paper-validation"):
                path = root / stage / "runs/confirm/test/model.pt"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"synthetic-checkpoint")
                path.with_name("stdout.log").write_text("not part of the weight archive")
                paths.append(path)
            for name in ("data/raw/private", "external/upstream/code.py", "token"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("excluded")
            release.package(out)
            record = json.loads((out / "package.json").read_text())
            assert record["checkpoints"] == record["formal_fits"] == 2
            parts = sorted(out.glob("*.part-*"))
            assert len(parts) == 1
            assert sha(parts[0]) == record["assets"][0]["sha256"]
            subprocess.run(["zstd", "-q", "-t", str(parts[0])], check=True)
            decoded = subprocess.check_output(["zstd", "-q", "-dc", str(parts[0])])
            assert decoded == (out / f"crossatten-{release.TAG}.tar").read_bytes()
            with tarfile.open(out / f"crossatten-{release.TAG}.tar") as archive:
                names = set(archive.getnames())
                assert names == {str(p.relative_to(root)) for p in paths} | {
                    "artifacts/paper-validation-checkpoints.sha256"}
                for path in paths:
                    assert archive.extractfile(str(path.relative_to(root))).read() == path.read_bytes()
            token = root / "not-needed-token"
            original = parts[0].read_bytes()
            parts[0].write_bytes(b"tampered")
            with patch.object(release.requests, "Session") as network:
                rejected(lambda: release.upload(out, token), "asset changed")
                network.assert_not_called()
            parts[0].write_bytes(original)
            report_path.write_text("{}")
            with patch.object(release.requests, "Session") as network:
                rejected(lambda: release.upload(out, token), "report changed")
                network.assert_not_called()
            report_path.write_text(json.dumps(report))
            token.write_text("synthetic-not-a-real-credential")
            candidates = parts + [
                root / "artifacts/PAPER_VALIDATION_SHA256SUMS",
                root / "artifacts/paper-validation-checkpoints.sha256",
                root / "artifacts/PAPER_VALIDATION_PACKAGE.json",
            ]
            assets = [{"name": p.name, "size": p.stat().st_size, "digest": f"sha256:{sha(p)}"}
                      for p in candidates]
            published = {"id": 1, "assets": assets, "upload_url": "https://example.invalid/assets{?name}"}
            response = MagicMock(status_code=200)
            response.json.return_value = published
            tag = MagicMock(status_code=200)
            tag.json.return_value = {"object": {"sha": "synthetic-commit"}}
            session = MagicMock()
            session.get.side_effect = [response, tag, response]
            with patch.object(release.requests, "Session", return_value=session), \
                    patch.object(release.subprocess, "check_output", return_value="synthetic-commit\n"), \
                    patch.dict("os.environ", {"ENV_VENUS_PROXY": "http://example.invalid"}):
                release.upload(out, token)
            session.post.assert_not_called()
            assert json.loads((out / "published-release.json").read_text())["id"] == 1
            tag.json.return_value = {"object": {"sha": "other-commit"}}
            session.get.side_effect = [response, tag]
            with patch.object(release.requests, "Session", return_value=session), \
                    patch.object(release.subprocess, "check_output", return_value="synthetic-commit\n"), \
                    patch.dict("os.environ", {"ENV_VENUS_PROXY": "http://example.invalid"}):
                rejected(lambda: release.upload(out, token), "different commit")
            session.post.assert_not_called()
    print("PASS release completion/weight gates, archive scope, manifest hashes and pre-network tamper rejection")


if __name__ == "__main__":
    main()
