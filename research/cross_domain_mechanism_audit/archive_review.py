"""Build and verify the review archive; reads text/code only, never experiment IQ."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = "9815503880291f4b66c3566b07769f6a5fc8c9a6"
EVIDENCE = [
    "11-spectral-generalization/RESULTS.md",
    "12-spectral-validation/RESULTS.md",
    "17-rx-distortion/D0_INTERPRETATION.zh-CN.md",
    "17b-phase-intervention/INTERPRETATION.zh-CN.md",
    "18-selective-rx/RESULTS.zh-CN.md",
    "19-tx-spectral-evidence/RESULTS.zh-CN.md",
    "19-tx-spectral-evidence/historical_matrix.md",
    "20-asknet-source-comparison/RESULTS.zh-CN.md",
    "20-asknet-source-comparison/UPSTREAM.json",
    "lib/xa/explore_models.py",
    "lib/legacy_rfhstu/mixstyle.py",
    "lib/legacy_rfhstu/models.py",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_bytes(path):
    payload = path.read_bytes()
    return payload.replace(b"\r\n", b"\n") if path.suffix in {".md", ".json", ".py"} else payload


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def verify_links():
    checked = 0
    for doc in HERE.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"):
                continue
            path = (doc.parent / target.split("#")[0]).resolve()
            assert path.is_relative_to(ROOT) and path.is_file(), (doc.name, target)
            checked += 1
    return checked


def write():
    assert git("rev-parse", "HEAD").decode().strip() == BASE, "Build on recorded S20 base"
    for rel in EVIDENCE:
        # Respect the repository's checkout line-ending filters on Windows.
        git("diff", "--exit-code", BASE, "--", rel)
    checks = json.loads((HERE / "SYNTHETIC_CHECKS.json").read_text())
    assert checks["status"] == "PASS" and checks["checks_passed"] == 6
    scope = dict(
        status="REVIEW_COMPLETE_NOT_EMPIRICAL_GO", review_date="2026-11-10", base_commit=BASE,
        formal_preregistration=False, hypotheses_empirically_passed=False,
        new_model_fits=0, new_experimental_iq_reads=0, new_checkpoint_loads=0,
        new_prediction_array_reads=0, target_reads=0, dates_3_4_reads=0,
        manysig_blind_data_reads=0, shen_blind_data_reads=0,
        historical_decisions_modified=False, paper_novelty_established=False,
        literature_scope="Targeted primary-source screening; not exhaustive novelty clearance",
        access_accounting="Declared actions in this review and its scripts; not OS-wide instrumentation",
        synthetic_only_checks=6, python=platform.python_version(),
        author_mixstyle_blob="7eb151bf06a181ed4bc6f54f562bffc49f35deaf",
        cited_local_evidence=[dict(path=rel, git_blob_sha256=hashlib.sha256(git("show", BASE+":"+rel)).hexdigest(),
                                   worktree_sha256_at_review=sha(ROOT/rel)) for rel in EVIDENCE],
        historical_comparison="Git canonical blobs and clean diff; Windows checkout CRLF may differ",
        reproduction=["python synthetic_checks.py", "python archive_review.py --verify"],
    )
    (HERE / "REVIEW_SCOPE.json").write_text(json.dumps(scope, indent=2)+"\n", encoding="utf-8", newline="\n")
    # MANIFEST intentionally excludes itself; all other archive members are covered.
    files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != "MANIFEST.json")
    manifest = dict(base_commit=BASE, scope="review archive; excludes MANIFEST.json itself",
                    text_encoding="UTF-8 with LF canonical newlines; PNG bytes unchanged",
                    files=[dict(path=p.name, bytes=len(archive_bytes(p)),
                                sha256=hashlib.sha256(archive_bytes(p)).hexdigest()) for p in files])
    (HERE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8", newline="\n")


def verify(index=False):
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    members = {p.name for p in HERE.iterdir() if p.is_file() and p.name != "MANIFEST.json"}
    assert members == {item["path"] for item in manifest["files"]}
    for item in manifest["files"]:
        path = HERE / item["path"]
        payload = archive_bytes(path)
        assert len(payload) == item["bytes"] and hashlib.sha256(payload).hexdigest() == item["sha256"], item["path"]
    scope = json.loads((HERE / "REVIEW_SCOPE.json").read_text())
    for item in scope["cited_local_evidence"]:
        git("diff", "--exit-code", BASE, "--", item["path"])
        assert hashlib.sha256(git("show", BASE+":"+item["path"])).hexdigest() == item["git_blob_sha256"]
    prefix = HERE.relative_to(ROOT).as_posix()+"/"
    changed = git("diff", "--name-only", BASE).decode().splitlines()
    assert all(p.startswith(prefix) for p in changed), "Changes outside review archive"
    if index:
        tracked = git("ls-files", "--", prefix).decode().splitlines()
        expected = {prefix+p.name for p in HERE.iterdir() if p.is_file()}
        assert set(tracked) == expected, "Missing/unexpected staged archive files"
        for rel in tracked:
            assert git("show", ":"+rel) == archive_bytes(ROOT/rel), rel
    print(json.dumps(dict(status="PASS", files_hashed=len(manifest["files"]),
                          local_links_checked=verify_links(), historical_evidence_unchanged=len(EVIDENCE),
                          git_index_checked=index)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--index", action="store_true")
    args = parser.parse_args()
    if args.write:
        write()
    verify(index=args.index)
