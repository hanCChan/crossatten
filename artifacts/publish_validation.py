#!/usr/bin/env python3
"""Package local S03/S06 checkpoints and publish additive GitHub release assets."""
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
TAG = "paper-validation-2026-10-25"
API = "https://api.github.com/repos/hanCChan/crossatten"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def package(out):
    report = json.loads((ROOT / "06-paper-validation/summary/report.json").read_text())
    checks = report["checks"]
    if not checks["complete"] or checks["problems"]:
        raise SystemExit("refusing to package an incomplete or invalid validation matrix")
    if checks.get("saved_weights_checked") != checks["validated"]:
        raise SystemExit("run the complete saved-weight audit before packaging")
    checkpoints = sorted(
        p for stage in ("03-chirp-structure", "06-paper-validation")
        for p in (ROOT / stage / "runs").glob("*/*/model.pt"))
    if not checkpoints:
        raise SystemExit("no checkpoints found")
    out.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "artifacts/paper-validation-checkpoints.sha256"
    sums = "".join(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in checkpoints)
    manifest.write_text(sums)
    archive = out / f"crossatten-{TAG}.tar"
    with tarfile.open(archive, "w") as stream:
        for path in checkpoints:
            stream.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
        stream.add(manifest, arcname=manifest.relative_to(ROOT).as_posix(), recursive=False)
    subprocess.run(["zstd", "-q", "-T2", "-3", "-f", str(archive)], check=True)
    compressed = archive.with_suffix(".tar.zst")
    parts = []
    with compressed.open("rb") as stream:
        index = 0
        while block := stream.read(128 * 1024 * 1024):
            path = out / f"{compressed.name}.part-{index:03d}"
            path.write_bytes(block)
            parts.append(path)
            index += 1
    for stale in out.glob(f"{compressed.name}.part-*"):
        if stale not in parts:
            raise SystemExit(f"unexpected existing asset: {stale.name}")
    parts_manifest = ROOT / "artifacts/PAPER_VALIDATION_SHA256SUMS"
    parts_manifest.write_text("".join(f"{digest(path)}  {path.name}\n" for path in parts))
    record = dict(tag=TAG, checkpoints=len(checkpoints),
                  formal_fits=checks["validated"],
                  bytes=sum(path.stat().st_size for path in checkpoints),
                  compressed_bytes=compressed.stat().st_size,
                  assets=[dict(name=p.name, bytes=p.stat().st_size, sha256=digest(p)) for p in parts],
                  checkpoint_manifest_sha256=digest(manifest),
                  parts_manifest_sha256=digest(parts_manifest),
                  report_sha256=digest(ROOT / "06-paper-validation/summary/report.json"),
                  scope="Local trained S03 and S06 weights, including smoke. No datasets or third-party source.")
    serialized = json.dumps(record, indent=1) + "\n"
    (out / "package.json").write_text(serialized)
    (ROOT / "artifacts/PAPER_VALIDATION_PACKAGE.json").write_text(serialized)
    print(json.dumps(record, indent=1))


def upload(out, token_path):
    record = json.loads((out / "package.json").read_text())
    if record["tag"] != TAG:
        raise SystemExit("package tag differs from the release tag")
    if json.loads((ROOT / "artifacts/PAPER_VALIDATION_PACKAGE.json").read_text()) != record:
        raise SystemExit("package record changed after packaging")
    report_path = ROOT / "06-paper-validation/summary/report.json"
    if record["report_sha256"] != digest(report_path):
        raise SystemExit("report changed after packaging")
    manifests = [
        (ROOT / "artifacts/paper-validation-checkpoints.sha256", "checkpoint_manifest_sha256"),
        (ROOT / "artifacts/PAPER_VALIDATION_SHA256SUMS", "parts_manifest_sha256"),
    ]
    for path, key in manifests:
        if digest(path) != record[key]:
            raise SystemExit(f"manifest changed after packaging: {path.name}")
    for asset in record["assets"]:
        path = out / asset["name"]
        if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
            raise SystemExit(f"asset changed after packaging: {path.name}")
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token_path.read_text().strip()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    proxy = os.environ.get("ENV_VENUS_PROXY")
    if not proxy:
        raise SystemExit("ENV_VENUS_PROXY is not configured")
    session.proxies.update(http=proxy, https=proxy)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    lookup = session.get(f"{API}/releases/tags/{TAG}", timeout=60)
    if lookup.status_code == 404:
        response = session.post(f"{API}/releases", timeout=60, json=dict(
            tag_name=TAG, target_commitish=commit, name="S03 and paper-validation checkpoint archive (2026-10-25)",
            body=(
                "Additive checkpoint archive for the frozen S03 and completed S06 matrix. "
                f"S06 has {record['formal_fits']} audited formal fits; "
                f"this archive has {record['checkpoints']} weights including smoke.\n\n"
                "Results, configurations, curves, predictions and failure records are in Git. "
                "Datasets, credentials and third-party source are excluded. "
                "Shen confirm RX were repeatedly inspected; WiSig did not support the general-method claim.\n\n"
                "Verify PAPER_VALIDATION_SHA256SUMS, concatenate ordered .part-* assets, "
                "extract using tar --zstd, then verify paper-validation-checkpoints.sha256 from the repository root."
            ), draft=False, prerelease=False))
    else:
        response = lookup
    response.raise_for_status()
    release = response.json()
    remote_tag = session.get(f"{API}/git/ref/tags/{TAG}", timeout=60)
    remote_tag.raise_for_status()
    if remote_tag.json()["object"]["sha"] != commit:
        raise SystemExit("release tag targets a different commit; refusing to overwrite")
    assets = {asset["name"]: asset for asset in release.get("assets", [])}
    candidates = [out / asset["name"] for asset in record["assets"]]
    candidates += [ROOT / "artifacts/PAPER_VALIDATION_SHA256SUMS",
                   ROOT / "artifacts/paper-validation-checkpoints.sha256",
                   ROOT / "artifacts/PAPER_VALIDATION_PACKAGE.json"]
    for path in candidates:
        existing = assets.get(path.name)
        expected = digest(path)
        if existing:
            if existing["size"] != path.stat().st_size or existing.get("digest") != f"sha256:{expected}":
                raise SystemExit(f"existing release asset differs or lacks verified digest: {path.name}")
            print(f"verified existing asset: {path.name}", flush=True)
            continue
        with path.open("rb") as stream:
            uploaded = session.post(
                release["upload_url"].split("{", 1)[0], params={"name": path.name},
                data=stream, headers={"Content-Type": "application/octet-stream",
                                      "Content-Length": str(path.stat().st_size)}, timeout=900)
        uploaded.raise_for_status()
        asset = uploaded.json()
        if asset["size"] != path.stat().st_size or asset.get("digest") != f"sha256:{expected}":
            raise SystemExit(f"uploaded asset checksum mismatch: {path.name}")
        print(f"verified uploaded asset: {path.name}", flush=True)
    final = session.get(f"{API}/releases/tags/{TAG}", timeout=60)
    final.raise_for_status()
    (out / "published-release.json").write_text(json.dumps(final.json(), indent=1) + "\n")
    print(json.dumps(dict(release_id=release["id"], tag=TAG, commit=commit, verified_assets=len(candidates))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("package", "upload"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/crossatten-validation-20261025"))
    parser.add_argument("--token", type=Path, default=Path("/workspace/user_code/kuozhan/token"))
    args = parser.parse_args()
    if args.action == "package":
        package(args.out)
    else:
        upload(args.out, args.token)


if __name__ == "__main__":
    main()
