"""Replay exactly the admitted capture and export its packet identity ledger."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import numpy as np
from scipy.io import loadmat

from fetch_pilot import CAPTURE, SIZE

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "wisig_w3"))
from selective_pickle import index, read_rows

TREE = "6530c6ed6be1d08544ac295040ed50f5babde86a"
TX = "20-15"
MEMBER_SIZE = 102400000
ARCHIVE_SHA = "70e7ab66b63f65064bcf13c1d9e73ec86ffa7e973e7d5a734325423da71b7109"
MEMBER_NAME = ("node23-5/capture/tx{node:node20-15}_rx{node:node23-5-rxFreq:2462e6-"
               "rxGain:0.5-capLen:0.512-rxSampRate:25e6}.dat")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root, tree_path):
    tree = json.loads(tree_path.read_text(encoding="utf-8"))
    assert tree["sha"] == TREE
    pinned = {node["path"]:node["sha"] for node in tree["tree"] if node["type"] == "blob"}
    used = {}
    for path in (root / "upstream").rglob("*.m"):
        relative = path.relative_to(root / "upstream").as_posix()
        raw = path.read_bytes()
        git_sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        assert git_sha == pinned[relative]
        used[relative] = git_sha

    adapted = root / "adapted"
    adapted.mkdir(exist_ok=True)
    detect = root / "upstream/matlab_detection_screening/split_signal_fun.m"
    detection = detect.read_text(encoding="utf-8").replace("function split_signal_fun(",
                                                            "function split_signal_fixed(", 1)
    detection, changes = re.subn(r"transmitter=regexp[\s\S]*?transmitter=transmitter\{1\};",
                                 f"transmitter='{TX}';", detection)
    assert changes == 1
    (adapted / "split_signal_fixed.m").write_text(detection, encoding="utf-8")
    eq = (root / "upstream/matlab_equalization/equalize_channel.m").read_text(encoding="utf-8")
    original = "helperNoiseEstimate(demodSig,nht.ChannelBandwidth,1)"
    assert eq.count(original) == 1
    (adapted / "equalize_channel.m").write_text(
        eq.replace(original, "wlanLLTFNoiseEstimate(demodSig)"), encoding="utf-8")

    meta = json.loads((root / "ARCHIVE.json").read_text(encoding="utf-8"))
    assert (meta["date"], meta["rx"], meta["group"]) == CAPTURE
    archive = root / meta["archive"]
    assert archive.stat().st_size == SIZE and digest(archive) == ARCHIVE_SHA == meta["sha256"]
    capture = root / "detect/data/node23-5/capture/capture.dat"
    capture.parent.mkdir(parents=True, exist_ok=True)
    assert all(other == capture for other in capture.parent.glob("*.dat"))
    member_sha = hashlib.sha256()
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(MEMBER_NAME)
        assert member.isfile() and member.size == MEMBER_SIZE
        temp = capture.with_suffix(".part")
        with tar.extractfile(member) as source, temp.open("wb") as dest:
            while block := source.read(1024 * 1024):
                member_sha.update(block)
                dest.write(block)
        assert temp.stat().st_size == MEMBER_SIZE
        temp.replace(capture)
    return used, member_sha.hexdigest()


def compact_raw(path, source_sha):
    assert digest(path) == source_sha
    day, rx, _ = CAPTURE
    with path.open("rb") as stream:
        obj, _ = index(stream)
    t, r, d = [obj[label].index(value) for label, value in
               (("tx_list", TX), ("rx_list", rx), ("capture_date_list", day))]
    assert d == 0 and obj["equalized_list"] == [0, 1]
    ref = obj["data"][t][r][d][0]
    cell = f"{TX}/{rx}/{day}/Raw"
    bounds = dict(key=cell, shape=list(ref.shape), dtype=ref.dtype.str, order=ref.order,
                  rows=(0,ref.shape[0]), offset=ref.payload.offset)
    return read_rows(path, bounds, {cell}), ref.payload.length


def fingerprint(row):
    return hashlib.sha256(np.ascontiguousarray(row, dtype="<f4").tobytes()).hexdigest()


def audit(root, archive_member_sha, used, compact, source_sha):
    with (root / "screened.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mat = loadmat(root / "VIEWS.mat", squeeze_me=True)
    original = compact_raw(compact, source_sha)
    compact_rows, bytes_read = original
    raw = np.stack((mat["raw"].real, mat["raw"].imag), axis=-1)
    eq = mat["eq"]
    assert mat["raw"].dtype == eq.dtype == np.complex64
    assert raw.shape == (len(rows), 256, 2) and eq.shape == (len(rows), 256)
    packet_ids = set()
    ledger = []
    compact_hits = defaultdict(list)
    raw_hits = defaultdict(list)
    for i, row in enumerate(compact_rows):
        compact_hits[fingerprint(row)].append(i)
    generated_pairs = 0
    for i, row in enumerate(rows):
        base = dict(**row, packet_id="", compact_raw_row_0based="", raw_sha256="", eq_sha256="")
        if int(row["matches"]) == 1:
            start, stop = int(row["start"]), int(row["stop"])
            assert 1 <= start <= stop - 255 < 12800000
            packet_id = hashlib.sha256(f"{archive_member_sha}:{start}:{stop}".encode()).hexdigest()
            assert packet_id not in packet_ids
            packet_ids.add(packet_id)
            raw_sha = fingerprint(raw[i])
            raw_hits[raw_sha].append(i)
            base.update(packet_id=packet_id, raw_sha256=raw_sha)
            if row["eq_status"] == "ok":
                assert np.isfinite(eq[i]).all() and int(row["eq_length"]) >= 256
                base["eq_sha256"] = fingerprint(np.stack((eq[i].real,eq[i].imag),axis=-1))
                generated_pairs += 1
            else:
                assert not np.isfinite(eq[i]).any()
        ledger.append(base)

    exact = 0
    for i, row in enumerate(ledger):
        key = row["raw_sha256"]
        matches = compact_hits.get(key, [])
        if len(matches) == 1 and len(raw_hits[key]) == 1:
            assert np.array_equal(raw[i], compact_rows[matches[0]])
            row["compact_raw_row_0based"] = matches[0]
            exact += 1
    with (root / "PACKET_LEDGER.csv").open("w", newline="", encoding="utf-8") as stream:
        output = csv.DictWriter(stream, fieldnames=list(ledger[0]))
        output.writeheader()
        output.writerows(ledger)
    result = dict(day=CAPTURE[0], rx=CAPTURE[1], tx=TX, archive_sha256=ARCHIVE_SHA,
                  member=MEMBER_NAME, member_bytes=MEMBER_SIZE, member_sha256=archive_member_sha,
                  detector_tree=TREE, upstream_matlab_git_blobs=used, release=str(mat["release"]),
                  adapter="Windows filename supplies fixed TX; EQ uses wlanLLTFNoiseEstimate",
                  detected_segments=int(mat["detected"]), screened=len(rows), unique_ids=len(packet_ids),
                  identity_unresolved=len(rows)-len(packet_ids), duplicate_ids=0,
                  eq_failed=len(packet_ids)-generated_pairs, raw_eq_pairs=generated_pairs,
                  compact_raw_rows=len(compact_rows), compact_raw_bytes=bytes_read,
                  compact_raw_unique_exact_matches=exact,
                  those_with_new_eq=sum(row["compact_raw_row_0based"] != "" and bool(row["eq_sha256"])
                                        for row in ledger), compact_eq_rows_read=0,
                  ledger_sha256=digest(root / "PACKET_LEDGER.csv"), training_admitted=False)
    (root / "PILOT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def matlab_literal(path):
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("tree", type=Path)
    parser.add_argument("matlab", type=Path)
    parser.add_argument("compact", type=Path)
    parser.add_argument("metadata_audit", type=Path)
    args = parser.parse_args()
    source_sha = json.loads(args.metadata_audit.read_text(encoding="utf-8"))["compact_file_sha256"]
    used, capture_sha = prepare(args.root, args.tree)
    print("Verified one archive member", capture_sha, flush=True)
    invocation = f"addpath({matlab_literal(HERE)}); replay_capture({matlab_literal(args.root)});"
    subprocess.run([str(args.matlab), "-batch", invocation], check=True)
    audit(args.root, capture_sha, used, args.compact, source_sha)
