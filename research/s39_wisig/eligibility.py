"""Metadata-only Raw/EQ eligibility and frozen auxiliary-supervision audit."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "s29_wisig"))
sys.path.insert(0, str(HERE.parent / "wisig_w3"))
from selective_pickle import ArrayRef, index
from index_manytx import AuditReader
from source import C, read, sha, write
from summarize import pair_summary

VIEWS = {"V0":"wisig_w4", "V1":"wisig_w5"}


def raw_schema(path):
    with path.open("rb") as stream:
        reader = AuditReader(stream)
        obj, raw_spans = index(reader)
    meta = read(C / "PRE_REGISTRATION.json")
    roles = set(meta["fit_rx"])
    assert obj["equalized_list"] == [0,1]
    records, spans = [], []
    for t, tx in enumerate(obj["tx_list"]):
        for r, rx in enumerate(obj["rx_list"]):
            for d, day in enumerate(obj["capture_date_list"]):
                pair = obj["data"][t][r][d]
                assert len(pair) == 2 and all(isinstance(v, ArrayRef) for v in pair)
                assert all(v.shape[1:] == (256,2) for v in pair)
                spans.extend(v.payload for v in pair)
                if rx in roles and d < 2:
                    records.append(dict(tx=tx, rx=rx, day=day, raw=pair[0].shape[0], eq=pair[1].shape[0]))
    spans = sorted((v.offset,v.offset+v.length) for v in spans if v.length)
    k = 0
    for start,end in reader.reads:
        while k < len(spans) and spans[k][1] <= start:
            k += 1
        assert k == len(spans) or end <= spans[k][0]
    assert set(obj) == {"tx_list","rx_list","capture_date_list","equalized_list","max_sig","data"}
    return dict(top_level_keys=sorted(obj), indexed_array_count=len(raw_spans),
                metadata_bytes=sum(end-start for start,end in reader.reads),
                waveform_bytes=0, source_cells=len(records),
                unequal_support_cells=sum(row["raw"] != row["eq"] for row in records),
                source_counts=records, raw_offsets_present=False, packet_id_present=False,
                decision="STOP_COMPACT_IDENTITY_UNAVAILABLE",
                reason="No identity fields in the actual local pickle; source and EQ arrays are independently indexed.",
                source_manifest_sha256=sha(path))


def assets(view, seed):
    home = HERE.parent / VIEWS[view]
    manifest = {r["relative"]:r["sha256"] for r in read(home/"REPLAY_FILES.json")["files"]}
    base = {}
    for method in ("A0","A1"):
        p = home/"artifacts"/f"P_{method}_S{seed}_selection.npz"
        assert sha(p) == manifest[p.relative_to(ROOT).as_posix()]
        with np.load(p,allow_pickle=False) as z:
            base[method] = {k:z[k] for k in z.files}
        weights = home/"artifacts"/"review_checkpoints"/(
            f"A0_S{seed}.pt" if method=="A0" else f"A1_S{seed}_downstream.pt")
        assert sha(weights) == manifest[weights.relative_to(ROOT).as_posix()]
    for field in ("pool_ids","keys","y","day","rx"):
        assert np.array_equal(base["A0"][field],base["A1"][field])
    return base


def a1_audit():
    pairs, ids = [], []
    for view, name in VIEWS.items():
        home = HERE.parent/name
        with (home/"P_TRAINING_COST.csv").open(encoding="utf-8") as stream:
            cost = list(csv.DictReader(stream))
        by_view = []
        for s in (0,1):
            files = assets(view,s)
            a0,a1 = (files[k] for k in ("A0","A1"))
            align = a0["rx"]
            x = (a1["logits"].argmax(1)==a0["y"]).astype(float)
            y = (a0["logits"].argmax(1)==a0["y"]).astype(float)
            by_view.append([100*(x-y)[align==r].mean() for r in sorted(set(align))])
            rows = [r for r in cost if r["job"] in (f"A1_S{s}_aux",f"A1_S{s}_downstream")]
            assert sorted(int(r["updates"]) for r in rows) == [4320,5080]
            ids.append(dict(view=view,seed=s,rows=len(align),receivers=sorted(set(align))))
        pairs.append(by_view)
    effect = pair_summary(np.asarray(pairs))
    qualified = min(effect["view_pp"]) >= 1 and effect["positive_view_seed"] >= 3
    return dict(effect=effect, matched=ids, pure_extra_compute_control_already_available=False,
                auxiliary_and_compute_jointly_changed=True, qualified=qualified)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("payload",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exists_ok=True)
    write(args.output/"A.json",raw_schema(args.payload))
    write(args.output/"C.json",a1_audit())
