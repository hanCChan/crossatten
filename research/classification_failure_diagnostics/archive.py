"""Archive checks for isolated diagnostics; no scientific arrays are opened."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PREFIX=HERE.relative_to(ROOT).as_posix()+"/"
BASE="7ae38a7b27521c9dc45d2ed1a332e5e584c3e8b8"
REGISTRATION="0a968be00cfe8a144effe0e17cb992c23d2c498e"


def git(*args): return subprocess.check_output(["git","-C",str(ROOT),*args])
def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(p):
    payload=p.read_bytes()
    return payload if p.suffix==".png" else payload.replace(b"\r\n",b"\n")
def members():
    return sorted(p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc" and p.name!="MANIFEST.json")


def check(manifest_pending=False):
    protocol=read(HERE/"G0_PROTOCOL.json"); registry=read(HERE/"INPUT_REGISTRY.json")
    assert sha(HERE/"INPUT_REGISTRY.json")==protocol["registry_sha256"]
    for rel in ["G0_PROTOCOL.json","INPUT_REGISTRY.json",*protocol["code_sha256"]]:
        blob=git("show",REGISTRATION+":"+PREFIX+rel)
        assert canonical(HERE/rel)==blob, "Registered input/code changed: "+rel
    for name,h in protocol["code_sha256"].items(): assert sha(HERE/name)==h
    for path,h in registry["metadata_sha256"].items(): assert sha(Path(path))==h
    changed=git("diff","--name-only",BASE).decode().splitlines()
    untracked=git("ls-files","--others","--exclude-standard").decode().splitlines()
    assert all(p.startswith(PREFIX) for p in changed+untracked), "Change outside isolated review"
    links=0
    for doc in HERE.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)",doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"): continue
            path=(doc.parent/target.split("#")[0]).resolve()
            exists=path.exists() or (manifest_pending and path==HERE/"MANIFEST.json")
            assert path.is_relative_to(ROOT) and exists, (doc.name,target)
            links+=1
    qa=read(HERE/"QA_REPORT.json"); complete=read(HERE/"ANALYSIS_COMPLETE.json")
    assert qa["status"]=="PASS" and qa["tx_rx_day_groups_checked"]==3840
    assert qa["paired_groups_checked"]==1920 and qa["archived_rx_confusions_exactly_reproduced"]==192
    assert complete["registry_sha256"]==sha(HERE/"INPUT_REGISTRY.json")
    assert complete["protocol_sha256"]==sha(HERE/"G0_PROTOCOL.json")
    for key in ("scientific_iq_reads","checkpoint_loads","model_forward_calls","new_model_fits","forbidden_target_reads","dates_3_4_reads","cross_rx_packet_pairings"):
        assert complete[key]==0
    access=read(HERE/"ACCESS_LOG.json")["records"]
    qa_access=read(HERE/"QA_ACCESS_LOG.json")
    registered={r["path"]:r for r in registry["arrays"]}
    assert len(access)==36 and len(registered)==36 and len(qa_access)==16
    assert {r["path"] for r in access}==set(registered)
    for r in access+qa_access:
        assert r["path"] in registered and r["sha256"]==registered[r["path"]]["sha256"]
    decision=read(HERE/"DECISION.json")
    assert not decision["route_I_algorithm_training_admitted"] and not decision["historical_STOP_modified"]
    return dict(status="PASS",base_commit=BASE,preread_registration_commit=REGISTRATION,
                registered_code_unchanged=True,historical_tracked_tree_unchanged=True,
                local_links_checked=links,unique_registered_array_paths=36,diagnostic_array_opens=36,qa_prediction_rereads=16,
                new_iq_reads=0,model_forward_calls=0,new_classifier_fits=0,
                scientific_metadata_files_pinned=len(registry["metadata_sha256"]),
                access_accounting="Registered script actions and declared tool actions; not OS-wide instrumentation",
                scientific_array_payloads_opened_by_archive_audit=False)


def write():
    result=check(manifest_pending=True)
    (HERE/"ARCHIVE_QA.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8",newline="\n")
    files=members()
    manifest=dict(base_commit=BASE,registration_commit=REGISTRATION,
                  scope="All diagnostic archive files except MANIFEST.json itself and ignored Python bytecode",
                  text_encoding="UTF-8 canonical LF; PNG byte-exact",
                  files=[dict(path=p.relative_to(HERE).as_posix(),bytes=len(canonical(p)),sha256=hashlib.sha256(canonical(p)).hexdigest()) for p in files])
    (HERE/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")


def verify(index=False):
    result=check()
    manifest=read(HERE/"MANIFEST.json")
    assert {p.relative_to(HERE).as_posix() for p in members()}=={r["path"] for r in manifest["files"]}
    for row in manifest["files"]:
        blob=canonical(HERE/row["path"])
        assert len(blob)==row["bytes"] and hashlib.sha256(blob).hexdigest()==row["sha256"],row["path"]
    if index:
        tracked=git("ls-files","--",PREFIX).decode().splitlines()
        expected={PREFIX+p.relative_to(HERE).as_posix() for p in [*members(),HERE/"MANIFEST.json"]}
        assert set(tracked)==expected, "Unexpected staged files"
        for rel in tracked: assert git("show",":"+rel)==canonical(ROOT/rel), rel
    print(json.dumps(dict(**result,files_hashed=len(manifest["files"]),git_index_checked=index)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true")
    parser.add_argument("--verify",action="store_true")
    parser.add_argument("--index",action="store_true")
    args=parser.parse_args()
    if args.write: write()
    verify(index=args.index)
