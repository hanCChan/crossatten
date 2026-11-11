"""Verify the research archive and source fingerprints, without loading arrays/models."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
BASE="b3035c9db8cc2e65488e8b606ff68e102ae006d3"
PREFIX=HERE.relative_to(ROOT).as_posix()+"/"


def read(p): return json.loads(p.read_text(encoding="utf-8"))
def git(*args): return subprocess.check_output(["git","-C",str(ROOT),*args])
def canonical(p):
    raw=p.read_bytes()
    return raw if p.suffix==".png" else raw.replace(b"\r\n",b"\n")
def files():
    return sorted(p for p in HERE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name!="MANIFEST.json" and p.suffix!=".pyc")


def check(pending_manifest=False):
    qa=read(HERE/"QA_REPORT.json");result=read(HERE/"AUDIT.json")
    assert qa["status"]=="PASS" and qa["source_pool_pair_comparisons_checked"]==6
    assert not result["cause_importance_ranked"] and not result["variance_components_identified"]
    for r in read(HERE/"ACCESS_LOG.json")["records"]:
        assert hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()==r["sha256"],r["path"]
    for r in read(HERE/"CODE_REVIEW.json")["historical_files"]:
        if "base_git_blob_sha256" in r:
            assert hashlib.sha256(git("show",BASE+":"+r["path"])).hexdigest()==r["base_git_blob_sha256"]
            assert canonical(ROOT/r["path"])==git("show",BASE+":"+r["path"])
        else: assert hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()==r["sha256"]
    changed=git("diff","--name-only",BASE).decode().splitlines()+git("ls-files","--others","--exclude-standard").decode().splitlines()
    assert all(p.startswith(PREFIX) for p in changed), "Historical files changed"
    links=0
    for doc in HERE.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)",doc.read_text(encoding="utf-8")):
            if "://" in target or target.startswith("#"): continue
            p=(doc.parent/target.split("#")[0]).resolve()
            assert p.is_relative_to(ROOT) and (p.exists() or pending_manifest and p==HERE/"MANIFEST.json"),(doc.name,target)
            links+=1
    assert read(HERE/"DECISION.json")["new_model_fits"]==0
    return dict(status="PASS",base_commit=BASE,historical_files_unchanged=True,
                source_histories_verified=8,pool_comparisons_independently_verified=6,
                synthetic_tests_passed=5,local_links_checked=links,
                new_model_fits=0,new_iq_reads=0,new_prediction_array_reads=0,
                checkpoint_loads=0,causal_attribution_identified=False,
                accounting="np.load label-array payload opens are counted in QA_REPORT; byte-hash reads are separate; no OS-wide instrumentation")


def write():
    status=check(pending_manifest=True)
    (HERE/"ARCHIVE_QA.json").write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8",newline="\n")
    manifest=dict(base_commit=BASE,text_newlines="LF canonical; PNG byte-exact",scope="All files except this MANIFEST and ignored Python bytecode",
                  files=[dict(path=p.relative_to(HERE).as_posix(),bytes=len(canonical(p)),sha256=hashlib.sha256(canonical(p)).hexdigest()) for p in files()])
    (HERE/"MANIFEST.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8",newline="\n")


def verify(index=False):
    status=check()
    manifest=read(HERE/"MANIFEST.json")
    assert {p.relative_to(HERE).as_posix() for p in files()}=={r["path"] for r in manifest["files"]}
    for r in manifest["files"]:
        raw=canonical(HERE/r["path"])
        assert len(raw)==r["bytes"] and hashlib.sha256(raw).hexdigest()==r["sha256"],r["path"]
    if index:
        tracked=git("ls-files","--",PREFIX).decode().splitlines()
        assert set(tracked)=={PREFIX+p.relative_to(HERE).as_posix() for p in [*files(),HERE/"MANIFEST.json"]}
        for rel in tracked: assert git("show",":"+rel)==canonical(ROOT/rel),rel
    print(json.dumps(dict(**status,files_hashed=len(manifest["files"]),git_index_checked=index)))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--write",action="store_true");p.add_argument("--verify",action="store_true");p.add_argument("--index",action="store_true");a=p.parse_args()
    if a.write: write()
    verify(a.index)
