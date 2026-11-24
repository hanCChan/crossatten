"""Fetch exactly one pinned eligible original archive, without executing upstream code."""

import argparse
import hashlib
import io
import json
import pickle
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path

INDEX_BLOB="f780763723fef4521e907709691a821791fc0ad2"
CAPTURE=("2021_03_01","23-5",9)
SIZE=568961970


class Metadata(pickle.Unpickler):
    def find_class(self,module,name):
        raise ValueError(f"Unexpected metadata global {module}.{name}")


def main(index_file,out,proxy):
    out.mkdir(parents=True,exist_ok=True)
    raw=index_file.read_bytes()
    assert hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()==INDEX_BLOB
    meta=Metadata(io.BytesIO(raw)).load()
    day,rx,group=CAPTURE
    name,identifier,size=meta[day][rx][group]
    assert name=="9.tar" and size==SIZE
    url="https://drive.usercontent.google.com/download?"+urllib.parse.urlencode(
        {"id":identifier,"export":"download","confirm":"t","audit":"legal-source-s39a2"})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({"https":proxy} if proxy else {}))
    path=out/"capture.tar.gz"
    if not path.exists():
        temp=path.with_suffix(".part")
        with opener.open(url,timeout=24) as response, temp.open("wb") as stream:
            assert response.status==200 and response.headers.get("Content-Range") is None
            digest=hashlib.sha256()
            total=0
            while b:=response.read(1024*1024):
                total+=len(b)
                if total>SIZE:
                    raise ValueError("Archive exceeds pinned source-group budget")
                stream.write(b)
                digest.update(b)
                if total%(64*1024*1024)==0:
                    print("received bytes",total,flush=True)
        if total!=SIZE:
            raise ValueError(f"Expected {SIZE} bytes but received {total}")
        temp.replace(path)
        checksum=digest.hexdigest()
    else:
        assert path.stat().st_size==SIZE
        with path.open("rb") as stream:
            checksum=hashlib.file_digest(stream,"sha256").hexdigest()
    with tarfile.open(path,"r:gz") as pkg:
        members=[dict(name=m.name,bytes=m.size,regular=bool(m.isfile())) for m in pkg]
    outcome=dict(archive=path.name,sha256=checksum,bytes=SIZE,date=day,rx=rx,group=group,
                 decoded_iq_files=0,members=members)
    (out/"ARCHIVE.json").write_text(json.dumps(outcome,indent=2)+"\n",encoding="utf-8")
    print("verified one capture archive",len(members),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("index",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--proxy")
    a=parser.parse_args()
    main(a.index,a.output,a.proxy)
