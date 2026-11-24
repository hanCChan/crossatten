"""Audit the public original-capture index and attempt one bounded tar-header read."""

import argparse
import hashlib
import io
import json
import pickle
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

EXPECTED = "f780763723fef4521e907709691a821791fc0ad2"
ROOT = Path(__file__).resolve().parents[2]


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f"Unexpected metadata global {module}.{name}")


def read_header(file_id, proxy):
    address = "https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(
        {"id":file_id,"export":"download","confirm":"t"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({"https":proxy} if proxy else {}))
    request = urllib.request.Request(address,headers={"Range":"bytes=0-511","User-Agent":"WiSig-metadata-audit"})
    try:
        with opener.open(request,timeout=12) as response:
            status = response.status
            content_range = response.headers.get("Content-Range")
            # A range request that the server ignores is never consumed.
            if status != 206 or not content_range or not content_range.startswith("bytes 0-511/"):
                return dict(http_status=status,reported_range=content_range,bytes_consumed=0,
                            status="NO_BOUNDED_RANGE")
            block=response.read(512)
            report=dict(http_status=status,bytes_consumed=len(block),reported_range=content_range,
                        content_type=response.headers.get("Content-Type"))
            if block.startswith(b"\x1f\x8b"):
                report["compression"]="gzip"
                try:
                    block=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(block,512)
                except zlib.error:
                    return dict(**report,status="GZIP_PREFIX_ONLY")
            if len(block)!=512:
                return dict(**report,status="INCOMPLETE_HEADER")
            check=sum(block[:148])+8*32+sum(block[156:])
            try:
                expected=int(block[148:156].strip(b" \0") or b"0",8)
            except ValueError:
                return dict(**report,status="NOT_TAR_HEADER",tar_magic=block[257:262]==b"ustar")
            if check!=expected:
                return dict(**report,status="NOT_TAR_HEADER")
            filename=block[:100].split(b"\0",1)[0].decode("utf-8",errors="replace")
            return dict(**report,status="TAR_HEADER_ONLY",attempt_header_name=filename)
    except urllib.error.HTTPError as exc:
        return dict(http_status=exc.code,bytes_consumed=0,status="RANGE_ENDPOINT_REFUSED")
    except (urllib.error.URLError,TimeoutError,ValueError) as exc:
        return dict(bytes_consumed=0,status="RANGE_UNAVAILABLE",failure=type(exc).__name__)


def audit(file,proxy):
    raw=file.read_bytes()
    git_sha=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    assert git_sha==EXPECTED
    index=PrimitiveUnpickler(io.BytesIO(raw)).load()
    data=json.loads((ROOT/"research/standard_method_competition_c/SOURCE_PLAN.json").read_text())
    eligible=set.intersection(*(set(v["fit_rx"]) for v in data["views"]))
    profiles=[dict(date=day,rx=rx,group=int(g),size=row[2],archive=row[0],identifier=row[1])
              for day in ("2021_03_01","2021_03_08") for rx in sorted(eligible)
              for g,row in index[day].get(rx,{}).items()]
    chosen=min(profiles,key=lambda row:row["size"])
    return dict(index_blob_sha=git_sha,allowed_common_fit_rx=sorted(eligible),
                source_days=["2021_03_01","2021_03_08"],candidate_group_count=len(profiles),
                grouped_bytes=sum(r["size"] for r in profiles),
                smallest={k:v for k,v in chosen.items() if k!="identifier"},
                same_packet_mapping_provided=False,tar_index_probe=read_header(chosen["identifier"],proxy))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("file",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--proxy")
    args=parser.parse_args()
    args.output.write_text(json.dumps(audit(args.file,args.proxy),indent=2)+"\n",encoding="utf-8")
