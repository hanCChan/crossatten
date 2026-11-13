"""Acquire the official archive as opaque bytes; do not deserialize waveforms."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
import urllib.request
import zipfile
from w3_a import HERE,session,write

URL='https://drive.usercontent.google.com/download?id=17EnvGFoflJEh1xhFC8wx5fhCuPYhWt2l&export=download&confirm=t'
SIZE=2631104718
CHUNK=32*1024*1024
LOCK=threading.Lock()
NETWORK=0


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def part(i):
    global NETWORK
    start=i*CHUNK;end=min(SIZE,start+CHUNK)-1
    dest=HERE/'artifacts/ManyTx.parts'/f'{i:03d}.part'
    if dest.exists() and dest.stat().st_size==end-start+1:return dest
    pos=start+(dest.stat().st_size if dest.exists() else 0)
    req=urllib.request.Request(URL,headers={'Range':f'bytes={pos}-{end}'})
    with session().open(req,timeout=90) as r:
        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {pos}-{end}/{SIZE}':raise ValueError('Server did not honor exact range: '+str(dict(r.headers)))
        with dest.open('ab') as f:
            while b:=r.read(1024*1024):
                with LOCK:
                    NETWORK+=len(b)
                    if NETWORK>3500000000:raise ValueError('Opaque acquisition network budget exceeded')
                f.write(b)
    if dest.stat().st_size!=end-start+1:raise ValueError('Incomplete range')
    return dest


def main():
    meta=json.loads((HERE/'A_METADATA_RESULT.json').read_text(encoding='utf8'))
    if not all(meta['gates'].values()):raise ValueError('Metadata gate did not pass')
    folder=HERE/'artifacts';(folder/'ManyTx.parts').mkdir(exist_ok=True)
    archive=folder/'ManyTx.official.zip';started=time.time()
    if not archive.exists():
        count=(SIZE+CHUNK-1)//CHUNK
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(part,i) for i in range(count)]
            for n,future in enumerate(concurrent.futures.as_completed(futures),1):
                future.result();write('A_ACQUISITION_LIVE.json',dict(completed_parts=n,parts=count,network_bytes=NETWORK,elapsed_seconds=time.time()-started))
                if n%8==0 or n==count:print('OPAQUE_DOWNLOAD',n,'/',count,round(NETWORK/1e6),'MB',flush=True)
        with archive.open('xb') as out:
            for i in range(count):
                with (folder/'ManyTx.parts'/f'{i:03d}.part').open('rb') as f:shutil.copyfileobj(f,out,8*1024*1024)
    if archive.stat().st_size!=SIZE:raise ValueError('Archive size differs')
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist();members=[p for p in entries if Path(p.filename).name=='ManyTx.pkl']
        if len(members)!=1 or len(entries)>10:raise ValueError('Unexpected archive structure')
        member=members[0]
        if member.file_size>9000000000:raise ValueError('Extraction bound exceeded')
        dest=folder/'ManyTx.pkl'
        if not dest.exists():
            with z.open(member) as f,dest.open('xb') as out:shutil.copyfileobj(f,out,8*1024*1024)
        if dest.stat().st_size!=member.file_size:raise ValueError('Opaque extraction incomplete')
    result=dict(status='OFFICIAL_MANYTX_STORED_OPAQUE_NOT_DECODED',url=URL,archive_path=str(archive),archive_bytes=SIZE,archive_sha256=digest(archive),
        pickle_path=str(dest),pickle_bytes=dest.stat().st_size,pickle_sha256=digest(dest),
        network_bytes_this_attempt=NETWORK,elapsed_seconds=time.time()-started,IQ_decodes=0,
        note='Downloading and ZIP decompression store opaque bytes. A selective index and committed B manifest are still required before any IQ row is decoded.')
    write('A_ACQUISITION.json',result);print(result['status'],result['pickle_bytes'],flush=True)


if __name__=='__main__':main()
