"""Materialize connector snapshots or restore exact Git blobs; never import them."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from s24_common import HERE, write_json, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if args.restore:
        man = json.loads((HERE / 'UPSTREAM.json').read_text(encoding='utf-8'))
        for repo in man['repos']:
            for f in repo['files']:
                dest = HERE / '_upstream' / repo['repo'] / f['path']
                data = urllib.request.urlopen(f['raw_url'], timeout=60).read()
                if sha(data) != f['sha256']:
                    raise ValueError(f'SHA256 mismatch: {dest}')
                dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
        print('Restored pinned upstream files. No training.')
        return
    bundle = json.loads((HERE / '_upstream/github_bundle.json').read_text(encoding='utf-8'))
    repos = []
    for repo in bundle['repos']:
        files = []
        for f in repo['files']:
            data = f['content'].encode('utf-8')
            blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            if blob != f['sha']:
                raise ValueError(f'Git blob mismatch: {repo["repo"]}/{f["path"]}')
            dest = HERE / '_upstream' / repo['repo'] / f['path']
            dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
            files.append(dict(path=f['path'], git_blob_sha1=blob, sha256=sha(data), bytes=len(data),
                              raw_url=f'https://raw.githubusercontent.com/{repo["repo"]}/{repo["commit"]}/{f["path"]}',
                              url=f['display_url']))
        repos.append(dict(repo=repo['repo'], commit=repo['commit'], files=files))
    write_json(HERE / 'UPSTREAM.json', dict(retrieved_on=bundle['retrieved_on'],
               mechanism='GitHub connector pinned commit contents, byte-verified Git blob SHA1 and SHA256', repos=repos))
    print(f'Verified {sum(len(r["files"]) for r in repos)} files across {len(repos)} pinned repositories.')


if __name__ == '__main__': main()
