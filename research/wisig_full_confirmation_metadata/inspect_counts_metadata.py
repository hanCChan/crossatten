"""Fetch only the preregistered public count metadata; never fetch WiSig signals."""
from pathlib import Path
import hashlib
import json
import pickletools
import urllib.request

HERE = Path(__file__).resolve().parent
URL = 'https://raw.githubusercontent.com/WiSig-dataset/wisig-examples/afbf0a95bc888a7e3d9f94fa3442d36dace6e1fb/data_summary.pkl'
SIZE = 459700
BLOB = '3df26215f59e349b1ba9da202de44fc503690039'

def admitted_blob():
    """Size-limited retrieval, exact Git blob identity, then static opcode inventory."""
    admission = json.loads((HERE / 'METADATA_FILE_ADMISSION.json').read_text(encoding='utf-8'))
    assert admission['expected_size_bytes'] == SIZE and admission['git_blob_sha'] == BLOB
    destination = HERE / 'artifacts' / 'data_summary.metadata.pkl'
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': 'http://127.0.0.1:78'}))
        with opener.open(URL, timeout=30) as response:
            assert response.geturl() == URL
            payload = response.read(SIZE + 1)
        assert len(payload) == SIZE
        assert hashlib.sha1(b'blob ' + str(SIZE).encode() + b'\0' + payload).hexdigest() == BLOB
        destination.write_bytes(payload)
    payload = destination.read_bytes()
    assert len(payload) == SIZE
    assert hashlib.sha1(b'blob ' + str(SIZE).encode() + b'\0' + payload).hexdigest() == BLOB
    operations = list(pickletools.genops(payload))
    globals_found = sorted({str(arg) for op, arg, _ in operations if op.name in ('GLOBAL', 'STACK_GLOBAL')})
    short_strings = sorted({str(arg) for op, arg, _ in operations if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE') and isinstance(arg, str) and len(arg) < 80})
    inspection = {'url': URL, 'bytes': SIZE, 'git_blob_sha': BLOB, 'sha256': hashlib.sha256(payload).hexdigest(), 'pickle_globals': globals_found, 'short_strings': short_strings}
    (HERE / 'STATIC_PICKLE_INSPECTION.json').write_text(json.dumps(inspection, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in inspection.items() if k != 'short_strings'}, indent=2))
    print('short string count:', len(short_strings))
    return payload

if __name__ == '__main__':
    admitted_blob()
