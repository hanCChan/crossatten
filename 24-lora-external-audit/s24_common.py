"""Stage-one-only LoRa adapters. No optimizer, fitting, or target loader."""
from __future__ import annotations
import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL_PATH = ROOT / 'configs/protocol.json'
P = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def source_path(raw_dir: Path, rx: str, fold: int) -> Path:
    # Validate identity and role BEFORE checking/opening any filesystem path.
    folds = {f['fold']: f for f in P['folds']}
    if fold not in folds or rx not in folds[fold]['source']:
        raise PermissionError(f'not a source receiver for fold {fold}: {rx}')
    root = raw_dir.resolve()
    path = root / P['dataset']['file_pattern'].format(rx=rx)
    if path.resolve().parent != root:
        raise PermissionError('source path escapes configured raw directory')
    return path


def selected_rows(labels: np.ndarray) -> np.ndarray:
    if set(labels.tolist()) != set(range(31, 41)):
        raise ValueError('expected exactly raw TX labels 31..40')
    rng = np.random.default_rng(P['packet_selection']['seed'])
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == c), 200, replace=False)
        for c in range(31, 41)]))


def split_indices(y, rows):
    tr, va = [], []
    if len(set(np.asarray(rows).tolist())) != len(rows):
        raise ValueError('duplicate raw rows')
    for t in range(10):
        idx = np.flatnonzero(y == t)
        if len(idx) != 200:
            raise ValueError('expected 200 packets per TX')
        idx = idx[np.argsort(rows[idx], kind='stable')]
        tr.extend(idx[:160]); va.extend(idx[160:])
    return np.asarray(tr), np.asarray(va)


def unpack_iq(a: np.ndarray) -> np.ndarray:
    if a.ndim != 2 or a.shape[1] != 16384:
        raise ValueError(f'expected [N,16384], got {a.shape}')
    iq = np.stack([a[:, :8192], a[:, 8192:]], axis=1).astype(np.float32)
    if not np.isfinite(iq).all() or np.any(np.sum(iq.astype(np.float64)**2, axis=(1,2)) <= 0):
        raise ValueError('nonfinite or zero-energy IQ')
    return iq


def project_ib(iq: np.ndarray) -> np.ndarray:
    # Same discrete mask as xa.frontends.band_limit; retain +/-512 and DC (1025 bins).
    z = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    keep = np.abs(np.fft.fftfreq(8192, 1 / 1e6)) <= 62500
    z = np.fft.ifft(np.fft.fft(z, axis=-1) * keep, axis=-1)
    return np.stack([z.real, z.imag], axis=1).astype(np.float32)


def power_norm(iq):
    power = np.mean(iq[:, 0]**2 + iq[:, 1]**2, axis=-1)
    return iq / np.sqrt(np.maximum(power, 1e-12))[:, None, None]


def verified_source(repo: str, path: str) -> bytes:
    man = json.loads((HERE / 'UPSTREAM.json').read_text(encoding='utf-8'))
    entry = next(f for r in man['repos'] if r['repo'] == repo for f in r['files'] if f['path'] == path)
    data = (HERE / '_upstream' / repo / path).read_bytes()
    if sha(data) != entry['sha256']:
        raise ValueError(f'upstream SHA256 mismatch: {repo}/{path}')
    return data


def official_preprocessor(repo: str, path: str, symbols: list[str]):
    # Compile only named original definitions. Never import/execute upstream main/loaders.
    from scipy import signal
    source = verified_source(repo, path).decode('utf-8')
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in symbols]
    if {n.name for n in nodes} != set(symbols):
        raise ValueError('requested upstream definition missing')
    namespace = {'np': np, 'signal': signal, 'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f'{repo}/{path}', 'exec'), namespace)
    return namespace


def method_input(method: str, iq: np.ndarray, view: str):
    if view not in ('native', 'strict_ib'):
        raise ValueError(view)
    if view == 'strict_ib':
        iq = project_ib(iq)
    z = iq[:, 0].astype(np.float64) + 1j * iq[:, 1].astype(np.float64)
    if method == 'shen_tmc2024':
        ns = official_preprocessor('gxhen/receiverAgnosticRFFI', 'dataset_preparation.py', ['ChannelIndSpectrogram'])
        return ns['ChannelIndSpectrogram']().channel_ind_spectrogram(z)
    if method == 'shen_tifs2022':
        ns = official_preprocessor('gxhen/LoRa_RFFI', 'Openset_RFFI_TIFS/dataset_preparation.py', ['ChannelIndSpectrogram'])
        return ns['ChannelIndSpectrogram']().channel_ind_spectrogram(z)
    if method == 'asknet_lora':
        # Official loader power normalization, then original ASK/CIS/ResNet remains intact.
        return power_norm(iq)
    if method == 'shen_tifs2023':
        ns = official_preprocessor('gxhen/lengthVersatileRFFI', 'utils.py',
                                  ['normalization','spec_crop','ch_ind_spectrogram','gen_ch_ind_spectrogram'])
        return ns['ch_ind_spectrogram'](z, win_len=64, crop_ratio=0)[:, :, :, 0].transpose(0,2,1)
    # No inferred LoRa recipe for IB-RFF: it must be explicitly designed/frozen separately.
    raise ValueError(f'method lacks an audited LoRa adapter: {method}')
