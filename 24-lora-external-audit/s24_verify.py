"""Numerical/permission engineering checks; zero fits and zero optimizer steps."""
import ast
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from s24_common import (HERE, ROOT, P, sha, write_json, source_path, split_indices,
                        method_input, project_ib, verified_source)

sys.path.insert(0, str(ROOT / 'lib'))
from xa.frontends import shen_cis, band_limit
from s24_dann import Reverse, ShenDANN


def main():
    torch.set_num_threads(4)
    torch.manual_seed(24)
    checks = []
    def record(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
        if not passed: raise AssertionError(name)
    # No prohibited filename is opened even when the caller passes a fabricated directory.
    for rx in P['blind_receivers_never_open'] + P['confirm_receivers'] + ['../rtl_1', 'unknown', 'rtl_7']:
        try: source_path(Path('NONEXISTENT'), rx, 0)
        except PermissionError: continue
        raise AssertionError(f'guard accepted {rx}')
    record('forbidden_and_wrong_fold_rejected_before_io', True, cases=13)
    y = np.repeat(np.arange(10), 200); rows = np.arange(2000)
    tr, va = split_indices(y, rows)
    record('160_40_identity_disjoint', not set(tr)&set(va) and len(tr)==1600 and len(va)==400)
    try: split_indices(y, np.zeros(2000, dtype=int))
    except ValueError: record('duplicate_row_rejected', True)
    else: raise AssertionError('duplicate row not rejected')
    probe = np.load(HERE / 'local/source_probe.npz')['iq']
    random_iq = np.random.default_rng(24).standard_normal((2,2,8192)).astype(np.float32)
    data = np.concatenate([probe, random_iq])
    frontend = []
    for method, expected in [('shen_tmc2024',(52,126,1)),('shen_tifs2022',(102,62,1)),
                             ('asknet_lora',(2,8192)),('shen_tifs2023',(254,64))]:
        for view in ('native','strict_ib'):
            start = time.perf_counter(); out = method_input(method, data, view)
            elapsed = time.perf_counter() - start
            record(f'{method}_{view}_shape_finite', out.shape[1:]==expected and np.isfinite(out).all())
            frontend.append(dict(method=method, view=view, input_packets=len(data), shape=list(out.shape),
                                 finite=True, output_sha256=sha(out.tobytes()),
                                 cpu_total_seconds=elapsed,
                                 timing_scope='single cold diagnostic call, not a latency benchmark'))
    x = torch.from_numpy(data[:4])
    for view in ('native','strict_ib'):
        iq = data[:4] if view=='native' else project_ib(data[:4])
        reference = method_input('shen_tmc2024', data[:4], view).transpose(0,3,1,2)
        port = shen_cis(torch.from_numpy(iq)).numpy()
        err = float(np.max(np.abs(port-reference)))
        record(f'shen_cis_original_scipy_vs_existing_torch_{view}', np.allclose(port,reference,atol=1e-5,rtol=1e-5), max_abs=err)
    projected = project_ib(data)
    z = projected[:,0].astype(np.float64) + 1j*projected[:,1].astype(np.float64)
    spec = np.fft.fft(z)
    keep = np.abs(np.fft.fftfreq(8192,1e-6)) <= 62500
    ratio = float(np.sum(np.abs(spec[:,~keep])**2) / np.sum(np.abs(spec)**2))
    record('strict_ib_discrete_support', keep.sum()==1025 and ratio<1e-12,
           kept_bins=int(keep.sum()), float32_residual_energy_ratio=ratio,
           numpy_vs_existing_torch_max_abs=float(np.max(np.abs(projected[:4]-band_limit(x,62500).numpy()))))
    # Execute original ASK class definitions (no main, no official data loader).
    src = verified_source('BeechburgPieStar/ASKNet-RFF','Lora/backbones/ResCISNet.py')
    ns = {'np':np, 'torch':torch, 'nn':torch.nn, 'F':torch.nn.functional}
    exec(compile(ast.parse(src.decode('utf-8')), 'pinned_ASKNet_ResCISNet.py','exec'),ns)
    cis = ns['ChannelIndSpectrogramTorch']()
    record('official_ask_cis_8192', tuple(cis(x).shape)==(4,1,52,126))
    cutoff = ns['LearnableCutoff'](8192)
    record('official_ask_default_masks', cutoff.max_modes==8191,
           candidates=cutoff.candidates, bins_per_mask=cutoff.masks.sum(1).int().tolist())
    with torch.device('meta'):
        full = ns['ResCISNet'](num_classes=10)
    ask_elements = sum(p.numel() for p in full.parameters())
    ask_dof = sum(p.numel() * (2 if p.is_complex() else 1) for p in full.parameters())
    write_json(HERE/'ASKNET_RESOURCES.json', dict(
        constructor='official ResCISNet(num_classes=10), all defaults; meta tensors only',
        parameter_elements=ask_elements, real_degrees_of_freedom=ask_dof,
        parameter_bytes=sum(p.numel()*p.element_size() for p in full.parameters()),
        koopman_complex_matrix_shape=[8191,8191],
        one_complex64_matrix_bytes=8191**2*8,
        matrix_weight_grad_adam_two_moments_bytes_lower_bound=8191**2*8*4,
        qr_scaling='O(8191^3) each forward, plus backward and matrix products',
        full_default_gpu_forward_backward='NOT_RUN',
        reduced_candidates_used_for_main_result=False,
        note='Memory lower bound excludes QR workspace/graph, other matrices, backbone and activations. Feasibility is unverified.'))
    # Tiny operator test only. Explicitly not a substitute for the full 8191-mode model.
    koopman = ns['KoopmanLayer'](7)
    toy = torch.randn(2,7,dtype=torch.cfloat,requires_grad=True)
    evolved = koopman(toy)
    evolved.real.sum().backward()
    record('official_koopman_toy_gradient_only', torch.isfinite(toy.grad).all(), n_modes=7,
           scope='unit operator check; default full-model GPU forward/backward remains unverified')
    v = torch.ones(2,requires_grad=True); Reverse.apply(v).sum().backward()
    record('dann_gradient_sign', torch.equal(v.grad, -torch.ones_like(v)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ShenDANN().to(device)
    inp = x[:2].to(device)
    tx, rx = model.forward_heads(inp)
    loss = torch.nn.functional.cross_entropy(tx, torch.tensor([0,1],device=device)) + torch.nn.functional.cross_entropy(rx,torch.tensor([0,1],device=device))
    loss.backward()
    params = sum(p.numel() for p in model.parameters())
    record('dann_full_port_shape_and_backward', tuple(tx.shape)==(2,10) and tuple(rx.shape)==(2,5)
           and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),
           params=params, device=device, tensorflow_weight_transfer_parity=False, optimizer_steps=0)
    write_json(HERE/'FRONTEND_AUDIT.json',dict(frontends=frontend, checks=checks,
               environment=dict(torch=torch.__version__, numpy=np.__version__,
                                tensorflow_installed=bool(importlib.util.find_spec('tensorflow'))),
               passed=all(c['passed'] for c in checks), fits=0, optimizer_steps=0))
    print(f'{len(checks)} engineering checks passed; no fits or optimizer steps.')


if __name__ == '__main__': main()
