"""Derived random-input costs for the sealed S19 source models.

This benchmark cannot select or train a model. It reads source checkpoint
weights after all20 source jobs are sealed, and uses synthetic IQ only. Its
five fold0 checkpoint reads are listed in COSTS.json, separately from the
scientific data access log. No held-source or target arrays are opened.
"""
from __future__ import annotations

import gc
import math
import time

import numpy as np
import torch

from s19_core import HERE, committed, job_id, now, read, sha, validate, write
from s19_models import build_model


ARMS = ('B0', 'B1', 'B2', 'B3', 'B4')
WARMUP = 10
REPEATS = 30
SEED = 19101


@torch.no_grad()
def arithmetic_proxy(model, arm):
    """Count dense module products and declare the remaining proxies.

    Conv/Linear outputs determine their MAC counts. The fixed two-layer MHA
    has two attention products per block. Spectral descriptor/gain matrices
    are accounted for separately because they are not nn.Linear modules.
    Every subtotal remains a partial arithmetic proxy, never total FLOPs.
    """
    macs = dict(conv1d=0, linear=0)
    handles = []

    def hook(module, args, output):
        if isinstance(module, torch.nn.Conv1d):
            macs['conv1d'] += int(output.numel()*module.kernel_size[0]*module.in_channels/module.groups)
        elif isinstance(module, torch.nn.Linear):
            macs['linear'] += int(output.numel()*module.in_features)

    try:
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                handles.append(module.register_forward_hook(hook))
        model(torch.zeros(1, 2, 256, device='cuda'))
    finally:
        for handle in handles:
            handle.remove()
    # Two blocks x two products x length^2 x width; two FLOPs per MAC.
    attention_flops = 2*2*2*32*32*64
    spectral_products = 0
    inventory = []
    if arm != 'B0':
        spectral_products += 2*8*256
        inventory.append('8band gains @ masks: 2*8*256')
    if arm in {'B2', 'B3', 'B4'}:
        spectral_products += 2*256*8
        inventory.append('band power pooling: 2*256*8')
    if arm in {'B3', 'B4'}:
        spectral_products += 2*2*256*8
        inventory.append('phase Re and Im band pooling: 2 products * 2*256*8')
    # A common rough radix-2 complex FFT convention, not cuFFT instruction
    # counting. Every inference arm executes one FFT and no inverse FFT.
    fft_estimate = int(5*256*math.log2(256))
    dense = 2*(macs['conv1d']+macs['linear'])
    return dict(
        conv1d_macs=macs['conv1d'], linear_macs=macs['linear'],
        conv_linear_flops=2*(macs['conv1d']+macs['linear']),
        attention_product_flops=attention_flops,
        spectral_band_matrix_product_flops=spectral_products,
        spectral_product_inventory=inventory,
        complex_fft_count=1, complex_ifft_count=0,
        complex_fft_flops_proxy=fft_estimate,
        complex_fft_convention='5*N*log2(N), N256 complex FFT; algorithmic estimate, not exact cuFFT FLOPs',
        partial_arithmetic_subtotal=dense+attention_flops+spectral_products+fft_estimate,
        extra_rf_descriptor_reconstructions=int(arm in {'B3', 'B4'}),
        excluded_from_arithmetic_subtotal=[
            'RMS/Parseval reductions, square roots, epsilon and divisions',
            'complex absolute values and pointwise complex/real multiplications',
            'LogMag log1p, centering, standard deviation, normalization',
            'phase validity comparisons, conjugation, products, padding',
            'GELU, tanh, softmax, LayerNorm, BatchNorm, bias additions',
            'pooling reductions, concatenations, indexing and memory movement',
            'the extra B3/B4 RF descriptor rebuild except its listed matrix products',
        ],
        complete_total_flops=False,
    )


@torch.no_grad()
def latency(model, batch):
    generator = torch.Generator(device='cuda').manual_seed(SEED+batch)
    x = torch.randn(batch, 2, 256, device='cuda', dtype=torch.float32, generator=generator)
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()
    measurements = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        measurements.append(1000*(time.perf_counter()-start))
    values = np.asarray(measurements)
    return dict(batch=batch, warmup=WARMUP, repetitions=REPEATS,
                median_batch_ms=float(np.median(values)),
                p10_batch_ms=float(np.quantile(values, .1)),
                p90_batch_ms=float(np.quantile(values, .9)),
                median_packet_ms=float(np.median(values))/batch,
                samples_batch_ms=measurements,
                input='fixed-seed synthetic standard-normal IQ; no scientific samples')


@torch.no_grad()
def operator_profile(model):
    """Profile one warmed synthetic batch128 and expose per-key scope."""
    generator = torch.Generator(device='cuda').manual_seed(SEED+128)
    x = torch.randn(128, 2, 256, device='cuda', dtype=torch.float32, generator=generator)
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, with_flops=True,
        ) as profile:
            model(x)
            torch.cuda.synchronize()
        events = []
        for event in profile.key_averages():
            events.append(dict(
                key=event.key, count=int(event.count),
                self_cpu_time_us=float(event.self_cpu_time_total),
                total_cpu_time_us=float(event.cpu_time_total),
                self_device_time_us=float(getattr(event, 'self_device_time_total', 0.)),
                total_device_time_us=float(getattr(event, 'device_time_total', 0.)),
                profiler_reported_flops=int(event.flops or 0),
            ))
        return dict(status='PASS', batch=128, events=events,
                    scope='One warmed synthetic inference; CPU/CUDA profiler overhead excluded from latency measurements',
                    profiler_flops_are_complete=False,
                    note='ATen time keys are an operator inventory; profiler FLOPs support only some operators and are not a total')
    except Exception as exc:
        # Missing local Kineto support must not turn a completed scientific
        # experiment into a different model or trigger a new training run.
        return dict(status='UNAVAILABLE', error_type=type(exc).__name__, error=str(exc),
                    profiler_flops_are_complete=False)


def inference_path(arm):
    path = ['legacy packet RMS in tokens()', 'legacy packet RMS in features()',
            'one complex FFT256 and fftshift']
    if arm == 'B1':
        path += ['8 static learned logits; mean-center; tanh; smooth partition-of-unity gain']
    elif arm == 'B2':
        path += ['8 relative log1p band powers',
                 'Linear8x16 + GELU + Linear16x8(no bias)',
                 'mean-center logits; tanh; smooth partition-of-unity gain']
    elif arm in {'B3', 'B4'}:
        path += ['8 relative log1p band powers',
                 'EXTRA full original 5channel RF reconstruction for phase descriptors',
                 '8 averaged phase-product Re and 8 Im descriptors',
                 'Linear24x8 + GELU + Linear8x8',
                 'mean-center logits; tanh; smooth partition-of-unity gain']
    if arm != 'B0':
        path += ['complex spectrum multiplied by gain',
                 'Parseval relative RMS correction against original spectrum']
    path += ['rebuild Re/Im/standardized LogMag and masked adjacent phase-product Re/Im',
             'HStem: two local convolutions plus 3 strided tokenization convolutions',
             '32 tokens x64; token preparation and pre-LayerNorm',
             'two ordinary MHA blocks including QK and AV products and softmax',
             'final LayerNorm; token mean pool; TX classifier']
    return path


def training_costs():
    new = []
    historical = []
    evidence = []
    for fold in range(4):
        for arm in ARMS:
            record = read(HERE/'runs'/job_id(fold, arm)/'metrics.json')
            if record['status'] != 'COMPLETE':
                raise ValueError('All formal source fits must complete before costs')
            if arm == 'B0':
                historical.append(dict(fold=fold, seconds=record['historical_teacher_fit_seconds'],
                                       checkpoint_sha256=record['model_sha256'],
                                       role='same reused checkpoint serves as B0 comparator and teacher; counted once'))
            else:
                new.append(dict(fold=fold, arm=arm, seconds=record['wall_seconds'],
                                epochs=record['epochs'],
                                peak_allocated_gib=record['peak_allocated_gib']))
        record = read(HERE/f'evidence_f{fold}.json')
        evidence.append(dict(fold=fold, seconds=record['seconds'], packets=record['packets'],
                             teacher_sha256=record['teacher_sha256']))
    new_seconds = float(sum(r['seconds'] for r in new))
    old_seconds = float(sum(r['seconds'] for r in historical))
    evidence_seconds = float(sum(r['seconds'] for r in evidence))
    return dict(
        new_fit_count=len(new), new_fit_seconds=new_seconds, new_fits=new,
        reused_historical_fit_count=len(historical), historical_reused_fit_seconds=old_seconds,
        historical_fits_counted_once=historical,
        teacher_evidence_inference_seconds=evidence_seconds, teacher_evidence=evidence,
        current_stage_measured_component_seconds=new_seconds+evidence_seconds,
        training_seconds_if_four_teachers_had_to_be_trained_here=new_seconds+old_seconds,
        complete_stage_wall_time=False,
        timing_scope='Fit timers include source-loop bookkeeping and source validation; evidence timers cover extraction and cache serialization before the evidence record. B0 reuse validation, seal, held-source scoring, reports and QA are not fully timed here.',
    )


def run():
    validate()
    committed(HERE/'seal_pilot.json')
    seal = read(HERE/'seal_pilot.json')
    if not seal['all_checkpoints_complete'] or len(seal['checkpoints']) != 20:
        raise ValueError('All20 source checkpoints must be sealed before costs')
    output = HERE/'COSTS.json'
    if output.exists():
        raise RuntimeError('Cost measurement already exists; retain the archived benchmark')
    training = training_costs()
    if training['new_fit_count'] != 16:
        raise ValueError('Expected fixed16 new fit budget')
    if not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA benchmark environment required')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    rows = []
    checkpoint_access = []
    for arm in ARMS:
        identifier = job_id(0, arm)
        cfg_path = HERE/'runs'/identifier/'config.json'
        cfg = read(cfg_path)
        metadata = seal['metadata_sha256'][identifier]
        for name in ['config.json', 'metrics.json']:
            if sha(HERE/'runs'/identifier/name) != metadata[name]:
                raise ValueError('Sealed benchmark metadata changed')
        reference = seal['checkpoints'][identifier]
        if sha(reference['path']) != reference['sha256']:
            raise ValueError('Sealed benchmark checkpoint changed')
        checkpoint_access.append(dict(kind='cost_checkpoint', fold=0, arm=arm,
                                      path=reference['path'], sha256=reference['sha256'],
                                      seal_sha256=sha(HERE/'seal_pilot.json'),
                                      input_role='synthetic IQ only; no scientific data read'))
        model = build_model(arm, 'ManyRx', cfg['data_seed']).cuda().eval()
        model.load_state_dict(torch.load(reference['path'], map_location='cpu', weights_only=True))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        torch.cuda.reset_peak_memory_stats()
        count = sum(p.numel() for p in model.parameters())
        row = dict(arm=arm, fold=0, checkpoint_sha256=reference['sha256'],
                   parameters=count, added_parameters_vs_B0=count-82426,
                   arithmetic_proxy=arithmetic_proxy(model, arm),
                   inference_path=inference_path(arm),
                   latency=[latency(model, b) for b in [1, 128]],
                   peak_benchmark_allocated_gib=float(torch.cuda.max_memory_allocated()/2**30),
                   operator_profile=operator_profile(model))
        rows.append(row)
        print('cost', arm, 'PASS', count, 'parameters', flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write(output, dict(
        time=now(), status='COMPLETE', benchmark_kind='derived random-input source-model benchmark',
        benchmark_code_sha256=sha(__file__), registration_sha256=sha(HERE/'registration.json'),
        checkpoint_seal_sha256=sha(HERE/'seal_pilot.json'),
        scientific_model_selection=False, new_training_runs=0,
        device=torch.cuda.get_device_name(), torch=torch.__version__, precision='FP32',
        cudnn_benchmark=False, cudnn_deterministic=True, host_threads=2,
        seed=SEED, input_length=256, batches=[1, 128], warmup=WARMUP, repetitions=REPEATS,
        synchronization='host perf_counter around each model call and torch.cuda.synchronize()',
        timing_scope='End-to-end Python model call on GPU; includes Python dispatch and synchronization. Synthetic batch inputs are generated before timed loops.',
        complete_total_flops=False,
        gain_bound_scope='Pre-packet-normalization complex-spectrum amplitude gain only; normalized RF channels are not bounded to +/-20%',
        source_checkpoint_access=checkpoint_access,
        scientific_access_log_modified=False, held_source_data_read=False, real_target_data_read=False,
        training_costs=training, arms=rows,
    ))


if __name__ == '__main__': run()
