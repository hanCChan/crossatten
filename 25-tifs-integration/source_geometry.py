"""No-fit feature diagnostics on all frozen H0/P0 source-validation cohorts."""
import csv
import time
from pathlib import Path

import numpy as np
import torch

from geometry_stats import geometry
from s25_common import HERE, P, digest, load_model, now, read, source_validation, validate, write


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def extract(model, x):
    captured = {}
    handles = []
    def hook(name):
        def receive(module, args, output):
            assert output.ndim == 3 and output.shape[1:] == (32, 64)
            captured[name] = output.mean(1).cpu().numpy()
        return receive
    handles.append(model.net.embedder.cnn_stem.register_forward_hook(hook('tokens')))
    handles.append(model.pre_ln.register_forward_hook(hook('pre_mha')))
    def final(module, args):
        assert args[0].shape[1:] == (64,)
        captured['final'] = args[0].cpu().numpy()
    handles.append(model.net.classifier.register_forward_pre_hook(final))
    outputs = {name: [] for name in ['tokens', 'pre_mha', 'final', 'logits']}
    try:
        for start in range(0, len(x), 128):
            batch = torch.from_numpy(np.ascontiguousarray(x[start:start+128])).cuda()
            logits = model(batch)
            for name in ['tokens', 'pre_mha', 'final']:
                outputs[name].append(captured[name].copy())
            outputs['logits'].append(logits.cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    return {name: np.concatenate(values) for name, values in outputs.items()}


@torch.inference_mode()
def benchmark(model):
    output = {}
    for batch in (1, 128):
        x = torch.randn(batch, 2, 8192, device='cuda')
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        samples = []
        for _ in range(15):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model(x)
            torch.cuda.synchronize()
            samples.append((time.perf_counter()-started)*1000)
        output[str(batch)] = dict(p50_ms=float(np.median(samples)), p95_ms=float(np.quantile(samples, .95)), samples=samples)
    return output


def main():
    registration = validate()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(25025)
    started = time.perf_counter()
    metrics, receiver_rows, costs = [], [], []
    feature_dir = HERE/'features'
    feature_dir.mkdir(exist_ok=True)
    for fold in (0, 1):
        data = source_validation(fold)
        for record in [r for r in registration['models'] if r['fold'] == fold]:
            model = load_model(record)
            features = extract(model, data['x'])
            destination = feature_dir/(record['id']+'.npz')
            np.savez(destination, **features, y=data['y'], rx=data['rx'], rows=data['rows'])
            for layer in ['tokens', 'pre_mha', 'final']:
                metrics.append(dict(arm=record['arm'], fold=fold, seed=record['seed'], layer=layer,
                                    **geometry(features[layer], data['y'], data['rx'])))
            predictions = features['logits'].argmax(1)
            for rx_index, rx in enumerate(P['folds'][fold]['source']):
                ids = data['rx'] == rx_index
                receiver_rows.append(dict(arm=record['arm'], fold=fold, seed=record['seed'], rx=rx,
                                          packets=int(ids.sum()), accuracy=float(np.mean(predictions[ids] == data['y'][ids]))))
            if fold == 0 and record['seed'] == 201:
                costs.append(dict(arm=record['arm'], params=record['params'], batches=benchmark(model)))
            assert digest(Path(record['checkpoint'])) == record['checkpoint_sha256']
            write(HERE/'SOURCE_PROGRESS.json', dict(status='RUNNING', completed_models=len(metrics)//3, total_models=12,
                                                  active_model=record['id'], new_fits=0, new_optimizer_steps=0))
            print(record['id'], 'source_acc', float(np.mean(predictions == data['y'])), flush=True)
            del model
            torch.cuda.empty_cache()
    contrasts = []
    keys = ['between_tx', 'conditional_rx', 'within_tx_rx', 'tx_to_rx_centroid_ratio', 'tx_to_within_class_ratio', 'tx_fraction', 'rx_fraction']
    for layer in ['tokens', 'pre_mha', 'final']:
        for key in keys:
            differences = []
            for fold in (0, 1):
                for seed in (201, 202, 203):
                    values = {r['arm']: r[key] for r in metrics if r['fold'] == fold and r['seed'] == seed and r['layer'] == layer}
                    differences.append(values['H0']-values['P0'])
            contrasts.append(dict(layer=layer, metric=key, mean_h0_minus_p0=float(np.mean(differences)),
                                  positive_pairs=sum(float(v)>0 for v in differences), paired_deltas=[float(v) for v in differences]))
    csv_write(HERE/'source_geometry.csv', metrics)
    csv_write(HERE/'source_receiver_accuracy.csv', receiver_rows)
    write(HERE/'SOURCE_GEOMETRY.json', dict(status='COMPLETE', time=now(), metrics=metrics, paired_descriptive_contrasts=contrasts,
          models=12, feature_predictions=24000, new_fits=0, new_optimizer_steps=0, new_target_waveform_reads=0,
          scope='Exploratory source validation, already used for checkpoint selection; fixed models and no new independent generalization or causal claim.',
          registration_sha256=digest(HERE/'registration.json'), wall_seconds=time.perf_counter()-started))
    write(HERE/'MATCHED_COSTS.json', dict(status='COMPLETE', gpu=torch.cuda.get_device_name(0), rows=costs,
          scope='New paired H0/P0 benchmark session; FP32/TF32-off, GPU-resident raw IQ to logits. No host transfer. Do not merge absolute timings with older sessions.'))
    write(HERE/'SOURCE_PROGRESS.json', dict(status='COMPLETE', completed_models=12, total_models=12, new_fits=0, new_optimizer_steps=0))


if __name__ == '__main__':
    main()
