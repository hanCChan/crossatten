"""Prespecified paired composition analysis; no IQ or checkpoint decoding."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('frozen_A_metrics', HERE.parent/'controlled_training_a/analyze.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
RUNS = ('B0_S00', 'B1_S00', 'B0_S11', 'B1_S11')
STREAMS = ('S00', 'S11')
ENDPOINTS = ('selected', 'epoch10')
SCALES = dict(accuracy=.01, macro_f1=.01, ce=.05, disagreement=.05)
REPLICATES, SEED = 10000, 22402


def interval(values):
    return np.quantile(values, (.025, .975)).tolist()


def local_risk_consistency(pairs, minimum_support=50, practical_scale=.05):
    maps = {flow: {(r['rx'], r['tx_class_index']): r for r in pairs[flow]['tx_rx']} for flow in STREAMS}
    if set(maps['S00']) != set(maps['S11']):
        raise ValueError('Streams do not evaluate the same local groups')
    rows = []
    for key in sorted(maps['S00']):
        records = [maps[flow][key] for flow in STREAMS]
        changes = [-r['accuracy_delta'] for r in records]
        supported = min(r['packets'] for r in records) >= minimum_support
        same_direction = changes[0]*changes[1] > 0
        large = all(abs(d) >= practical_scale-1e-12 for d in changes)
        rows.append(dict(rx=key[0], tx_class_index=key[1], support_per_arm={flow: maps[flow][key]['packets'] for flow in STREAMS},
                         risk_change={flow: changes[i] for i, flow in enumerate(STREAMS)},
                         mean_risk_change=float(np.mean(changes)), sufficient_support=supported,
                         same_direction=same_direction, both_cross_scale=large,
                         repeated_local_change=supported and same_direction and large,
                         interpretation='composition1 minus composition0 error rate; exploratory multiple-group screen'))
    repeated = [r for r in rows if r['repeated_local_change']]
    return dict(minimum_support=minimum_support, practical_scale=practical_scale, total_groups=len(rows),
                repeated_local_groups=len(repeated), composition1_lower_risk=sum(r['mean_risk_change'] < 0 for r in repeated),
                composition1_higher_risk=sum(r['mean_risk_change'] > 0 for r in repeated), groups=rows,
                no_groupwise_population_significance_claim=True)


def analyze_endpoint(predictions, replicates=REPLICATES, seed=SEED):
    if set(predictions) != set(RUNS):
        raise ValueError('Exactly the four Stage B runs are required')
    aligned = {RUNS[0]: M.validate_predictions(predictions[RUNS[0]])}
    for run in RUNS[1:]:
        _, aligned[run] = M.align_pair(aligned[RUNS[0]], predictions[run])
    summaries = {run: M.summarize_predictions(data) for run, data in aligned.items()}
    pairs = {flow: M.compare_predictions(aligned[f'B0_{flow}'], aligned[f'B1_{flow}']) for flow in STREAMS}
    receivers = sorted(summaries[RUNS[0]]['per_rx'])
    draws = np.random.default_rng(seed).integers(0, len(receivers), size=(replicates, len(receivers)))
    effects = {}
    for metric in M.METRICS:
        delta = {flow: np.asarray([summaries[f'B1_{flow}']['per_rx'][rx][metric]-summaries[f'B0_{flow}']['per_rx'][rx][metric] for rx in receivers]) for flow in STREAMS}
        mean = (delta['S00']+delta['S11'])/2
        difference = delta['S11']-delta['S00']
        per_flow = {flow: dict(estimate=float(values.mean()), paired_rx_bootstrap_95_interval=interval(values[draws].mean(1)), per_rx=dict(zip(receivers, values.tolist()))) for flow, values in delta.items()}
        ci = interval(mean[draws].mean(1))
        signs_agree = per_flow['S00']['estimate']*per_flow['S11']['estimate'] > 0
        all_cross = all(abs(per_flow[flow]['estimate']) >= SCALES[metric] for flow in STREAMS) and abs(float(mean.mean())) >= SCALES[metric]
        effects[metric] = dict(streams=per_flow, average=dict(estimate=float(mean.mean()), paired_rx_bootstrap_95_interval=ci,
                                per_rx=dict(zip(receivers, mean.tolist())), positive_rx=int((mean>0).sum()), negative_rx=int((mean<0).sum()), zero_rx=int((mean==0).sum())),
                               between_stream_difference=dict(estimate=float(difference.mean()), paired_rx_bootstrap_95_interval=interval(difference[draws].mean(1))),
                               same_direction=signs_agree, practical_scale=SCALES[metric], both_streams_and_average_cross_scale=all_cross,
                               consistent_finite_effect_support=signs_agree and all_cross and (ci[0]>0 or ci[1]<0))
    for flow, pair in pairs.items():
        pair['paired_rx_bootstrap_95_intervals'] = {}
        for metric in ('disagreement', 'accuracy_delta', 'macro_f1_delta', 'ce_delta', 'corrected_fraction', 'new_error_fraction'):
            values = np.asarray([pair['per_rx'][rx][metric] for rx in receivers])
            pair['paired_rx_bootstrap_95_intervals'][metric] = interval(values[draws].mean(1))
    return dict(runs=summaries, composition_effects=effects, paired_changes=pairs, local_risk_consistency=local_risk_consistency(pairs),
                uncertainty=dict(unit='paired observed RX', receivers=receivers, replicates=replicates, seed=seed,
                                 interpretation='conditional resampling of fixed historical RX; two streams fixed, no RX or seed population claim'),
                direction='composition1 minus composition0')


def verified_paths(stage):
    def read(name): return json.loads((stage/name).read_text(encoding='utf-8'))
    reg, seal, manifest = read('PRE_REGISTRATION.json'), read('STAGE_B_SEAL.json'), read('EVALUATION_MANIFEST.json')
    if seal['status'] != 'SEALED_STAGE_B' or set(seal['runs']) != set(RUNS) or not all(seal['pairing_checks'].values()):
        raise ValueError('All four runs must pass the pairing seal')
    if manifest['stage_b_seal_sha256'] != M.file_sha256(stage/'STAGE_B_SEAL.json'):
        raise ValueError('Evaluation seal hash differs')
    digest = M.file_sha256(stage/'PRE_REGISTRATION.json')
    if seal['registration_sha256'] != digest or manifest['registration_sha256'] != digest:
        raise ValueError('Registration hash differs')
    expected = {f'{run}/{ep}_{split}.npz' for run in RUNS for ep in ENDPOINTS for split in ('val', 'held')}
    if set(manifest['files']) != expected:
        raise ValueError('Require all 16 predictions before analysis')
    artifacts = stage/'artifacts'
    for run in RUNS:
        complete_path = artifacts/run/'COMPLETE.json'
        if M.file_sha256(complete_path) != seal['runs'][run]['complete_sha256']:
            raise ValueError('Run completion record changed')
        complete = json.loads(complete_path.read_text(encoding='utf-8'))
        for relative, expected_sha in complete['files_sha256'].items():
            path = artifacts/run/relative
            if not path.resolve().is_relative_to((artifacts/run).resolve()) or M.file_sha256(path) != expected_sha:
                raise ValueError('Sealed training artifact changed')
    result = {}
    for relative, expected_sha in manifest['files'].items():
        path = artifacts/relative
        if M.file_sha256(path) != expected_sha:
            raise ValueError('Prediction hash differs')
        result[relative] = path
    return result


def render_markdown(results):
    lines = ['# 阶段 B：两个特定源 RX 集合的替换结果', '',
             'selected 为主要端点；epoch10 为辅助端点。差值始终为 composition1 − composition0，Accuracy/Macro-F1 为0–1比例，CE为nat。', '',
             '只有两个固定随机流与六个历史探索 held RX；不比较 A/B 绝对性能，不推断 RX 总体效应或硬件属性。', '']
    for split in ('val', 'held'):
        lines += [f'## {split}', '']
        for endpoint in ENDPOINTS:
            r = results[split][endpoint]
            lines += [f'### {endpoint}（{"主要" if endpoint == "selected" else "辅助"}）', '',
                      '| 运行 | RX等权 Accuracy | Macro-F1 | CE |', '|---|---:|---:|---:|']
            for run in RUNS:
                m = r['runs'][run]['rx_equal']
                lines.append(f"| {run} | {m['accuracy']:.6f} | {m['macro_f1']:.6f} | {m['ce']:.6f} |")
            lines += ['', '| 指标 | ΔS00 | ΔS11 | 平均差 | 平均差条件95%区间 | 一致有限效应支持 |', '|---|---:|---:|---:|---|---|']
            for metric, effect in r['composition_effects'].items():
                mean = effect['average']; lo, hi = mean['paired_rx_bootstrap_95_interval']
                lines.append(f"| {metric} | {effect['streams']['S00']['estimate']:.6f} | {effect['streams']['S11']['estimate']:.6f} | {mean['estimate']:.6f} | [{lo:.6f}, {hi:.6f}] | {'是' if effect['consistent_finite_effect_support'] else '否'} |")
            lines += ['', '| 随机流 | RX等权预测分歧 | 纠正包数 | 新增错误包数 | 均错而改类包数 |', '|---|---:|---:|---:|---:|']
            for flow, pair in r['paired_changes'].items():
                p = pair['packet_weighted']
                lines.append(f"| {flow} | {pair['rx_equal']['disagreement']:.6f} | {p['corrected_errors']} | {p['new_errors']} | {p['both_wrong_changed_class']} |")
            local = r['local_risk_consistency']
            lines += ['', f"预定局部风险筛查：{local['repeated_local_groups']}/{local['total_groups']} 个 TX×RX 组在两流同方向且各变化≥5pp、支持≥50；composition1 风险较低 {local['composition1_lower_risk']} 组、较高 {local['composition1_higher_risk']} 组。全部组同时列于完整 JSON，不能将该筛查称为单组总体显著性。", '']
    lines += ['配对 RX cluster bootstrap：10000次，seed22402。区间仅描述固定 RX 上的条件重采样，不是新 RX 或 seed 分布推断。selected/epoch10 不事后择优。', '',
              '完整支持、最差组、逐RX差异、margin及逐包诊断摘要见 ANALYSIS.json.gz。4次拟合完成后停止归因训练扩张；标准方法和任何新机制另行登记。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=HERE)
    args = parser.parse_args()
    stage = args.stage
    if (stage/'ANALYSIS.json').exists() or (stage/'ANALYSIS.json.gz').exists():
        raise FileExistsError('Analysis exists; do not overwrite')
    paths = verified_paths(stage)
    result, diagnostics = {}, {}
    for split in ('val', 'held'):
        result[split] = {}
        for endpoint in ENDPOINTS:
            predictions = {run: M.load_prediction(paths[f'{run}/{endpoint}_{split}.npz']) for run in RUNS}
            result[split][endpoint] = analyze_endpoint(predictions)
            for run in RUNS:
                path = stage/'artifacts/analysis/packet_diagnostics'/f'{run}_{endpoint}_{split}.npz'
                diagnostics[f'{run}_{endpoint}_{split}'] = M.export_packet_diagnostics(path, predictions[run])
    result['artifacts'] = dict(raw_prediction_manifest_sha256=M.file_sha256(stage/'EVALUATION_MANIFEST.json'), packet_diagnostics=diagnostics)
    raw = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    (stage/'ANALYSIS.json').write_bytes(raw)
    archive = gzip.compress(raw, compresslevel=9, mtime=0)
    assert gzip.decompress(archive) == raw
    (stage/'ANALYSIS.json.gz').write_bytes(archive)
    (stage/'ARTIFACT_ARCHIVE.json').write_text(json.dumps(dict(archive_sha256=hashlib.sha256(archive).hexdigest(), uncompressed_sha256=hashlib.sha256(raw).hexdigest(), archive_bytes=len(archive), uncompressed_bytes=len(raw), lossless_roundtrip_verified=True), indent=2)+'\n', encoding='utf-8', newline='\n')
    (stage/'RESULTS.zh-CN.md').write_text(render_markdown(result), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
