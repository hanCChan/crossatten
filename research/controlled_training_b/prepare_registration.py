"""Record metadata and raw-byte hashes; never decode IQ, labels or checkpoints."""
from __future__ import annotations
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(16 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    target = HERE/'PRE_REGISTRATION.json'
    if target.exists() or (HERE/'artifacts').exists():
        raise RuntimeError('Refusing to overwrite registration or existing attempts')
    base = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    assert base.startswith('7ff8986'), 'Freeze against the reviewed Stage A completion'
    previous = read(HERE.parent/'controlled_training_a/PRE_REGISTRATION.json')
    previous_seal = read(HERE.parent/'controlled_training_a/STAGE_A_SEAL.json')
    plan_path = ROOT/'research/training_instance_attribution/CONTROL_PANEL_PLAN.json'
    plan = read(plan_path)
    assert plan['arms'][0]['train_packets'] == plan['arms'][1]['train_packets'] == 15849
    assert plan['arms'][0]['validation_packets'] == plan['arms'][1]['validation_packets'] == 3551
    hashed = []
    for item in list(previous['inputs'].values())+previous['held_inputs']:
        path = Path(item['path'])
        assert sha(path) == item['sha256']
        hashed.append(str(path))
    code = dict(previous['code_sha256'])
    for relative, digest in code.items():
        assert sha(ROOT/relative) == digest, 'Historical registered code changed'
    for path in list(HERE.glob('*.py'))+[HERE/'PREREGISTRATION.zh-CN.md', ROOT/'tests/test_controlled_training_b.py']:
        code[path.relative_to(ROOT).as_posix()] = sha(path)
    evidence_paths = [plan_path]+[HERE.parent/'controlled_training_a'/name for name in ('PRE_REGISTRATION.json', 'REPLAY_GATE.json', 'STAGE_A_SEAL.json', 'DECISION.json', 'EXECUTION.json')]
    evidence = {path.relative_to(ROOT).as_posix(): sha(path) for path in evidence_paths}
    runs = [dict(id=f'B{arm}_{stream}', composition=arm, stream=stream, initialization_seed=init, order_seed=order, stochastic_seed=22301)
            for stream, init, order in (('S00', 22101, 22201), ('S11', 22102, 22202)) for arm in (0, 1)]
    record = dict(schema_version=1, stage='B', status='PREREGISTERED_BEFORE_NEW_FIT', base_commit=base,
                  registered_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                  maximum_new_fits=4, run_sequence=[j['id'] for j in runs], runs=runs,
                  panel_plan=plan_path.relative_to(ROOT).as_posix(),
                  initial_model_hashes={str(seed): previous_seal['runs'][name]['endpoints']['initial']['model_sha256'] for seed, name in ((22101, 'A00'), (22102, 'A11'))},
                  code_sha256=code, evidence_sha256=evidence,
                  output_dir='research/controlled_training_b/artifacts',
                  training=dict(previous['training'], selection='common_Core6_RX_equal_validation_CE_strict_less'),
                  endpoints=previous['endpoints'],
                  analysis=dict(bootstrap_replicates=10000, bootstrap_seed=22402, confidence_level=.95,
                                accuracy_scale=.01, macro_f1_scale=.01, ce_scale=.05, disagreement_scale=.05,
                                local_risk_scale=.05, local_minimum_support=50,
                                consistent_metric_support='same direction across two streams; both and mean cross scale; average conditional interval excludes zero',
                                primary_contrast='composition1 minus composition0 averaged over fixed S00/S11; report each stream'),
                  held_access_gate='ALL_FOUR_RUNS_AND_ENDPOINTS_SEALED_AND_COMMITTED',
                  historical_arrays_and_weights_to_decode=0, new_replay_fits=0,
                  stop_rule='Exactly four attempts maximum; no silent retries or added seeds/BN/SWA/EMA/new loss experiments.',
                  metadata_access_before_registration=dict(raw_file_hash_reads=hashed, iq_payload_loads=0, label_array_loads=0,
                                                           model_checkpoint_loads=0, model_forwards=0, fits=0,
                                                           note='Only text metadata and raw-byte file hashes; hash reads may cover full legal fit18 cache, no array decoding.'))
    for key in ('inputs', 'held_inputs', 'rx_names', 'fit_rx', 'held_rx', 'target_rx', 'source_cell_sha256'):
        record[key] = previous[key]
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(status=record['status'], registration_sha256=sha(target), code_files=len(code), new_fits=0, iq_payload_loads=0)))


if __name__ == '__main__':
    main()
