"""Build reviewable phase-one outputs. This does not register or launch training."""
import csv
import argparse
import json
import subprocess
from pathlib import Path
from s24_common import HERE, ROOT, P, PROTOCOL_PATH, sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='Verify delivered artifacts without opening IQ or model files')
    args = parser.parse_args()
    if args.check:
        manifest = json.loads((HERE/'ARTIFACTS.json').read_text(encoding='utf-8'))
        for name, entry in manifest['artifacts'].items():
            assert sha((HERE/name).read_bytes()) == entry['sha256'], name
        for name in ('native_full.csv','strict_ib_full.csv'):
            with (HERE/name).open(encoding='utf-8',newline='') as f:
                rows = list(csv.DictReader(f))
            assert len(rows)==18 and all(r['status']=='NOT_TRAINED' and r['acc_rx_equal']=='' for r in rows)
        with (HERE/'PACKETS.csv').open(encoding='utf-8',newline='') as f:
            packets = list(csv.DictReader(f))
        assert len(packets)==20000 and len({p['packet_id'] for p in packets})==20000
        for fold in (0,1):
            current = [p for p in packets if int(p['fold'])==fold]
            assert sum(p['split']=='train' for p in current)==8000
            assert sum(p['split']=='source_val' for p in current)==2000
        for access in json.loads((HERE/'ACCESS_AUDIT.json').read_text(encoding='utf-8'))['accesses']:
            source_path = next(f['source'] for f in P['folds'] if f['fold']==access['fold'])
            assert access['rx'] in source_path and access['fields']==['label','data']
        plan = json.loads((HERE/'DRAFT_PLAN.json').read_text(encoding='utf-8'))
        assert plan['maximum_authorized_fits']==0 and plan['status']=='DRAFT_NOT_FROZEN'
        print('Artifact hashes, 20,000 packet identities, permissions and empty result tables verified.')
        return
    data = json.loads((HERE/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    frontend = json.loads((HERE/'FRONTEND_AUDIT.json').read_text(encoding='utf-8'))
    history = json.loads((HERE/'HISTORY_AUDIT.json').read_text(encoding='utf-8'))
    assert data['passed'] and frontend['passed'] and history['passed']
    blockers = [
        dict(method='shen_tmc2024', item='TF-to-port weight transfer parity unverified; fix RX validation indexing and freeze source-only selection/augmentation rule'),
        dict(method='asknet_lora', item='Default 8191-mode full-model GPU forward/backward and batch budget unverified; no silent cutoff reduction'),
        dict(method='ib_rff_shen_adapted', item='Official LoRa recipe absent; freeze and audit seq_len/patch/alpha and source-only two-stage initialization'),
        dict(method='shen_tifs2022', item='Triplet/channel-augmentation/gallery closed-set adaptation not integrated; no auxiliary TX or target enrollment'),
    ]
    # Draft only: no command path or launcher consumes these jobs.
    plan = dict(stage='S24', status='DRAFT_NOT_FROZEN', maximum_authorized_fits=0,
                phase1_completed=True, all_method_training_gates_passed=False,
                protocol_sha256=sha(PROTOCOL_PATH.read_bytes()), base_git='e041f09',
                views=['native','strict_ib'], folds=P['folds'], seeds=[201,202,203],
                per_source_rx_tx=dict(train=160, source_val=40),
                candidate_primary_methods=['shen_tmc2024','asknet_lora','ib_rff_shen_adapted'],
                candidate_secondary_methods=['shen_tifs2022','shen_tifs2023','ma_tifs2025','ricai2024_gmlp'],
                proposed_primary_final_checkpoints=36,
                proposed_primary_fit_upper_bound=48,
                proposed_fit_count_explanation='2 views x 2 folds x 3 seeds x (1 DANN + 1 ASKNet + 2 IB stages), conditional on gates, configuration freeze and authorization; no hyperparameter search included',
                frozen_hstem_reuse=dict(checkpoints=6, params=133690, source='S13 A1_mha / S23 H0',
                                        reuse_same_result_in_both_views=True, new_fits=0),
                checkpoint_selection='source_val only; exact per-method scalar/loss/noise/restore rules must be frozen before fits',
                source_only_search='No search authorized; IB alpha/patch remain unresolved. Any search needs explicit source-only budget and separate registration.',
                evaluation=dict(dev_val='report only, after checkpoint/configuration seal; no adaptation',
                                confirm='historically exposed; excluded from current access, optional report only after separate frozen decision',
                                blind='never open', primary='receiver-equal packet accuracy',
                                secondary=['receiver-equal macro-F1','worst receiver','all per-RX values','seed SD','paired fold/seed deltas']),
                costs=['parameter elements and real degrees of freedom','MACs with FFT/QR coverage disclosed',
                       'same-machine GPU p50/p95 and batch throughput','native preprocessing time','total stage1+stage2 training time'],
                pairing='Same raw packet identities/fold/seed; use original H0 data_seed where compatible. Different architectures and augmentation do not imply identical RNG streams.',
                failure_policy='Keep every attempt/error; no target-driven reruns, RX changes, alpha tuning or best-seed reporting',
                blockers=blockers)
    write_json(HERE/'DRAFT_PLAN.json', plan)
    for view in plan['views']:
        rows = []
        for method in plan['candidate_primary_methods']:
            for fold in (0,1):
                for seed in (201,202,203):
                    rows.append(dict(view=view,method=method,fold=fold,seed=seed,status='NOT_TRAINED',
                                     acc_rx_equal='',macro_f1_rx_equal='',worst_rx_acc='',params='',
                                     macs='',gpu_p50_ms='',preprocess_ms='',total_train_seconds='',checkpoint_sha256=''))
        with (HERE/f'{view}_full.csv').open('w',newline='',encoding='utf-8') as out:
            writer=csv.DictWriter(out,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    tracked = subprocess.check_output(['git','diff','--name-only','e041f09'],cwd=ROOT,text=True).splitlines()
    assert all(p.startswith('24-lora-external-audit/') for p in tracked), tracked
    write_json(HERE/'STATUS.json',dict(stage='S24',phase='phase1',
                status='AUDIT_DELIVERED_WITH_TRAINING_GATES_PENDING',
                completed_engineering_checks=len(frontend['checks']), source_packets=data['records'],
                fitted_models=0, optimizer_steps=0, new_performance_scores=0,
                blind_reads=0, confirm_reads=0, changed_tracked_historical_files=[],
                first_stage_permission_and_data_checks_passed=True,
                all_method_training_gates_passed=False,training_plan_frozen=False,
                blockers=blockers))
    # Preserve byte hashes across Windows/Linux and Git checkout, without touching upstream bytes.
    for p in HERE.iterdir():
        if p.is_file() and p.suffix in ('.json','.csv'):
            data = p.read_bytes()
            if b'\r\n' in data: p.write_bytes(data.replace(b'\r\n',b'\n'))
    artifacts = {}
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name!='ARTIFACTS.json' and not p.name.startswith('.'):
            artifacts[p.name]=dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size)
    write_json(HERE/'ARTIFACTS.json',dict(artifacts=artifacts, raw_waveforms_in_git=False,
                upstream_source_in_git=False, fits=0))
    print('Phase-one package generated; training gates pending; authorized fits=0.')


if __name__ == '__main__': main()
