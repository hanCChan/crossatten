"""Stage B controls. Historical A modules supply pure, frozen training helpers."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
spec = importlib.util.spec_from_file_location('frozen_stage_a_runtime', HERE.parent/'controlled_training_a/runner.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)  # Sets deterministic environment before torch import.
import numpy as np
import torch
import torch.nn.functional as F

read, write, frozen, now, file_sha = A.read, A.write, A.frozen, A.now, A.file_sha
semantic_hash, compare_objects = A.semantic_hash, A.compare_objects
setup, build_model, cpu_tree = A.setup, A.build_model, A.cpu_tree
checkpoint, snapshot_digests = A.checkpoint, A.snapshot_digests
predict, save_predictions = A.predict, A.save_predictions
require_committed, make_keys = A.require_committed, A.make_keys
EXPECTED = {
    'B0_S00': (0, 'S00', 22101, 22201, 22301),
    'B1_S00': (1, 'S00', 22101, 22201, 22301),
    'B0_S11': (0, 'S11', 22102, 22202, 22301),
    'B1_S11': (1, 'S11', 22102, 22202, 22301),
}


def validate_runs(reg):
    if reg.get('stage') != 'B' or reg.get('schema_version') != 1 or reg.get('maximum_new_fits') != 4:
        raise ValueError('Only the four registered Stage B attempts are allowed')
    if reg.get('run_sequence') != list(EXPECTED) or [j['id'] for j in reg['runs']] != list(EXPECTED):
        raise ValueError('Unregistered run sequence')
    for job in reg['runs']:
        actual = tuple(job[k] for k in ('composition', 'stream', 'initialization_seed', 'order_seed', 'stochastic_seed'))
        if actual != EXPECTED[job['id']]:
            raise ValueError('Changed paired random streams')
    fit, held, target = (set(reg[k]) for k in ('fit_rx', 'held_rx', 'target_rx'))
    if (len(fit), len(held), len(target)) != (18, 6, 8) or fit & held or (fit | held) & target:
        raise ValueError('Invalid fold2 roles')


def validate_registration(path):
    A.require_committed(path)
    reg = read(path)
    validate_runs(reg)
    for group in ('code_sha256', 'evidence_sha256'):
        for relative, digest in reg[group].items():
            source = (ROOT/relative).resolve()
            source.relative_to(ROOT)
            if file_sha(source) != digest:
                raise ValueError(f'Frozen {group} changed: {relative}')
            A.require_committed(source)
    gate = read(ROOT/'research/controlled_training_a/REPLAY_GATE.json')
    if gate['status'] != 'PASS_EXACT_REPLAY' or not all(c['exact'] for c in gate['checks'].values()):
        raise ValueError('Stage A engineering replay is a prerequisite')
    return reg


def artifact_root(reg):
    path = (ROOT/reg['output_dir']).resolve()
    path.relative_to(HERE)
    return path


def access(reg, item, action, run=None):
    import json
    with (HERE/'ACCESS_LOG.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(dict(time=now(), action=action, run=run, path=str(item['path']),
                                     sha256=item['sha256']), ensure_ascii=False)+'\n')


def verify_input(reg, name, item, run=None):
    access(reg, item, 'hash_registered_'+name, run)
    path = Path(item['path'])
    if file_sha(path) != item['sha256']:
        raise ValueError('Registered input changed: '+name)
    return path


def tuple_keys(y, rx, day, names, hashes):
    counters, result = {}, []
    for r, d in zip(rx, day):
        name, d = names[int(r)], int(d)
        ordinal = counters.get((name, d), 0)
        counters[name, d] = ordinal+1
        result.append((name, d, hashes[f'{name}|{d}'], ordinal))
    return result


def build_panels(y, rx, day, train, val, names, cell_hashes, plan):
    """Exactly the historical tuple-repr ranking, without IQ or prediction access."""
    keys = tuple_keys(y, rx, day, names, cell_hashes)
    rx_names = np.asarray([names[int(r)] for r in rx])
    core = plan['arms'][0]['shared_core_rx']
    extensions = [arm['extension_rx'] for arm in plan['arms']]
    if core != plan['arms'][1]['shared_core_rx'] or set(extensions[0]) & set(extensions[1]):
        raise ValueError('Composition sets overlap or have different cores')
    common = sorted([int(i) for i in train if rx_names[i] in core], key=lambda i: keys[i])
    validation = np.asarray(sorted([int(i) for i in val if rx_names[i] in core], key=lambda i: keys[i]), dtype=np.int32)
    selected, caps = [[], []], []
    rank = lambda i: hashlib.sha256(('source-panel-v1|'+repr(keys[i])).encode()).hexdigest()
    for tx in range(10):
        for date in (1, 2):
            available = [[int(i) for i in train if rx_names[i] in ext and y[i] == tx and day[i] == date] for ext in extensions]
            cap = min(map(len, available))
            caps.append(dict(tx=tx, day=date, extension_counts=list(map(len, available)), retained_per_extension=cap))
            for arm in range(2):
                selected[arm].extend(sorted(available[arm], key=rank)[:cap])
    panels = []
    for arm in range(2):
        extension = sorted(selected[arm], key=lambda i: (int(y[i]), int(day[i]), keys[i]))
        ids = np.asarray(common+extension, dtype=np.int64)
        expected = plan['arms'][arm]
        if len(ids) != expected['train_packets'] or len(validation) != expected['validation_packets']:
            raise ValueError('Panel sample budget differs')
        if len(common) != expected['fixed_core_train_packets'] or len(extension) != expected['extension_packets']:
            raise ValueError('Core/extension budget differs')
        if hashlib.sha256(ids.tobytes()).hexdigest() != expected['train_indices_sha256']:
            raise ValueError('Historical panel train index SHA differs')
        if hashlib.sha256(validation.tobytes()).hexdigest() != expected['validation_indices_sha256']:
            raise ValueError('Historical panel validation index SHA differs')
        if len(np.unique(ids)) != len(ids) or set(ids) & set(validation):
            raise ValueError('Duplicate or overlapping panel members')
        if not set(rx_names[ids]) <= set(core+extensions[arm]):
            raise ValueError('RX outside the registered arm')
        panels.append(ids)
    if caps != plan['extension_caps'] or not np.array_equal(y[panels[0]], y[panels[1]]) or not np.array_equal(day[panels[0]], day[panels[1]]):
        raise ValueError('Class/day slot matching or caps differ')
    if not np.array_equal(panels[0][:len(common)], panels[1][:len(common)]):
        raise ValueError('Core positions differ')
    return panels, validation


def load_fit(reg, run):
    required = {'fit_x', 'fit_y', 'fit_rx', 'fit_day', 'train_ids', 'val_ids', 'fit_manifest'}
    if set(reg['inputs']) != required:
        raise ValueError('Unexpected input whitelist')
    paths = {name: verify_input(reg, name, item, run) for name, item in reg['inputs'].items()}
    manifest = read(paths['fit_manifest'])
    if manifest['rx_names'] != reg['rx_names'] or manifest['fit_rx'] != reg['fit_rx'] or manifest['held_rx'] != reg['held_rx']:
        raise ValueError('Source manifest roles differ')
    pool = {}
    for name in ('fit_y', 'fit_rx', 'fit_day', 'train_ids', 'val_ids'):
        access(reg, reg['inputs'][name], 'load_registered_'+name, run)
        pool[name] = np.load(paths[name], allow_pickle=False)
    y, rx, day = [pool['fit_'+name] for name in ('y', 'rx', 'day')]
    if y.shape != (34668,) or rx.shape != y.shape or day.shape != y.shape or set(day) != {1, 2}:
        raise ValueError('Source metadata shape/date differs')
    if set(reg['rx_names'][int(i)] for i in np.unique(rx)) != set(reg['fit_rx']) or np.min(y) < 0 or np.max(y) >= 10:
        raise ValueError('Source metadata outside registered RX/TX')
    if not np.array_equal(np.sort(np.r_[pool['train_ids'], pool['val_ids']]), np.arange(len(y))):
        raise ValueError('Original split is not a disjoint partition')
    plan = read(ROOT/reg['panel_plan'])
    panels, validation = build_panels(y, rx, day, pool['train_ids'], pool['val_ids'], reg['rx_names'], reg['source_cell_sha256'], plan)
    job = next(j for j in reg['runs'] if j['id'] == run)
    train = panels[job['composition']]
    keys = A.make_keys(y, rx, day, reg['rx_names'], reg['source_cell_sha256'])
    labels = np.stack((y[train], day[train]), 1)
    evidence = dict(composition=job['composition'], arm_rx=plan['arms'][job['composition']]['shared_core_rx']+plan['arms'][job['composition']]['extension_rx'],
                    train_packets=len(train), validation_packets=len(validation), common_core_packets=8149,
                    train_indices_sha256=hashlib.sha256(train.tobytes()).hexdigest(),
                    validation_indices_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
                    label_slots_sha256=semantic_hash(labels), core_keys_sha256=semantic_hash(keys[train[:8149]]),
                    validation_keys_sha256=semantic_hash(keys[validation]),
                    train_class_day_slots_matched=True, original_panel_index_hashes_verified=True)
    out = artifact_root(reg)/run
    write(out/'PANEL.json', evidence)
    access(reg, reg['inputs']['fit_x'], 'load_registered_fit_x_panel_rows_only', run)
    x = np.load(paths['fit_x'], mmap_mode='r', allow_pickle=False)
    if x.shape != (34668, 2, 256) or x.dtype != np.float32:
        raise ValueError('Source IQ shape/dtype differs')
    return {part: dict(x=np.array(x[ids], copy=True), y=np.asarray(y[ids], dtype=np.int64),
                       rx=np.asarray(rx[ids], dtype=np.int64), day=np.asarray(day[ids], dtype=np.int64),
                       keys=keys[ids], pool_ids=np.asarray(ids, dtype=np.int64))
            for part, ids in (('train', train), ('val', validation))}


def check_sequence(reg, reg_path, run):
    index = reg['run_sequence'].index(run)
    for preceding in reg['run_sequence'][:index]:
        verify_complete(reg, reg_path, preceding)


def panel_preflight(reg, reg_path):
    values = {}
    for name in ('fit_y', 'fit_rx', 'fit_day', 'train_ids', 'val_ids'):
        path = verify_input(reg, name, reg['inputs'][name], 'panel_preflight')
        access(reg, reg['inputs'][name], 'load_registered_'+name, 'panel_preflight')
        values[name] = np.load(path, allow_pickle=False)
    plan = read(ROOT/reg['panel_plan'])
    panels, val = build_panels(values['fit_y'], values['fit_rx'], values['fit_day'], values['train_ids'], values['val_ids'], reg['rx_names'], reg['source_cell_sha256'], plan)
    result = dict(status='PASS_EXACT_PANEL_INDICES', registration_sha256=file_sha(reg_path),
                  train_packets=list(map(len, panels)), validation_packets=len(val),
                  train_index_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in panels],
                  validation_index_sha256=hashlib.sha256(val.tobytes()).hexdigest(),
                  iq_payload_loads=0, model_forwards=0, new_fits=0, time=now())
    frozen(HERE/'PANEL_PREFLIGHT.json', result)
    return result


def prepare_order(train_ids, keys, generator):
    permutation = torch.randperm(len(train_ids), generator=generator, device='cuda')
    positions = permutation.cpu().numpy()
    digest = hashlib.sha256(('\n'.join(keys[positions].tolist())+'\n').encode()).hexdigest()
    return permutation, digest


# The train/verify_complete/load_held/evaluate functions below preserve A's
# training loop and inference rules. Only the B inputs, gate and evidence change.


def train(reg, reg_path, run):
    job = next(j for j in reg["runs"] if j["id"] == run)
    check_sequence(reg, reg_path, run)
    A.require_committed(HERE/'PANEL_PREFLIGHT.json')
    preflight = read(HERE/'PANEL_PREFLIGHT.json')
    if preflight['status'] != 'PASS_EXACT_PANEL_INDICES' or preflight['registration_sha256'] != file_sha(reg_path):
        raise ValueError('Exact historical panel indices must pass before any fit')
    out = artifact_root(reg) / run
    if out.exists():
        raise RuntimeError("Scientific attempt already exists; no overwrite or automatic retry")
    out.mkdir(parents=True)
    write(out / "ATTEMPT.json", dict(time=now(), status="RUNNING", job=job,
                                     registration_sha256=file_sha(reg_path)))
    started = time.perf_counter()
    try:
        setup(job["initialization_seed"])
        data = load_fit(reg, run)
        tr, va = data["train"], data["val"]
        model = build_model(job["initialization_seed"])
        initial_model = cpu_tree(model.state_dict())
        if semantic_hash(initial_model) != reg['initial_model_hashes'][str(job['initialization_seed'])]:
            raise ValueError('Initialization differs from the sealed A level')
        model.load_state_dict(initial_model)
        # The model RNG starts at the same position irrespective of construction.
        torch.manual_seed(job["stochastic_seed"])
        np.random.seed(job["stochastic_seed"])
        random.seed(job["stochastic_seed"])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.1, patience=5, min_lr=1e-6)
        order = torch.Generator(device="cuda").manual_seed(job["order_seed"])
        write(out / "CONFIG.json", dict(
            job=job, registration_sha256=file_sha(reg_path),
            torch_version=torch.__version__, numpy_version=np.__version__,
            python_version=sys.version, cuda_version=torch.version.cuda,
            cudnn_version=torch.backends.cudnn.version(),
            device_name=torch.cuda.get_device_name(),
            device_capability=list(torch.cuda.get_device_capability()),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
            cpu_threads=torch.get_num_threads(),
            train_packets=len(tr["y"]), val_packets=len(va["y"]),
            train_keys_sha256=semantic_hash(tr["keys"]),
            val_keys_sha256=semantic_hash(va["keys"]),
            training=reg["training"], panel=read(out / "PANEL.json")))
        initial = checkpoint(model, opt, scheduler, order, 0, float("inf"), 0, 0, 0)
        torch.save(initial, out / "initial.pt")
        write(out / "INITIAL.json", snapshot_digests(initial))
        np.savez_compressed(out / "SAMPLE_KEYS.npz", train_keys=tr["keys"], val_keys=va["keys"],
                            train_pool_ids=tr["pool_ids"], val_pool_ids=va["pool_ids"])
        (out / "orders").mkdir()
        (out / "epoch_states").mkdir()
        tensors = {part: {k: torch.as_tensor(d[k], device="cuda") for k in ("x", "y", "rx")}
                   for part, d in data.items()}
        best, best_epoch, wait, updates = float("inf"), 0, 0, 0
        history = []
        for epoch in range(1, 41):
            model.train()
            loss_sum, correct, seen = 0.0, 0, 0
            used_lr = opt.param_groups[0]["lr"]
            permutation, order_sha = prepare_order(tr["pool_ids"], tr["keys"], order)
            positions = permutation.cpu().numpy()
            np.save(out / "orders" / f"positions_{epoch:02d}.npy", positions)
            ordered_ids = tr["pool_ids"][positions]
            if not np.array_equal(np.sort(ordered_ids), np.sort(tr["pool_ids"])):
                raise ValueError("Epoch order does not cover every training key exactly once")
            np.save(out / "orders" / f"epoch_{epoch:02d}.npy", ordered_ids)
            for ids in permutation.split(128):
                x, y = tensors["train"]["x"][ids], tensors["train"]["y"][ids]
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training CE")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                updates += 1
                loss_sum += float(loss.detach()) * len(ids)
                correct += int((logits.argmax(1) == y).sum())
                seen += len(ids)
            model.eval()
            with torch.no_grad():
                lv = torch.cat([model(tensors["val"]["x"][i:i + 128])
                                for i in range(0, len(va["y"]), 128)])
            labels, receivers = tensors["val"]["y"], tensors["val"]["rx"]
            ce = float(torch.stack([F.cross_entropy(lv[receivers == r], labels[receivers == r])
                                    for r in torch.unique(receivers)]).mean())
            if not math.isfinite(ce):
                raise ValueError("Nonfinite RX-equal source validation CE")
            improved = ce < best
            if improved:
                best, best_epoch, wait = ce, epoch, 0
            else:
                wait += 1
            scheduler.step(ce)
            state = checkpoint(model, opt, scheduler, order, epoch, best, best_epoch, wait, updates)
            torch.save(state, out / "epoch_states" / f"epoch_{epoch:02d}.pt")
            row = dict(epoch=epoch, train_ce=loss_sum / seen, train_accuracy=correct / seen,
                       val_rx_ce=ce, val_packet_accuracy=float((lv.argmax(1) == labels).float().mean()),
                       lr_used=used_lr, lr_next=opt.param_groups[0]["lr"], best_epoch=best_epoch,
                       wait=wait, updates=updates, order_sha256=order_sha,
                       relative_order_sha256=semantic_hash(positions),
                       class_day_order_sha256=semantic_hash(np.stack((tr['y'][positions], tr['day'][positions]), 1)),
                       core_order_keys_sha256=semantic_hash(tr['keys'][positions[positions < 8149]]),
                       order_pool_ids_sha256=semantic_hash(ordered_ids),
                       val_logits_sha256=semantic_hash(lv.detach().cpu()), **snapshot_digests(state))
            history.append(row)
            if improved:
                torch.save(state, out / "selected.pt")
                save_predictions(out / "selected_val.npz", lv.cpu().numpy(), va, reg)
            if epoch == 10:
                torch.save(state, out / "epoch10.pt")
                save_predictions(out / "epoch10_val.npz", lv.cpu().numpy(), va, reg)
            write(out / "HISTORY.json", history)
            write(out / "LIVE.json", dict(time=now(), seconds=time.perf_counter() - started, **row))
            print(f"{run} epoch {epoch:02d} RXvalCE={ce:.8f} selected={best_epoch} lr={used_lr:g}", flush=True)
            if wait >= 10:
                break
        if not (out / "epoch10.pt").exists():
            raise RuntimeError("Predeclared epoch10 endpoint missing")
        torch.save(state, out / "final.pt")
        # Re-run inference through the inference API and verify endpoint output.
        prediction_replay = {}
        for endpoint in ("selected", "epoch10"):
            saved = torch.load(out / f"{endpoint}.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            logits = predict(model, va["x"])
            with np.load(out / f"{endpoint}_val.npz", allow_pickle=False) as z:
                prediction_replay[endpoint] = compare_objects(z["logits"], logits)
            if not prediction_replay[endpoint]["exact"]:
                raise RuntimeError("Saved endpoint validation inference did not reproduce exactly")
        checkpoint_files = ("initial.pt", "selected.pt", "epoch10.pt", "final.pt")
        complete = dict(status="COMPLETE", run=run, registration_sha256=file_sha(reg_path),
                        selected_epoch=best_epoch, epochs=len(history), updates=updates,
                        best_val_rx_ce=best, active_parameters=sum(p.numel() for p in model.parameters()),
                        endpoints={name[:-3]: snapshot_digests(torch.load(out / name, map_location="cpu", weights_only=False))
                                   for name in checkpoint_files},
                        files_sha256={p.relative_to(out).as_posix(): file_sha(p)
                                      for p in sorted(out.rglob("*")) if p.is_file()},
                        val_inference_replay=prediction_replay, wall_seconds=time.perf_counter() - started,
                        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
                        finished_at=now())
        write(out / "COMPLETE.json", complete)
        return complete
    except Exception:
        write(out / "FAILURE.json", dict(time=now(), status="FAILED", automatic_retry=False,
                                         traceback=traceback.format_exc()))
        raise


def verify_complete(reg, reg_path, run):
    out = artifact_root(reg) / run
    complete = read(out / "COMPLETE.json")
    if complete["status"] != "COMPLETE" or complete["registration_sha256"] != file_sha(reg_path):
        raise ValueError("Run registration differs")
    for name, digest in complete["files_sha256"].items():
        if file_sha(out / name) != digest:
            raise ValueError(f"Completed artifact changed: {run}/{name}")
    return out, complete


def seal(reg, reg_path):
    evidence, all_runs = {}, {}
    for job in reg['runs']:
        run = job['id']
        out, item = verify_complete(reg, reg_path, run)
        evidence[run] = dict(complete_sha256=file_sha(out/'COMPLETE.json'),
                             selected_sha256=file_sha(out/'selected.pt'), epoch10_sha256=file_sha(out/'epoch10.pt'),
                             endpoints=item['endpoints'])
        all_runs[run] = dict(CONFIG=read(out/'CONFIG.json'), HISTORY=read(out/'HISTORY.json'), COMPLETE=item)
    checks = {}
    for stream in ('S00', 'S11'):
        left, right = (all_runs[f'B{i}_{stream}'] for i in (0, 1))
        checks[stream+'_initial_full_state_equal'] = left['COMPLETE']['endpoints']['initial']['state_sha256'] == right['COMPLETE']['endpoints']['initial']['state_sha256']
        for field in ('label_slots_sha256', 'core_keys_sha256', 'validation_keys_sha256'):
            checks[stream+'_'+field] = left['CONFIG']['panel'][field] == right['CONFIG']['panel'][field]
        shared = min(len(left['HISTORY']), len(right['HISTORY']))
        for field in ('relative_order_sha256', 'class_day_order_sha256', 'core_order_keys_sha256'):
            checks[stream+'_'+field] = all(left['HISTORY'][i][field] == right['HISTORY'][i][field] for i in range(shared))
        checks[stream+'_validation_keys_equal'] = left['CONFIG']['val_keys_sha256'] == right['CONFIG']['val_keys_sha256']
    checks['all_validation_keys_equal'] = len({x['CONFIG']['val_keys_sha256'] for x in all_runs.values()}) == 1
    checks['distinct_composition_members'] = all_runs['B0_S00']['CONFIG']['panel']['train_indices_sha256'] != all_runs['B1_S00']['CONFIG']['panel']['train_indices_sha256']
    if not all(checks.values()):
        raise RuntimeError('Stage B pairing controls failed; do not access held inputs')
    frozen(HERE/'RUN_EVIDENCE.json', all_runs)
    result = dict(status='SEALED_STAGE_B', registration_sha256=file_sha(reg_path),
                  engineering_replay_inherited_from_A=True, new_B_replay_fits=0,
                  runs=evidence, pairing_checks=checks, held_input_opened_by_seal=False,
                  additional_attribution_fits_admitted=False, time=now())
    frozen(HERE/'STAGE_B_SEAL.json', result)
    return result


def load_held(reg):
    inputs = reg["held_inputs"]
    if {(x["rx"], x["day"]) for x in inputs} != {(rx, d) for rx in reg["held_rx"] for d in (1, 2)} or len(inputs) != 12:
        raise ValueError("Held whitelist must contain precisely the six held source RX × days1–2")
    parts = {key: [] for key in ("x", "y", "rx", "day")}
    for item in inputs:
        if item["sha256"] != reg["source_cell_sha256"][f"{item['rx']}|{item['day']}"]:
            raise ValueError("Held cell lineage differs")
        path = verify_input(reg, "held_cell", item)
        access(reg, item, "load_registered_held_cell_after_all_four_sealed")
        with np.load(path, allow_pickle=False) as z:
            x, y = z["x"], z["y"]
        if x.dtype != np.float32 or x.shape != (len(y), 2, 256):
            raise ValueError("Held source cell shape/dtype differs")
        parts["x"].append(x)
        parts["y"].append(np.asarray(y, dtype=np.int64))
        parts["rx"].append(np.full(len(y), reg["rx_names"].index(item["rx"]), dtype=np.int64))
        parts["day"].append(np.full(len(y), item["day"], dtype=np.int64))
    data = {key: np.concatenate(values) for key, values in parts.items()}
    data["keys"] = make_keys(data["y"], data["rx"], data["day"], reg["rx_names"], reg["source_cell_sha256"])
    data["pool_ids"] = np.arange(len(data["y"]), dtype=np.int64)
    return data


def evaluate(reg, reg_path):
    seal_path = HERE / "STAGE_B_SEAL.json"
    require_committed(seal_path)
    sealed = read(seal_path)
    if (sealed["status"] != "SEALED_STAGE_B" or sealed["registration_sha256"] != file_sha(reg_path)
            or set(sealed['runs']) != set(EXPECTED) or not all(sealed['pairing_checks'].values())):
        raise ValueError("All four registered fits must be sealed before held access")
    marker = HERE / "HELD_EVALUATION.json"
    if marker.exists() or (HERE / "HELD_EVALUATION_ATTEMPT.json").exists():
        raise RuntimeError("Held evaluation attempt already exists; no automatic retry")
    for job in reg["runs"]:
        out, _ = verify_complete(reg, reg_path, job["id"])
        if file_sha(out / "COMPLETE.json") != sealed["runs"][job["id"]]["complete_sha256"]:
            raise ValueError("Sealed run changed")
    write(HERE / "HELD_EVALUATION_ATTEMPT.json", dict(time=now(), status="RUNNING", seal_sha256=file_sha(seal_path)))
    setup(22301)
    data = load_held(reg)
    evidence = {}
    for job in reg["runs"]:
        run = job["id"]
        out = artifact_root(reg) / run
        model = build_model(job["initialization_seed"])
        evidence[run] = {}
        for endpoint in ("selected", "epoch10"):
            path = out / f"{endpoint}.pt"
            if file_sha(path) != sealed["runs"][run][endpoint + "_sha256"]:
                raise ValueError("Sealed endpoint changed")
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"])
            logits = predict(model, data["x"])
            predpath = out / f"{endpoint}_held.npz"
            save_predictions(predpath, logits, data, reg)
            evidence[run][endpoint] = dict(predictions_sha256=file_sha(predpath),
                                          logits_sha256=semantic_hash(logits), keys_sha256=semantic_hash(data["keys"]))
        del model
    frozen(marker, dict(status="COMPLETE", seal_sha256=file_sha(seal_path),
                        registration_sha256=file_sha(reg_path), runs=evidence,
                        packets=len(data["y"]), source_held_only=True, time=now()))
    predictions = {}
    for job in reg["runs"]:
        for endpoint in ("selected", "epoch10"):
            for scope in ("val", "held"):
                name = f"{job['id']}/{endpoint}_{scope}.npz"
                predictions[name] = file_sha(artifact_root(reg) / name)
    frozen(HERE / "EVALUATION_MANIFEST.json", dict(
        status="COMPLETE", stage_b_seal_sha256=file_sha(seal_path),
        registration_sha256=file_sha(reg_path), files=predictions))
    return read(marker)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=HERE/'PRE_REGISTRATION.json')
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--run', choices=list(EXPECTED), required=True)
    sub.add_parser('seal')
    sub.add_parser('evaluate')
    sub.add_parser('panel-check')
    args = parser.parse_args()
    reg = validate_registration(args.registration)
    if args.command == 'train':
        result = train(reg, args.registration, args.run)
    elif args.command == 'seal':
        result = seal(reg, args.registration)
    elif args.command == 'evaluate':
        result = evaluate(reg, args.registration)
    else:
        result = panel_preflight(reg, args.registration)
    print(json.dumps(dict(status=result['status'], command=args.command)))


if __name__ == '__main__':
    main()
