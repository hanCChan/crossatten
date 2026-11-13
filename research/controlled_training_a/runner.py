"""Registered Stage A training. Never uses the historical stage writers/loaders.

Initialisation, CUDA sample order and model RNG are independent streams. A replay
gate precedes the remaining three fits; held source input is inaccessible until
all five fits and both predeclared endpoints have been sealed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

# Set before importing torch / constructing any CUDA context.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXPECTED_RUNS = {
    "A00": (22101, 22201, 22301),
    "A00-replay": (22101, 22201, 22301),
    "A10": (22102, 22201, 22301),
    "A01": (22101, 22202, 22301),
    "A11": (22102, 22202, 22301),
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def frozen(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Refusing to overwrite frozen evidence: {path}")
    else:
        write(path, value)


def semantic_hash(obj):
    """Content digest includes names, types, shapes and exact bytes; no pickle IDs."""
    h = hashlib.sha256()

    def feed(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            h.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            feed(tuple(tensor.shape))
            h.update(tensor.numpy().tobytes())
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            h.update(b"ndarray\0" + array.dtype.str.encode() + b"\0")
            feed(tuple(array.shape))
            h.update(array.tobytes())
        elif isinstance(value, dict):
            h.update(b"dict\0")
            for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
                feed(key)
                feed(value[key])
            h.update(b"enddict\0")
        elif isinstance(value, (list, tuple)):
            h.update(type(value).__name__.encode() + b"\0")
            feed(len(value))
            for item in value:
                feed(item)
        elif isinstance(value, np.generic):
            feed(value.item())
        elif isinstance(value, bytes):
            h.update(b"bytes\0" + str(len(value)).encode() + b"\0" + value)
        else:
            h.update(type(value).__name__.encode() + b"\0" + repr(value).encode() + b"\0")

    feed(obj)
    return h.hexdigest()


def compare_objects(a, b):
    """Return exact equality and largest finite numeric difference recursively."""
    differences = []
    maximum = 0.0

    def compare(left, right, path):
        nonlocal maximum
        if torch.is_tensor(left):
            if not torch.is_tensor(right):
                differences.append(path + ":type")
                return
            left, right = left.detach().cpu().numpy(), right.detach().cpu().numpy()
        if isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray) or left.shape != right.shape or left.dtype != right.dtype:
                differences.append(path + ":shape_or_dtype")
                return
            if left.tobytes() != right.tobytes():
                differences.append(path)
                if left.size and np.issubdtype(left.dtype, np.number):
                    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
                    finite = delta[np.isfinite(delta)]
                    maximum = max(maximum, float(finite.max()) if finite.size else 0.0)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or set(left) != set(right):
                differences.append(path + ":keys")
                return
            for key in left:
                compare(left[key], right[key], path + "/" + str(key))
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                differences.append(path + ":sequence")
                return
            for index, (x, y) in enumerate(zip(left, right)):
                compare(x, y, path + "/" + str(index))
        elif type(left) is not type(right) or left != right:
            differences.append(path)
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                delta = abs(float(left) - float(right))
                if math.isfinite(delta):
                    maximum = max(maximum, delta)

    compare(a, b, "root")
    return dict(exact=not differences, max_abs_numeric_difference=maximum,
                differing_paths=differences[:200], differing_path_count=len(differences))


def validate_runs(reg):
    if reg.get("schema_version") != 1 or reg.get("stage") != "A":
        raise ValueError("Only registered Stage A schema version 1 is supported")
    if len(reg["runs"]) != 5 or {j["id"] for j in reg["runs"]} != set(EXPECTED_RUNS):
        raise ValueError("Stage A has exactly the five predeclared fits")
    if reg["maximum_new_fits"] != 5 or reg["run_sequence"] != list(EXPECTED_RUNS):
        raise ValueError("Fit budget/sequence differs from the predeclared five attempts")
    for job in reg["runs"]:
        if tuple(job[k] for k in ("initialization_seed", "order_seed", "stochastic_seed")) != EXPECTED_RUNS[job["id"]]:
            raise ValueError("Run differs from the registered factorial seed levels")
    fit, held, target = (set(reg[k]) for k in ("fit_rx", "held_rx", "target_rx"))
    if len(fit) != 18 or len(held) != 6 or len(target) != 8 or fit & held or fit & target or held & target:
        raise ValueError("Invalid fold2 receiver roles")


def require_committed(path):
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "-C", str(ROOT), "show", f"HEAD:{relative}"])
    checkout = path.read_bytes()
    # Git's Windows checkout may use CRLF while its text blob stores LF.
    # This check tests committed text; code_sha256 separately pins the exact
    # checkout bytes used at runtime, including its newline representation.
    if path.suffix.lower() in {".py", ".md", ".json"}:
        committed, checkout = (value.replace(b"\r\n", b"\n") for value in (committed, checkout))
    if committed != checkout:
        raise ValueError(f"Uncommitted registered evidence: {relative}")


def validate_registration(path):
    path = Path(path).resolve()
    require_committed(path)
    reg = read(path)
    validate_runs(reg)
    if not reg.get("code_sha256"):
        raise ValueError("Registered implementation hashes required")
    for name, digest in reg["code_sha256"].items():
        source = (ROOT / name).resolve()
        source.relative_to(ROOT)
        if file_sha(source) != digest:
            raise ValueError(f"Registered code changed: {name}")
        require_committed(source)
    return reg


def artifact_root(reg):
    path = (ROOT / reg["output_dir"]).resolve()
    path.relative_to(HERE)
    return path


def access(reg, item, action, run=None):
    path = HERE / "ACCESS_LOG.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(time=now(), action=action, run=run,
                                     path=str(item["path"]), sha256=item["sha256"]),
                                ensure_ascii=False) + "\n")


def verify_input(reg, name, item, run=None):
    path = Path(item["path"])
    access(reg, item, "hash_registered_" + name, run)
    if file_sha(path) != item["sha256"]:
        raise ValueError(f"Registered input changed: {name}")
    return path


def make_keys(y, rx, day, rx_names, cell_hashes):
    if not (len(y) == len(rx) == len(day)):
        raise ValueError("Metadata lengths differ")
    counters = {}
    keys = []
    for _, receiver, date in zip(y, rx, day):
        name, date = rx_names[int(receiver)], int(date)
        cell = f"{name}|{date}"
        ordinal = counters.get(cell, 0)
        counters[cell] = ordinal + 1
        keys.append(f"{name}|{date}|{cell_hashes[cell]}|{ordinal}")
    result = np.asarray(keys)
    if len(set(result)) != len(result):
        raise ValueError("Canonical keys are not unique")
    return result


def load_fit(reg, run):
    names = ("fit_x", "fit_y", "fit_rx", "fit_day", "train_ids", "val_ids", "fit_manifest")
    if set(reg["inputs"]) != set(names):
        raise ValueError("Unexpected fit input whitelist")
    paths = {name: verify_input(reg, name, reg["inputs"][name], run) for name in names}
    manifest = read(paths["fit_manifest"])
    if manifest["rx_names"] != reg["rx_names"] or set(manifest["fit_rx"]) != set(reg["fit_rx"]):
        raise ValueError("Manifest receiver roles differ")
    pool = {}
    for name in names[:-1]:
        access(reg, reg["inputs"][name], "load_registered_" + name, run)
        pool[name] = np.load(paths[name], mmap_mode="r", allow_pickle=False)
    for name, shape in reg["expected_shapes"].items():
        if list(pool[name].shape) != shape:
            raise ValueError(f"Registered shape differs: {name}")
    y, rx, day = (pool[f"fit_{name}"] for name in ("y", "rx", "day"))
    if pool["fit_x"].shape != (len(y), 2, 256) or pool["fit_x"].dtype != np.float32:
        raise ValueError("Expected immutable FP32 [N,2,256] IQ cache")
    if set(day) != {1, 2} or {reg["rx_names"][int(i)] for i in np.unique(rx)} != set(reg["fit_rx"]):
        raise ValueError("Fit data outside registered RX/date scope")
    if np.min(y) < 0 or np.max(y) >= 10:
        raise ValueError("Unexpected ManyRx class labels")
    tr, va = pool["train_ids"], pool["val_ids"]
    if not np.array_equal(np.sort(np.r_[tr, va]), np.arange(len(y))):
        raise ValueError("Train/validation membership is not a disjoint full partition")
    keys = make_keys(y, rx, day, reg["rx_names"], reg["source_cell_sha256"])
    data = {}
    for part, ids in (("train", tr), ("val", va)):
        ids = np.asarray(ids, dtype=np.int64)
        data[part] = dict(x=np.array(pool["fit_x"][ids], copy=True),
                          y=np.asarray(y[ids], dtype=np.int64),
                          rx=np.asarray(rx[ids], dtype=np.int64),
                          day=np.asarray(day[ids], dtype=np.int64),
                          keys=keys[ids], pool_ids=ids)
    return data


def setup(seed):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    if not torch.cuda.is_available():
        raise RuntimeError("Registered CUDA device required")
    torch.cuda.reset_peak_memory_stats()


def build_model(seed):
    for relative in ("lib", "13-mixer-cfo-mechanisms", "17-rx-distortion"):
        path = str(ROOT / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
    from s17_models import build
    return build("ManyRx", seed).cuda()


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def rng_state(order):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                order=order.get_state())


def checkpoint(model, opt, scheduler, order, epoch, best, best_epoch, wait, updates):
    return cpu_tree(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                         scheduler=scheduler.state_dict(), rng=rng_state(order),
                         epoch=epoch, best=best, best_epoch=best_epoch, wait=wait,
                         updates=updates))


def snapshot_digests(state):
    bn = {k: v for k, v in state["model"].items()
          if k.endswith(("running_mean", "running_var", "num_batches_tracked"))}
    return dict(state_sha256=semantic_hash(state),
                model_sha256=semantic_hash(state["model"]),
                bn_buffers_sha256=semantic_hash(bn),
                optimizer_sha256=semantic_hash(state["optimizer"]),
                scheduler_sha256=semantic_hash(state["scheduler"]),
                rng_sha256=semantic_hash(state["rng"]))


def prepare_order(train_ids, keys, generator):
    permutation = torch.randperm(len(train_ids), generator=generator, device="cuda")
    positions = permutation.cpu().numpy()
    digest = hashlib.sha256(("\n".join(keys[positions].tolist()) + "\n").encode()).hexdigest()
    return permutation, digest


@torch.no_grad()
def predict(model, x):
    model.eval()
    return torch.cat([model(torch.as_tensor(np.array(x[i:i + 128], copy=True), device="cuda"))
                      for i in range(0, len(x), 128)]).cpu().numpy()


def save_predictions(path, logits, data, reg):
    rx = np.asarray([reg["rx_names"][int(r)] for r in data["rx"]])
    np.savez_compressed(path, logits=np.asarray(logits, dtype=np.float32), y=data["y"],
                        rx=rx, day=data["day"], keys=data["keys"], pool_ids=data["pool_ids"])


def check_gate(reg, reg_path):
    gate = read(HERE / "REPLAY_GATE.json")
    if gate["status"] != "PASS_EXACT_REPLAY" or gate["registration_sha256"] != file_sha(reg_path):
        raise ValueError("Exact replay gate has not passed")
    outputs = artifact_root(reg)
    for name in ("A00", "A00-replay"):
        if file_sha(outputs / name / "COMPLETE.json") != gate["complete_sha256"][name]:
            raise ValueError("Replay evidence changed after gate")


def train(reg, reg_path, run):
    job = next(j for j in reg["runs"] if j["id"] == run)
    if run not in ("A00", "A00-replay"):
        check_gate(reg, reg_path)
    if run == "A00-replay" and not (artifact_root(reg) / "A00" / "COMPLETE.json").exists():
        raise ValueError("A00 must finish before its independent replay")
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
            training=reg["training"]))
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
            ordered_ids = tr["pool_ids"][permutation.cpu().numpy()]
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


def replay_check(reg, reg_path):
    left, lc = verify_complete(reg, reg_path, "A00")
    right, rc = verify_complete(reg, reg_path, "A00-replay")
    checks = {}
    configs = [read(p / "CONFIG.json") for p in (left, right)]
    for cfg in configs:
        cfg["job"].pop("id")
    checks["environment_and_configuration"] = compare_objects(*configs)
    with np.load(left / "SAMPLE_KEYS.npz", allow_pickle=False) as a, \
            np.load(right / "SAMPLE_KEYS.npz", allow_pickle=False) as b:
        checks["sample_keys_and_membership"] = compare_objects(dict(a), dict(b))
    checks["epoch_history"] = compare_objects(read(left / "HISTORY.json"), read(right / "HISTORY.json"))
    histories = [read(p / "HISTORY.json") for p in (left, right)]
    for epoch in range(1, min(map(len, histories)) + 1):
        a, b = [torch.load(path / "epoch_states" / f"epoch_{epoch:02d}.pt", map_location="cpu", weights_only=False)
                for path in (left, right)]
        checks[f"epoch_{epoch:02d}_full_state"] = compare_objects(a, b)
    for endpoint in ("initial", "selected", "epoch10", "final"):
        a, b = [torch.load(path / f"{endpoint}.pt", map_location="cpu", weights_only=False)
                for path in (left, right)]
        checks[endpoint + "_state"] = compare_objects(a, b)
    for endpoint in ("selected", "epoch10"):
        with np.load(left / f"{endpoint}_val.npz", allow_pickle=False) as a, \
                np.load(right / f"{endpoint}_val.npz", allow_pickle=False) as b:
            checks[endpoint + "_outputs"] = compare_objects(dict(a), dict(b))
            checks[endpoint + "_outputs"]["classification_disagreements"] = int(
                np.count_nonzero(a["logits"].argmax(1) != b["logits"].argmax(1)))
    result = dict(status="PASS_EXACT_REPLAY" if all(c["exact"] for c in checks.values()) else "FAIL_REPLAY",
                  registration_sha256=file_sha(reg_path), checks=checks,
                  complete_sha256={name: file_sha(artifact_root(reg) / name / "COMPLETE.json")
                                   for name in ("A00", "A00-replay")},
                  interpretation="Exact semantic tensors and stream state; no accuracy tolerance.", time=now())
    frozen(HERE / "REPLAY_GATE.json", result)
    if result["status"] != "PASS_EXACT_REPLAY":
        raise RuntimeError("A00 replay failed; stop remaining factors and Stage B")
    return result


def seal(reg, reg_path):
    check_gate(reg, reg_path)
    evidence = {}
    complete = {}
    for job in reg["runs"]:
        run = job["id"]
        out, item = verify_complete(reg, reg_path, run)
        evidence[run] = dict(complete_sha256=file_sha(out / "COMPLETE.json"),
                            selected_sha256=file_sha(out / "selected.pt"),
                            epoch10_sha256=file_sha(out / "epoch10.pt"), endpoints=item["endpoints"])
        complete[run] = item
    # Cross-order initial states must agree; cross-init order prefixes agree.
    checks = {}
    for names in (("A00", "A01"), ("A10", "A11")):
        checks["same_init_" + "_".join(names)] = (complete[names[0]]["endpoints"]["initial"]["model_sha256"] ==
                                                     complete[names[1]]["endpoints"]["initial"]["model_sha256"])
    for names in (("A00", "A10"), ("A01", "A11")):
        histories = [read(artifact_root(reg) / name / "HISTORY.json") for name in names]
        shared = min(map(len, histories))
        checks["same_order_" + "_".join(names)] = all(histories[0][i]["order_sha256"] == histories[1][i]["order_sha256"]
                                                           for i in range(shared))
    checks["distinct_initialization_levels"] = (
        complete["A00"]["endpoints"]["initial"]["model_sha256"] !=
        complete["A10"]["endpoints"]["initial"]["model_sha256"])
    histories = [read(artifact_root(reg) / name / "HISTORY.json") for name in ("A00", "A01")]
    checks["distinct_order_levels"] = histories[0][0]["order_sha256"] != histories[1][0]["order_sha256"]
    if not all(checks.values()):
        raise RuntimeError("Factor isolation evidence failed")
    result = dict(status="SEALED_STAGE_A", registration_sha256=file_sha(reg_path),
                  replay_pass=True,
                  replay_gate_sha256=file_sha(HERE / "REPLAY_GATE.json"), runs=evidence,
                  factor_isolation_checks=checks, held_input_opened_by_seal=False,
                  stage_b_admitted=False, time=now())
    frozen(HERE / "STAGE_A_SEAL.json", result)
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
        access(reg, item, "load_registered_held_cell_after_all_five_sealed")
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
    seal_path = HERE / "STAGE_A_SEAL.json"
    require_committed(seal_path)
    sealed = read(seal_path)
    if sealed["status"] != "SEALED_STAGE_A" or sealed["registration_sha256"] != file_sha(reg_path):
        raise ValueError("All five registered fits must be sealed before held access")
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
        status="COMPLETE", stage_a_seal_sha256=file_sha(seal_path),
        registration_sha256=file_sha(reg_path), files=predictions))
    return read(marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=HERE / "PRE_REGISTRATION.json")
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("train")
    fit_parser.add_argument("--run", choices=list(EXPECTED_RUNS), required=True)
    for name in ("replay-check", "seal", "evaluate"):
        sub.add_parser(name)
    args = parser.parse_args()
    reg = validate_registration(args.registration)
    if args.command == "train":
        result = train(reg, args.registration, args.run)
    elif args.command == "replay-check":
        result = replay_check(reg, args.registration)
    elif args.command == "seal":
        result = seal(reg, args.registration)
    else:
        result = evaluate(reg, args.registration)
    print(json.dumps(dict(status=result["status"], command=args.command), ensure_ascii=False))


if __name__ == "__main__":
    main()
