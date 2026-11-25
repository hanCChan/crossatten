"""Recompute public ASKNet log summaries and audit the released LoRa configurations."""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
import subprocess
from time import perf_counter

import torch
from torch.nn import functional as F


ASK_COMMIT = "f00846d68425315b68ae69b8c6b712e08559c2fe"
IB_COMMIT = "5a03a04da86c4bbf5cf90f41a55c268bf2152203"
ASK_LOG = "Lora/logs/h5_rescisnet_k10_per800_trainrxn210_1_day1_seed2023.k_trace.txt"
ASK_MODEL = "Lora/backbones/ResCISNet.py"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def pinned_reader(repo, commit):
    if git(repo, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError(f"Expected pinned revision {commit} in {repo}")
    hashes = {}

    def read(name):
        payload = git(repo, "show", f"{commit}:{name}")
        hashes[name] = sha256(payload).hexdigest()
        return payload.decode("utf-8")

    return read, hashes


def argument_defaults(source):
    arguments = {}
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            continue
        key = ast.literal_eval(node.args[0]).removeprefix("--")
        for param in node.keywords:
            if param.arg in ("default", "choices"):
                try:
                    arguments.setdefault(key, {})[param.arg] = ast.literal_eval(param.value)
                except ValueError:
                    continue
    return arguments


def ask_log(text):
    header, *blocks = text.split("[test]")
    candidates = ast.literal_eval(re.search(r"(?m)^Candidates: (\[.*\])$", header).group(1))
    last_epoch = max(int(i) for i in re.findall(r"(?m)^\s*(\d+)\s*\|", header))
    evaluations = {}
    for block in blocks:
        rx = re.search(r"(?m)^test_rx\s*:\s*\['([^']+)'\]", block).group(1)
        days = {int(day): float(value) for day, value in re.findall(r"(?m)^Day(\d+)\s*:\s*([0-9.]+)", block)}
        reported = float(re.search(r"(?m)^Average\s*:\s*([0-9.]+)", block).group(1))
        training_rx = re.search(r"(?m)^train_rx\s*:\s*\['([^']+)'\]", block).group(1)
        training_day = int(re.search(r"(?m)^train_day\s*:\s*\[(\d+)\]", block).group(1))
        selected = int(re.search(r"(?m)^test_k\s*:\s*(\d+)", block).group(1))
        if list(days) != [1, 2, 3, 4] or abs(mean(days.values()) - reported) > 0.00005:
            raise ValueError(f"Unexpected author log values for {rx}")
        evaluations[rx] = dict(train_rx=training_rx, train_day=training_day, selected_k=selected,
                               day_accuracy=days, reported_mean=reported,
                               four_day_mean=mean(days.values()), days_2_to_4_mean=mean(list(days.values())[1:]))
    if set(evaluations) != {"n210_1", "rtl_6"}:
        raise ValueError("Author log is missing an evaluation receiver")
    return candidates, last_epoch, evaluations


def architecture(model_class, candidates):
    with torch.device("meta"):
        model = model_class(num_classes=10, candidates=candidates)
    return dict(candidates=candidates, koopman_dimension=model.ask.max_modes,
                parameter_elements=sum(p.numel() for p in model.parameters()),
                real_parameter_dofs=sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()))


def synthetic_ask_step(model_class, candidates):
    """Measure compute feasibility without opening an RF dataset or producing a score."""
    seed = 4401
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()
    with torch.device("cuda"):
        model = model_class(num_classes=10, candidates=candidates)
        iq = torch.randn(128, 2, 8192)
        tx = torch.arange(128) % 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    torch.cuda.synchronize()
    steps = []
    for _ in range(3):
        before = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(iq, tau=5.0)
        loss = F.cross_entropy(logits, tx)
        if logits.shape != (128, 10) or not torch.isfinite(loss):
            raise ValueError("ASKNet data-free step produced invalid logits or loss")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        steps.append(perf_counter() - before)
    return dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__, seed=seed, batch=128,
                synthetic_optimizer_steps=3, fit_count=0, seconds=steps,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                fewer_candidate_train_hours_at_148_epochs=mean(steps[1:]) * 44 * 148 / 3600)


def run(ask_repo, ib_repo, include_synthetic):
    ask_read, ask_hashes = pinned_reader(ask_repo, ASK_COMMIT)
    logged_candidates, last_epoch, accuracy = ask_log(ask_read(ASK_LOG))
    args = argument_defaults(ask_read("Lora/run_CIS.py"))
    defaults = args["candidates"]["default"]
    namespace = {}
    exec(compile(ask_read(ASK_MODEL), ASK_MODEL, "exec"), namespace)
    model_class = namespace["ResCISNet"]
    num_pool = args["last_k_classes"]["default"] * args["per_class"]["default"]
    file_list = git(ask_repo, "ls-tree", "-r", "--name-only", ASK_COMMIT).decode().splitlines()
    ask_read("README.md")
    ask_read("weights&datasets.txt")
    ask = dict(commit=ASK_COMMIT, evidence="author log, engineering only locally", sha256=ask_hashes,
               accuracy=accuracy, last_logged_epoch=last_epoch, selected_pool=num_pool,
               train_val_sizes=[int(num_pool * 0.7), int(num_pool * 0.3)],
               author_log_architecture=architecture(model_class, logged_candidates),
               released_default_architecture=architecture(model_class, defaults),
               tracked_weights=[p for p in file_list if Path(p).suffix in (".pt", ".pth", ".ckpt")])
    if include_synthetic:
        ask["synthetic_logged_architecture"] = synthetic_ask_step(model_class, logged_candidates)

    ib_read, ib_hashes = pinned_reader(ib_repo, IB_COMMIT)
    for file in ("README.md", "dataset/link.txt", "utils/load_data.py"):
        ib_read(file)
    ib_defaults = argument_defaults(ib_read("main.py"))
    shell = ib_read("run_all.sh")
    patch_sizes = {dataset: int(size) for dataset, size in re.findall(r"(ManyRx|ManySig)\)\s+echo (\d+)", shell)}
    ib_files = git(ib_repo, "ls-tree", "-r", "--name-only", IB_COMMIT).decode().splitlines()
    weights = [p for p in ib_files if p.startswith("weights/")]
    logs = [p for p in ib_files if p.startswith("logs/")]
    return {
        "asknet": ask,
        "ib_rff": dict(commit=IB_COMMIT, sha256=ib_hashes, launcher_patch_sizes=patch_sizes,
                       datasets=ib_defaults["dataset_name"]["choices"], weights=weights, logs=logs,
                       tracked_lora_weights=[p for p in weights if not re.search(r"Many(Rx|Sig)", p)],
                       tracked_lora_logs=[p for p in logs if not re.search(r"Many(Rx|Sig)", p)]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask", type=Path, required=True)
    parser.add_argument("--ib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    result = run(args.ask, args.ib, args.synthetic)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
