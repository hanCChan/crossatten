"""Run the balanced, all-client Gul FL adaptation on the registered Shen split."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "s42_lora_baselines"))
from fit import Data, ROOT, SEEDS, atomic, cpu_state, git_file, read, save, sha
from xa.frontends import awgn, shen_cis
from xa.models import ShenCISPipeline

ROUNDS = 200
LOCAL_EPOCHS = 5
BATCH = 64
CLIENTS = 5
CLIENT_PACKETS = 1600
LR = 0.001
GRADIENT_REVERSAL = 0.01
DEVICE = "cuda"


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature):
        return feature.view_as(feature)

    @staticmethod
    def backward(ctx, gradient):
        return -GRADIENT_REVERSAL * gradient


class GulModel(ShenCISPipeline):
    def __init__(self):
        super().__init__()
        del self.tx_input
        self.classifier = nn.Linear(512, 10)
        self.receiver = nn.Linear(512, CLIENTS)
        for head in (self.classifier, self.receiver):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def logits(self, features):
        return self.classifier(self.features(features))

    def objective(self, features, tx, rx):
        feature = self.features(features)
        tx_logits = self.classifier(feature)
        rx_logits = self.receiver(Reverse.apply(feature))
        return F.cross_entropy(tx_logits.float(), tx) + F.cross_entropy(rx_logits.float(), rx)

    def forward(self, iq):
        return self.logits(shen_cis(iq).contiguous(memory_format=torch.channels_last))


def binding():
    paths = [
        HERE / "fit_gul.py", HERE / "PROTOCOL.zh-CN.md",
        HERE.parent / "s42_lora_baselines/fit.py",
        ROOT / "lib/xa/models.py", ROOT / "lib/xa/frontends.py",
        ROOT / "lib/xa/data.py", ROOT / "configs/protocol.json",
        ROOT / "26-long-patch-control/registration.json",
    ]
    result = {}
    for file in paths:
        name = file.relative_to(ROOT).as_posix()
        content = file.read_bytes().replace(b"\r\n", b"\n")
        if content != git_file(name):
            raise ValueError(f"Registration input differs from Git: {name}")
        result[name] = sha256(content).hexdigest()
    return result


def configure():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def source_metrics(model, inputs, labels):
    model.eval()
    logits = torch.cat([model.logits(s) for s in inputs.split(128)])
    return float(F.cross_entropy(logits, labels)), float((logits.argmax(1) == labels).float().mean())


def fit(source, validation, directory, registered, fold, seed):
    directory.mkdir(exist_ok=True)
    done = directory / "DONE.json"
    selected = directory / "selected.pt"
    history_file = directory / "HISTORY.json"
    if done.exists():
        row = read(done)
        if (row["files"], row["weights"], row["history"]) != (registered, sha(selected), sha(history_file)):
            raise ValueError(f"Completed fit changed: {directory}")
        return row

    torch.manual_seed(seed + fold * 10000)
    generator = torch.Generator(device=DEVICE).manual_seed(seed + fold * 10000 + 43000)
    iq, tx_labels, receiver_labels = source
    x = torch.as_tensor(iq, device=DEVICE)
    label = torch.as_tensor(tx_labels, dtype=torch.long, device=DEVICE)
    rx = torch.as_tensor(receiver_labels, dtype=torch.long, device=DEVICE)
    clients = [torch.where(rx == r)[0] for r in range(CLIENTS)]
    if any(len(i) != CLIENT_PACKETS for i in clients):
        raise ValueError("Unequal client counts in registered balanced protocol")
    val_iq, val_y, _ = validation
    val = torch.cat([shen_cis(batch) for batch in torch.as_tensor(val_iq, device=DEVICE).split(128)])
    val = val.contiguous(memory_format=torch.channels_last)
    val_y = torch.as_tensor(val_y, dtype=torch.long, device=DEVICE)

    model = GulModel().to(DEVICE, memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, foreach=True)
    global_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    aggregate = {name: torch.zeros_like(value) for name, value in global_state.items()}
    history = []
    best, best_round, best_weights = float("inf"), 0, None
    last = directory / "LAST.pt"
    if last.exists():
        checkpoint = torch.load(last, weights_only=True, map_location="cpu")
        if checkpoint["files"] != registered:
            raise ValueError("Checkpoint registration mismatch")
        for name, value in checkpoint["weights"].items():
            global_state[name].copy_(value)
        generator.set_state(checkpoint["generator"])
        history = checkpoint["history"]
        best, best_round, best_weights = (checkpoint[k] for k in ("best", "best_round", "best_weights"))
    before = time.perf_counter()
    previous = history[-1]["seconds"] if history else 0.
    for round_index in range(len(history) + 1, ROUNDS + 1):
        torch._foreach_zero_(list(aggregate.values()))
        for index in clients:
            model.load_state_dict(global_state)
            model.train()
            for _ in range(LOCAL_EPOCHS):
                order = index[torch.randperm(CLIENT_PACKETS, device=DEVICE, generator=generator)]
                for rows in order.split(BATCH):
                    iq = awgn(x[rows], 0., 49., generator)
                    features = shen_cis(iq).contiguous(memory_format=torch.channels_last)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                        loss = model.objective(features, label[rows], rx[rows])
                    loss.backward()
                    optimizer.step()
            with torch.no_grad():
                torch._foreach_add_(list(aggregate.values()), list(model.state_dict().values()),
                                    alpha=1 / CLIENTS)
        model.load_state_dict(aggregate)
        torch._foreach_copy_(list(global_state.values()), list(aggregate.values()))
        ce, acc = source_metrics(model, val, val_y)
        if not np.isfinite((ce, acc)).all():
            raise FloatingPointError((fold, seed, round_index, ce, acc))
        if ce < best:
            best, best_round, best_weights = ce, round_index, cpu_state(model)
        history.append(dict(round=round_index, source_ce=ce, source_acc=acc,
                            seconds=previous + time.perf_counter() - before))
        if round_index == 1 or round_index % 5 == 0:
            save(last, dict(weights=global_state, generator=generator.get_state(), history=history,
                            best=best, best_round=best_round, best_weights=best_weights, files=registered))
            print(f"f{fold} s{seed} round {round_index}/{ROUNDS} source {acc:.4f} CE {ce:.5f} "
                  f"best {best_round} wall {history[-1]['seconds']:.1f}s", flush=True)

    model.load_state_dict(best_weights)
    ce, acc = source_metrics(model, val, val_y)
    if abs(ce - best) > 2e-6:
        raise ValueError("Selected checkpoint fails source CE replay")
    save(selected, best_weights)
    atomic(history_file, history)
    result = dict(fold=fold, seed=seed, rounds=ROUNDS, selected_round=best_round, source_ce=ce,
                  source_accuracy=acc, final_source_accuracy=history[-1]["source_acc"],
                  updates=ROUNDS * CLIENTS * LOCAL_EPOCHS * (CLIENT_PACKETS // BATCH),
                  fit_seconds=history[-1]["seconds"], params=sum(p.numel() for p in model.parameters()),
                  files=registered, weights=sha(selected), history=sha(history_file))
    atomic(done, result)
    return result


def run(out):
    configure()
    out.mkdir(parents=True, exist_ok=True)
    registered = binding()
    jobs = [(f, s) for f in (0, 1) for s in SEEDS]
    plan = dict(protocol="P-H5x160 Native", jobs=[list(job) for job in jobs], files=registered)
    if (out / "PLAN.json").exists() and read(out / "PLAN.json") != plan:
        raise ValueError("Output directory contains a different plan")
    atomic(out / "PLAN.json", plan)
    completed = []
    for fold in (0, 1):
        source, validation = Data(out).source(fold)
        for seed in SEEDS:
            completed.append(fit(source, validation, out / f"f{fold}_s{seed}", registered, fold, seed))
    atomic(out / "FIT_SEAL.json", dict(plan=sha(out / "PLAN.json"), fits=completed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
