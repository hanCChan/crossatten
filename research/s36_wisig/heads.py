"""Source-only linear, shallow nonlinear, and calibrated fusion readouts."""

import sys
from pathlib import Path

import numpy as np

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / "s34_wisig"))
from common import DEVICE, ROLES, tensor, torch

F = torch.nn.functional
REG = (.1, .01, .0001)
ALPHA = (0., .125, .25, .5, 1.)
SCALES = (.5, 1., 2., 4., 8., 16., 32.)


def weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return tensor(np.float32(1 / (len(counts)*counts[inverse])))


def select_linear(encoded, labels, classes=10, factors=REG, validation_groups=None):
    x, val, held = (encoded[role] for role in ROLES)
    data, target = labels["fit_train"], labels["checkpoint_val"]
    w = weights(data["rx"])
    groups = target["rx"] if validation_groups is None else validation_groups
    masks = [groups == r for r in np.unique(groups)]
    onehot = F.one_hot(tensor(data["y"]).long(), classes).float()
    xbar, ybar = w @ x, w @ onehot
    xc, yc = x - xbar, onehot - ybar
    covariance = (xc.T * w) @ xc
    rhs = (xc.T * w) @ yc
    eye = torch.eye(covariance.shape[0], device=DEVICE)
    scale = covariance.diag().mean().clamp_min(1e-8)
    score, trace = -1., []
    for lam in factors:
        model = torch.linalg.solve(covariance + lam*scale*eye, rhs)
        v = (val-xbar) @ model + ybar
        correct = v.argmax(1).cpu().numpy() == target["y"]
        a = float(np.mean([correct[mask].mean() for mask in masks]))
        trace.append(a)
        if a > score:
            score, factor, chosen, val_scores = a, lam, model, v
    return val_scores, (held-xbar) @ chosen + ybar, dict(factor=factor, validation_accuracy=trace)


def loss(logits, y, w):
    return float(F.cross_entropy(logits, y, reduction="none") @ w)


def calibrate(validation, held, data):
    y = tensor(data["y"]).long()
    w = weights(data["rx"])
    validation = validation - validation.mean(1, keepdim=True)
    held = held - held.mean(1, keepdim=True)
    rms = validation.square().mean().sqrt().clamp_min(1e-8)
    validation, held = validation/rms, held/rms
    ce = [loss(scale*validation, y, w) for scale in SCALES]
    scale = SCALES[int(np.argmin(ce))]
    return validation*scale, held*scale, dict(rms=float(rms), scale=scale, calibration_ce=ce)


def fuse(main, alternate, val, deep=None):
    deep_val, deep_held, deep_record = calibrate(*main, val) if deep is None else deep
    alt_val, alt_held, alt_record = calibrate(*alternate, val)
    y = tensor(val["y"]).long()
    w = weights(val["rx"])
    ce = [loss(deep_val + mix*alt_val, y, w) for mix in ALPHA]
    mix = ALPHA[int(np.argmin(ce))]
    output = deep_held + mix*alt_held
    return output, dict(alpha=mix, fusion_ce=ce, deep=deep_record, second=alt_record)


class ProbeBank(torch.nn.Module):
    def __init__(self, number, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layer1 = torch.nn.Linear(64, 8)
            layer2 = torch.nn.Linear(8, 10)
        self.left = self.repeat(layer1.weight.T, number)
        self.hidden = self.repeat(layer1.bias[None], number)
        self.right = self.repeat(layer2.weight.T, number)
        self.output = self.repeat(layer2.bias[None], number)

    @staticmethod
    def repeat(parameter, count):
        return torch.nn.Parameter(parameter.detach().to(DEVICE).unsqueeze(0).repeat(count, 1, 1))

    def forward(self, x):
        return torch.bmm(F.gelu(torch.bmm(x, self.left) + self.hidden), self.right) + self.output


def nonlinear(encoded, data, seed):
    names = tuple(encoded)
    fit, val, held = (torch.stack([encoded[k][role] for k in names]) for role in ROLES)
    y0 = tensor(data["fit_train"]["y"]).long()
    y1 = tensor(data["checkpoint_val"]["y"]).long()
    w = weights(data["checkpoint_val"]["rx"])
    orders = torch.Generator(device=DEVICE).manual_seed(seed + 71)
    bank = ProbeBank(len(names), seed)
    assert sum(p.numel() for p in bank.parameters()) == 610*len(names)
    opt = torch.optim.Adam(bank.parameters(), lr=.003, weight_decay=.001)
    best = torch.full((len(names),), torch.inf, device=DEVICE)
    epoch = torch.zeros(len(names), device=DEVICE, dtype=torch.int64)
    selected = [torch.empty_like(item) for item in bank.parameters()]
    history = []
    updates = 0
    for step in range(1, 21):
        for batch in torch.randperm(len(y0), device=DEVICE, generator=orders).split(256):
            output = bank(fit[:, batch])
            target = y0[batch].expand(len(names), -1)
            training_ce = F.cross_entropy(output.flatten(0, 1), target.reshape(-1), reduction="sum")/len(batch)
            opt.zero_grad(set_to_none=True)
            training_ce.backward()
            opt.step()
            updates += 1
        with torch.no_grad():
            scores = bank(val)
            ce = F.cross_entropy(scores.transpose(1, 2), y1.expand(len(names), -1), reduction="none") @ w
            assert torch.isfinite(ce).all()
            improved = ce < best
            for copy, parameter in zip(selected, bank.parameters()):
                copy[improved] = parameter[improved]
            best[improved] = ce[improved]
            epoch[improved] = step
            history.append(ce.cpu().numpy().tolist())
    with torch.no_grad():
        for parameter, copy in zip(bank.parameters(), selected):
            parameter.copy_(copy)
        decisions = bank(held).argmax(2).cpu().numpy().astype(np.uint8)
    return dict(zip(names, decisions)), dict(selected_epochs=dict(zip(names, epoch.cpu().numpy().tolist())),
            ce_history=history, vectorized_optimizer_steps=updates, heads=len(names))
