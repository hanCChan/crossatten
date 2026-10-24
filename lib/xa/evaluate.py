"""Per-receiver metrics (packet accuracy, macro-F1, confusion) and RX-equal summaries."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def predict(model, x: torch.Tensor, batch: int = 500) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch]).float() for i in range(0, len(x), batch)])


def confusion(pred: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    return np.bincount(y * k + pred, minlength=k * k).reshape(k, k)


def rx_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    pred = logits.argmax(1)
    cm = confusion(pred, y)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    return dict(acc=float((pred == y).mean()), macro_f1=float(f1.mean()), recall=rec.round(4).tolist(),
                confusion=cm.tolist())


def summarize(per_rx: dict) -> dict:
    accs = {r: m["acc"] for r, m in per_rx.items()}
    f1s = {r: m["macro_f1"] for r, m in per_rx.items()}
    worst = min(accs, key=accs.get)
    return dict(acc_mean=float(np.mean(list(accs.values()))), f1_mean=float(np.mean(list(f1s.values()))),
                worst_rx=worst, worst_acc=accs[worst], per_rx_acc=accs, per_rx_f1=f1s)
