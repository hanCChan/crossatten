"""Read-only HStem probes and bounded, label-free feature interventions.

Pooled feature probes average spatial/token positions but preserve channels.
Interventions use the same channel-space operator at every position; the
resulting network can therefore be evaluated without TX/RX metadata.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


# First convolution activation, first tokenization activation, HStem tokens,
# and normalized mixer output. Axis refers to the unpooled module output.
LAYERS = {
    "stem1": ("frequency.stem.2", 1),
    "stem_mid": ("frequency.tokenize.2", 1),
    "tokens": ("frequency", 2),
    "mixer": ("encoder.final_norm", 2),
}


class Collector:
    """CPU numpy batches from one or more forward passes, in pass order."""

    def __init__(self):
        self.batches = {name: [] for name in LAYERS}

    def arrays(self):
        """Concatenate each layer's batches as an ``[N, C]`` array."""
        if any(not rows for rows in self.batches.values()):
            raise ValueError("No forward pass captured for every requested layer")
        return {name: np.concatenate(rows, axis=0)
                for name, rows in self.batches.items()}

    def clear(self):
        for rows in self.batches.values():
            rows.clear()


def _module(model, layer):
    if layer not in LAYERS:
        raise ValueError(f"Unknown S18 layer: {layer}")
    path, axis = LAYERS[layer]
    return model.get_submodule(path), axis


@contextmanager
def capture(model):
    """Yield a :class:`Collector`; remove all hooks even when a pass fails.

    Caller controls ``model.eval()`` and ``torch.no_grad()``. This context never
    changes either setting, model weights, buffers, or module outputs.
    """
    collector = Collector()
    handles = []

    def make_hook(name, channel_axis):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError(f"Unexpected tensor shape at {name}")
            position_axis = 2 if channel_axis == 1 else 1
            pooled = output.detach().mean(dim=position_axis)
            collector.batches[name].append(pooled.cpu().numpy().copy())
        return hook

    try:
        for name in LAYERS:
            module, axis = _module(model, name)
            handles.append(module.register_forward_hook(make_hook(name, axis)))
        yield collector
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def intervene(model, layer, U, center, alpha=.5):
    """Apply ``z - alpha * U U^T (z-center)`` to each channel vector.

    U must have orthonormal columns, so 0 <= alpha <= 1 bounds every singular
    value of the centered operator in [0, 1]. No labels enter this operation.
    The hook creates new outputs and leaves all parameters/buffers unchanged.
    """
    module, axis = _module(model, layer)
    basis = np.array(U, dtype=np.float64, copy=True)
    mean = np.array(center, dtype=np.float64, copy=True)
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    if basis.ndim != 2 or mean.shape != (basis.shape[0],):
        raise ValueError("Expected U[C, rank] and center[C]")
    if not 1 <= basis.shape[1] <= basis.shape[0]:
        raise ValueError("Invalid intervention rank")
    if not np.isfinite(basis).all() or not np.isfinite(mean).all():
        raise ValueError("Intervention parameters must be finite")
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-6):
        raise ValueError("Intervention basis columns must be orthonormal")
    tensor_cache = {}

    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise ValueError(f"Unexpected tensor shape at {layer}")
        if output.shape[axis] != basis.shape[0]:
            raise ValueError(f"Channel dimension does not match basis at {layer}")
        if alpha == 0:
            return output
        key = (output.device, output.dtype)
        if key not in tensor_cache:
            tensor_cache[key] = (torch.as_tensor(basis, device=output.device,
                                                dtype=output.dtype),
                                 torch.as_tensor(mean, device=output.device,
                                                 dtype=output.dtype))
        u, mu = tensor_cache[key]
        channel_last = output.movedim(axis, -1)
        corrected = channel_last - alpha * ((channel_last - mu) @ u) @ u.T
        return corrected.movedim(-1, axis)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
