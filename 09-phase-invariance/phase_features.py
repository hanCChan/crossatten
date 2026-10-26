"""S09 RF spectral phase controls; input is S08's frequency_views(x).

All variants return [Re(X), Im(X), LogMag(X), extra_Re, extra_Im]
except `frequency`, which preserves the existing three-channel baseline.
The first-order implementation is exactly the S08 masked product for parity.

No claim that time-shift invariance implies receiver-independent TX identity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def phase_features_from_base(
    base: torch.Tensor,
    kind: str = "first",
    threshold: float = 0.05,
    min_resultant: float = 0.1,
) -> torch.Tensor:
    """Build controlled phase views from [batch,3,freq_bins] S08 FFT features.

    kind: frequency | null | first | curvature | centered.

    `centered` and `curvature` are invariant to ideal integer *circular*
    time shifts (aside from numerical roundoff, or centered's fallback).
    A real noncircular crop, CFO, multipath and receiver filtering are not
    guaranteed nuisances removed by these transforms.
    """
    if base.ndim != 3 or base.size(1) != 3 or base.size(2) < 3:
        raise ValueError("base must have shape [batch,3,N>=3]")
    if not (0 < threshold < 1) or not (0 <= min_resultant <= 1):
        raise ValueError("invalid numerical thresholds")
    if kind == "frequency":
        return base
    if kind not in {"null", "first", "curvature", "centered"}:
        raise ValueError(f"unknown phase view: {kind}")
    z = torch.complex(base[:, 0], base[:, 1])
    amp = z.abs()
    valid = (amp[:, 1:] >= threshold) & (amp[:, :-1] >= threshold)
    p = z[:, 1:] * z[:, :-1].conj()
    p = p / (amp[:, 1:] * amp[:, :-1]).clamp_min(1e-6)
    p = p * valid

    if kind == "first":
        extra = F.pad(torch.stack((p.real, p.imag), dim=1), (1, 0))
    elif kind == "null":
        extra = torch.zeros((base.shape[0], 2, base.shape[-1]),
                            device=base.device, dtype=base.dtype)
    elif kind == "curvature":
        # Second-order difference removes any constant inter-bin rotation,
        # e.g. the phase slope caused by an ideal circular sample shift.
        q = p[:, 1:] * p[:, :-1].conj()
        extra = F.pad(torch.stack((q.real, q.imag), dim=1), (2, 0))
    else:
        # The circular mean is differentiable without a phase branch cut.
        # Weak resultant => avoid normalizing noise into a confident direction.
        resultant = p.sum(dim=-1, keepdim=True)
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        strength = resultant.abs() / count
        anchor = resultant.conj() / resultant.abs().clamp_min(1e-6)
        anchor = torch.where(strength >= min_resultant,
                             anchor, torch.ones_like(anchor))
        pc = p * anchor
        extra = F.pad(torch.stack((pc.real, pc.imag), dim=1), (1, 0))
    return torch.cat((base, extra), dim=1)
