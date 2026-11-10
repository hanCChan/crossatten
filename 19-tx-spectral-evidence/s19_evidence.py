"""Source-training-only counterfactual TX spectral evidence.

The teacher is frozen and the band definition is fixed before outcomes. A band
mask acts on the same complex spectrum before all canonical RF channels are
recomputed by the teacher. Counterfactuals are circular spectral interventions,
not identified receiver hardware filters. Negative controls are diagnostic and
never alter q, mask strength, or the frequency partition.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT/'lib') not in sys.path:
    sys.path.insert(0, str(ROOT/'lib'))
from xa.frontends import rms_normalize


def _labels(values, n, name):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(f'{name} must have one label per packet')
    if values.dtype.kind in 'fc' and not np.isfinite(values).all():
        raise ValueError(f'nonfinite {name}')
    return values, np.unique(values)


def _list(values):
    return np.asarray(values).tolist()


def aggregate_contributions(drop, y, rx, day):
    """Cell mean -> equal observed days per RX -> cross-RX lower quartile.

    Missing TX/RX/day cells are not imputed. A receiver's observed days have
    equal weight, regardless of packet count. A TX needs >=2 source receivers.
    Returns JSON-safe lists/scalars, including q and normalized bar_q [T,K].
    TX label order is explicit and must be mapped by the caller during training.
    """
    drop = np.asarray(drop, dtype=np.float64)
    if drop.ndim != 2 or not len(drop) or not drop.shape[1]:
        raise ValueError('drop must be a nonempty [packets,bands] array')
    if not np.isfinite(drop).all() or np.any(drop < 0):
        raise ValueError('margin contributions must be finite and nonnegative')
    y, tx_labels = _labels(y, len(drop), 'TX')
    rx, rx_labels = _labels(rx, len(drop), 'RX')
    day, day_labels = _labels(day, len(drop), 'day')
    nt, nr, nd, k = len(tx_labels), len(rx_labels), len(day_labels), drop.shape[1]
    q = np.empty((nt, k), dtype=np.float64)
    counts = np.zeros((nt, nr, nd), dtype=np.int64)
    # None is an explicit absent cell, never a numeric placeholder.
    cell_means = [[[None for _ in day_labels] for _ in rx_labels] for _ in tx_labels]
    rx_day_means = [[None for _ in rx_labels] for _ in tx_labels]
    supports = []
    for ti, t in enumerate(tx_labels):
        means = []
        for ri, r in enumerate(rx_labels):
            by_day = []
            for di, d in enumerate(day_labels):
                ids = (y == t) & (rx == r) & (day == d)
                counts[ti, ri, di] = int(ids.sum())
                if ids.any():
                    value = drop[ids].mean(axis=0)
                    by_day.append(value)
                    cell_means[ti][ri][di] = value.tolist()
            if by_day:
                value = np.mean(by_day, axis=0)
                means.append(value)
                rx_day_means[ti][ri] = value.tolist()
        if len(means) < 2:
            raise ValueError('each TX requires at least two observed source receivers')
        supports.append(len(means))
        q[ti] = np.quantile(np.asarray(means), .25, axis=0, method='linear')
    epsilon = 1e-6*max(float(q.mean()), 1e-6)
    bar_q = (q+epsilon)/(q+epsilon).sum(axis=1, keepdims=True)
    return dict(q=q.tolist(), bar_q=bar_q.tolist(), epsilon=epsilon,
                tx_labels=_list(tx_labels), rx_labels=_list(rx_labels), day_labels=_list(day_labels),
                rx_support_per_tx=supports, cell_counts=counts.tolist(),
                cell_support=(counts > 0).tolist(), cell_means=cell_means,
                rx_day_means=rx_day_means, quantile=.25, quantile_method='linear',
                weighting='equal observed day cell means per TX/RX, then equal RX lower quartile',
                missing_support='not imputed; observed days/RX can differ across TX',
                evidence_controls_used_for_selection=False)


def margins(logits, y):
    """True-TX logit minus the strongest other class; labels are scoring only."""
    if logits.ndim != 2 or len(logits) != len(y) or logits.shape[1] < 2:
        raise ValueError('invalid teacher logits')
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite teacher logits')
    y = torch.as_tensor(y, dtype=torch.long, device=logits.device)
    if y.ndim != 1 or (y < 0).any() or (y >= logits.shape[1]).any():
        raise ValueError('TX label outside teacher classes')
    other = logits.clone()
    other.scatter_(1, y[:, None], -torch.inf)
    return logits.gather(1, y[:, None])[:, 0]-other.max(dim=1).values


def counterfactuals(x, masks=None, attenuation=.5):
    """Return clean/masked IQ and exactly energy-matched amplitude/phase controls.

    Clean and all variants start from the same RMS-normalized IQ packet. The
    returned energy/EVM measures precede the teacher's canonical normalization.
    Positive real spectral masks preserve spectral phase where magnitude is
    nonzero, but IFFT remains a circular filter with possible temporal ringing.
    """
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 2:
        raise ValueError('expected IQ [packets,2,length] tensor')
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError('IQ must be finite floating point')
    if not 0 < attenuation < 1:
        raise ValueError('attenuation must be strictly between zero and one')
    if masks is None:
        from s19_models import band_masks
        masks = band_masks(x.shape[-1], bands=8, transition=2, device=x.device, dtype=x.dtype)
    masks = torch.as_tensor(masks, device=x.device, dtype=x.dtype)
    if masks.ndim != 2 or masks.shape[1] != x.shape[-1] or not torch.isfinite(masks).all():
        raise ValueError('mask shape/values invalid')
    if (masks < 0).any() or (masks > 1).any() or not torch.allclose(masks.sum(0), torch.ones_like(masks[0]), atol=1e-6):
        raise ValueError('masks must be a nonnegative partition of unity')
    if ((x[:, 0].square()+x[:, 1].square()).sum(-1) <= 0).any():
        raise ValueError('zero-energy IQ packet')
    clean = rms_normalize(x)
    z = torch.complex(clean[:, 0], clean[:, 1])
    spectrum = torch.fft.fftshift(torch.fft.fft(z, dim=-1), dim=-1)
    gain = 1-attenuation*masks
    masked_spectrum = spectrum[:, None, :]*gain[None, :, :]
    source_energy = spectrum.abs().square().sum(-1)
    evm = ((masked_spectrum-spectrum[:, None, :]).abs().square().sum(-1)/source_energy[:, None]).sqrt()
    retained = masked_spectrum.abs().square().sum(-1)/source_energy[:, None]
    amplitude_spectrum = spectrum[:, None, :]*(1-evm[..., None])
    theta = 2*torch.asin((evm/2).clamp(0, 1))
    phase_spectrum = spectrum[:, None, :]*torch.polar(torch.ones_like(theta), theta)[..., None]

    def iq(s):
        time = torch.fft.ifft(torch.fft.ifftshift(s, dim=-1), dim=-1)
        return torch.stack([time.real, time.imag], dim=2)

    amp_evm = ((amplitude_spectrum-spectrum[:, None, :]).abs().square().sum(-1)/source_energy[:, None]).sqrt()
    phase_evm = ((phase_spectrum-spectrum[:, None, :]).abs().square().sum(-1)/source_energy[:, None]).sqrt()
    # Explicit artifact descriptor for this fixed circular intervention. Tail is
    # outside +/-4 samples around the zero-lag residual impulse response.
    impulse = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(-attenuation*masks, dim=-1), dim=-1), dim=-1)
    energy = impulse.abs().square()
    keep = (torch.arange(x.shape[-1], device=x.device)-x.shape[-1]//2).abs() <= 4
    tail = energy[:, ~keep].sum(-1)/energy.sum(-1).clamp_min(torch.finfo(x.dtype).tiny)
    return dict(clean=clean, masked=iq(masked_spectrum), amplitude=iq(amplitude_spectrum), phase=iq(phase_spectrum),
                evm=evm, retained_energy=retained, amplitude_evm=amp_evm, phase_evm=phase_evm,
                masks=masks, residual_impulse_tail_fraction=tail)


@torch.no_grad()
def collect_evidence(model, x, y, rx, day, device='cuda', batch=128, *, role='train', max_cell_samples=16):
    """Collect fixed teacher evidence only from externally guarded fit training IQ.

    Role/date/cell-cap checks are local defenses. The caller must additionally
    enforce fold-specific source RX identities and train/validation indices.
    No validation/held/target information is accepted or used to construct q.
    Arrays include all teacher logits, permitting independent margin/QA checks.
    """
    if role != 'train':
        raise ValueError('evidence estimation is source training only')
    if not isinstance(batch, int) or batch < 1:
        raise ValueError('batch must be a positive integer')
    if not isinstance(max_cell_samples, int) or max_cell_samples < 1:
        raise ValueError('max_cell_samples must be a positive integer')
    n = len(x)
    y, tx_labels = _labels(y, n, 'TX')
    rx, rx_labels = _labels(rx, n, 'RX')
    day, day_labels = _labels(day, n, 'day')
    if not n or not set(day_labels.tolist()) <= {1, 2}:
        raise ValueError('ManyRx evidence uses source dates1-2 only')
    if np.asarray(x).shape != (n, 2, 256):
        raise ValueError('S19 pilot expects 256-point WiSig IQ')
    for t in tx_labels:
        for r in rx_labels:
            for d in day_labels:
                if np.sum((y == t) & (rx == r) & (day == d)) > max_cell_samples:
                    raise ValueError('evidence per-cell source cap exceeded')
    arrays = {k: [] for k in ['clean_logits', 'masked_logits', 'amplitude_logits', 'phase_logits',
                              'signed_delta', 'drop', 'amplitude_signed_delta', 'amplitude_drop',
                              'phase_signed_delta', 'phase_drop', 'evm', 'amplitude_evm', 'phase_evm', 'retained_energy']}
    was_training = model.training
    model.eval()
    artifact = None
    try:
        for start in range(0, n, batch):
            stop = min(start+batch, n)
            xb = torch.as_tensor(np.array(x[start:stop], copy=True), dtype=torch.float32, device=device)
            yb = torch.as_tensor(y[start:stop], dtype=torch.long, device=device)
            cf = counterfactuals(xb)
            clean_logits = model(cf['clean'])
            clean_margin = margins(clean_logits, yb)
            arrays['clean_logits'].append(clean_logits.cpu().numpy())
            outputs = {}
            for kind, key in [('masked', 'masked_logits'), ('amplitude', 'amplitude_logits'), ('phase', 'phase_logits')]:
                outputs[kind] = torch.stack([model(cf[kind][:, j]) for j in range(8)], dim=1)
                arrays[key].append(outputs[kind].cpu().numpy())
            for kind, signed_key, drop_key in [('masked', 'signed_delta', 'drop'),
                                                ('amplitude', 'amplitude_signed_delta', 'amplitude_drop'),
                                                ('phase', 'phase_signed_delta', 'phase_drop')]:
                changed_margin = torch.stack([margins(outputs[kind][:, j], yb) for j in range(8)], dim=1)
                signed = clean_margin[:, None]-changed_margin
                arrays[signed_key].append(signed.cpu().numpy())
                arrays[drop_key].append(signed.clamp_min(0).cpu().numpy())
            for key in ['evm', 'amplitude_evm', 'phase_evm', 'retained_energy']:
                arrays[key].append(cf[key].cpu().numpy())
            artifact = dict(masks=cf['masks'].cpu().numpy(), residual_impulse_tail_fraction=cf['residual_impulse_tail_fraction'].cpu().numpy())
    finally:
        model.train(was_training)
    arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    arrays.update(y=np.array(y, copy=True), rx=np.array(rx, copy=True), day=np.array(day, copy=True), **artifact)
    summary = aggregate_contributions(arrays['drop'], y, rx, day)
    summary.update(role='train', packets=n, bands=8, attenuation=.5, max_cell_samples=max_cell_samples,
                   clean_source_accuracy=float((arrays['clean_logits'].argmax(1) == y).mean()),
                   artifacts=dict(evm_mean=float(arrays['evm'].mean()), evm_q95=float(np.quantile(arrays['evm'], .95)),
                                  retained_energy_mean=float(arrays['retained_energy'].mean()), retained_energy_min=float(arrays['retained_energy'].min()),
                                  amplitude_evm_max_error=float(np.max(np.abs(arrays['amplitude_evm']-arrays['evm']))),
                                  phase_evm_max_error=float(np.max(np.abs(arrays['phase_evm']-arrays['evm']))),
                                  residual_impulse_tail_fraction=artifact['residual_impulse_tail_fraction'].tolist(),
                                  tail_definition='Residual circular filter impulse energy outside +/-4 samples',
                                  interpretation='Spectral counterfactual may ring in time; these are not identified receiver hardware filters'),
                   controls=dict(amplitude_signed_mean=float(arrays['amplitude_signed_delta'].mean()),
                                 amplitude_drop_mean=float(arrays['amplitude_drop'].mean()),
                                 phase_signed_mean=float(arrays['phase_signed_delta'].mean()),
                                 phase_drop_mean=float(arrays['phase_drop'].mean()),
                                 role='diagnostic only; not subtracted from q and not used to choose masks'),
                   truth_labels_scope='Only source-training evidence and offline teacher scoring; never model inference')
    return arrays, summary
