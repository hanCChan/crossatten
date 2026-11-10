"""S19 WiSig spectral controls on a common complex spectrum.

Only the added spectral module differs between arms. The legacy HStem, MHA,
normalization, FFT convention and first-order phase-feature rule remain intact.
The +/-20% bound applies to the filter gain before packet RMS normalization;
it is not a bound on individual normalized RF feature values.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
from torch import nn

from s17_models import build as build_baseline
from wifi import phase
from xa.frontends import rms_normalize


BANDS = 8
ALPHA = .2
TRANSITION = 2


def band_masks(length=256, bands=8, transition=2, device=None,
               dtype=torch.float32):
    """Eight contiguous bands with a periodic raised-sine edge transition.

    Return ``[bands, length]`` nonnegative weights in fftshift bin order. The
    masks form a partition of unity. At each boundary, including the periodic
    Nyquist boundary, ``transition`` bins on either side cross-fade between
    adjacent bands; no frequencies are discarded or given a LoRa cutoff.
    """
    if (isinstance(length, bool) or isinstance(bands, bool)
            or int(length) != length or int(bands) != bands
            or length < 3 or bands < 1 or length % bands):
        raise ValueError('length must be divisible by a positive band count')
    if (isinstance(transition, bool) or int(transition) != transition
            or transition < 0 or 2*transition > length//bands):
        raise ValueError('invalid transition width')
    if not dtype.is_floating_point:
        raise ValueError('band masks require a floating dtype')
    length, bands, transition = int(length), int(bands), int(transition)
    width = length//bands
    indices = torch.arange(length, device=device)
    masks = torch.nn.functional.one_hot(indices//width, num_classes=bands).T.to(dtype)
    if transition and bands > 1:
        offsets = torch.arange(-transition, transition, device=device)
        position = offsets.to(dtype)+.5
        right = .5*(1+torch.sin(math.pi*position/(2*transition)))
        for j in range(bands):
            bins = (j*width+offsets) % length
            masks[:, bins] = 0
            masks[(j-1) % bands, bins] = 1-right
            masks[j, bins] = right
    return masks


def spectrum_from_iq(x):
    """Legacy RMS normalization followed by unscaled, fftshift-ordered FFT."""
    if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 2 or x.shape[-1] < 3:
        raise ValueError('IQ must be a real [batch,2,length>=3] tensor')
    if not x.is_floating_point():
        raise ValueError('IQ must have a floating dtype')
    raw = rms_normalize(x)
    return torch.fft.fftshift(torch.fft.fft(torch.complex(raw[:, 0], raw[:, 1])), dim=-1)


def features_from_spectrum(s, reference_spectrum=None):
    """Rebuild all five canonical RF channels from one complex spectrum.

    With a reference spectrum ``S=FFT(rms(x))``, the relative Parseval factor
    ``rsqrt(1 + mean(|S'|^2-|S|^2)/length)`` makes ``S'`` equivalent to applying
    the spectral filter to ``x`` before its original RMS normalization. This
    avoids IFFT/FFT round trips and has exactly unit correction at identity.
    The formula retains the legacy ``+1e-6`` RMS convention algebraically.
    All amplitude thresholds and phase products are then rebuilt from the
    corrected spectrum; features are never scaled as independent channels.
    """
    if not isinstance(s, torch.Tensor) or not s.is_complex() or s.ndim != 2 or s.shape[-1] < 3:
        raise ValueError('spectrum must be complex [batch,length>=3]')
    if reference_spectrum is not None:
        if (not isinstance(reference_spectrum, torch.Tensor)
                or not reference_spectrum.is_complex()
                or reference_spectrum.shape != s.shape
                or reference_spectrum.device != s.device
                or reference_spectrum.dtype != s.dtype):
            raise ValueError('reference spectrum must match the spectrum')
        difference = (s.abs().square()-reference_spectrum.abs().square()).mean(-1, keepdim=True)/s.shape[-1]
        relative_rms = (1+difference).clamp_min(1e-12).rsqrt()
        s = s*relative_rms
    magnitude = s.abs()
    norm = magnitude.square().mean(-1, keepdim=True).clamp_min(1e-6).sqrt()
    logmag = torch.log1p(magnitude)
    logmag = (logmag-logmag.mean(-1, keepdim=True))/logmag.std(-1, keepdim=True).clamp_min(1e-6)
    base = torch.stack((s.real/norm, s.imag/norm, logmag), dim=1)
    return phase.phase_features_from_base(base, 'first')


class SpectralModulator(nn.Module):
    """Static filter or a small label-free input-dependent bounded filter."""

    def __init__(self, arm, length=256):
        super().__init__()
        if arm not in {'B1', 'B2', 'B3', 'B4'}:
            raise ValueError('unknown spectral arm')
        self.arm = arm
        self.alpha = ALPHA
        self.register_buffer('masks', band_masks(length), persistent=True)
        if arm == 'B1':
            self.theta = nn.Parameter(torch.zeros(BANDS))
        elif arm == 'B2':
            # 8*16+16+16*8 = 272 active parameters.
            self.gate = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8, bias=False))
        else:
            # 24*8+8+8*8+8 = 272 active parameters, shared by B3 and B4.
            self.gate = nn.Sequential(nn.Linear(24, 8), nn.GELU(), nn.Linear(8, 8, bias=True))
        if arm != 'B1':
            nn.init.zeros_(self.gate[-1].weight)
            if self.gate[-1].bias is not None:
                nn.init.zeros_(self.gate[-1].bias)
        self.last_gain = None

    def descriptors(self, spectrum):
        weights = self.masks.to(dtype=spectrum.real.dtype)
        mass = weights.sum(-1).clamp_min(1e-12)
        power = spectrum.abs().square()
        band_power = power@weights.T/mass
        # Relative log1p power prevents an arbitrary FFT/RMS scale from
        # dominating the small gate. No class or receiver labels are used.
        log_power = torch.log1p(band_power/power.mean(-1, keepdim=True).clamp_min(1e-12))
        if self.arm == 'B2':
            return log_power
        original = features_from_spectrum(spectrum)
        phase_re = original[:, 3]@weights.T/mass
        phase_im = original[:, 4]@weights.T/mass
        return torch.cat((log_power, phase_re, phase_im), dim=-1)

    def forward(self, spectrum):
        if (not spectrum.is_complex() or spectrum.ndim != 2
                or spectrum.shape[-1] != self.masks.shape[-1]):
            raise ValueError('spectral module expects its declared complex FFT layout')
        if self.arm == 'B1':
            logits = self.theta.unsqueeze(0).expand(len(spectrum), -1)
        else:
            logits = self.gate(self.descriptors(spectrum))
        centered = logits-logits.mean(-1, keepdim=True)
        # A convex combination of bounded band gains stays in [.8,1.2].
        gain = 1+self.alpha*(centered.tanh()@self.masks.to(dtype=spectrum.real.dtype))
        self.last_gain = gain
        return spectrum*gain, logits


def _spectral_features(self, x):
    spectrum = spectrum_from_iq(x)
    filtered, logits = self.spectral_modulator(spectrum)
    self.last_evidence_logits = logits
    self.last_spectral_gain = self.spectral_modulator.last_gain
    return features_from_spectrum(filtered, reference_spectrum=spectrum)


def build_model(arm, dataset, seed):
    """Build a paired legacy backbone; only B4's training loss uses evidence.

    B3 and B4 are the same module and initialization, with equal backbone RNG
    state, so the parent training loop can isolate evidence-loss supervision.
    The bound feature method and spectral module are recreated by this factory
    before loading a checkpoint; checkpoint files contain ordinary state_dicts.
    """
    if arm not in {'B0', 'B1', 'B2', 'B3', 'B4'}:
        raise ValueError('unknown experiment arm')
    if dataset not in {'ManyRx', 'ManySig'}:
        raise ValueError('this registered frontend supports WiSig only')
    model = build_baseline(dataset, seed)
    model.spectral_arm = arm
    model.last_evidence_logits = None
    model.last_spectral_gain = None
    if arm != 'B0':
        # Extra-module initialization cannot change the backbone or consume
        # random draws that the common training recipe might otherwise use.
        with torch.random.fork_rng(devices=[]):
            # The added modules are constructed on CPU. Seed only its default
            # generator: torch.manual_seed would also alter CUDA RNG state,
            # which a fork with devices=[] intentionally does not save.
            torch.random.default_generator.manual_seed(seed+19000)
            model.spectral_modulator = SpectralModulator(arm)
        model.features = MethodType(_spectral_features, model)
    return model
