"""Source-only preamble decomposition and matched-energy frozen interventions."""

import argparse
import math
from pathlib import Path

import numpy as np

from common import (DEVICE, HERE, VIEWS, FrozenStudy, centers, featurize,
                    restrict_torch, sha, tensor, torch, write)
from experiment import predict
from extract import fit_reference, physical as s33_physical
from physics import complex_iq, lag_coherence

CONDITIONS = ("native", "white", "phase_walk", "frequency_shape", "constant_cfo")
FRACTION = .12
WINDOWS = ((0, 80), (40, 120), (80, 160), (160, 240))
PHYSICAL = ("rank1_residual", "phase_residual_rad", "repeat_loss", "spectral_distance",
            "early_loss", "middle_loss", "late_loss", "tail_loss")


def decompose(x, reference, y, day, features=False):
    z = complex_iq(x)
    grid = z[:, 20:140].reshape(len(z), 6, 20)
    gram = grid @ grid.mH
    eig, vec = np.linalg.eigh(gram.cpu().numpy())
    residual_energy = np.clip(1 - eig[:, -1] / np.maximum(eig.sum(1), 1e-12), 0, 1)

    step = (grid[:, 1:] * grid[:, :-1].conj()).sum(-1)
    powers = grid.abs().square().sum(-1)
    cc = step.abs() / (powers[:, 1:] * powers[:, :-1]).sqrt().clamp_min(1e-12)
    accumulated = torch.nn.functional.pad(step.angle().cumsum(dim=1), (1, 0))
    axis = torch.arange(6, device=DEVICE, dtype=torch.float32) - 2.5
    detrended = remove_line(accumulated, axis)
    phase_residual = detrended.square().mean(1).sqrt()

    fft = torch.fft.fftshift(torch.fft.fft(z), dim=1).abs().square()
    current = fft.reshape(len(z), 16, 16).sum(-1)
    current = (current / current.sum(1, keepdim=True)).cpu().numpy()
    spectral = np.empty(len(y), dtype=np.float32)
    for (tx, date), (_, _, profile) in reference.items():
        mask = (y == tx) & (day == date)
        spectral[mask] = np.linalg.norm(np.sqrt(current[mask]) - np.sqrt(profile), axis=1) / np.sqrt(2)
    result = dict(rank1_residual=residual_energy, phase_residual_rad=phase_residual.cpu().numpy(),
                  repeat_loss=(1 - cc.amin(1).clamp_max(1)).cpu().numpy(), spectral_distance=spectral)
    for (left, right), name in zip(WINDOWS, ("early", "middle", "late", "tail")):
        result[name + "_loss"] = (1 - lag_coherence(z[:, left:right], 20)).cpu().numpy()
    assert all(np.isfinite(a).all() for a in result.values())
    if not features:
        return result
    left = tensor(vec[:, :, -1])
    shared = left[:, :, None] * (left.conj()[:, None, :] @ grid)
    residue = grid - shared
    energy = lambda z: z.abs().square().flatten(1).sum(1)
    real_residual = energy(residue) / energy(grid)
    assert np.max(np.abs(real_residual.cpu().numpy() - residual_energy)) < 8e-7
    return result, {name: spectral_readout(packets.flatten(1)) for name, packets in
                    dict(raw=grid, shared=shared, residual=residue).items()}


def spectral_readout(z):
    spectrum = torch.fft.fft(z).abs()
    relative = spectrum / spectrum.square().mean(1, keepdims=True).sqrt().clamp_min(1e-8)
    return torch.nn.functional.normalize(torch.log1p(relative))


def remove_line(wave, axis):
    return wave - wave.mean(1, keepdim=True) - (wave @ axis / axis.square().sum())[:, None] * axis


def ratio(delta, z):
    return delta.norm(dim=1) / z.norm(dim=1).clamp_min(1e-12)


def phase_match(z, phase):
    # EVM is monotone before any sample crosses pi. The upper endpoint is
    # derived from weighted RMS and checked on every packet.
    phase = phase.expand_as(z.real)
    power = z.abs().square()
    rms = ((power * phase.square()).sum(1) / power.sum(1)).sqrt().clamp_min(1e-12)
    phase = phase / rms[:, None]

    def err(scale):
        return ratio(z * (torch.polar(torch.ones_like(phase), phase * scale[:, None]) - 1), z)

    hi = torch.full((len(z),), 2 * FRACTION, device=DEVICE)
    lo = torch.zeros_like(hi)
    assert (err(hi) > FRACTION).all()
    assert (phase.abs().amax(1) * hi < math.pi).all()
    for _ in range(16):
        middle = (lo + hi) / 2
        above = err(middle) > FRACTION
        lo, hi = torch.where(above, lo, middle), torch.where(above, middle, hi)
    angle = phase * ((lo + hi) / 2)[:, None]
    return z * torch.polar(torch.ones_like(angle), angle)


def interventions(x, seed):
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    z = complex_iq(x)
    n = z.shape[1]
    axis = torch.arange(n, device=DEVICE, dtype=x.dtype) - (n-1)/2
    white = complex_iq(torch.randn(x.shape, device=DEVICE, generator=generator))
    walk = remove_line(torch.randn(z.shape, device=DEVICE, generator=generator).cumsum(1), axis)
    signs = torch.where(torch.rand((len(x), 1), device=DEVICE, generator=generator) < .5, -1., 1.)
    filtered = (z.roll(1, 1) + z.roll(-1, 1)) / 2 - z
    for name, result in (
        ("white", z + white * (FRACTION / ratio(white, z))[:, None]),
        ("phase_walk", phase_match(z, walk)),
        ("frequency_shape", z + filtered * (FRACTION / ratio(filtered, z))[:, None]),
        ("constant_cfo", phase_match(z, signs * axis)),
    ):
        relative = ratio(result - z, z).cpu().numpy()
        assert np.max(np.abs(relative - FRACTION)) < 5e-6
        yield name, torch.stack((result.real, result.imag), dim=1), relative


def engineering():
    rng = torch.Generator(device=DEVICE).manual_seed(34117)
    tiles = torch.randn((20, 2, 20), device=DEVICE, generator=rng).repeat(1, 1, 13)[..., :256]
    ref = {(0, 1): (None, None, np.full(16, 1 / 16))}
    zeros, day = np.zeros(20, dtype=int), np.ones(20, dtype=int)
    by = dict(native=tiles)
    by.update({name: x for name, x, _ in interventions(tiles, 34121)})
    outputs = {k: decompose(v, ref, zeros, day) for k, v in by.items()}
    assert np.max(outputs["constant_cfo"]["rank1_residual"]) < 6e-7
    assert np.max(outputs["constant_cfo"]["phase_residual_rad"]) < 2e-6
    assert np.max(outputs["frequency_shape"]["rank1_residual"]) < 6e-7
    assert np.min(outputs["white"]["rank1_residual"]) > 0
    return {name: {k: float(a.max()) for k, a in x.items()} for name, x in outputs.items()}


@torch.inference_mode()
def execute(s29, s33, out):
    if any(out.iterdir()):
        raise FileExistsError("S34-B requires an empty output directory")
    study = FrozenStudy(s29, s33, out)
    write(out / "EXECUTION.json", dict(protocol=sha(HERE / "PROTOCOL.zh-CN.md"), engine=sha(Path(__file__)),
              common=sha(HERE / "common.py"), physical_dose=FRACTION, equipment=str(DEVICE),
              backbone_trainings=0, optimizer_updates=0,
              raw_eq=dict(decision="STOP_NO_VERIFIABLE_LOCAL_PAIRING",
              previous_sha=sha(HERE.parent / "wisig_raw_eq_m0" / "DECISION.json")),
              engineering=engineering()))
    hashes = {}
    from experiment import build_original
    fmodel = build_original("ManyRx", 24101)
    for view_index, v in enumerate(VIEWS):
        data, audit = study.data(v, ("fit_train", "source_selection"), "s34_preamble")
        fit, held = data["fit_train"], data["source_selection"]
        roster = fit_reference(s33_physical(tensor(fit["x"])), fit)
        _, train_features = decompose(tensor(fit["x"]), roster, fit["y"], fit["day"], features=True)
        x = tensor(held["x"])
        native, held_features = decompose(x, roster, held["y"], held["day"], features=True)
        assert np.allclose(native["spectral_distance"], audit["spectral_distance"], atol=1e-6, rtol=1e-6)
        assert np.allclose(native["repeat_loss"], audit["repeat_loss"], atol=1e-6, rtol=1e-6)
        payload = {k: held[k] for k in ("y", "rx", "ids", "day")}
        payload.update({f"native_{k}": n for k, n in native.items()})
        for name, fit_x in train_features.items():
            centroid = centers(fit_x, fit)
            payload[f"{name}_prototype"] = (held_features[name] @ centroid.T).argmax(1).cpu().numpy().astype(np.uint8)
        models = list(study.models(v))
        for seed, arch, model in models:
            payload[f"{seed}_{arch}_native"] = audit[f"{seed}_{arch}_source_selection_logits"].argmax(1).astype(np.uint8)
        for kind, z, evm in interventions(x, 34100 + view_index):
            info = decompose(z, roster, held["y"], held["day"])
            payload.update({f"{kind}_{k}": n for k, n in info.items()})
            payload[f"{kind}_evm"] = evm
            features = featurize(fmodel, z)
            for seed, arch, model in models:
                payload[f"{seed}_{arch}_{kind}"] = predict(model, features).argmax(1).cpu().numpy().astype(np.uint8)
        path = out / f"PRED_{v}.npz"
        np.savez_compressed(path, **payload)
        hashes[v] = sha(path)
        print(v, "preamble and matched interventions done", flush=True)
    write(out / "RECEIPT.json", dict(predictions=hashes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("s29", "s33", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    restrict_torch()
    execute(args.s29, args.s33, args.output)
