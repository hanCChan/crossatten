"""Deterministic mathematical counterexamples; no RF data or model loading.

Run with Python + numpy + matplotlib. Outputs are synthetic illustrations,
not WiSig measurements, hypothesis acceptance, or a novel identifiability theorem.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(out: Path) -> dict:
    rng = np.random.default_rng(20261110)
    n = 256
    f = (np.arange(n) + 0.5) / n
    # Orthonormal DCT-II synthesis basis, explicitly formed in float64.
    q = np.cos(np.pi * f[:, None] * np.arange(n)[None, :])
    q[:, 0] *= np.sqrt(1 / n)
    q[:, 1:] *= np.sqrt(2 / n)
    assert np.max(np.abs(q.T @ q - np.eye(n))) < 1e-12
    records = []

    def add(name, **values):
        records.append(dict(name=name, status="PASS", **values))

    # 1: A shared gauge does not destroy relative transmitter ratios.
    x = np.exp(rng.normal(0, 0.2, (2, n)) + 1j*rng.normal(0, 0.2, (2, n)))
    h = np.exp(rng.normal(0, 0.2, (3, n)) + 1j*rng.normal(0, 0.2, (3, n)))
    a = np.exp(0.3*np.cos(2*np.pi*f) + 0.2j*np.sin(2*np.pi*f))
    y = x[:, None, :] * h[None, :, :]
    y_alt = (x*a)[:, None, :] * (h/a)[None, :, :]
    gauge_error = float(np.max(np.abs(y-y_alt)))
    ratio_error = float(np.max(np.abs((x*a)[0]/(x*a)[1] - x[0]/x[1])))
    assert gauge_error < 1e-12 and ratio_error < 1e-12
    assert np.linalg.norm(x*a-x) > 0.1
    add("shared_gauge_preserves_observations_and_relative_tx", observation_max_error=gauge_error,
        relative_tx_max_error=ratio_error)

    # 2: Even a smooth positive multiplier can connect two distinct TX spectra.
    xa = np.exp(0.12*np.cos(9*np.pi*f) + 0.1j*np.sin(5*np.pi*f))
    g = np.exp(0.24*np.cos(np.pi*f))
    xb = xa*g
    orbit_error = float(np.max(np.abs(xb-g*xa)))
    tx_distance = float(np.linalg.norm(xb-xa))
    assert orbit_error < 1e-12 and tx_distance > 0.1
    assert g.min() > 0 and np.max(np.abs(g*(1/g)-1)) < 1e-12
    add("smooth_multiplier_tx_orbit_collision", mapping_max_error=orbit_error,
        distinct_tx_distance=tx_distance, gain_min=float(g.min()), gain_max=float(g.max()),
        implication="Exact invariance to this group identifies these distinct spectra.")

    # 3: Exact low-quefrency projection cancels a smooth TX difference as well as RX.
    common = 0.35*np.cos(19*np.pi*f)
    tx = np.stack([common-0.18*np.cos(np.pi*f), common+0.18*np.cos(np.pi*f)])
    rx = np.stack([-0.22*np.cos(3*np.pi*f), np.zeros(n), 0.22*np.cos(3*np.pi*f)])
    z = tx[:, None, :] + rx[None, :, :]
    low = q[:, :8]
    residual = z - (z @ low) @ low.T
    before = float(np.linalg.norm(z[1, 0]-z[0, 0]))
    after = float(np.linalg.norm(residual[1, 0]-residual[0, 0]))
    rx_after = float(np.linalg.norm(residual[:, 2]-residual[:, 0]))
    assert before > 0.1 and after < 1e-12 and rx_after < 1e-12
    add("frequency_projection_can_remove_tx_and_rx", tx_distance_before=before,
        tx_distance_after=after, rx_distance_after=rx_after, discarded_dct_modes=8)

    # 4: Frame-axis mean subtraction removes static identity as well as channel.
    k = np.arange(8)
    dynamics = np.sin(2*np.pi*k[:, None]/8)*np.cos(13*np.pi*f)[None, :]
    frames = z[:, :, None, :] + dynamics[None, None, :, :]
    centered = frames - frames.mean(axis=2, keepdims=True)
    temporal_tx_after = float(np.linalg.norm(centered[1]-centered[0]))
    temporal_rx_after = float(np.linalg.norm(centered[:, 2]-centered[:, 0]))
    assert temporal_tx_after < 1e-12 and temporal_rx_after < 1e-12
    assert np.linalg.norm(centered) > 1
    add("temporal_centering_can_remove_static_identity", tx_distance_after=temporal_tx_after,
        rx_distance_after=temporal_rx_after, nonzero_dynamic_norm=float(np.linalg.norm(centered)))

    # 5: Complex linearity permits U(1) equivariance; split ReLU generally breaks it.
    w = rng.normal(size=(5, 7)) + 1j*rng.normal(size=(5, 7))
    v = rng.normal(size=7) + 1j*rng.normal(size=7)
    phase = np.exp(0.61j)
    def relu_split(value):
        return np.maximum(value.real, 0) + 1j*np.maximum(value.imag, 0)
    linear_error = float(np.linalg.norm(w@(phase*v)-phase*(w@v)))
    split_error = float(np.linalg.norm(relu_split(w@(phase*v))-phase*relu_split(w@v)))
    assert linear_error < 1e-12 and split_error > 0.1
    add("complex_linear_vs_split_relu_equivariance", linear_residual=linear_error,
        split_relu_residual=split_error, phase_radians=0.61)

    # 6: Balanced binary classes, predictor h(X)=X; hardware-like conditional flip shift.
    def joint(error):
        # Rows Y, columns X; class priors remain 1/2 in every domain.
        return np.array([[1-error, error], [error, 1-error]]) / 2
    loss = np.array([[0, 1], [1, 0]])
    source = np.stack([joint(e) for e in (0.1, 0.2, 0.3)])
    weights = np.array([0.2, 0.3, 0.5])
    risks = (source*loss).sum(axis=(1, 2))
    mixture = np.einsum("r,rij->ij", weights, source)
    mix_risk = float((mixture*loss).sum())
    target = joint(0.9)
    target_risk = float((target*loss).sum())
    assert np.allclose(source.sum(axis=2), 0.5) and np.allclose(target.sum(axis=1), 0.5)
    assert np.isclose(mix_risk, weights@risks) and mix_risk <= risks.max()
    assert target_risk > risks.max()
    add("source_mixture_bound_does_not_cover_arbitrary_target", source_risks=risks.tolist(),
        weights=weights.tolist(), mixture_risk=mix_risk, target_risk=target_risk,
        target_outside_source_convex_hull=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(f, tx[0], label="TX A")
    axes[0].plot(f, tx[1], label="TX B")
    axes[0].set(title="A. Distinct smooth TX signatures", xlabel="Normalized frequency", ylabel="Synthetic log magnitude")
    axes[0].legend(frameon=False)
    axes[1].plot(f, residual[0, 0], label="TX A after projection")
    axes[1].plot(f, residual[1, 0], "--", label="TX B after projection")
    axes[1].set(title="B. Removing RX can also erase TX", xlabel="Normalized frequency")
    axes[1].legend(frameon=False)
    axes[2].bar(["Source 1", "Source 2", "Source 3", "Mixture", "New target"],
                [*risks, mix_risk, target_risk], color=["#4C78A8"]*3+["#72B7B2", "#E45756"])
    axes[2].axhline(risks.max(), color="#333333", linestyle=":", label="Worst source")
    axes[2].set(title="C. Source groups do not bound all targets", ylabel="Synthetic classification risk", ylim=(0, 1))
    axes[2].tick_params(axis="x", labelrotation=25)
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Mathematical counterexamples only — no WiSig data, fits, or performance estimates", fontsize=12)
    fig.savefig(out / "SYNTHETIC_COUNTEREXAMPLES.png", dpi=180)
    plt.close(fig)
    result = dict(status="PASS", scope="SYNTHETIC_MATHEMATICAL_CHECKS_ONLY", seed=20261110,
                  checks_passed=len(records), checks=records, numpy_version=np.__version__,
                  matplotlib_version=matplotlib.__version__, experimental_iq_reads=0,
                  checkpoint_loads=0, model_fits=0)
    (out / "SYNTHETIC_CHECKS.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    result = run(Path(__file__).resolve().parent)
    print(json.dumps({k: result[k] for k in ("status", "scope", "checks_passed", "model_fits")}))
