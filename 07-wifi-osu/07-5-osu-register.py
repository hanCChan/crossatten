#!/usr/bin/env python3
"""Register OSU capture/time-block protocol before complete detection or fitting."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    pilot = json.loads((HERE / "osu_pilot.json").read_text())
    if len(pilot) != 6 or any(len(r["records"]) < 2 for r in pilot):
        raise ValueError("OSU pilot failed")
    registration = dict(
        date="2026-10-25", dataset="OSU Diff_Receivers Indoor SameTx", classes=25,
        dtype="<c8", sample_rate=1000000, capture_samples=20000000,
        chirp_samples=1024, nominal_bandwidth=125000, nominal_sf=7,
        parameter_evidence="source pilot chirp sweep/tone/repetition; metadata declares sample rate only",
        detector_sha256=digest(HERE / "07-4-osu-detect.py"), pilot_sha256=digest(HERE / "osu_pilot.json"),
        source_train_seconds=[0, 10], source_val_seconds=[10.5, 14], target_seconds=[15, 20],
        guard_samples=500000, packet_crop_samples=8192, detector_chunk_samples=2000000,
        detection=dict(coherence=.8, tone_min=.25, tone_spread_bins=2, down_tone_min=.25,
                       symbol_energy_min_ratio=.4, reset_threshold=.07),
        minimum_per_class=dict(train=12, val=6, target=8),
        methods=["patch_hstu", "hstem_hstu", "hstem_cnn", "cis_rank16"],
        input_fmax=62500, input_filter="shared nominal strict-IB; pilot-derived bandwidth, not OSU metadata claim",
        seeds=[5101, 5102, 5103], directions=["RX1_to_RX2", "RX2_to_RX1"],
        recipe=dict(optimizer="AdamW", lr=.001, weight_decay=.01, epochs=200,
                    warmup_epochs=5, awgn=[0, 50], batch=64, checkpoint="minimum source temporal-val CE"),
        max_formal_fits=24, target_gate="commit complete data audit before first target model evaluation",
        limitation="one capture per TX/RX; temporal source validation is not independent capture validation",
        cross_day_status="not local; official requests denied by proxy 403; no cross-day claim",
        independent_record_sha256_required=True)
    path = HERE / "osu_registration.json"
    if path.exists() and json.loads(path.read_text()) != registration:
        raise ValueError("different OSU registration already exists")
    path.write_text(json.dumps(registration, indent=1) + "\n")
    jobs = []
    for source, target in (("RX1", "RX2"), ("RX2", "RX1")):
        for method in registration["methods"]:
            for seed in registration["seeds"]:
                jobs.append(dict(source=source, target=target, method=method, seed=seed,
                                 id=f"{source}_to_{target}_{method}__s{seed}"))
    (HERE / "jobs_osu.json").write_text(json.dumps(jobs, indent=1) + "\n")
    print(json.dumps(dict(osu_jobs=len(jobs), training="blocked until complete data audit")))


if __name__ == "__main__":
    main()
