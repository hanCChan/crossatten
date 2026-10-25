#!/usr/bin/env python3
"""Fixed LoRa preamble detector: periodicity, chirp reset, and downchirp guard."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
RAW = Path("/workspace/gai/data/raw/osu_lora/Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx")
SAMPLES = 1024
LENGTH = 8192


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def coherent_windows(x, hop=256, threshold=.8):
    product = x[SAMPLES:] * x[:-SAMPLES].conj()
    width = LENGTH - SAMPLES
    starts = np.arange(0, len(x) - LENGTH + 1, hop)

    def sums(a):
        c = np.concatenate(([0], np.cumsum(a, dtype=np.complex128 if np.iscomplexobj(a) else np.float64)))
        return c[starts + width] - c[starts]

    numerator = np.abs(sums(product))
    p0, p1 = sums(np.abs(x[:-SAMPLES])**2), sums(np.abs(x[SAMPLES:])**2)
    score = numerator / np.sqrt(np.maximum(p0 * p1, 1e-30))
    good = starts[score >= threshold]
    groups = np.split(good, np.flatnonzero(np.diff(good) > hop) + 1)
    return [g for g in groups if len(g)], score, starts


def spectral_tone(x, sign, bw=.125):
    n = np.arange(SAMPLES)
    reference = np.exp(sign * 1j * np.pi * bw * (n*n / SAMPLES - n))
    power = np.abs(np.fft.fft(x * reference.conj()))**2
    return float(power.max() / max(power.sum(), 1e-30)), int(power.argmax())


def detect(x, offset=0):
    groups, score, coarse = coherent_windows(x)
    records, rejected = [], []
    for group in groups:
        left, right = max(1, int(group[0]) - SAMPLES), min(len(x) - LENGTH, int(group[-1]) + SAMPLES)
        region = x[left:right + LENGTH]
        phase = np.angle(region[1:] * region[:-1].conj()) / (2*np.pi)
        smooth = uniform_filter1d(phase, size=17, mode="nearest")
        drop = smooth[:-17] - smooth[17:]
        resets, _ = find_peaks(drop, height=.07, distance=900)
        detected_resets = resets + left + 9
        boundaries = np.unique(np.concatenate((detected_resets, detected_resets - SAMPLES)))
        choices = []
        for start in boundaries:
            if start < left or start + LENGTH + 4*SAMPLES >= len(x):
                continue
            window = x[start:start + LENGTH]
            coherence = float(abs(np.vdot(window[:-SAMPLES], window[SAMPLES:])) /
                              np.sqrt(max(float(np.vdot(window[:-SAMPLES], window[:-SAMPLES]).real *
                                                np.vdot(window[SAMPLES:], window[SAMPLES:]).real), 1e-30)))
            tones = [spectral_tone(window[k*SAMPLES:(k+1)*SAMPLES], 1) for k in range(8)]
            energies = np.mean(np.abs(window.reshape(8, SAMPLES))**2, axis=1)
            concentration = float(np.median([v[0] for v in tones]))
            bins = np.array([v[1] for v in tones])
            bin_spread = float(np.max(np.abs(np.angle(np.exp(2j*np.pi*(bins-bins[0])/SAMPLES)))) * SAMPLES/(2*np.pi))
            tail = x[start + LENGTH:start + LENGTH + 4*SAMPLES]
            down = max(spectral_tone(tail[j:j+SAMPLES], -1)[0]
                       for j in range(0, len(tail)-SAMPLES+1, 128))
            # Repeating payload symbols must not pass the later SFD downchirp guard.
            if (coherence >= .8 and min(v[0] for v in tones) >= .25
                    and energies.min() / max(energies.max(), 1e-30) >= .4
                    and bin_spread <= 2 and down >= .25):
                choices.append(dict(start=offset + int(start), coherence=coherence,
                                    up_tone=concentration, tone_spread=bin_spread, down_tone=down))
        if choices:
            # Earliest qualifying eight-upchirp run, one crop per detected packet.
            records.append(min(choices, key=lambda r: r["start"]))
        else:
            rejected.append(dict(coarse_start=offset + int(group[0]),
                                 reason="chirp-boundary/tone/SFD gate failed"))
    unique = []
    for r in sorted(records, key=lambda r: r["start"]):
        if not unique or r["start"] - unique[-1]["start"] >= 16384:
            unique.append(r)
    return unique, rejected


def read_iq(path):
    meta = json.loads(path.with_suffix(".sigmf-meta").read_text())["_metadata"]
    if meta["global"]["core:datatype"] != "cf32" or meta["global"]["core:sample_rate"] != "1 MS/s":
        raise ValueError("unregistered OSU datatype/sample rate")
    x = np.memmap(path, dtype="<c8", mode="r")
    if len(x) != meta["annotations"][0]["core:sample_count"]:
        raise ValueError("OSU capture size mismatch")
    return x, meta


def prepare():
    import sys
    sys.path.insert(0, str(HERE.parent / "lib"))
    from xa.paper_train import require_committed
    path = HERE / "osu_registration.json"
    reg = json.loads(path.read_text())
    require_committed(path)
    if reg["detector_sha256"] != file_sha(Path(__file__)):
        raise ValueError("detector changed after freeze")
    cache = HERE / "cache" / "osu"
    cache.mkdir(parents=True, exist_ok=True)
    inventory, counts = [], []
    for rx in ("RX1", "RX2"):
        for tx in range(1, 26):
            raw = RAW / rx / f"Device{tx}_IQ.dat"
            mm, meta = read_iq(raw)
            dest = cache / f"{rx}_TX{tx}.npz"
            index_path = HERE / "osu_detection" / f"{rx}_TX{tx}.json"
            if index_path.exists() and dest.exists():
                entry = json.loads(index_path.read_text())
            else:
                records, rejected = [], []
                chunk = reg["detector_chunk_samples"]
                for start in range(0, len(mm), chunk):
                    end = min(len(mm), start + chunk + 6*SAMPLES + LENGTH)
                    x = np.asarray(mm[start:end])
                    if not np.isfinite(x).all():
                        raise ValueError(f"nonfinite IQ in {rx}/TX{tx}")
                    found, bad = detect(x, start)
                    records.extend(r for r in found if r["start"] < start + chunk)
                    rejected.extend(r for r in bad if r["coarse_start"] < start + chunk)
                unique = []
                for r in sorted(records, key=lambda r: r["start"]):
                    if not unique or r["start"] - unique[-1]["start"] >= 16384:
                        unique.append(r)
                parts = {}
                split_counts = {}
                for split, seconds in (("train", reg["source_train_seconds"]),
                                       ("val", reg["source_val_seconds"]), ("target", reg["target_seconds"])):
                    lo, hi = [int(t * reg["sample_rate"]) for t in seconds]
                    eligible = [r for r in unique if r["start"] >= lo and r["start"] + LENGTH <= hi]
                    starts = np.array([r["start"] for r in eligible], dtype=np.int64)
                    iq = np.array([np.stack((mm[s:s+LENGTH].real, mm[s:s+LENGTH].imag))
                                   for s in starts], dtype=np.float32).reshape(-1, 2, LENGTH)
                    parts[split], parts[f"{split}_start"] = iq, starts
                    split_counts[split] = len(starts)
                np.savez_compressed(dest, **parts)
                entry = dict(rx=rx, tx=tx, raw_bytes=raw.stat().st_size, raw_sha256=file_sha(raw),
                             metadata_sha256=file_sha(raw.with_suffix(".sigmf-meta")),
                             registration_sha256=file_sha(path), records=unique, rejected=rejected,
                             counts=split_counts, cache_sha256=file_sha(dest), finite=True)
                index_path.parent.mkdir(exist_ok=True)
                index_path.write_text(json.dumps(entry, indent=1) + "\n")
            inventory.append(entry)
            counts.append(dict(rx=rx, tx=tx, **entry["counts"]))
            print(json.dumps(dict(rx=rx, tx=tx, counts=entry["counts"])), flush=True)
    failures = [dict(rx=r["rx"], tx=r["tx"], split=split, count=r[split], minimum=minimum)
                for r in counts for split, minimum in reg["minimum_per_class"].items() if r[split] < minimum]
    manifest = dict(registration_sha256=file_sha(path), records=inventory, failures=failures,
                    passed=not failures, target_model_access=False)
    (HERE / "osu_data_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    with (HERE / "osu_detection_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rx", "tx", "train", "val", "target"])
        writer.writeheader()
        writer.writerows(counts)
    print(json.dumps(dict(data_gate=not failures, failures=failures)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if not args.pilot and not args.prepare:
        parser.error("choose --pilot or --prepare")
    if args.prepare:
        prepare()
        return
    rows = []
    for rx in ("RX1", "RX2"):
        for tx in (1, 2, 3):
            path = RAW / rx / f"Device{tx}_IQ.dat"
            mm, meta = read_iq(path)
            x = np.asarray(mm[:1000000])
            if not np.isfinite(x).all():
                raise ValueError("nonfinite little-endian pilot")
            records, rejected = detect(x)
            row = dict(rx=rx, tx=tx, pilot_samples=len(x), records=records,
                       rejected=rejected, dtype="<c8", metadata_sha256=file_sha(path.with_suffix(".sigmf-meta")))
            rows.append(row)
            print(json.dumps(dict(rx=rx, tx=tx, detections=len(records), rejected=len(rejected),
                                  starts=[r["start"] for r in records])), flush=True)
    path = HERE / "osu_pilot.json"
    if path.exists():
        previous = json.loads(path.read_text())
        archive = HERE / f"osu_pilot_attempt_{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}.json"
        if not archive.exists():
            archive.write_text(json.dumps(previous, indent=1) + "\n")
    path.write_text(json.dumps(rows, indent=1) + "\n")
    passed = all(len(row["records"]) >= 2 for row in rows)
    print(json.dumps(dict(pilot_gate=passed, formal_detection="blocked pending reviewed registration")), flush=True)


if __name__ == "__main__":
    main()
