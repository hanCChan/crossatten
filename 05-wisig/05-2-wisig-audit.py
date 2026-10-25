#!/usr/bin/env python3
"""Freeze signal structure only from fold-0 training receivers on days 1 and 2."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
from xa.wisig import load_cell, receivers


def correlation(z, lag):
    a, b = z[:, :-lag], z[:, lag:]
    numerator = np.abs((a.conj() * b).sum(-1))
    return numerator / np.sqrt((np.abs(a) ** 2).sum(-1) * (np.abs(b) ** 2).sum(-1) + 1e-30)


def sliding_correlation(z, lag, width=32):
    a, b = z[:, :-lag], z[:, lag:]
    ab = a.conj() * b
    pa, pb = np.abs(a) ** 2, np.abs(b) ** 2
    sums = []
    for v in (ab, pa, pb):
        c = np.pad(v.cumsum(-1), ((0, 0), (1, 0)))
        sums.append(c[:, width:] - c[:, :-width])
    return np.abs(sums[0]) / np.sqrt(sums[1] * sums[2] + 1e-30)


def audit(dataset):
    manifest = json.loads((HERE / "out" / f"05-1-{dataset}-manifest.json").read_text())
    if not manifest["counts_match"]:
        raise ValueError("count gate not passed")
    train, _ = receivers(dataset, 0)
    zs = []
    for rx in train:
        for day in (1, 2):
            x, _ = load_cell(dataset, rx, day)
            zs.append(x[:, 0] + 1j * x[:, 1])
    z = np.concatenate(zs)
    lags = list(range(8, 41))
    corr = [float(np.mean(correlation(z[:, :160], lag))) for lag in lags]
    peak = lags[int(np.argmax(corr))]
    c16, c20 = corr[lags.index(16)], corr[lags.index(20)]
    # Integer multiples share STF peaks; the 0.8us period candidates identify the sample rate.
    period = 16 if c16 > c20 else 20
    fs = period / 0.8e-6
    confident = max(c16, c20) > 0.5 and abs(c16 - c20) > 0.15
    psd = np.mean(np.abs(np.fft.fftshift(np.fft.fft(z, axis=-1), axes=-1)) ** 2, axis=0)
    db = 10 * np.log10(psd / psd.max() + 1e-30)
    f = np.fft.fftshift(np.fft.fftfreq(256, 1 / fs))
    occupied = f[db >= -10]
    lo, hi = float(occupied.min()), float(occupied.max())
    spacing = fs / (period * 4)
    cutoff = max(abs(lo), abs(hi)) + spacing / 2
    stf_trace = sliding_correlation(z, period).mean(0)
    cp_start = 10 * period
    cp_len = 2 * period
    ltf_start = cp_start + cp_len
    ltf_lag = 4 * period
    cp_corr = None
    if cp_start + ltf_lag + cp_len <= z.shape[-1]:
        a = z[:, cp_start:cp_start + cp_len]
        b = z[:, cp_start + ltf_lag:cp_start + ltf_lag + cp_len]
        cp_corr = float(np.mean(np.abs((a.conj() * b).sum(-1)) /
                               np.sqrt((abs(a) ** 2).sum(-1) * (abs(b) ** 2).sum(-1) + 1e-30)))
    trace64 = sliding_correlation(z, ltf_lag).mean(0)
    return dict(dataset=dataset, source_scope=dict(fold=0, rx=train, days=[1, 2], packets=len(z)),
                fs_hz=fs, fs_confident=confident, stf_period_samples=period,
                autocorrelation_first160=dict(lags=lags, values=corr, strongest_lag=peak, lag16=c16, lag20=c20),
                psd=dict(frequencies_hz=f.tolist(), relative_db=db.tolist(), threshold_db=-10,
                         occupied_lo_hz=lo, occupied_hi_hz=hi),
                band_limit_frozen=confident, strict_ib_cutoff_hz=cutoff if confident else None,
                subcarrier_spacing_hz=spacing,
                stf=dict(expected_repetitions=10, expected_start=0, expected_end=cp_start,
                         correlation_width=32, sliding_period_correlation=stf_trace.tolist()),
                ltf=dict(expected_cp_start=cp_start, expected_cp_length=cp_len,
                         expected_first_symbol_start=ltf_start, symbol_length=ltf_lag,
                         cp_tail_correlation=cp_corr, sliding_ltf_lag_correlation=trace64.tolist(),
                         note="positions are protocol candidates; correlation evidence is reported, not assumed"))


def main():
    results = [audit(d) for d in ("ManyRx", "ManySig")]
    out = HERE / "out/05-2-audit.json"
    if out.exists():
        raise SystemExit("audit cutoff already frozen; refusing silent overwrite")
    out.write_text(json.dumps(dict(selection_scope="fold0 source RX days1,2 only", datasets=results), indent=1) + "\n")
    for result in results:
        print(json.dumps({k: v for k, v in result.items() if k not in ("psd", "stf", "ltf")}), flush=True)
    if not all(r["fs_confident"] for r in results):
        raise SystemExit("sample rate ambiguous; STOP WiSig method design")


if __name__ == "__main__":
    main()
