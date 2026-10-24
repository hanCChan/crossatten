#!/usr/bin/env python3
"""00-2  Model-free signal audit on the 10 source/dev_val receivers (confirmation RX are never read here).

A  physical checks   : dechirp alignment (Fs/BW/SF), CFO compensation, 8-chirp coherence, per-symbol residual
B  comb profile      : periodic (comb, every 8th bin of the 8192-pt FFT) vs aperiodic (off-comb) power by |f|, L/R
C  band information  : linear probes (within-RX and cross-RX under the protocol folds) on comb / off-comb
                       band features, band-power scalars, and the SNR / CFO metadata fields
D  SNR shortcut      : eta^2 of SNR field and far-band power by class; correlation far-band power vs SNR field
E  CIS coordinates   : STFT row frequencies of Shen's crop, and where the chirp energy actually sits
F  chirp residual    : class information in the coherent mean chirp vs the symbol-to-symbol residual
Output: 00-signal-audit/out/00-2-audit.json and out/00-2-tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from xa import ROOT  # noqa: E402
from xa.data import fold_roles, load_rx, source_split  # noqa: E402

FS, BW, L, NSYM, NS = 1e6, 125e3, 8192, 8, 1024
OUT = ROOT / "00-signal-audit" / "out"
WHO = "00-2-signal-audit"
F = np.fft.fftshift(np.fft.fftfreq(L, 1 / FS))
K = np.arange(L) - L // 2
ON = (K % NSYM) == 0
BANDS = {"IB": (0, 62.5e3), "near": (62.5e3, 100e3), "mid": (100e3, 200e3), "far": (200e3, 500e3 + 1)}
GROUP = 32  # 32 FFT bins = 3.906 kHz = 4 comb lines per feature group


def upchirp():
    t = np.arange(NS) / FS
    T = NS / FS
    return np.exp(1j * (np.pi * BW / T * t ** 2 - np.pi * BW * t))


def band_mask(lo, hi, side=None):
    m = (np.abs(F) > lo) & (np.abs(F) <= hi) if lo > 0 else (np.abs(F) <= hi)
    if side == "L":
        m &= F < 0
    elif side == "R":
        m &= F > 0
    return m


def group_feats(P, mask, comb):
    """log10 mean power in consecutive groups of GROUP bins (restricted to comb or off-comb bins)."""
    idx = np.flatnonzero(mask)
    feats = []
    for s in range(0, len(idx) - GROUP + 1, GROUP):
        g = idx[s:s + GROUP]
        g = g[ON[g]] if comb else g[~ON[g]]
        feats.append(np.log10(P[:, g].mean(1) + 1e-30))
    return np.stack(feats, 1)


def load(rx):
    d = load_rx(rx, "audit", WHO)
    x = d["iq"][:, 0].astype(np.float64) + 1j * d["iq"][:, 1].astype(np.float64)
    x /= np.sqrt(np.mean(np.abs(x) ** 2, 1, keepdims=True))
    snr = np.load(ROOT / "data/cache" / rx / "snr_field.npy")
    cfo = np.load(ROOT / "data/cache" / rx / "cfo_field.npy")
    return dict(x=x, y=d["y"], rows=d["rows"], snr=snr, cfo=cfo)


def eta2(v, y):
    mu = np.array([v[y == c].mean() for c in range(10)])
    return float(mu.var() / (v.var() + 1e-30))


def probe(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xtr), ytr)
    return float((clf.predict(sc.transform(Xte)) == yte).mean())


def main():
    rx10 = sorted(set(fold_roles(0)["source"]) | set(fold_roles(0)["dev_val"]))
    c = upchirp()
    data, phys, comb_rows, snrsc = {}, {}, {}, {}
    edges = [0, 25e3, 50e3, 62.5e3, 75e3, 100e3, 125e3, 150e3, 175e3, 200e3, 250e3, 300e3, 400e3, 500e3 + 1]
    for rx in rx10:
        d = load(rx)
        x = d["x"]
        P = np.abs(np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)) ** 2
        s = x.reshape(len(x), NSYM, NS)
        dc = np.abs(np.fft.fft(s * np.conj(c), axis=2)) ** 2
        dsum = dc.sum(1)
        pk = dsum.argmax(1)
        conc = np.take_along_axis(dsum, np.stack([(pk - 1) % NS, pk, (pk + 1) % NS], 1), 1).sum(1) / dsum.sum(1)
        m = s.mean(1)
        coh = np.sum(np.abs(m) ** 2, 1) / np.mean(np.sum(np.abs(s) ** 2, 2), 1)
        resid = np.sum(np.abs(s - m[:, None]) ** 2, 2) / np.sum(np.abs(m) ** 2, 1)[:, None]   # [N, 8]
        cen = (P * F).sum(1) / P.sum(1)
        phys[rx] = dict(dechirp_peak_bin_zero_frac=float(np.mean(np.minimum(pk, NS - pk) == 0)),
                        dechirp_energy_in_3_bins_median=float(np.median(conc)),
                        coherence_median=float(np.median(coh)),
                        residual_per_symbol_median=[float(v) for v in np.median(resid, 0)],
                        # independent noise gives -1/7 between adjacent residuals (they sum to zero); larger = slow drift
                        residual_adjacent_corr_median=float(np.median(np.mean(
                            [np.real(np.sum(r[:, k] * np.conj(r[:, k + 1]), 1)) / np.sqrt(np.sum(np.abs(r[:, k]) ** 2, 1) * np.sum(np.abs(r[:, k + 1]) ** 2, 1))
                             for r in [s - m[:, None]] for k in range(NSYM - 1)], 0))),
                        centroid_hz_median=float(np.median(cen)), centroid_hz_iqr=float(np.subtract(*np.percentile(cen, [75, 25]))),
                        cfo_field_khz_range=[float(d["cfo"].min() / 1e3), float(d["cfo"].max() / 1e3)],
                        spearman_centroid_vs_cfo_field=float(spearmanr(cen, d["cfo"])[0]),
                        snr_field_median=float(np.median(d["snr"])))
        rows = []
        for side in ("L", "R"):
            for lo, hi in zip(edges[:-1], edges[1:]):
                mk = band_mask(lo, hi, side)
                rows.append(float(10 * np.log10(np.median(P[:, mk & ON].mean(1) / P[:, mk & ~ON].mean(1)))))
        comb_rows[rx] = rows
        far = np.log10(P[:, band_mask(*BANDS["far"])].mean(1))
        snrsc[rx] = dict(eta2_snr_field=eta2(d["snr"], d["y"]), eta2_cfo_field=eta2(d["cfo"], d["y"]),
                         eta2_far_power=eta2(far, d["y"]), spearman_far_power_vs_snr=float(spearmanr(far, d["snr"])[0]))
        # features for probes
        feats = {}
        for b, (lo, hi) in BANDS.items():
            sides = [None] if b == "IB" else [None, "L", "R"]
            for sd in sides:
                mk = band_mask(lo, hi, sd)
                name = b if sd is None else f"{b}-{sd}"
                feats[f"{name}|comb"] = group_feats(P, mk, True)
                feats[f"{name}|offcomb"] = group_feats(P, mk, False)
                if b != "IB":
                    feats[f"{name}|ratio_scalar"] = np.log10(P[:, mk].mean(1) / P[:, band_mask(*BANDS["IB"])].mean(1))[:, None]
        feats["field|snr"] = d["snr"][:, None].astype(np.float64)
        feats["field|cfo"] = d["cfo"][:, None].astype(np.float64)
        Pm = np.abs(np.fft.fftshift(np.fft.fft(m, axis=1), axes=1)) ** 2              # mean chirp, 1024 bins (976.6 Hz)
        Pr = (np.abs(np.fft.fftshift(np.fft.fft(s - m[:, None], axis=2), axes=2)) ** 2).mean(1)
        f1k = np.fft.fftshift(np.fft.fftfreq(NS, 1 / FS))
        for nm, PP in (("meanchirp", Pm), ("residual", Pr)):
            for b, (lo, hi) in (("IB", (0, 62.5e3)), ("OOB<=200k", (62.5e3, 200e3))):
                mk = (np.abs(f1k) <= hi) if lo == 0 else ((np.abs(f1k) > lo) & (np.abs(f1k) <= hi))
                idx = np.flatnonzero(mk)
                g = [np.log10(PP[:, idx[i:i + 4]].mean(1) + 1e-30) for i in range(0, len(idx) - 3, 4)]
                feats[f"{nm}-{b}|chirp"] = np.stack(g, 1)
        tr, va = source_split(d["y"], d["rows"])
        data[rx] = dict(feats=feats, y=d["y"], tr=tr, va=va)
        print(rx, "loaded", flush=True)

    names = list(next(iter(data.values()))["feats"].keys())
    probes = {}
    for nm in names:
        within = [probe(data[rx]["feats"][nm][data[rx]["tr"]], data[rx]["y"][data[rx]["tr"]],
                        data[rx]["feats"][nm][data[rx]["va"]], data[rx]["y"][data[rx]["va"]]) for rx in rx10]
        cross = []
        for fold in (0, 1):
            r = fold_roles(fold)
            Xtr = np.concatenate([data[rx]["feats"][nm][data[rx]["tr"]] for rx in r["source"]])
            ytr = np.concatenate([data[rx]["y"][data[rx]["tr"]] for rx in r["source"]])
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Xtr), ytr)
            cross.append([float((clf.predict(sc.transform(data[rx]["feats"][nm])) == data[rx]["y"]).mean()) for rx in r["dev_val"]])
        probes[nm] = dict(dim=int(data[rx10[0]]["feats"][nm].shape[1]), within_rx_mean=float(np.mean(within)),
                          within_rx_min=float(np.min(within)), within_per_rx=dict(zip(rx10, within)),
                          cross_rx_mean=float(np.mean([np.mean(c) for c in cross])),
                          cross_rx_worst=float(np.min(cross)), cross_per_fold=cross)
        print(f"{nm:28s} dim={probes[nm]['dim']:4d} within={probes[nm]['within_rx_mean']:.3f} cross={probes[nm]['cross_rx_mean']:.3f}", flush=True)

    # E: CIS coordinates (Shen crop) and where the chirp energy sits in the 128-bin STFT
    x0 = load(rx10[0])["x"][:200]
    fr = np.lib.stride_tricks.sliding_window_view(x0, 128, axis=1)[:, ::64]
    S = np.abs(np.fft.fftshift(np.fft.fft(fr, axis=2), axes=2)) ** 2
    row_db = 10 * np.log10(S.mean((0, 1)) / S.mean((0, 1)).max())
    rows_hz = (np.arange(128) - 64) * FS / 128
    hot = np.flatnonzero(row_db > -10)
    cis = dict(receiver=rx10[0], crop_rows=[38, 89], crop_hz=[float(rows_hz[38]), float(rows_hz[89])], n_rows=52,
               rows_within_10db_of_peak=[int(hot.min()), int(hot.max())],
               rows_within_10db_hz=[float(rows_hz[hot.min()]), float(rows_hz[hot.max()])],
               row_db_profile={f"{rows_hz[i] / 1e3:+.1f}k": float(row_db[i]) for i in range(30, 99, 4)})
    res = dict(receivers=rx10, physical=phys, comb_profile=dict(edges_hz=edges, per_rx=comb_rows),
               snr_shortcut=snrsc, probes=probes, cis_coordinates=cis)
    (OUT / "00-2-audit.json").write_text(json.dumps(res, indent=1))
    print("wrote", OUT / "00-2-audit.json")


if __name__ == "__main__":
    main()
