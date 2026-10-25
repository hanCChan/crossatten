# S07 Code And Historical Diagnostics

Date: 2026-10-25. Baseline: `ebe3225`.
Historical target results are descriptive, never new selection inputs.

## Code Facts

- Legacy channels are IQ, FFT magnitude, log amplitude, absolute phase.
- FFT indices are frequency; the others are time. Shared indices are not physical alignment.
- W1 uses 32x64 tokens, two HSTU blocks, Pre-LN, and minimum source packet CE.
- R0 keeps exact state-dictionary initialization parity with S06 W1.
- S07 separates frequency positions and globally pools spectral tokens before fusion.

## Source Audit

| Dataset | Fold | STF median coherence | STF p10 | Absolute phase jumps |
|---|---:|---:|---:|---:|
| ManyRx | 0 | 0.9919 | 0.7876 | 0.1827 |
| ManyRx | 1 | 0.9934 | 0.8125 | 0.1845 |
| ManyRx | 2 | 0.9916 | 0.8046 | 0.1838 |
| ManyRx | 3 | 0.9921 | 0.8397 | 0.1822 |
| ManySig | 0 | 0.9924 | 0.8463 | 0.1855 |
| ManySig | 1 | 0.9931 | 0.8447 | 0.1868 |
| ManySig | 2 | 0.9922 | 0.8686 | 0.1874 |
| ManySig | 3 | 0.9932 | 0.8629 | 0.1860 |

Raw target RX/days are not read by this audit. Source access is logged per fold.
Lag-20 coherence confirms repetition but does not prove a complete LTF or an exact STF boundary.
The auxiliary lag-20 feature uses only [20,160); it is a fixed prefix hypothesis, not synchronization.

## Mechanism Hypotheses

Removing mixed FFT, phase differencing, and receiver-level selection may reduce source overfit.
These are hypotheses; historical source/target gaps do not identify their cause.
Periodic products retain CFO and receiver noise; they are not receiver-invariant fingerprints.

## OSU Inventory

50 metadata files; 8000000000 IQ bytes.
Only Diff_Receivers/Indoor/SameTx is local; no Diff_Days recordings were found.
Each TX/RX has one continuous capture. Old G1 periodic windows are not aligned preambles.
OSU requires packet/boundary verification and capture/time-block leakage controls.

Detailed old scores/confusions: `historical_*_diagnostics.csv`.
Measured source features: `source_signal_audit.json`; OSU metadata: `osu_inventory.json`.
