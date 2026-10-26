# S09 Source-IQ Diagnostics

Fixed 16 packets/TX from first/last legal source RX, days1-2. No target access. Fisher ratios are in-sample feature diagnostics, not trained classifier accuracy or independent identity preservation evidence.

| Dataset | Feature | Circular extra RMS | Crop extra RMS | Channel extra RMS | Fallback fraction |
|---|---|---:|---:|---:|---:|
| ManyRx | first | 0.144110 | 0.401331 | 0.088098 | 0.0281 |
| ManyRx | centered | 0.004108 | 0.392411 | 0.088727 | 0.0281 |
| ManyRx | curvature | 0.000001 | 0.442654 | 0.089363 | 0.0281 |
| ManySig | first | 0.146230 | 0.392936 | 0.081706 | 0.0039 |
| ManySig | centered | 0.000574 | 0.382522 | 0.082198 | 0.0039 |
| ManySig | curvature | 0.000001 | 0.436335 | 0.086010 | 0.0039 |

The retained Re/Im channels are still shift sensitive. Centered weak-resultant fallback retains first-order timing sensitivity. Curvature invariance applies to ideal circular shifts only; ordinary truncation and channel perturbations need not cancel. Changes do not prove a dominant receiver nuisance.
