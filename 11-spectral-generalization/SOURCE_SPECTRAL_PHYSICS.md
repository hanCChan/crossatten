# Source Spectral Physics

Zero-fit route decision: **STOP**, selected=None.
All weights use only the 70% fit packets from legal fit RX/day1-2; packet validation and held RX never enter the prior. Cartesian phase avoids unwrap/branch cuts.
Equal TX/RX/day centroid scatter, complete day pairs, cell uncertainty included, minimum10 fit packets/cell and3RX/TX. Per-cell split-half reproducibility is descriptive support.
Scores use fixed5-bin reflect smoothing and floor0.5; no thresholds are searched.
Source nearest-centroid diagnostics do use held RX labels solely for route admission.

## Route Evidence
```json
{
 "F1": {
  "admitted": false,
  "checks": {
   "manyrx_positive_folds": true,
   "manyrx_centroid_gain": true,
   "manysig_not_degraded": true,
   "tx_retention": false,
   "nuisance_ratio": true,
   "prior_reproducibility": true
  },
  "details": {
   "ManyRx": {
    "mean_centroid_gain_pp": 7.902421437694878,
    "positive_folds": 4,
    "tx_retention": 0.3331184892162327,
    "nuisance_ratio_change": 0.6575297181645888,
    "half_weight_correlation": 0.9980353911021652
   },
   "ManySig": {
    "mean_centroid_gain_pp": 1.4270833333333357,
    "positive_folds": 3,
    "tx_retention": 0.3309678779417907,
    "nuisance_ratio_change": 0.6238880376851961,
    "half_weight_correlation": 0.9936007686275637
   }
  }
 },
 "F2": {
  "admitted": false,
  "checks": {
   "manyrx_positive_folds": true,
   "manyrx_centroid_gain": true,
   "manysig_not_degraded": true,
   "tx_retention": false,
   "nuisance_ratio": true,
   "prior_reproducibility": true
  },
  "details": {
   "ManyRx": {
    "mean_centroid_gain_pp": 4.170212483981782,
    "positive_folds": 4,
    "tx_retention": 0.6528011434315601,
    "nuisance_ratio_change": 0.80125945054805,
    "half_weight_correlation": 0.9980353911021652
   },
   "ManySig": {
    "mean_centroid_gain_pp": 2.3645833333333353,
    "positive_folds": 4,
    "tx_retention": 0.6151900906862601,
    "nuisance_ratio_change": 0.8315444101598692,
    "half_weight_correlation": 0.9936007686275637
   }
  }
 }
}
```

F1 k32 is a diagnostic control, not novelty. F2 is a Fisher-like variance-ratio prior, not a calibrated causal separation or new mathematical primitive.
No target-derived per-fold cutoff or receiver-specific model is constructed.
Missing TX/RX cells are excluded, not filled with zero. Both audits and final scores are RX equal.
Raw per-bin/cell evidence: `source_spectral_statistics.csv`, `priors/`, `source_centroid_probes.csv`.
