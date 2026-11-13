# Relationship to the supplied OOB study

The related work supplied with this project is titled “Diagnosing
Receiver-Sensitive Out-of-Band Dependencies in LoRa Radio-Frequency Fingerprint
Identification,” by Ziyang Wang, Chengcheng Han, Jianhuan Chen, and Xin Zhang.
The present draft does not assert an unverified publication status or DOI.

The two studies share the public 20-SDR/10-transmitter LoRa corpus, RMS and
strict in-band preprocessing, five signal channels, the local CNN front-end,
and a 32-token/64-dimensional interface. These shared elements are explicitly
identified as inherited in the new manuscript and are not offered as new
contributions. The old study's six blind receivers have already been used by
that work and are not new blind receivers for the present study.

The OOB study examines dependence on spectral pathways using frozen-model
interventions and OOB-enabled models. The present manuscript studies strictly
in-band token formation, replacing a large patch projection with a four-stage
hierarchical tokenizer, and examines parameter-matched controls, local-support
and full-rank parameterization interventions, normalization controls, and
deployment materialization. It uses ordinary MHA as its main mixer and includes
no OOB fusion branch.

Numerical gains are supported by comparisons within the present consolidated
protocol. The letter does not subtract old-paper accuracies from the new
numbers to claim architectural causality. The supplement includes the full
component/experiment overlap table and states the prior exposure of the
development, historical-confirmation and former blind receiver groups.

Before actual submission, the authors should replace the related manuscript's
temporary bibliographic description with its verified formal record and provide
the editor with the related work as required by the selected journal. This
statement describes the technical overlap; it is not an editorial ruling that
the new manuscript necessarily meets that journal's originality threshold.
