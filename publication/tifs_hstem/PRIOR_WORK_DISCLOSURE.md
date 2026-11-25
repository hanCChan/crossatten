# Relationship to the companion LoRa study

The related manuscript supplied to the project is “Diagnosing Receiver-Sensitive Out-of-Band Dependencies in LoRa Radio-Frequency Fingerprint Identification,” by Ziyang Wang, Chengcheng Han, Jianhuan Chen, and Xin Zhang. The corresponding author should supply its verified publication or submission record and the review copy required by the journal.

The studies share the public multi-receiver LoRa corpus, RMS normalization, strictly in-band reconstruction, five views, the kernel-7/kernel-5 local front end, and the 32-by-64 token interface. The receiver ledger records development, historical confirmation, and the companion's six-receiver blind evaluation. The current paper uses historically developed receivers.

| Dimension | OOB study | HStem study |
|---|---|---|
| Question | How do trained classifiers depend on receiver-sensitive digital out-of-band information? | How does a classifier's cross-receiver performance change when the RF tokenizer changes at a fixed interface? |
| Intervention | Freeze classifiers and intervene on digital representation pathways, with strictly in-band reconstruction as a control. | Train paired tokenizer configurations under the same recipe. Compare four-stage HStem with a freely learned Long-Patch projection at fixed nominal support, stride, center, and attention backend. |
| Architecture | Large-kernel patch tokenization, an HSTU-style mixer, and an OOB pathway in the corresponding diagnostic arm. | A four-stage tokenizer within the strictly in-band path and two ordinary MHA blocks shared by the architectural controls. |
| Evidence | Pathway dependence across the companion's development and blind cohorts. | Six same-protocol fit pairs, parameterization and normalization controls, and external Native and common-band comparisons. |

Architecture effects are estimated from paired configurations within the current protocol. Cross-paper accuracy differences also reflect changes in mixer, information pathways, and training and evaluation cohorts. See the [technical overlap audit](../../23-hstem-mechanism-evidence/PAPER_OVERLAP_REVIEW.zh-CN.md) for page references to the author-supplied manuscript.
