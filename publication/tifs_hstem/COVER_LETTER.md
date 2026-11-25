# Cover letter

Dear Editor,

We submit “Hierarchical RF Tokenization for Parameter-Efficient Cross-Receiver LoRa Fingerprint Identification” for consideration as a Regular Paper in IEEE Transactions on Information Forensics and Security.

For receiver-side identification of enrolled transmitters, tokenization determines how local RF features reach the attention backbone. The paper isolates this design choice by matching the input representation, nominal convolutional support, token grid, and attention architecture. Our four-stage HStem tokenizer improves LoRa accuracy by 2.835 percentage points over a freely learned Long-Patch projection. All six paired fits favor HStem, and the conditional 95% receiver interval is [2.253, 3.530] percentage points. Kernel controls, a BN and GELU factorial comparison, and layerwise source feature analysis explain different aspects of this result.

Under the common Native receiver and packet protocol, HStem reaches 97.333% receiver-equal accuracy, 97.325% macro-F1, and 93.683% worst-receiver accuracy with 133,690 parameters. Relative to the Ma Spec+Siamese reimplementation at 95.890% accuracy, HStem reduces the parameter count by 99.04% and provides 1.96 times the batch-128 throughput in a shared GPU session. The manuscript reports receiver-level results, single-packet latency, and peak memory. A separate WiSig comparison measures the response of matched tokenizers to controlled frequency and timing perturbations.

We disclose the companion manuscript “Diagnosing Receiver-Sensitive Out-of-Band Dependencies in LoRa Radio-Frequency Fingerprint Identification.” The studies share the LoRa corpus, in-band reconstruction, five-view representation, local front-end, and token interface. The companion study intervenes on frozen classifier pathways to assess receiver-sensitive out-of-band dependence. This paper uses paired training on a common attention backend to study progressive RF token formation. The accompanying disclosure describes the shared components and receiver exposure.

Sincerely,

The corresponding author
