# Cover letter

Dear Editor,

We wish to submit “Hierarchical RF Tokenization for Parameter-Efficient Cross-Receiver LoRa Fingerprint Identification” for consideration as a Regular Paper in IEEE Transactions on Information Forensics and Security.

The paper asks how much RF token construction can change cross-receiver identification when the signal representation, nominal local support, token geometry, and attention architecture are matched. HStem, a four-stage tokenizer, improves LoRa accuracy over the freely learned Long-Patch projection by 2.835 percentage points and leads in all six paired fits. The conditional 95% receiver interval is [2.253, 3.530] percentage points. Kernel parameterization, intermediate normalization, and embedding geometry help explain the observed gain.

On aligned LoRa windows, HStem reaches 97.333% receiver-equal accuracy, 93.683% worst-receiver accuracy, and 97.325% macro-F1 with 133,690 parameters. The Spec+Siamese reimplementation reaches 95.890% under the same receiver and packet protocol. HStem uses 0.96% of its parameters and delivers 1.96 times its batch-128 throughput in a shared GPU session. The manuscript reports paired receiver intervals, single-packet latency, and memory use. A matched WiSig stress test extends the structural comparison to controlled frequency and timing perturbations.

We disclose the companion manuscript “Diagnosing Receiver-Sensitive Out-of-Band Dependencies in LoRa Radio-Frequency Fingerprint Identification.” It studies digital pathway dependence through interventions on frozen classifiers. The present paper shares the corpus and in-band front end and studies token formation through paired training with a common attention backend. The separate relationship statement details the shared components and receiver exposure.

Sincerely,

The corresponding author
