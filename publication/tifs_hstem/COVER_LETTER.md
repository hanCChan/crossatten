# Cover letter

Dear Editor,

We wish to submit “Hierarchical RF Tokenization for Parameter-Efficient Cross-Receiver LoRa Fingerprint Identification” for consideration as a Regular Paper in IEEE Transactions on Information Forensics and Security.

The paper introduces HStem, a four-stage RF tokenizer that forms compact tokens for standard multihead attention. It improves cross-receiver identification by 2.835 percentage points over a freely learned Long-Patch projection with the same input, token geometry, nominal convolutional support, and attention backend. Long-Patch has 3.48% more parameters. Matched parameterization and normalization controls show how local token construction shapes accuracy.

On aligned multi-receiver LoRa windows, HStem reaches 97.333% receiver-equal accuracy with 133,690 parameters. Compared with the Spec+Siamese implementation, HStem uses approximately one hundredth of the parameters and offers higher batch-128 throughput on the same GPU. The manuscript includes conditional paired intervals and a full account of single-packet latency, memory, and receiver-level outcomes. A WiSig stress test extends the comparison to controlled frequency and timing perturbations.

We disclose the companion manuscript “Diagnosing Receiver-Sensitive Out-of-Band Dependencies in LoRa Radio-Frequency Fingerprint Identification.” It studies digital pathway dependence through interventions on frozen classifiers. The present paper shares the corpus and in-band front end and studies token formation through paired training with a common attention backend. The separate relationship statement details the shared components and receiver exposure.

Sincerely,

The corresponding author
