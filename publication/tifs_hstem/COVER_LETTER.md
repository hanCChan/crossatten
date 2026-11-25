# Cover letter draft for author review

Dear Editor,

We wish to submit “Hierarchical RF Tokenization for Parameter-Efficient Cross-Receiver LoRa Fingerprint Identification” for consideration as a Regular Paper in IEEE Transactions on Information Forensics and Security.

We study how token construction affects closed-set identification of enrolled LoRa transmitters across receivers. HStem forms tokens through four local stages before standard multihead attention. In a registered comparison with the RF input, token geometry, nominal convolutional support, and attention backend held fixed, HStem exceeds a freely learned Long-Patch projection by 2.835 percentage points. The Long-Patch model has 3.48% more parameters. Parameterization and normalization controls provide further evidence about how the tokenizer configuration affects performance.

Under the Native protocol, HStem attains 97.333% receiver-equal accuracy with 133,690 parameters. The closest external comparison, our paper-derived Spec+Siamese implementation, attains 95.890% with 13,935,882 parameters. Their paired accuracy interval includes zero. The same-session IQ-to-logits comparison shows higher batch-128 throughput and higher single-packet latency for HStem. A small WiSig stress test provides evidence under specified perturbations. Severe sensitivity to LoRa window shifts and the prior use of the historical development receivers delimit the main claim.

We disclose the companion manuscript “Diagnosing Receiver-Sensitive Out-of-Band Dependencies in LoRa Radio-Frequency Fingerprint Identification.” It studies digital pathway dependence through interventions on frozen classifiers. The present paper shares the corpus and in-band front end and studies token formation through paired training with a common attention backend. The separate relationship statement details the shared components and receiver exposure.

Sincerely,

The corresponding author
