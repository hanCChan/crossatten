# W5-A exact review replay

From the repository root, with Python 3.10+ and NumPy installed:

    python research/wisig_w5/audit_w5.py --portable

The same command works after extracting output/review/wisig_w5_review.zip into a new directory. No D: path, GPU, IQ file or network access is needed for this replay. Absolute historical paths in records are resolved through REPLAY_FILES.json to exact archived relative files.

This recalculates six saved prediction sets into Accuracy, all-ten-label Macro-F1, CE, all RX/TX risks, the registered conditional RX bootstrap (10,000, seed43101), and the three promotion gates. It also checks every archived weight/prediction digest, official count metadata qualification, packet/role identities, unchanged W4 A1 source, all training budgets/selection histories and paired downstream sample orders.

There are nine selected weight files: six new auxiliary/downstream checkpoints and three exactly reused A0 checkpoints. Raw IQ, optimizer/intermediate epoch files and full data caches are deliberately excluded. This is prediction-to-result and recorded-training-evidence verification; it does not retrain the neural models or independently rerun the raw-IQ collection process.

Original local neural training is one-shot guarded. train.py and p_evaluate.py require committed matching registrations and reject a second attempt. Do not rerun or modify those to reproduce the statistics. W5-B and target CRD have not been run.
