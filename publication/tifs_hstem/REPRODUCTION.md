# Reproduction guide

HStem identifies ten enrolled LoRa transmitters from aligned windows. Native retains each method's representation on the same IQ packet identities. Strict-IB imposes the common band restriction before the external front-ends. The matched GPU comparison keeps the input views and attention architecture fixed while changing the tokenizer.

## Rebuild the submission package

From the submission source directory, use Python 3 with NumPy and Matplotlib, TeX Live with IEEEtran, latexmk, LuaLaTeX and the Markdown package, plus the Poppler utilities.

    python build_tables.py --check
    python build_figures.py
    python build_pdfs.py

The evidence ledger records the Git object and SHA-256 of each aggregate input. Fetch the named historical branches if those objects are absent. Main figures export editable SVG and embedded-font vector PDF; the architecture diagrams use the frozen method derivation. The only waveform illustration is analytic. All experimental marks derive from frozen aggregate evidence. The commands use no model or waveform I/O. The optional comparison build uses Pillow and PyMuPDF and renders both revisions at 115 dpi.

    python build_review.py

The four submission PDFs and the comparison PDF are in output/pdf. Populate an author_metadata.tex file using the example once the metadata are confirmed.

## Trace the LoRa and WiSig results

Each LoRa fold uses five source receivers and 160 training plus 40 source-validation packets per TX/RX cell. The remaining five receivers provide held evaluation, with 200 packets per cell. Seeds 201, 202 and 203 give six fits, and clean source RX-equal TX cross-entropy selects the checkpoints. Fold 1 reverses the receiver groups of fold 0.

The external and paired tokenizer records are under 24-lora-external-audit and 26-long-patch-control. Research/s42_lora_baselines contains the source-selected Ma fit and the four-method Native receiver means. Research/s43_gul_federated contains the shared timing session. Its synchronized median batch-128 time in milliseconds yields throughput as 128,000 divided by that time. Parameter plots use full checkpoint counts, while timed inference for Shen-DANN follows its transmitter path. The two HStem means, 97.333% Native and 97.335% in the matched GPU session, retain their original session assignments. Fit SD describes six-fit dispersion. Conditional bootstrap intervals resample average paired differences over receivers.

The WiSig ledger records ManyRx days 1 and 2, two source-domain views, the 11,468 and 11,500 evaluation packets, and two paired seeds per view. Its intervals resample within view. The supplement records the scope of the model-specific PCA and retains all six LoRa cohorts. Repeating historical model fits or predictions would additionally require the registered raw inputs, source caches, and selected checkpoints, which are stored separately from the submission package.

The additional baseline audit records the pinned ASKNet source, public receiver-drift protocol, registered arguments, selected checkpoint hash, and correct counts. Additional external reimplementations and their original protocol context are documented in the supplement.
