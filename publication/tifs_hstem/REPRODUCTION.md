# Reproduction guide

The paper studies closed-set identification of ten enrolled LoRa transmitters using fixed, aligned windows. The Native comparison retains each pipeline's RF preprocessing. Strict-IB applies the common band restriction before the external front-ends. The main results pair all methods on the same raw packet identities and receiver roles.

## Rebuild tables, figures, and submission PDFs

From the submission source directory, use Python 3 with Matplotlib and a TeX Live installation that includes IEEEtran, LuaLaTeX, the Markdown package, and the Poppler PDF tools.

    python build_tables.py
    python build_tables.py --check
    python build_figures.py
    python build_pdfs.py

The evidence ledger records the Git revision, source path, and SHA-256 for every generated table and results plot. Fetch the historical experiment branches to make those Git objects available. PDF outputs are in the output/pdf directory. The architecture and matched tokenizer diagrams are generated from the LaTeX source.

## Reproduce LoRa pairing

Use the receiver and packet manifests under configs and the Long-Patch registration record for the exact cache hashes and selected packet rows. Each fold trains on five source receivers with 160 training and 40 validation packets per TX/RX. The other five receivers supply held-out predictions. Run seeds 201, 202, and 203 in each fold. The checkpoint rule uses source validation, and the ten historical evaluation receivers define the paired bootstrap units.

The external and paired tokenizer records are stored under 24-lora-external-audit and 26-long-patch-control. The Spec+Siamese source fit and report are under research/s42_lora_baselines. Each report binds source-selected weights before matching the historical raw row and true TX label. The same-session timing script starts from GPU-resident synthetic IQ and measures complete preprocessing and TX logits. The released aggregate evidence supports table and plot reconstruction. A full model rerun also requires the local waveforms, cached packet rows, and archived checkpoints specified by those records.

## Receiver and deployment scope

Each accuracy is an equal-weight mean across ten historical development receivers and their three fits. The paired interval resamples receiver identities with the fitted models fixed. Batch-128 throughput is derived from median synchronized latency. Peak CUDA allocation includes the deployed PyTorch weights, input, and temporary tensors. The evaluation uses pre-extracted windows, with no target-side adaptation.
