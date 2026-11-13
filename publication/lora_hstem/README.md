# LoRa HStem manuscript review package

This is an English IEEE-style review draft, prepared from already completed
experiments on 13 November 2026. It is not an uploaded journal submission.
The verified main model, historical results, receiver roles and numerical
failure records are unchanged. No new scientific fit, IQ read, checkpoint load,
or receiver scoring was performed during manuscript preparation.

## Files

- `main.tex`: six-page main manuscript after compilation.
- `supplement.tex`: supplementary protocol, receiver tables, mechanism gates,
  conversion/precision checks, historical comparisons and prior-paper disclosure.
- `references.bib`: primary-source references and the explicitly identified
  author-supplied OOB manuscript, without an invented DOI/publication status.
- `figures/`, `tables/`, `data/`: English figures, generated LaTeX tables,
  frozen source CSV/JSON files and relevant implementation snapshots.
- `EVIDENCE_MANIFEST.json`: original paths and SHA256 hashes for exact source
  copies; checks that receiver aggregates reproduce the main result tables.
- `SUBMISSION_REVIEW.zh-CN.md`: concrete readiness assessment and remaining
  author/editorial inputs.
- `COVER_LETTER_DRAFT.md` and `PRIOR_WORK_DISCLOSURE.md`: editable draft texts.

## Build

With Python and TeX Live (including IEEEtran), run:

    python build.py

Poppler `pdftoppm`/`pdfinfo` are used for rendering and page checks. Alternatively,
in this directory run `pdflatex main`, `bibtex main`, and `pdflatex main` twice;
run `pdflatex supplement` twice for the supplement. Upload this directory's
contents to Overleaf and select `main.tex` as the main document if preferred.

The supplied tables and figures are already generated. Rebuilding the LaTeX
does not require the original D: drive or IQ datasets. `prepare_evidence.py`
records the original local preparation workflow and intentionally requires its
frozen source archive; it is not needed to compile or read this package elsewhere.

The main and supplementary PDFs are supplied separately and inside the source
archive. No raw LoRa IQ, model weights or full prior-paper PDF is redistributed.
The original source/data licenses and inherited implementation provenance remain
applicable to the included snapshots. This package does not relicense them.

## Scope

The main comparison concerns parameter efficiency under an aligned, strictly
in-band LoRa protocol. It does not establish lower MACs/latency, synchronization
robustness, transmitter-only hardware separation, a universal LoRa/Wi-Fi method,
an exhaustive SOTA ranking, or fresh independent receiver confirmation.

Author names, affiliations, funding, final venue and the related OOB paper's
publication metadata must be supplied/confirmed by the authors. The PDFs use
an anonymous review author line until that information is settled. The checklist
distinguishes these editorial inputs from scientific limitations that cannot be
removed by editing.
