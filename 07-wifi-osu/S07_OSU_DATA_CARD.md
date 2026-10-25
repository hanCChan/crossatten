# OSU Data Card And Detection Gate

Date: 2026-10-25. Dataset: Diff Receivers / Indoor / SameTx.

## Available Records

25 transmitter classes and two receiver captures per class, 50 records, 8 GB.
Metadata describes 1 MS/s, complex float32, 915 MHz and 2022-03-08.
Little-endian decoding is finite in the pilot; opposite-endian decoding is not.
Metadata does not declare bandwidth or spreading factor. The fixed 1024-sample,
125 kHz, SF7 model is supported by source pilot periodicity, chirp slope and dechirp
tone concentration, not asserted as metadata-certified transmitter configuration.

## Pipeline

Do not reuse the historical GAI arbitrary 8192-sample periodic windows.
The new detector uses lag-1024 periodicity, eight individual upchirp tone/energy
checks, chirp-reset boundaries and a later downchirp/SFD guard. One crop per packet.
Synthetic tests cover a known boundary with CFO, noise and repeating-payload rejection.
RX1 pilot records 1–3 define the rule; RX2 records are pipeline checks for the
bidirectional experiment, not target classifier selection. No accuracy is measured.
The first failed detector attempt is retained.

The six pilot records each yield two guarded preambles in the first second.
This is a limited pipeline acceptance check, not proof of perfect packet recovery.
All 25 classes and both receivers must pass the fixed minimum counts before training.
Detection rejections and crop locations are saved; no class may be silently omitted.

## Fixed Splits

In each direction, source RX samples at 0–10 s train, 10.5–14 s validate,
and the other RX samples at 15–20 s test. Entire crops must fit within blocks.
The 0.5/1-second gaps exceed complete packet/crop lengths and measured pilot
receiver start offsets. Same-transmission pairs are separated by time blocks.
Reverse direction repeats the same rule; no per-direction detector retuning.

Only one continuous capture exists for each transmitter/receiver. Source temporal
validation is not independent-capture validation. Bidirectional fits are correlated,
not two independent receiver populations. Capture hashes and full class counts
are required, and raw data and caches stay outside Git.

## Registered Comparisons

25-class retraining: ordinary Stem + HSTU, HStem + HSTU, HStem CNN-only, CIS rank16.
All use the same detected packets and nominal strict-IB cutoff, source validation
checkpoint rule and 200-epoch budget, with seeds 5101/5102/5103, both RX directions.
Maximum 24 formal fits. The cutoff is a shared source-supported nominal assumption,
not a target-selected optimal passband. Never load Shen ten-class weights for scoring.

## Limits And Download Status

The official OSU directory and arXiv requests on 2026-10-25 failed at the mandated
proxy with `403 ERR_ACCESS_DENIED`. Cross-day captures are not local.
No bulk download was attempted and no cross-day result may be reported.
OSU success would support independent LoRa validation, not establish OSU SOTA
without matching external published protocols and independently trained baselines.
Full data inspection and frozen data-manifest gates remain required.
