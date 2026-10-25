#!/usr/bin/env bash
# Fetch third-party reference material for the literature review (read-only downloads into external/).
# Network only works through star-proxy in this environment.
set -u
export https_proxy=http://star-proxy.oa.com:3128 http_proxy=http://star-proxy.oa.com:3128
cd "$(dirname "$0")"
mkdir -p papers/meta

[ -d WaveMLP-RFF ] || timeout 240 git clone -q --depth 1 https://github.com/BeechburgPieStar/WaveMLP-RFF.git

# OpenAlex metadata (title, venue, abstract_inverted_index, open-access locations)
declare -A DOI=(
  [ibrff_tifs2026]=10.1109/TIFS.2026.3723081
  [wavemlp_imns2026]=10.1109/IMNS67862.2026.11655284
  [lightrff_phycom2026]=10.1016/j.phycom.2026.103138
  [lora_prep_icc2026]=10.1109/ICC59461.2026.11587750
  [ma_tifs2025]=10.1109/TIFS.2025.3630316
  [rifle_icccn2026]=10.1109/ICCCN69946.2026.11662855
  [triadgan_icc2026]=10.1109/ICC59461.2026.11588052
  [qi_oob_tifs2026]=10.1109/TIFS.2026.3723193
  [riei_iotj2025]=10.1109/JIOT.2024.3486068
  [bertkd_lcomm2026]=10.1109/LCOMM.2026.3656814
  [pnpdg_adhoc2026]=10.1016/j.adhoc.2026.104199
)
for k in "${!DOI[@]}"; do
  [ -s "papers/meta/$k.openalex.json" ] || timeout 60 curl -sS -m 55 "https://api.openalex.org/works/doi:${DOI[$k]}" -o "papers/meta/$k.openalex.json"
  [ -s "papers/meta/$k.crossref.json" ] || timeout 60 curl -sS -m 55 "https://api.crossref.org/works/${DOI[$k]}" -o "papers/meta/$k.crossref.json"
done

# Ma et al. (TIFS 2025) open preprint
[ -s papers/ma2025_arxiv2512.12070.pdf ] || timeout 300 curl -sS -L -m 290 -C - "https://export.arxiv.org/pdf/2512.12070" -o papers/ma2025_arxiv2512.12070.pdf
# Liverpool repository page of the ICC 2026 LoRa preprocessing paper (accepted manuscript link inside)
[ -s papers/lora_prep_icc2026_landing.html ] || timeout 60 curl -sS -L -m 55 "https://livrepository.liverpool.ac.uk/3196888/" -o papers/lora_prep_icc2026_landing.html
# open-access full texts found via OpenAlex: Bothereau et al. ICC 2026 (HAL) and RIEI (arXiv)
[ -s papers/bothereau_icc2026_hal.pdf ] || timeout 120 curl -sS -L -m 110 "https://hal.science/hal-05721219/document" -o papers/bothereau_icc2026_hal.pdf
[ -s papers/riei_arxiv2411.03636.pdf ] || timeout 200 curl -sS -L -m 190 -C - "https://export.arxiv.org/pdf/2411.03636" -o papers/riei_arxiv2411.03636.pdf
ls -la . papers papers/meta
