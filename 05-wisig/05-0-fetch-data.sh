#!/usr/bin/env bash
# Reuse local ManyRx and resume UCLA-linked ManySig; never substitute another source.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data/wisig"
mkdir -p "$DATA"
export NO_PROXY=localhost,.woa.com,.oa.com,.tencent.com,.tencentcos.cn,.myqcloud.com
# ENV_VENUS_PROXY rejects the UCLA CONNECT request here; the documented star proxy works.
export HTTP_PROXY=http://star-proxy.oa.com:3128 HTTPS_PROXY=http://star-proxy.oa.com:3128
export no_proxy="$NO_PROXY" http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"

MANYRX="$ROOT/../gai/data/wisig-manyrx-kaggle/ManyRx.kaggle.v1.zip"
MANYRX_SHA=b0eca705816b03a47290286fd23ca1a10e02c6f9bd29b5461cfd92c13a86fbdd
MANYSIG_SHA=5c3d6f5aece87a86ecd8cf5a0f6bdc56535f34cfd3e8a32e8ce6dbc448c85bac
MANYSIG_ID=1szuns8MhcYocdbipK9t9TM9MLgEMklxk

printf '%s  %s\n' "$MANYRX_SHA" "$MANYRX" | sha256sum --check --status
if ! printf '%s  %s\n' "$MANYSIG_SHA" "$DATA/ManySig.official.zip" | sha256sum --check --status 2>/dev/null; then
    PAGE="$(curl --fail --location --silent --show-error --max-time 60 \
        https://cores.ee.ucla.edu/downloads/datasets/wisig/)"
    if ! printf '%s' "$PAGE" | rg --quiet "drive.google.com/file/d/$MANYSIG_ID/"; then
        printf 'STOP: UCLA official page no longer lists the expected ManySig file.\n' >&2
        exit 1
    fi
    curl --fail --location --silent --show-error --retry 4 --retry-delay 2 \
        --continue-at - --max-time 1800 \
        "https://drive.usercontent.google.com/download?id=$MANYSIG_ID&export=download&confirm=t" \
        --output "$DATA/ManySig.official.zip"
fi
printf '%s  %s\n' "$MANYSIG_SHA" "$DATA/ManySig.official.zip" | sha256sum --check
unzip -n "$MANYRX" -d "$DATA"
unzip -n "$DATA/ManySig.official.zip" -d "$DATA"
printf '%s  %s\n' \
    2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f "$DATA/ManySig.pkl" \
    | sha256sum --check
printf 'Official compact files ready; run 05-1-build-wisig-cache.py next.\n'
