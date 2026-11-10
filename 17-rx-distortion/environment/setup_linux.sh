#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this script inside an approved Linux/WSL2 environment.' >&2
  exit 1
fi
crossatten_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
crossatten_env_dir="${CROSSATTEN_LINUX_ENV:-$HOME/.venvs/crossatten4060}"
python3 -m venv "$crossatten_env_dir"
"$crossatten_env_dir/bin/python" -m pip install --upgrade pip setuptools wheel
"$crossatten_env_dir/bin/python" -m pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu124
"$crossatten_env_dir/bin/python" -m pip install -r "$crossatten_script_dir/requirements-analysis.txt"
"$crossatten_env_dir/bin/python" -m pip check
"$crossatten_env_dir/bin/python" -u "$crossatten_script_dir/verify_gpu.py"
echo "Environment ready: $crossatten_env_dir"
echo 'No scientific fits are started by this setup script.'
