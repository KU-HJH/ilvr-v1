#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$script_dir"
exec "$script_dir/venv/ilvr-venv/bin/python" -m src.multinode train-stage1 "$@"
