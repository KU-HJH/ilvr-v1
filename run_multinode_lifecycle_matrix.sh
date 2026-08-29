#!/usr/bin/env bash
set -euo pipefail

repository_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$repository_root:$repository_root/src"

exec "$repository_root/venv/ilvr-venv/bin/python" \
    -m src.multinode lifecycle-matrix "$@"
