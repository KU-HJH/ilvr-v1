#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false

: "${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT}"
: "${DATA_PATH:?set DATA_PATH}"
: "${IMAGE_ROOT:?set IMAGE_ROOT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${LOG_FILE:?set LOG_FILE}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint_path "$RESUME_CHECKPOINT")
fi

COMMAND=(
  "$REPO_ROOT/venv/ilvr-venv/bin/python"
  "$REPO_ROOT/src/main.py"
  --stage stage2
  --load_model_path "$STAGE1_CHECKPOINT"
  --data_path "$DATA_PATH"
  --image_root "$IMAGE_ROOT"
  --save_model_path "$OUTPUT_DIR"
  --log_file "$LOG_FILE"
  --per_device_train_batch_size 1
  "${RESUME_ARGS[@]}"
  "$@"
)

if [[ "${STAGE2_DRY_RUN:-false}" == "true" ]]; then
  printf 'PYTHONPATH=%q TOKENIZERS_PARALLELISM=%q ' \
    "$PYTHONPATH" "$TOKENIZERS_PARALLELISM"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
