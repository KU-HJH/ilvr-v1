#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false

: "${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT}"
: "${DATA_PATH:?set DATA_PATH}"
: "${IMAGE_ROOT:?set IMAGE_ROOT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

experiment_id=${OUTPUT_DIR##*/}
canonical_log_root='/root/workspace/ILVR/logs/zero-stage-compatibility'
result_dir="/root/workspace/ILVR/results/zero-stage-compatibility/$experiment_id"
length_manifest="$result_dir/stage2-length-manifest.json"
processor_source=${RESUME_CHECKPOINT:-$STAGE1_CHECKPOINT}
if [[ -z "${LOG_FILE:-}" ]]; then
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    LOG_FILE="$canonical_log_root/$experiment_id/resume-$(date +%Y%m%dT%H%M%S)-$$.log"
  else
    LOG_FILE="$canonical_log_root/$experiment_id/train.log"
  fi
fi

if [[ -v TUNING_METHOD ]]; then
  TUNING_METHOD_EXPLICIT=true
else
  TUNING_METHOD_EXPLICIT=false
fi
TUNING_METHOD=${TUNING_METHOD-full}
LORA_R=${LORA_R-32}
LORA_ALPHA=${LORA_ALPHA-64}
LORA_DROPOUT=${LORA_DROPOUT-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
LATENT_SIZE=${LATENT_SIZE:-8}
STAGE2_DIAGNOSTICS_INTERVAL=${STAGE2_DIAGNOSTICS_INTERVAL:-0}
SUPERVISE_LATENT_BOUNDARIES=${SUPERVISE_LATENT_BOUNDARIES:-true}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
ATTENTION_IMPLEMENTATION=${ATTENTION_IMPLEMENTATION:-sdpa}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-8192}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
PER_DEVICE_TRAIN_BATCH_SIZE=1

MANAGED_OPTIONS=(
  --stage
  --load_model_path
  --data_path
  --image_root
  --save_model_path
  --log_file
  --latent_size
  --stage2_diagnostics_interval
  --supervise_latent_boundaries
  --no-supervise_latent_boundaries
  --learning_rate
  --gradient_accumulation_steps
  --per_device_train_batch_size
  --attention_implementation
  --gradient_checkpointing
  --no-gradient_checkpointing
  --max_seq_length_train
  --deepspeed_config
  --resume_from_checkpoint_path
  --tuning_method
  --lora_r
  --lora_alpha
  --lora_dropout
  --lora_target_modules
  --stage2_length_manifest
  --stage2_length_manifest_sha256
  --dataloader_num_workers
)

reject_managed_arguments() {
  local argument
  local managed_option
  local option
  for argument in "$@"; do
    option=${argument%%=*}
    for managed_option in "${MANAGED_OPTIONS[@]}"; do
      if [[ "$managed_option" == "$option"* ]]; then
        printf 'launcher-owned option cannot be passed directly: %s\n' "$option" >&2
        exit 2
      fi
    done
  done
}

reject_managed_arguments "$@"

require_positive_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s must be a positive integer\n' "$name" >&2
    exit 2
  fi
}

require_non_negative_integer() {
  local name=$1
  local value=$2
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer\n' "$name" >&2
    exit 2
  fi
}

require_positive_integer LATENT_SIZE "$LATENT_SIZE"
require_non_negative_integer STAGE2_DIAGNOSTICS_INTERVAL "$STAGE2_DIAGNOSTICS_INTERVAL"
require_positive_integer GRADIENT_ACCUMULATION_STEPS "$GRADIENT_ACCUMULATION_STEPS"
require_positive_integer MAX_SEQUENCE_LENGTH "$MAX_SEQUENCE_LENGTH"
require_non_negative_integer DATALOADER_NUM_WORKERS "$DATALOADER_NUM_WORKERS"
require_positive_integer LORA_R "$LORA_R"
require_positive_integer LORA_ALPHA "$LORA_ALPHA"

case "$TUNING_METHOD" in
  full|lora|dora) ;;
  *)
    printf 'TUNING_METHOD must be full, lora, or dora\n' >&2
    exit 2
    ;;
esac

if ! "$REPO_ROOT/venv/ilvr-venv/bin/python" - "$LORA_DROPOUT" <<'PY'
import math
import sys

try:
    dropout = float(sys.argv[1])
except (OverflowError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(dropout) and 0 <= dropout < 1 else 1)
PY
then
  printf 'LORA_DROPOUT must be a finite number in [0, 1)\n' >&2
  exit 2
fi

declare -A seen_lora_targets=()
remaining_targets=$LORA_TARGET_MODULES
while true; do
  if [[ "$remaining_targets" == *,* ]]; then
    target=${remaining_targets%%,*}
    remaining_targets=${remaining_targets#*,}
    has_more_targets=true
  else
    target=$remaining_targets
    has_more_targets=false
  fi
  case "$target" in
    q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj) ;;
    *)
      printf 'LORA_TARGET_MODULES must contain only approved decoder targets\n' >&2
      exit 2
      ;;
  esac
  if [[ -n "${seen_lora_targets[$target]+present}" ]]; then
    printf 'LORA_TARGET_MODULES must not contain duplicate targets\n' >&2
    exit 2
  fi
  seen_lora_targets["$target"]=true
  if [[ "$has_more_targets" == false ]]; then
    break
  fi
done

if ! "$REPO_ROOT/venv/ilvr-venv/bin/python" - "$LEARNING_RATE" <<'PY'
import math
import sys

try:
    learning_rate = float(sys.argv[1])
except (OverflowError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(learning_rate) and learning_rate > 0 else 1)
PY
then
  printf 'LEARNING_RATE must be a positive number\n' >&2
  exit 2
fi

BOUNDARY_ARGS=()
case "$SUPERVISE_LATENT_BOUNDARIES" in
  true)
    BOUNDARY_ARGS+=(--supervise_latent_boundaries)
    MANIFEST_BOUNDARY_ARG=--supervise-latent-boundaries
    ;;
  false)
    BOUNDARY_ARGS+=(--no-supervise_latent_boundaries)
    MANIFEST_BOUNDARY_ARG=--no-supervise-latent-boundaries
    ;;
  *)
    printf 'SUPERVISE_LATENT_BOUNDARIES must be true or false\n' >&2
    exit 2
    ;;
esac

CHECKPOINTING_ARGS=()
case "$GRADIENT_CHECKPOINTING" in
  true) CHECKPOINTING_ARGS+=(--gradient_checkpointing) ;;
  false) CHECKPOINTING_ARGS+=(--no-gradient_checkpointing) ;;
  *)
    printf 'GRADIENT_CHECKPOINTING must be true or false\n' >&2
    exit 2
    ;;
esac

case "$ATTENTION_IMPLEMENTATION" in
  eager|sdpa|flash_attention_2) ;;
  *)
    printf 'ATTENTION_IMPLEMENTATION must be eager, sdpa, or flash_attention_2\n' >&2
    exit 2
    ;;
esac

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint_path "$RESUME_CHECKPOINT")
fi

TUNING_ARGS=()
if [[ -z "${RESUME_CHECKPOINT:-}" || "$TUNING_METHOD_EXPLICIT" == true ]]; then
  TUNING_ARGS+=(
    --tuning_method "$TUNING_METHOD"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_target_modules "$LORA_TARGET_MODULES"
  )
fi

DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  DEEPSPEED_ARGS+=(--deepspeed_config "$DEEPSPEED_CONFIG")
fi

MANIFEST_ARGS=()
TRAINING_ARGS=(
  --stage stage2
  --load_model_path "$STAGE1_CHECKPOINT"
  --data_path "$DATA_PATH"
  --image_root "$IMAGE_ROOT"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --save_model_path "$OUTPUT_DIR"
  --log_file "$LOG_FILE"
  --latent_size "$LATENT_SIZE"
  --stage2_diagnostics_interval "$STAGE2_DIAGNOSTICS_INTERVAL"
  "${BOUNDARY_ARGS[@]}"
  --learning_rate "$LEARNING_RATE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --attention_implementation "$ATTENTION_IMPLEMENTATION"
  "${CHECKPOINTING_ARGS[@]}"
  --max_seq_length_train "$MAX_SEQUENCE_LENGTH"
  "${TUNING_ARGS[@]}"
  "${RESUME_ARGS[@]}"
  "${DEEPSPEED_ARGS[@]}"
  "$@"
)

if [[ "${STAGE2_DRY_RUN:-false}" != "true" ]]; then
  "$REPO_ROOT/venv/ilvr-venv/bin/python" -m src.artifact_paths validate-run \
    --checkpoint "$OUTPUT_DIR" --log "$LOG_FILE" \
    --stage2-length-manifest "$length_manifest" >/dev/null
  if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
    STAGE2_NUM_PROCESSES=${STAGE2_NUM_PROCESSES:-2}
    require_positive_integer STAGE2_NUM_PROCESSES "$STAGE2_NUM_PROCESSES"
    "$REPO_ROOT/venv/ilvr-venv/bin/python" -m src.deepspeed_runtime validate \
      --config "$DEEPSPEED_CONFIG" --world-size "$STAGE2_NUM_PROCESSES" >/dev/null
  fi
  if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
    mkdir -p -- "$(dirname -- "$OUTPUT_DIR")" "$(dirname -- "$(dirname -- "$LOG_FILE")")" "$(dirname -- "$result_dir")"
    if ! mkdir -- "$OUTPUT_DIR"; then
      printf 'fresh run artifact directory already exists: %s\n' "$OUTPUT_DIR" >&2
      exit 2
    fi
    if ! mkdir -- "$(dirname -- "$LOG_FILE")"; then
      printf 'fresh run artifact directory already exists: %s\n' "$(dirname -- "$LOG_FILE")" >&2
      exit 2
    fi
    if ! mkdir -- "$result_dir"; then
      printf 'fresh run artifact directory already exists: %s\n' "$result_dir" >&2
      exit 2
    fi
    "$REPO_ROOT/venv/ilvr-venv/bin/python" -m stage2.length_manifest \
      --data-path "$DATA_PATH" \
      --image-root "$IMAGE_ROOT" \
      --processor-source "$processor_source" \
      --latent-size "$LATENT_SIZE" \
      "$MANIFEST_BOUNDARY_ARG" \
      --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
      --manifest-path "$length_manifest"
  elif [[ -e $LOG_FILE ]]; then
    printf 'resume log file must be distinct: %s\n' "$LOG_FILE" >&2
    exit 2
  fi
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    mkdir -p -- "$OUTPUT_DIR" "$(dirname -- "$LOG_FILE")"
    if [[ ! -f "$length_manifest" ]]; then
      printf 'resume requires Stage 2 length manifest: %s\n' "$length_manifest" >&2
      exit 2
    fi
    "$REPO_ROOT/venv/ilvr-venv/bin/python" - "$length_manifest" <<'PY'
import sys
from stage2.length_manifest import read_stage2_length_manifest

read_stage2_length_manifest(sys.argv[1])
PY
  fi
  read -r length_manifest_sha256 _ < <(sha256sum -- "$length_manifest")
  MANIFEST_ARGS=(
    --stage2_length_manifest "$length_manifest"
    --stage2_length_manifest_sha256 "$length_manifest_sha256"
  )
else
  MANIFEST_ARGS=(
    --stage2_length_manifest "$length_manifest"
    --stage2_length_manifest_sha256 "$(printf '0%.0s' {1..64})"
  )
fi

TRAINING_ARGS+=("${MANIFEST_ARGS[@]}")

if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
  STAGE2_NUM_PROCESSES=${STAGE2_NUM_PROCESSES:-2}
  require_positive_integer STAGE2_NUM_PROCESSES "$STAGE2_NUM_PROCESSES"
  COMMAND=(
    "$REPO_ROOT/venv/ilvr-venv/bin/accelerate" launch
    --num_processes "$STAGE2_NUM_PROCESSES"
    --main_process_port "${STAGE2_MAIN_PROCESS_PORT:-29502}"
    "$REPO_ROOT/src/main.py"
    "${TRAINING_ARGS[@]}"
  )
else
  COMMAND=(
    "$REPO_ROOT/venv/ilvr-venv/bin/python"
    "$REPO_ROOT/src/main.py"
    "${TRAINING_ARGS[@]}"
  )
fi

if [[ "${STAGE2_DRY_RUN:-false}" == "true" ]]; then
  printf 'PYTHONPATH=%q TOKENIZERS_PARALLELISM=%q ' \
    "$PYTHONPATH" "$TOKENIZERS_PARALLELISM"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
