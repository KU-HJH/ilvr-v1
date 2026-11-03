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
canonical_log_root='/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/logs/zero-stage-compatibility'
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
SUPERVISE_LATENT_BOUNDARIES=${SUPERVISE_LATENT_BOUNDARIES:-true}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
ATTENTION_IMPLEMENTATION=${ATTENTION_IMPLEMENTATION:-sdpa}
STAGE2_SPLIT_ATTENTION=${STAGE2_SPLIT_ATTENTION:-false}
STAGE2_DEBUGGING=${STAGE2_DEBUGGING:-false}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
rollout_backend_explicit=false
cache_checkpointing_explicit=false
if [[ -v STAGE2_ROLLOUT_BACKEND ]]; then rollout_backend_explicit=true; fi
if [[ -v STAGE2_CACHE_ACTIVATION_CHECKPOINTING ]]; then cache_checkpointing_explicit=true; fi
default_rollout_backend=no_cache
default_cache_checkpointing=false
if [[ "${STEP_PREFERENCE_PREFIX_MODE-clean_prefix}" == single_image_latent_gate \
      && -z "${RESUME_CHECKPOINT:-}" ]]; then
  default_rollout_backend=differentiable_cache
  if [[ "${STAGE2_ROLLOUT_BACKEND-$default_rollout_backend}" == differentiable_cache ]]; then
    default_cache_checkpointing=true
  fi
fi
STAGE2_ROLLOUT_BACKEND=${STAGE2_ROLLOUT_BACKEND-$default_rollout_backend}
STAGE2_CACHE_ACTIVATION_CHECKPOINTING=${STAGE2_CACHE_ACTIVATION_CHECKPOINTING:-$default_cache_checkpointing}
STAGE2_LATENT_ACTIVATION_CHECKPOINTING=${STAGE2_LATENT_ACTIVATION_CHECKPOINTING:-false}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-8192}
PER_DEVICE_TRAIN_BATCH_SIZE=1
STAGE2_PHASE=${STAGE2_PHASE:-standard}

MANAGED_OPTIONS=(
  --enable_stage2_objective
  --no-enable_stage2_objective
  --stage2_lambda_dyn
  --stage2_lambda_vis
  --stage2_reference_model_path
  --stage2_gamma_h
  --stage2_vision_margin
  --stage2_visual_ablation
  --stage2_entropy_eps
  --stage2_pre_window
  --stage2_post_window
  --stage
  --load_model_path
  --data_path
  --image_root
  --save_model_path
  --log_file
  --latent_size
  --supervise_latent_boundaries
  --no-supervise_latent_boundaries
  --learning_rate
  --gradient_accumulation_steps
  --dataloader_num_workers
  --per_device_train_batch_size
  --attention_implementation
  --stage2_split_attention
  --no-stage2_split_attention
  --stage2_debugging
  --no-stage2_debugging
  --gradient_checkpointing
  --no-gradient_checkpointing
  --stage2_rollout_backend
  --stage2_cache_activation_checkpointing
  --no-stage2_cache_activation_checkpointing
  --stage2_latent_activation_checkpointing
  --no-stage2_latent_activation_checkpointing
  --stage1_adapter_path
  --max_seq_length_train
  --deepspeed_config
  --resume_from_checkpoint_path
  --tuning_method
  --lora_r
  --lora_alpha
  --lora_dropout
  --lora_target_modules
  --stage2_phase
  --stage2a_checkpoint_path
  --step_negatives_path
  --stage2_total_optimizer_steps
  --stage2a_ratio
  --step_preference_beta
  --step_preference_lambda_dpo
  --step_preference_lambda_preference
  --step_preference_margin
  --step_preference_ablation
  --step_preference_modalities
  --step_preference_target_scope
  --step_preference_lambda_entropy
  --step_preference_entropy_margin
  --step_preference_execution_backend
)

case "$STAGE2_PHASE" in
  standard) ;;
  stage2a|stage2b) MANAGED_OPTIONS+=(--max_steps) ;;
  *)
    printf 'STAGE2_PHASE must be standard, stage2a, or stage2b\n' >&2
    exit 2
    ;;
esac

reject_managed_arguments() {
  local argument
  local managed_option
  local option
  for argument in "$@"; do
    option=${argument%%=*}
    for managed_option in "${MANAGED_OPTIONS[@]}"; do
      if [[ "$managed_option" == "$option"* ]]; then
        printf 'launcher-owned option cannot be passed directly (managed): %s\n' "$option" >&2
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

require_positive_integer LATENT_SIZE "$LATENT_SIZE"
require_positive_integer GRADIENT_ACCUMULATION_STEPS "$GRADIENT_ACCUMULATION_STEPS"
if [[ ! "$DATALOADER_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
  printf 'DATALOADER_NUM_WORKERS must be a nonnegative integer\n' >&2
  exit 2
fi
require_positive_integer MAX_SEQUENCE_LENGTH "$MAX_SEQUENCE_LENGTH"
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
  true) BOUNDARY_ARGS+=(--supervise_latent_boundaries) ;;
  false) BOUNDARY_ARGS+=(--no-supervise_latent_boundaries) ;;
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

case "$STAGE2_ROLLOUT_BACKEND" in
  no_cache|differentiable_cache) ;;
  *)
    printf 'STAGE2_ROLLOUT_BACKEND must be no_cache or differentiable_cache\n' >&2
    exit 2
    ;;
esac

CACHE_ACTIVATION_CHECKPOINTING_ARGS=()
LATENT_CHECKPOINTING_ARGS=()
case "$STAGE2_LATENT_ACTIVATION_CHECKPOINTING" in
  true)
    if [[ "$GRADIENT_CHECKPOINTING" == true || "$STAGE2_ROLLOUT_BACKEND" != no_cache ]]; then
      printf 'latent activation checkpointing requires no_cache and global gradient checkpointing disabled\n' >&2
      exit 2
    fi
    LATENT_CHECKPOINTING_ARGS+=(--stage2_latent_activation_checkpointing)
    ;;
  false) ;;
  *)
    printf 'STAGE2_LATENT_ACTIVATION_CHECKPOINTING must be true or false\n' >&2
    exit 2
    ;;
esac
case "$STAGE2_CACHE_ACTIVATION_CHECKPOINTING" in
  true) CACHE_ACTIVATION_CHECKPOINTING_ARGS+=(--stage2_cache_activation_checkpointing) ;;
  false) CACHE_ACTIVATION_CHECKPOINTING_ARGS+=(--no-stage2_cache_activation_checkpointing) ;;
  *)
    printf 'STAGE2_CACHE_ACTIVATION_CHECKPOINTING must be true or false\n' >&2
    exit 2
    ;;
esac

if [[ "$STAGE2_CACHE_ACTIVATION_CHECKPOINTING" == true \
      && "$STAGE2_ROLLOUT_BACKEND" != differentiable_cache \
      && ( -z "${RESUME_CHECKPOINT:-}" || "$rollout_backend_explicit" == true ) ]]; then
  printf 'stage2_cache_activation_checkpointing requires differentiable_cache\n' >&2
  exit 2
fi

# Omitted resume settings inherit the artifact's backend/checkpoint identity.
# Never turn defaults into explicit conflicting overrides of a saved variant.
ROLLOUT_BACKEND_ARGS=(--stage2_rollout_backend "$STAGE2_ROLLOUT_BACKEND")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  if [[ "$rollout_backend_explicit" == false ]]; then ROLLOUT_BACKEND_ARGS=(); fi
  if [[ "$cache_checkpointing_explicit" == false ]]; then CACHE_ACTIVATION_CHECKPOINTING_ARGS=(); fi
fi

if [[ "$STAGE2_ROLLOUT_BACKEND" == differentiable_cache \
      && "$GRADIENT_CHECKPOINTING" == true ]]; then
  printf 'differentiable_cache is incompatible with global gradient checkpointing\n' >&2
  exit 2
fi

SPLIT_ATTENTION_ARGS=()
DEBUGGING_ARGS=()
case "$STAGE2_DEBUGGING" in
  true) DEBUGGING_ARGS+=(--stage2_debugging) ;;
  false) ;;
  *) printf '%s\n' 'STAGE2_DEBUGGING must be true or false' >&2; exit 2 ;;
esac
case "$STAGE2_SPLIT_ATTENTION" in
  true)
    ATTENTION_IMPLEMENTATION=sdpa
    SPLIT_ATTENTION_ARGS+=(--stage2_split_attention)
    ;;
  false) ;;
  *)
    printf 'STAGE2_SPLIT_ATTENTION must be true or false\n' >&2
    exit 2
    ;;
esac

case "$ATTENTION_IMPLEMENTATION" in
  eager|sdpa|flash_attention_2|flex_attention) ;;
  *)
    printf 'ATTENTION_IMPLEMENTATION must be eager, sdpa, flash_attention_2, or flex_attention\n' >&2
    exit 2
    ;;
esac

if [[ "$STAGE2_PHASE" == stage2b \
      && "$ATTENTION_IMPLEMENTATION" != eager \
      && "$ATTENTION_IMPLEMENTATION" != sdpa \
      && "$ATTENTION_IMPLEMENTATION" != flex_attention ]]; then
  printf 'Stage 2B requires eager, sdpa, or flex_attention\n' >&2
  exit 2
fi

LIFECYCLE_ARGS=(--stage2_phase "$STAGE2_PHASE")
PHASE_MAX_STEPS_ARGS=()
if [[ "$STAGE2_PHASE" != standard ]]; then
  if [[ -z "${STAGE2_TOTAL_OPTIMIZER_STEPS:-}" ]]; then
    printf 'set STAGE2_TOTAL_OPTIMIZER_STEPS\n' >&2
    exit 2
  fi
  if [[ -z "${STAGE2A_RATIO:-}" ]]; then
    printf 'set STAGE2A_RATIO\n' >&2
    exit 2
  fi
  if ! phase_max_steps=$("$REPO_ROOT/venv/ilvr-venv/bin/python" -m step_preference.config phase-steps \
    --phase "$STAGE2_PHASE" \
    --total-steps "$STAGE2_TOTAL_OPTIMIZER_STEPS" \
    --stage2a-ratio "$STAGE2A_RATIO"); then
    exit 2
  fi
  LIFECYCLE_ARGS+=(
    --stage2_total_optimizer_steps "$STAGE2_TOTAL_OPTIMIZER_STEPS"
    --stage2a_ratio "$STAGE2A_RATIO"
  )
  PHASE_MAX_STEPS_ARGS=(--max_steps "$phase_max_steps")
fi

if [[ "$STAGE2_PHASE" == stage2b ]]; then
  if [[ -z "${STAGE2A_CHECKPOINT:-}" ]]; then
    if [[ "$STAGE2A_RATIO" != 0 && "$STAGE2A_RATIO" != 0.0 ]]; then
      printf 'Without STAGE2A_CHECKPOINT, set STAGE2A_RATIO=0 for direct preference\n' >&2
      exit 2
    fi
  else
    LIFECYCLE_ARGS+=(--stage2a_checkpoint_path "$STAGE2A_CHECKPOINT")
  fi
  if [[ -n "${STEP_NEGATIVES_PATH:-}" || -v STEP_PREFERENCE_LAMBDA_DPO ]]; then
    printf 'Stage 2B uses context preference: remove STEP_NEGATIVES_PATH and use STEP_PREFERENCE_LAMBDA_PREFERENCE\n' >&2
    exit 2
  fi
fi

PREFERENCE_ARGS=()
if [[ -v STEP_PREFERENCE_TARGET_SCOPE ]]; then
  case "$STEP_PREFERENCE_TARGET_SCOPE" in
    until_next_latent|immediate_text) ;;
    *) printf '%s\n' 'STEP_PREFERENCE_TARGET_SCOPE must be until_next_latent or immediate_text' >&2; exit 2 ;;
  esac
  PREFERENCE_ARGS+=(--step_preference_target_scope "$STEP_PREFERENCE_TARGET_SCOPE")
fi
if [[ -v STEP_PREFERENCE_MODALITIES ]]; then
  case "$STEP_PREFERENCE_MODALITIES" in
    visual|visual_text) ;;
    *) printf '%s\n' 'STEP_PREFERENCE_MODALITIES must be visual or visual_text' >&2; exit 2 ;;
  esac
  PREFERENCE_ARGS+=(--step_preference_modalities "$STEP_PREFERENCE_MODALITIES")
fi
if [[ -v STEP_PREFERENCE_ABLATION ]]; then
  case "$STEP_PREFERENCE_ABLATION" in
    attention_mask|remove_full_visual_block|remove_full_latent_block_keep_image) ;;
    *) printf '%s\n' 'STEP_PREFERENCE_ABLATION is unsupported' >&2; exit 2 ;;
  esac
  PREFERENCE_ARGS+=(--step_preference_ablation "$STEP_PREFERENCE_ABLATION")
fi
if [[ -v STEP_PREFERENCE_PREFIX_MODE ]]; then
  case "$STEP_PREFERENCE_PREFIX_MODE" in
    clean_prefix|v_off_insertion|single_image_latent_gate) ;;
    *) printf '%s\n' 'STEP_PREFERENCE_PREFIX_MODE must be clean_prefix, v_off_insertion, or single_image_latent_gate' >&2; exit 2 ;;
  esac
  PREFERENCE_ARGS+=(--step_preference_prefix_mode "$STEP_PREFERENCE_PREFIX_MODE")
fi
if [[ -v STEP_PREFERENCE_BRANCH_MODE ]]; then
  case "$STEP_PREFERENCE_BRANCH_MODE" in
    independent|cumulative) ;;
    *) printf '%s\n' 'STEP_PREFERENCE_BRANCH_MODE must be independent or cumulative' >&2; exit 2 ;;
  esac
  PREFERENCE_ARGS+=(--step_preference_branch_mode "$STEP_PREFERENCE_BRANCH_MODE")
fi
if [[ -v STEP_ON_LAYER_CHECKPOINT_STRIDE ]]; then
  case "$STEP_ON_LAYER_CHECKPOINT_STRIDE" in
    1|2|4) ;;
    *) printf '%s\n' 'STEP_ON_LAYER_CHECKPOINT_STRIDE must be 1, 2, or 4' >&2; exit 2 ;;
  esac
  PREFERENCE_ARGS+=(--step_on_layer_checkpoint_stride "$STEP_ON_LAYER_CHECKPOINT_STRIDE")
fi

effective_preference_prefix_mode=${STEP_PREFERENCE_PREFIX_MODE-clean_prefix}
effective_preference_modalities=${STEP_PREFERENCE_MODALITIES-visual_text}
effective_preference_ablation=${STEP_PREFERENCE_ABLATION-attention_mask}
if [[ "$effective_preference_prefix_mode" == single_image_latent_gate ]]; then
  if [[ "$GRADIENT_CHECKPOINTING" == true ]]; then
    printf '%s\n' 'single_image_latent_gate requires native checkpointing disabled' >&2
    exit 2
  fi
  if [[ "$STAGE2_PHASE" != stage2b ]]; then
    printf '%s\n' 'single_image_latent_gate requires STAGE2_PHASE=stage2b' >&2
    exit 2
  fi
  if [[ "$effective_preference_modalities" != visual ]]; then
    printf '%s\n' 'single_image_latent_gate requires STEP_PREFERENCE_MODALITIES=visual' >&2
    exit 2
  fi
  if [[ -v STEP_ON_LAYER_CHECKPOINT_STRIDE ]]; then
    printf '%s\n' 'STEP_ON_LAYER_CHECKPOINT_STRIDE is only supported for v_off_insertion' >&2
    exit 2
  fi
elif [[ "$STAGE2_PHASE" == stage2b && "$STAGE2_ROLLOUT_BACKEND" == differentiable_cache ]]; then
  printf '%s\n' "differentiable cache is not integrated with $effective_preference_prefix_mode preference" >&2
  exit 2
fi
if [[ -v STEP_PREFERENCE_BETA ]]; then
  PREFERENCE_ARGS+=(--step_preference_beta "$STEP_PREFERENCE_BETA")
fi
if [[ -v STEP_PREFERENCE_LAMBDA_PREFERENCE ]]; then
  PREFERENCE_ARGS+=(--step_preference_lambda_preference "$STEP_PREFERENCE_LAMBDA_PREFERENCE")
fi
if [[ -v STEP_PREFERENCE_MARGIN ]]; then
  PREFERENCE_ARGS+=(--step_preference_margin "$STEP_PREFERENCE_MARGIN")
fi
if [[ -v STEP_PREFERENCE_LAMBDA_ENTROPY ]]; then
  PREFERENCE_ARGS+=(--step_preference_lambda_entropy "$STEP_PREFERENCE_LAMBDA_ENTROPY")
fi
if [[ -v STEP_PREFERENCE_ENTROPY_MARGIN ]]; then
  PREFERENCE_ARGS+=(--step_preference_entropy_margin "$STEP_PREFERENCE_ENTROPY_MARGIN")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint_path "$RESUME_CHECKPOINT")
fi

STAGE1_ADAPTER_ARGS=()
if [[ -n "${STAGE1_ADAPTER_PATH:-}" ]]; then
  STAGE1_ADAPTER_ARGS+=(--stage1_adapter_path "$STAGE1_ADAPTER_PATH")
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

OBJECTIVE_ARGS=()
objective_environment_present=false
for objective_variable in ENABLE_STAGE2_OBJECTIVE STAGE2_LAMBDA_DYN STAGE2_LAMBDA_VIS STAGE2_REFERENCE_MODEL_PATH STAGE2_GAMMA_H STAGE2_VISION_MARGIN STAGE2_VISUAL_ABLATION STAGE2_ENTROPY_EPS STAGE2_PRE_WINDOW STAGE2_POST_WINDOW; do
  if [[ -v "$objective_variable" ]]; then objective_environment_present=true; fi
done
if [[ $objective_environment_present == true ]]; then
  case "${ENABLE_STAGE2_OBJECTIVE:-false}" in
    true) OBJECTIVE_ARGS+=(--enable_stage2_objective) ;;
    false) OBJECTIVE_ARGS+=(--no-enable_stage2_objective) ;;
    *) printf 'ENABLE_STAGE2_OBJECTIVE must be true or false\n' >&2; exit 2 ;;
  esac
  OBJECTIVE_ARGS+=(
    --stage2_lambda_dyn "${STAGE2_LAMBDA_DYN:-0.0}"
    --stage2_lambda_vis "${STAGE2_LAMBDA_VIS:-0.0}"
    --stage2_gamma_h "${STAGE2_GAMMA_H:-0.0}"
    --stage2_vision_margin "${STAGE2_VISION_MARGIN:-0.0}"
    --stage2_visual_ablation "${STAGE2_VISUAL_ABLATION:-latent_body}"
    --stage2_entropy_eps "${STAGE2_ENTROPY_EPS:-1e-6}"
    --stage2_pre_window "${STAGE2_PRE_WINDOW:-1}"
    --stage2_post_window "${STAGE2_POST_WINDOW:-1}"
  )
  if [[ -n "${STAGE2_REFERENCE_MODEL_PATH:-}" ]]; then
    OBJECTIVE_ARGS+=(--stage2_reference_model_path "$STAGE2_REFERENCE_MODEL_PATH")
  fi
fi

TRAINING_ARGS=(
  "${OBJECTIVE_ARGS[@]}"
  --stage stage2
  --load_model_path "$STAGE1_CHECKPOINT"
  "${STAGE1_ADAPTER_ARGS[@]}"
  --data_path "$DATA_PATH"
  --image_root "$IMAGE_ROOT"
  --save_model_path "$OUTPUT_DIR"
  --log_file "$LOG_FILE"
  --latent_size "$LATENT_SIZE"
  "${BOUNDARY_ARGS[@]}"
  --learning_rate "$LEARNING_RATE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --attention_implementation "$ATTENTION_IMPLEMENTATION"
  "${SPLIT_ATTENTION_ARGS[@]}"
  "${DEBUGGING_ARGS[@]}"
  "${CHECKPOINTING_ARGS[@]}"
  "${CACHE_ACTIVATION_CHECKPOINTING_ARGS[@]}"
  "${LATENT_CHECKPOINTING_ARGS[@]}"
  "${ROLLOUT_BACKEND_ARGS[@]}"
  "${LIFECYCLE_ARGS[@]}"
  "${PREFERENCE_ARGS[@]}"
  "${PHASE_MAX_STEPS_ARGS[@]}"
  --max_seq_length_train "$MAX_SEQUENCE_LENGTH"
  "${TUNING_ARGS[@]}"
  "${RESUME_ARGS[@]}"
  "${DEEPSPEED_ARGS[@]}"
  "$@"
)

if [[ "${STAGE2_DRY_RUN:-false}" != "true" ]]; then
  "$REPO_ROOT/venv/ilvr-venv/bin/python" -m src.artifact_paths validate-run \
    --checkpoint "$OUTPUT_DIR" --log "$LOG_FILE" >/dev/null
  if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
    STAGE2_NUM_PROCESSES=${STAGE2_NUM_PROCESSES:-2}
    require_positive_integer STAGE2_NUM_PROCESSES "$STAGE2_NUM_PROCESSES"
    "$REPO_ROOT/venv/ilvr-venv/bin/python" -m src.deepspeed_runtime validate \
      --config "$DEEPSPEED_CONFIG" --world-size "$STAGE2_NUM_PROCESSES" >/dev/null
  fi
  if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
    mkdir -p -- "$(dirname -- "$OUTPUT_DIR")" "$(dirname -- "$(dirname -- "$LOG_FILE")")"
    if ! mkdir -- "$OUTPUT_DIR"; then
      printf 'fresh run artifact directory already exists: %s\n' "$OUTPUT_DIR" >&2
      exit 2
    fi
    if ! mkdir -- "$(dirname -- "$LOG_FILE")"; then
      printf 'fresh run artifact directory already exists: %s\n' "$(dirname -- "$LOG_FILE")" >&2
      exit 2
    fi
  elif [[ -e $LOG_FILE ]]; then
    printf 'resume log file must be distinct: %s\n' "$LOG_FILE" >&2
    exit 2
  fi
  if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    mkdir -p -- "$OUTPUT_DIR" "$(dirname -- "$LOG_FILE")"
  fi
fi

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
