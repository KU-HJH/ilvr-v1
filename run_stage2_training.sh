#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export PYTHONPATH="$REPO_ROOT/transformers/src:$REPO_ROOT/src"
export TOKENIZERS_PARALLELISM=false

: "${STAGE1_CHECKPOINT:?set STAGE1_CHECKPOINT}"
: "${DATA_PATH:?set DATA_PATH}"
: "${IMAGE_ROOT:?set IMAGE_ROOT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

stage2_dry_run=${STAGE2_DRY_RUN:-false}
if [[ -v STAGE2_DRY_RUN_FORMAT && "$stage2_dry_run" != true ]]; then
  printf 'STAGE2_DRY_RUN_FORMAT is only valid with STAGE2_DRY_RUN=true\n' >&2
  exit 2
fi
STAGE2_DRY_RUN_FORMAT=${STAGE2_DRY_RUN_FORMAT:-text}
case "$STAGE2_DRY_RUN_FORMAT" in
  text|json) ;;
  *)
    printf 'STAGE2_DRY_RUN_FORMAT must be text or json\n' >&2
    exit 2
    ;;
esac

experiment_id=${OUTPUT_DIR##*/}
canonical_log_root="$REPO_ROOT/logs/zero-stage-compatibility"
result_dir="$REPO_ROOT/results/$experiment_id"
length_manifest="$result_dir/stage2-length-manifest.json"
processor_source=${RESUME_CHECKPOINT:-$STAGE1_CHECKPOINT}
if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
  stage2_launcher_arguments=("$@")
  for ((stage2_argument_index=0; stage2_argument_index<${#stage2_launcher_arguments[@]}; stage2_argument_index++)); do
    case "${stage2_launcher_arguments[stage2_argument_index]}" in
      --stage1_adapter_path) processor_source=${stage2_launcher_arguments[stage2_argument_index+1]:-} ;;
      --stage1_adapter_path=*) processor_source=${stage2_launcher_arguments[stage2_argument_index]#*=} ;;
    esac
  done
fi
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
CE_WEIGHT=${CE_WEIGHT:-1.0}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
ATTENTION_IMPLEMENTATION=${ATTENTION_IMPLEMENTATION:-sdpa}
if [[ -n "${VISION_ATTENTION_IMPLEMENTATION:-}" ]]; then
  vision_attention_implementation_explicit=true
else
  vision_attention_implementation_explicit=false
fi
VISION_ATTENTION_IMPLEMENTATION=${VISION_ATTENTION_IMPLEMENTATION:-$ATTENTION_IMPLEMENTATION}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-8192}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
PER_DEVICE_TRAIN_BATCH_SIZE=1

STEP_PREFERENCE_ENABLED=${STEP_PREFERENCE_ENABLED:-false}
STEP_PREFERENCE_OBJECTIVE=${STEP_PREFERENCE_OBJECTIVE:-context_preference}
case "${STEP_VISUAL_ONLY_CONTEXT_PREFERENCE-}" in
  ""|0|false|False) step_visual_only_context_preference=false ;;
  1|true|True) step_visual_only_context_preference=true ;;
  *)
    printf 'STEP_VISUAL_ONLY_CONTEXT_PREFERENCE must be 0, 1, false, or true\n' >&2
    exit 2
    ;;
esac
SEED=${SEED:-42}
STEP_MAX_PACKED_SEQUENCE_LENGTH=${STEP_MAX_PACKED_SEQUENCE_LENGTH:-4608}
ASSISTANT_PLAN_RENDERING=concatenated-v1
stage2b_checkpoint_initialization=false
case "${STEP_PREFERENCE_PACKING-}" in
  ""|0|false|False) step_preference_packing=false ;;
  1|true|True) step_preference_packing=true ;;
  *)
    printf 'STEP_PREFERENCE_PACKING must be 0, 1, false, or true\n' >&2
    exit 2
    ;;
esac
STEP_PREFERENCE_ARGS=()
case "${STEP_SHARED_PREFIX_ENABLED-}" in
  ""|0|false|False) step_shared_prefix_enabled=false ;;
  1|true|True) step_shared_prefix_enabled=true ;;
  *)
    printf 'STEP_SHARED_PREFIX_ENABLED must be 0, 1, false, or true\n' >&2
    exit 2
    ;;
esac
STEP_SIDECAR_ATTENTION_BACKEND=${STEP_SIDECAR_ATTENTION_BACKEND:-flex}
STEP_SIDECAR_QUERY_BUDGET=${STEP_SIDECAR_QUERY_BUDGET:-1024}
if [[ "$STEP_SIDECAR_ATTENTION_BACKEND" != dense && "$STEP_SIDECAR_ATTENTION_BACKEND" != flex ]]; then
  printf 'STEP_SIDECAR_ATTENTION_BACKEND must be dense or flex\n' >&2
  exit 2
fi
if [[ ! "$STEP_SIDECAR_QUERY_BUDGET" =~ ^[1-9][0-9]*$ ]]; then
  printf 'STEP_SIDECAR_QUERY_BUDGET must be a positive integer\n' >&2
  exit 2
fi
if [[ "$STEP_PREFERENCE_ENABLED" == true && "${STEP_PREFERENCE_MODE:-stage2a}" == stage2b && "$STEP_PREFERENCE_OBJECTIVE" == context_preference ]]; then
  if [[ -n "${STEP_NEGATIVE_PATH:-}${STEP_NEGATIVE_MANIFEST:-}" ]]; then
    printf 'Context preference does not consume negative artifacts; select legacy_dpo explicitly\n' >&2
    exit 2
  fi
  if [[ "$step_preference_packing" == true || -n "${STEP_PACKED_ADMISSION_MANIFEST:-}${STEP_PACKED_ADMISSION_MANIFEST_SHA256:-}${STEP_PACKED_ADMISSION_VERIFICATION:-}${STEP_PACKED_ADMISSION_VERIFICATION_SHA256:-}" ]]; then
    printf 'Context preference does not consume legacy packed admission\n' >&2
    exit 2
  fi
fi
if [[ "$step_shared_prefix_enabled" == true ]]; then
  if [[ "$STEP_PREFERENCE_ENABLED" != true || "${STEP_PREFERENCE_MODE:-stage2a}" != stage2b ]]; then
    printf 'Shared prefix requires Stage 2B\n' >&2
    exit 2
  fi
  if [[ "$step_preference_packing" == true || -n "${STEP_PACKED_ADMISSION_MANIFEST:-}${STEP_PACKED_ADMISSION_MANIFEST_SHA256:-}${STEP_PACKED_ADMISSION_VERIFICATION:-}${STEP_PACKED_ADMISSION_VERIFICATION_SHA256:-}" ]]; then
    printf 'Shared prefix cannot consume legacy packing or packed admission\n' >&2
    exit 2
  fi
  STEP_PREFERENCE_ARGS+=(--step_shared_prefix_enabled)
fi
if [[ "$step_preference_packing" == true ]]; then
  if [[ ! "$STEP_MAX_PACKED_SEQUENCE_LENGTH" =~ ^[1-9][0-9]*$ ]]; then
    printf 'STEP_MAX_PACKED_SEQUENCE_LENGTH must be a positive integer\n' >&2
    exit 2
  fi
  for admission_variable in \
    STEP_PACKED_ADMISSION_MANIFEST \
    STEP_PACKED_ADMISSION_MANIFEST_SHA256 \
    STEP_PACKED_ADMISSION_VERIFICATION \
    STEP_PACKED_ADMISSION_VERIFICATION_SHA256; do
    if [[ -z "${!admission_variable:-}" ]]; then
      printf '%s is required when STEP_PREFERENCE_PACKING=true\n' "$admission_variable" >&2
      exit 2
    fi
  done
  for admission_hash_variable in \
    STEP_PACKED_ADMISSION_MANIFEST_SHA256 \
    STEP_PACKED_ADMISSION_VERIFICATION_SHA256; do
    if [[ ! "${!admission_hash_variable}" =~ ^[0-9a-f]{64}$ ]]; then
      printf '%s must be a lowercase SHA-256\n' "$admission_hash_variable" >&2
      exit 2
    fi
  done
  STEP_PREFERENCE_ARGS+=(
    --step_preference_packing
    --step_max_packed_sequence_length "$STEP_MAX_PACKED_SEQUENCE_LENGTH"
    --step_packed_admission_manifest "$STEP_PACKED_ADMISSION_MANIFEST"
    --step_packed_admission_manifest_sha256 "$STEP_PACKED_ADMISSION_MANIFEST_SHA256"
    --step_packed_admission_verification "$STEP_PACKED_ADMISSION_VERIFICATION"
    --step_packed_admission_verification_sha256 "$STEP_PACKED_ADMISSION_VERIFICATION_SHA256"
  )
fi
case "$STEP_PREFERENCE_ENABLED" in
  true)
    ASSISTANT_PLAN_RENDERING=adjacent-text-space-lf-v1
    STEP_PREFERENCE_MODE=${STEP_PREFERENCE_MODE:-stage2a}
    STEP_DPO_BETA=${STEP_DPO_BETA:-0.1}
    STEP_DPO_WEIGHT=${STEP_DPO_WEIGHT:-1.0}
    STEP_PREFERENCE_BETA=${STEP_PREFERENCE_BETA:-$STEP_DPO_BETA}
    STEP_PREFERENCE_WEIGHT=${STEP_PREFERENCE_WEIGHT:-1.0}
    STEP_PREFERENCE_MARGIN=${STEP_PREFERENCE_MARGIN:-0.0}
    STEP_ENTROPY_WEIGHT=${STEP_ENTROPY_WEIGHT:-0.1}
    STEP_ENTROPY_MARGIN=${STEP_ENTROPY_MARGIN:-0.1}
    STAGE2A_RATIO=${STAGE2A_RATIO:-0.5}
    case "$STEP_PREFERENCE_MODE" in
      stage2a|stage2b) ;;
      *)
        printf 'STEP_PREFERENCE_MODE must be stage2a or stage2b\n' >&2
        exit 2
        ;;
    esac
    case "$STEP_PREFERENCE_OBJECTIVE" in
      context_preference|legacy_dpo) ;;
      *)
        printf 'Unknown STEP_PREFERENCE_OBJECTIVE\n' >&2
        exit 2
        ;;
    esac
    STEP_PREFERENCE_ARGS+=(
      --step_preference_enabled
      --step_preference_mode "$STEP_PREFERENCE_MODE"
      --step_preference_objective "$STEP_PREFERENCE_OBJECTIVE"
      --step_entropy_weight "$STEP_ENTROPY_WEIGHT"
      --step_entropy_margin "$STEP_ENTROPY_MARGIN"
      --stage2a_ratio "$STAGE2A_RATIO"
    )
    if [[ "$STEP_PREFERENCE_OBJECTIVE" == context_preference ]]; then
      STEP_PREFERENCE_ARGS+=(
        --step_preference_beta "$STEP_PREFERENCE_BETA"
        --step_preference_weight "$STEP_PREFERENCE_WEIGHT"
        --step_preference_margin "$STEP_PREFERENCE_MARGIN"
        --step_sidecar_attention_backend "$STEP_SIDECAR_ATTENTION_BACKEND"
        --step_sidecar_query_budget "$STEP_SIDECAR_QUERY_BUDGET"
      )
      if [[ "$step_visual_only_context_preference" == true ]]; then
        STEP_PREFERENCE_ARGS+=(--step_visual_only_context_preference)
      fi
    else
      STEP_PREFERENCE_ARGS+=(
        --step_dpo_beta "$STEP_DPO_BETA"
        --step_dpo_weight "$STEP_DPO_WEIGHT"
      )
      if [[ "$step_shared_prefix_enabled" == true ]]; then
        STEP_PREFERENCE_ARGS+=(
          --step_sidecar_attention_backend "$STEP_SIDECAR_ATTENTION_BACKEND"
          --step_sidecar_query_budget "$STEP_SIDECAR_QUERY_BUDGET"
        )
      fi
    fi
    if [[ -n "${STAGE2_TOTAL_STEPS:-}" ]]; then
      STEP_PREFERENCE_ARGS+=(--stage2_total_steps "$STAGE2_TOTAL_STEPS")
    fi
    if [[ "$STEP_PREFERENCE_MODE" == stage2b ]]; then
      if [[ "$vision_attention_implementation_explicit" == false ]]; then
        VISION_ATTENTION_IMPLEMENTATION=flash_attention_2
      fi
      if [[ "$ATTENTION_IMPLEMENTATION" != sdpa || "$VISION_ATTENTION_IMPLEMENTATION" != flash_attention_2 ]]; then
        printf "Stage 2B requires ATTENTION_IMPLEMENTATION=sdpa and VISION_ATTENTION_IMPLEMENTATION=flash_attention_2\n" >&2
        exit 2
      fi
      [[ -n "${STAGE2A_CHECKPOINT_PATH:-}" ]] || {
        printf 'Stage 2B requires STAGE2A_CHECKPOINT_PATH\n' >&2
        exit 2
      }
      if [[ "$STEP_PREFERENCE_OBJECTIVE" == legacy_dpo ]]; then
        [[ -n "${STEP_NEGATIVE_PATH:-}" ]] || {
          printf 'Stage 2B requires STEP_NEGATIVE_PATH\n' >&2
          exit 2
        }
        [[ -n "${STEP_NEGATIVE_MANIFEST:-}" ]] || {
          printf 'Stage 2B requires STEP_NEGATIVE_MANIFEST\n' >&2
          exit 2
        }
      fi
      if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
        printf 'Stage 2B Stage 2A initialization and RESUME_CHECKPOINT are mutually exclusive\n' >&2
        exit 2
      fi
      stage2b_checkpoint_initialization=true
      processor_source=$STAGE2A_CHECKPOINT_PATH
      if [[ "$STEP_PREFERENCE_OBJECTIVE" == legacy_dpo ]]; then
        STEP_PREFERENCE_ARGS+=(
          --step_negative_path "$STEP_NEGATIVE_PATH"
          --step_negative_manifest "$STEP_NEGATIVE_MANIFEST"
        )
      fi
      STEP_PREFERENCE_ARGS+=(--stage2a_checkpoint_path "$STAGE2A_CHECKPOINT_PATH")
    elif [[ -n "${STAGE2A_CHECKPOINT_PATH:-}" ]]; then
      STEP_PREFERENCE_ARGS+=(--stage2a_checkpoint_path "$STAGE2A_CHECKPOINT_PATH")
    fi
    ;;
  false) ;;
  *)
    printf 'STEP_PREFERENCE_ENABLED must be true or false\n' >&2
    exit 2
    ;;
esac

if [[ "$step_visual_only_context_preference" == true ]] &&
   [[ "$STEP_PREFERENCE_ENABLED" != true || "${STEP_PREFERENCE_MODE:-stage2a}" != stage2b || "$STEP_PREFERENCE_OBJECTIVE" != context_preference ]]; then
  printf 'STEP_VISUAL_ONLY_CONTEXT_PREFERENCE requires context-preference Stage 2B\n' >&2
  exit 2
fi

ilvr_topology_variables=(
  ILVR_NUM_MACHINES
  ILVR_NUM_PROCESSES
  ILVR_MACHINE_RANK
  ILVR_MAIN_PROCESS_IP
  ILVR_MAIN_PROCESS_PORT
  ILVR_SOCKET_IFNAME
)
ilvr_multinode=false
for topology_variable in "${ilvr_topology_variables[@]}"; do
  if [[ -v $topology_variable ]]; then
    ilvr_multinode=true
    break
  fi
done
if [[ "$ilvr_multinode" == true ]]; then
  for topology_variable in "${ilvr_topology_variables[@]}"; do
    if [[ ! -v $topology_variable || -z ${!topology_variable} ]]; then
      topology_label=$topology_variable
      if [[ "$topology_variable" == ILVR_SOCKET_IFNAME ]]; then
        topology_label=interface
      fi
      printf 'all distributed variables must be explicitly set and non-empty: %s\n' "$topology_label" >&2
      exit 2
    fi
  done
  ilvr_num_machines=$ILVR_NUM_MACHINES
  ilvr_num_processes=$ILVR_NUM_PROCESSES
  ilvr_machine_rank=$ILVR_MACHINE_RANK
  ilvr_main_process_ip=$ILVR_MAIN_PROCESS_IP
  ilvr_main_process_port=$ILVR_MAIN_PROCESS_PORT
  ilvr_socket_ifname=$ILVR_SOCKET_IFNAME
  [[ "$ilvr_num_machines" == 2 ]] || { printf 'ILVR_NUM_MACHINES must be exactly 2\n' >&2; exit 2; }
  [[ "$ilvr_num_processes" == 4 ]] || { printf 'ILVR_NUM_PROCESSES must be exactly 4\n' >&2; exit 2; }
  [[ "$ilvr_machine_rank" == 0 || "$ilvr_machine_rank" == 1 ]] || { printf 'ILVR_MACHINE_RANK must be a machine rank of 0 or 1\n' >&2; exit 2; }
  [[ "$ilvr_main_process_ip" == 10.1.102.41 ]] || { printf 'ILVR_MAIN_PROCESS_IP must be 10.1.102.41\n' >&2; exit 2; }
  [[ "$ilvr_socket_ifname" == bond1.102 ]] || { printf 'ILVR_SOCKET_IFNAME must be interface bond1.102\n' >&2; exit 2; }
  if [[ ! "$ilvr_main_process_port" =~ ^[0-9]+$ ]] || \
    (( 10#$ilvr_main_process_port < 1024 || 10#$ilvr_main_process_port > 65535 )); then
    printf 'ILVR_MAIN_PROCESS_PORT must be an integer between 1024 and 65535\n' >&2
    exit 2
  fi
  if [[ -v STAGE2_NUM_PROCESSES && "$STAGE2_NUM_PROCESSES" != "$ilvr_num_processes" ]]; then
    printf 'STAGE2_NUM_PROCESSES must match ILVR_NUM_PROCESSES\n' >&2
    exit 2
  fi
else
  ilvr_num_machines=1
  ilvr_num_processes=${STAGE2_NUM_PROCESSES:-2}
  ilvr_machine_rank=0
  ilvr_main_process_ip=
  ilvr_main_process_port=${STAGE2_MAIN_PROCESS_PORT:-29502}
  ilvr_socket_ifname=
fi

accelerate_topology=(
  --num_processes "$ilvr_num_processes"
  --main_process_port "$ilvr_main_process_port"
)
if [[ "$ilvr_num_machines" == 2 ]]; then
  accelerate_topology+=(
    --num_machines 2
    --machine_rank "$ilvr_machine_rank"
    --main_process_ip "$ilvr_main_process_ip"
  )
  export MASTER_ADDR="$ilvr_main_process_ip"
  export GLOO_SOCKET_IFNAME="$ilvr_socket_ifname"
  export NCCL_SOCKET_IFNAME="$ilvr_socket_ifname"
fi

MANAGED_OPTIONS=(
  --stage
  --load_model_path
  --data_path
  --image_root
  --dataloader_num_workers
  --save_model_path
  --log_file
  --latent_size
  --stage2_diagnostics_interval
  --step_preference_enabled
  --step_preference_packing
  --step_shared_prefix_enabled
  --step_sidecar_attention_backend
  --step_sidecar_query_budget
  --step_preference_mode
  --step_preference_objective
  --step_preference_beta
  --step_preference_weight
  --step_preference_margin
  --step_visual_only_context_preference
  --step_negative_path
  --step_negative_manifest
  --step_max_packed_sequence_length
  --step_packed_admission_manifest
  --step_packed_admission_manifest_sha256
  --step_packed_admission_verification
  --step_packed_admission_verification_sha256
  --step_packed_capacity_evidence_path
  --step_packed_capacity_sample_id
  --step_packed_gpu_snapshot
  --step_packed_gpu_snapshot_sha256
  --ce_weight
  --step_dpo_beta
  --step_dpo_weight
  --step_entropy_weight
  --step_entropy_margin
  --stage2a_ratio
  --stage2_total_steps
  --stage2a_checkpoint_path
  --stage2b_preflight_only
  --supervise_latent_boundaries
  --no-supervise_latent_boundaries
  --learning_rate
  --gradient_accumulation_steps
  --per_device_train_batch_size
  --attention_implementation
  --vision_attention_implementation
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
  --seed
  --benchmark_evidence_path
  --stage2_length_manifest
  --stage2_length_manifest_sha256
  --stage2_launcher_log_capture
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

require_finite_non_negative_number() {
  local name=$1
  local value=$2
  if ! "$REPO_ROOT/venv/ilvr-venv/bin/python" - "$value" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (OverflowError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value >= 0 else 1)
PY
  then
    printf '%s must be a finite non-negative number\n' "$name" >&2
    exit 2
  fi
}

require_finite_number() {
  local name=$1
  local value=$2
  if ! "$REPO_ROOT/venv/ilvr-venv/bin/python" - "$value" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (OverflowError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) else 1)
PY
  then
    printf '%s must be a finite number\n' "$name" >&2
    exit 2
  fi
}

require_finite_positive_number() {
  local name=$1
  local value=$2
  if ! "$REPO_ROOT/venv/ilvr-venv/bin/python" - "$value" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (OverflowError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
  then
    printf '%s must be a finite positive number\n' "$name" >&2
    exit 2
  fi
}

require_number_equal_one() {
  local name=$1
  local value=$2
  if ! "$REPO_ROOT/venv/ilvr-venv/bin/python" - "$value" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (OverflowError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value == 1.0 else 1)
PY
  then
    printf 'Context preference requires %s=1\n' "$name" >&2
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
require_finite_non_negative_number CE_WEIGHT "$CE_WEIGHT"
if [[ "$STEP_PREFERENCE_ENABLED" == true ]]; then
  if [[ "$STEP_PREFERENCE_OBJECTIVE" == context_preference ]]; then
    require_finite_positive_number STEP_PREFERENCE_BETA "$STEP_PREFERENCE_BETA"
    require_finite_non_negative_number STEP_PREFERENCE_WEIGHT "$STEP_PREFERENCE_WEIGHT"
    require_finite_number STEP_PREFERENCE_MARGIN "$STEP_PREFERENCE_MARGIN"
    if [[ "$STEP_PREFERENCE_MODE" == stage2b ]]; then
      require_number_equal_one CE_WEIGHT "$CE_WEIGHT"
    fi
  else
    require_finite_positive_number STEP_DPO_BETA "$STEP_DPO_BETA"
    require_finite_non_negative_number STEP_DPO_WEIGHT "$STEP_DPO_WEIGHT"
  fi
  require_finite_non_negative_number STEP_ENTROPY_WEIGHT "$STEP_ENTROPY_WEIGHT"
fi

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

case "$VISION_ATTENTION_IMPLEMENTATION" in
  eager|sdpa|flash_attention_2) ;;
  *)
    printf 'VISION_ATTENTION_IMPLEMENTATION must be eager, sdpa, or flash_attention_2\n' >&2
    exit 2
    ;;
esac

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=(--resume_from_checkpoint_path "$RESUME_CHECKPOINT")
fi

TUNING_ARGS=()
if [[ "$TUNING_METHOD_EXPLICIT" == true || ( -z "${RESUME_CHECKPOINT:-}" && "$stage2b_checkpoint_initialization" == false ) ]]; then
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

EVIDENCE_ARGS=()
capacity_variables=(STEP_PACKED_CAPACITY_EVIDENCE_PATH STEP_PACKED_CAPACITY_SAMPLE_ID STEP_PACKED_GPU_SNAPSHOT STEP_PACKED_GPU_SNAPSHOT_SHA256)
capacity_requested=false
for capacity_variable in "${capacity_variables[@]}"; do
  if [[ -v "$capacity_variable" ]]; then capacity_requested=true; fi
done
if [[ "$capacity_requested" == true ]]; then
  for capacity_variable in "${capacity_variables[@]}"; do
    if [[ -z "${!capacity_variable:-}" ]]; then
      printf 'capacity-only options require %s\n' "$capacity_variable" >&2
      exit 2
    fi
  done
  if [[ "$step_preference_packing" != true || "${STEP_PREFERENCE_MODE:-}" != stage2b || "$TUNING_METHOD" != dora || "$GRADIENT_ACCUMULATION_STEPS" != 2 || "$DATALOADER_NUM_WORKERS" != 0 || "${STAGE2_NUM_PROCESSES:-2}" != 2 || -z "${DEEPSPEED_CONFIG:-}" || -n "${RESUME_CHECKPOINT:-}" || -n "${BENCHMARK_EVIDENCE_PATH:-}" ]]; then
    printf 'capacity-only options require packed Stage 2B DoRA two-rank ZeRO-2 accumulation 2\n' >&2
    exit 2
  fi
  if [[ ! "$STEP_PACKED_GPU_SNAPSHOT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'STEP_PACKED_GPU_SNAPSHOT_SHA256 must be a lowercase SHA-256\n' >&2
    exit 2
  fi
  EVIDENCE_ARGS+=(--step_packed_capacity_evidence_path "$STEP_PACKED_CAPACITY_EVIDENCE_PATH"
    --step_packed_capacity_sample_id "$STEP_PACKED_CAPACITY_SAMPLE_ID"
    --step_packed_gpu_snapshot "$STEP_PACKED_GPU_SNAPSHOT"
    --step_packed_gpu_snapshot_sha256 "$STEP_PACKED_GPU_SNAPSHOT_SHA256")
fi
if [[ -n "${BENCHMARK_EVIDENCE_PATH:-}" ]]; then
  EVIDENCE_ARGS+=(--benchmark_evidence_path "$BENCHMARK_EVIDENCE_PATH")
fi

MANIFEST_ARGS=()
LAUNCHER_LOG_ARGS=()
if [[ "$step_preference_packing" == true && -z "${RESUME_CHECKPOINT:-}" ]]; then
  LAUNCHER_LOG_ARGS+=(--stage2_launcher_log_capture)
fi
TRAINING_ARGS=(
  --stage stage2
  --load_model_path "$STAGE1_CHECKPOINT"
  --data_path "$DATA_PATH"
  --image_root "$IMAGE_ROOT"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --save_model_path "$OUTPUT_DIR"
  --log_file "$LOG_FILE"
  "${LAUNCHER_LOG_ARGS[@]}"
  --latent_size "$LATENT_SIZE"
  --stage2_diagnostics_interval "$STAGE2_DIAGNOSTICS_INTERVAL"
  "${STEP_PREFERENCE_ARGS[@]}"
  --ce_weight "$CE_WEIGHT"
  "${BOUNDARY_ARGS[@]}"
  --learning_rate "$LEARNING_RATE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --attention_implementation "$ATTENTION_IMPLEMENTATION"
  --vision_attention_implementation "$VISION_ATTENTION_IMPLEMENTATION"
  "${CHECKPOINTING_ARGS[@]}"
  --max_seq_length_train "$MAX_SEQUENCE_LENGTH"
  "${TUNING_ARGS[@]}"
  --seed "$SEED"
  "${RESUME_ARGS[@]}"
  "${DEEPSPEED_ARGS[@]}"
  "${EVIDENCE_ARGS[@]}"
  "$@"
)

if [[ "$stage2_dry_run" != true && "$stage2b_checkpoint_initialization" == true ]]; then
  "$REPO_ROOT/venv/ilvr-venv/bin/python" -m main \
    "${TRAINING_ARGS[@]}" --stage2b_preflight_only
fi

if [[ "$stage2_dry_run" != true ]]; then
  "$REPO_ROOT/venv/ilvr-venv/bin/python" -m src.artifact_paths validate-run \
    --checkpoint "$OUTPUT_DIR" --log "$LOG_FILE" \
    --stage2-length-manifest "$length_manifest" >/dev/null
  if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
    if [[ "$ilvr_multinode" == false ]]; then
      STAGE2_NUM_PROCESSES=${STAGE2_NUM_PROCESSES:-2}
      require_positive_integer STAGE2_NUM_PROCESSES "$STAGE2_NUM_PROCESSES"
    fi
    "$REPO_ROOT/venv/ilvr-venv/bin/python" -m src.deepspeed_runtime validate \
      --config "$DEEPSPEED_CONFIG" --world-size "$ilvr_num_processes" >/dev/null
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
    if [[ "$step_preference_packing" == true ]]; then
      # The launcher owns this fresh log, including preprocessing and failures.
      # Python uses its stream handler only under this explicit capture mode.
      set -o noclobber
      exec >"$LOG_FILE" 2>&1
      set +o noclobber
    fi
    "$REPO_ROOT/venv/ilvr-venv/bin/python" -m stage2.length_manifest \
      --data-path "$DATA_PATH" \
      --image-root "$IMAGE_ROOT" \
      --processor-source "$processor_source" \
      --latent-size "$LATENT_SIZE" \
      "$MANIFEST_BOUNDARY_ARG" \
      --max-sequence-length "$MAX_SEQUENCE_LENGTH" \
      --assistant-plan-rendering "$ASSISTANT_PLAN_RENDERING" \
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

if [[ -n "${DEEPSPEED_CONFIG:-}" || "$ilvr_multinode" == true ]]; then
  if [[ "$ilvr_multinode" == false ]]; then
    STAGE2_NUM_PROCESSES=${STAGE2_NUM_PROCESSES:-2}
    require_positive_integer STAGE2_NUM_PROCESSES "$STAGE2_NUM_PROCESSES"
  fi
  ACCELERATE_DEEPSPEED_ARGS=()
  if [[ "$ilvr_num_machines" == 2 ]]; then
    ACCELERATE_DEEPSPEED_ARGS=(
      --use_deepspeed
      --deepspeed_multinode_launcher standard
      --deepspeed_config_file "$DEEPSPEED_CONFIG"
    )
  fi
  COMMAND=(
    "$REPO_ROOT/venv/ilvr-venv/bin/accelerate" launch
    "${accelerate_topology[@]}"
    "${ACCELERATE_DEEPSPEED_ARGS[@]}"
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

if [[ "$stage2_dry_run" == true ]]; then
  "$REPO_ROOT/venv/ilvr-venv/bin/python" -c \
    'import sys; from utils_deepseed import parse_args; parse_args(sys.argv[1:])' \
    "${TRAINING_ARGS[@]}"
  if [[ "$STAGE2_DRY_RUN_FORMAT" == json ]]; then
    "$REPO_ROOT/venv/ilvr-venv/bin/python" -c \
      'import json, sys; print(json.dumps(sys.argv[1:], separators=(",", ":")))' \
      "${COMMAND[@]}"
    exit 0
  fi
  printf 'PYTHONPATH=%q TOKENIZERS_PARALLELISM=%q ASSISTANT_PLAN_RENDERING=%q ' \
    "$PYTHONPATH" "$TOKENIZERS_PARALLELISM" "$ASSISTANT_PLAN_RENDERING"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
