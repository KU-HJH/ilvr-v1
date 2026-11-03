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
ATTENTION_IMPLEMENTATION=${ATTENTION_IMPLEMENTATION:-sdpa}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
MAX_SEQUENCE_LENGTH=${MAX_SEQUENCE_LENGTH:-8192}
PER_DEVICE_TRAIN_BATCH_SIZE=1

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
  [[ "$ilvr_main_process_ip" == 172.16.11.41 ]] || { printf 'ILVR_MAIN_PROCESS_IP must be 172.16.11.41\n' >&2; exit 2; }
  [[ "$ilvr_socket_ifname" == bond0.11 ]] || { printf 'ILVR_SOCKET_IFNAME must be interface bond0.11\n' >&2; exit 2; }
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
  --save_model_path
  --log_file
  --latent_size
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
  --benchmark_evidence_path
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

require_positive_integer LATENT_SIZE "$LATENT_SIZE"
require_positive_integer GRADIENT_ACCUMULATION_STEPS "$GRADIENT_ACCUMULATION_STEPS"
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

EVIDENCE_ARGS=()
if [[ -n "${BENCHMARK_EVIDENCE_PATH:-}" ]]; then
  EVIDENCE_ARGS+=(--benchmark_evidence_path "$BENCHMARK_EVIDENCE_PATH")
fi

TRAINING_ARGS=(
  --stage stage2
  --load_model_path "$STAGE1_CHECKPOINT"
  --data_path "$DATA_PATH"
  --image_root "$IMAGE_ROOT"
  --save_model_path "$OUTPUT_DIR"
  --log_file "$LOG_FILE"
  --latent_size "$LATENT_SIZE"
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
  "${EVIDENCE_ARGS[@]}"
  "$@"
)

if [[ "${STAGE2_DRY_RUN:-false}" != "true" ]]; then
  "$REPO_ROOT/venv/ilvr-venv/bin/python" -m src.artifact_paths validate-run \
    --checkpoint "$OUTPUT_DIR" --log "$LOG_FILE" >/dev/null
  if [[ -n "${DEEPSPEED_CONFIG:-}" ]]; then
    if [[ "$ilvr_multinode" == false ]]; then
      STAGE2_NUM_PROCESSES=${STAGE2_NUM_PROCESSES:-2}
      require_positive_integer STAGE2_NUM_PROCESSES "$STAGE2_NUM_PROCESSES"
    fi
    "$REPO_ROOT/venv/ilvr-venv/bin/python" -m src.deepspeed_runtime validate \
      --config "$DEEPSPEED_CONFIG" --world-size "$ilvr_num_processes" >/dev/null
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

if [[ "${STAGE2_DRY_RUN:-false}" == "true" ]]; then
  printf 'PYTHONPATH=%q TOKENIZERS_PARALLELISM=%q ' \
    "$PYTHONPATH" "$TOKENIZERS_PARALLELISM"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
