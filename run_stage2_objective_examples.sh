#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

usage() {
  cat >&2 <<'EOF'
usage: run_stage2_objective_examples.sh MODE --stage1 PATH --data FILE
       --image-root DIR --run-id ID --max-steps N --main-process-port PORT
       [--lambda-dyn VALUE] [--lambda-vis VALUE] [--dry-run | --launch]

MODE is ce, entropy, or combined. Dry-run is the default; --launch is required
to start the existing two-node/four-rank coordinator.
EOF
}

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

take_value() {
  local option=$1
  local remaining=$2
  if (( remaining < 2 )); then
    fail "$option requires a value"
  fi
}

positive_finite() {
  "$script_dir/venv/ilvr-venv/bin/python" - "$1" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (OverflowError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and value > 0 else 1)
PY
}

mode=${1-}
if (( $# )); then
  shift
fi
case "$mode" in
  ce|entropy|combined) ;;
  *)
    usage
    fail 'mode must be ce, entropy, or combined'
    ;;
esac

stage1=
data_path=
image_root=
run_id=
max_steps=
main_process_port=
lambda_dyn=
lambda_vis=
stage1_seen=false
data_seen=false
image_root_seen=false
run_id_seen=false
max_steps_seen=false
port_seen=false
lambda_dyn_seen=false
lambda_vis_seen=false
dry_run_seen=false
launch_seen=false

while (( $# )); do
  case "$1" in
    --stage1)
      take_value "$1" "$#"
      [[ $stage1_seen == false ]] || fail '--stage1 may be supplied exactly once'
      stage1=$2
      stage1_seen=true
      shift 2
      ;;
    --data)
      take_value "$1" "$#"
      [[ $data_seen == false ]] || fail '--data may be supplied exactly once'
      data_path=$2
      data_seen=true
      shift 2
      ;;
    --image-root)
      take_value "$1" "$#"
      [[ $image_root_seen == false ]] || fail '--image-root may be supplied exactly once'
      image_root=$2
      image_root_seen=true
      shift 2
      ;;
    --run-id)
      take_value "$1" "$#"
      [[ $run_id_seen == false ]] || fail '--run-id may be supplied exactly once'
      run_id=$2
      run_id_seen=true
      shift 2
      ;;
    --max-steps)
      take_value "$1" "$#"
      [[ $max_steps_seen == false ]] || fail '--max-steps may be supplied exactly once'
      max_steps=$2
      max_steps_seen=true
      shift 2
      ;;
    --main-process-port)
      take_value "$1" "$#"
      [[ $port_seen == false ]] || fail '--main-process-port may be supplied exactly once'
      main_process_port=$2
      port_seen=true
      shift 2
      ;;
    --lambda-dyn)
      take_value "$1" "$#"
      [[ $lambda_dyn_seen == false ]] || fail '--lambda-dyn may be supplied exactly once'
      lambda_dyn=$2
      lambda_dyn_seen=true
      shift 2
      ;;
    --lambda-vis)
      take_value "$1" "$#"
      [[ $lambda_vis_seen == false ]] || fail '--lambda-vis may be supplied exactly once'
      lambda_vis=$2
      lambda_vis_seen=true
      shift 2
      ;;
    --dry-run)
      [[ $dry_run_seen == false ]] || fail '--dry-run may be supplied exactly once'
      dry_run_seen=true
      shift
      ;;
    --launch)
      [[ $launch_seen == false ]] || fail '--launch may be supplied exactly once'
      launch_seen=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option: $1"
      ;;
  esac
done

[[ $stage1_seen == true ]] || fail '--stage1 is required'
[[ $data_seen == true ]] || fail '--data is required'
[[ $image_root_seen == true ]] || fail '--image-root is required'
[[ $run_id_seen == true ]] || fail '--run-id is required'
[[ $max_steps_seen == true ]] || fail '--max-steps is required'
[[ $port_seen == true ]] || fail '--main-process-port is required'

[[ $dry_run_seen != true || $launch_seen != true ]] || \
  fail '--dry-run and --launch are mutually exclusive'

[[ $stage1 == /* && -d $stage1 && -r $stage1 ]] || \
  fail '--stage1 must be a readable absolute directory'
[[ $data_path == /* && -f $data_path && -r $data_path ]] || \
  fail '--data must be a readable file at an absolute path'
[[ $image_root == /* && -d $image_root && -r $image_root ]] || \
  fail '--image-root must be a readable absolute directory'
[[ $run_id =~ ^[a-z0-9][a-z0-9-]{2,127}$ ]] || \
  fail '--run-id must use 3-128 lowercase letters, digits, or hyphens'
[[ $max_steps =~ ^[1-9][0-9]*$ ]] || \
  fail '--max-steps must be a positive integer'
if [[ ! $main_process_port =~ ^[0-9]+$ ]] || \
    (( 10#$main_process_port < 1024 || 10#$main_process_port > 65535 )); then
  fail '--main-process-port must be an integer between 1024 and 65535'
fi

case "$mode" in
  ce)
    [[ $lambda_dyn_seen == false && $lambda_vis_seen == false ]] || \
      fail 'ce mode does not accept auxiliary weights'
    lambda_dyn=0
    lambda_vis=0
    ;;
  entropy)
    [[ $lambda_dyn_seen == true ]] || fail '--lambda-dyn is required for entropy mode'
    positive_finite "$lambda_dyn" || fail '--lambda-dyn must be positive and finite'
    [[ $lambda_vis_seen == false ]] || fail 'entropy mode does not accept --lambda-vis'
    lambda_vis=0
    ;;
  combined)
    [[ $lambda_dyn_seen == true ]] || fail '--lambda-dyn is required for combined mode'
    [[ $lambda_vis_seen == true ]] || fail '--lambda-vis is required for combined mode'
    positive_finite "$lambda_dyn" || fail '--lambda-dyn must be positive and finite'
    positive_finite "$lambda_vis" || fail '--lambda-vis must be positive and finite'
    ;;
esac

for destination in \
  "$script_dir/checkpoints/$run_id" \
  "$script_dir/results/$run_id" \
  "$script_dir/logs/multinode/$run_id" \
  "$script_dir/logs/zero-stage-compatibility/$run_id"; do
  [[ ! -e $destination && ! -L $destination ]] || \
    fail "fresh run destination already exists: $destination"
done

coordinator=${ILVR_STAGE2_OBJECTIVE_COORDINATOR:-"$script_dir/run_multinode_stage2_training.sh"}
[[ $coordinator == /* && -x $coordinator ]] || \
  fail 'Stage 2 coordinator must be an executable absolute path'

objective_arguments=(
  --enable_stage2_objective
  --lambda_dyn "$lambda_dyn"
  --lambda_vis "$lambda_vis"
  --gamma_H 0
  --m_V 0
  --entropy_eps 1e-6
  --pre_window 1
  --post_window 1
  --stage2_m_execution packed_suffix
)
if [[ $mode != ce ]]; then
  objective_arguments+=(--reference_model_path "$stage1")
fi

coordinator_arguments=(
  --run-id "$run_id"
  --backend ddp
  --main-process-port "$main_process_port"
  --save_strategy steps
  --save_steps "$max_steps"
  --max_steps "$max_steps"
  --warm_up_steps 10
  --seed 42
  "${objective_arguments[@]}"
)
if [[ $launch_seen != true ]]; then
  coordinator_arguments+=(--dry-run)
fi

export STAGE1_CHECKPOINT=$stage1
export DATA_PATH=$data_path
export IMAGE_ROOT=$image_root
export TUNING_METHOD=dora
export LORA_R=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05
export LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export LATENT_SIZE=8
export SUPERVISE_LATENT_BOUNDARIES=true
export LEARNING_RATE=5e-5
export GRADIENT_ACCUMULATION_STEPS=2
export ATTENTION_IMPLEMENTATION=sdpa
export GRADIENT_CHECKPOINTING=true
export MAX_SEQUENCE_LENGTH=8192

exec "$coordinator" "${coordinator_arguments[@]}"
