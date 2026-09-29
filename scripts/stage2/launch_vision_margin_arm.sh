#!/usr/bin/env bash
# Launch one pre-registered Vision-margin arm through the existing two-node
# coordinator. Dry-run by default; pass --launch only after live preflight and
# an explicit deployment decision. See
# documents/stage2-vision-margin-preregistration.md.
set -euo pipefail

usage() {
  echo "usage: $0 {visiononly-mv05|combined-mv0|combined-mv05|visiononly-l03m0} --main-process-port PORT [--run-id ID] [--launch]" >&2
  exit 2
}

[[ $# -ge 1 ]] || usage
arm=$1; shift
port=
run_id=
launch=false
while [[ $# -gt 0 ]]; do
  case $1 in
    --main-process-port) port=${2:?}; shift 2 ;;
    --run-id) run_id=${2:?}; shift 2 ;;
    --launch) launch=true; shift ;;
    *) usage ;;
  esac
done
[[ $port =~ ^[0-9]+$ ]] || usage

lambda_vis=1.0
case $arm in
  visiononly-mv05) lambda_dyn=0; m_v=0.5 ;;
  # Coefficient/margin follow-up (2026-09-29): ours1 values, all else as visiononly-mv05.
  visiononly-l03m0) lambda_dyn=0; m_v=0.0; lambda_vis=0.3 ;;
  combined-mv0) lambda_dyn=1.0; m_v=0.0 ;;
  combined-mv05) lambda_dyn=1.0; m_v=0.5 ;;
  *) usage ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
repo=/root/workspace/ILVR
base=$repo/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
stage1=$repo/checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431
max_steps=212
run_id=${run_id:-stage2-$arm-gb16-s212-$(date -u +%Y%m%d-%H%M%S)}

objective_arguments=(
  --enable_stage2_objective
  --lambda_dyn "$lambda_dyn"
  --lambda_vis "$lambda_vis"
  --m_V "$m_v"
  --gamma_H 0
  --entropy_eps 1e-6
  --pre_window 1
  --post_window 1
  --vision_token_reduction score_mean
  --stage2_m_execution packed_suffix
)
if [[ $lambda_dyn != 0 ]]; then
  objective_arguments+=(--reference_model_path "$stage1")
fi

coordinator_arguments=(
  --run-id "$run_id"
  --backend ddp
  --main-process-port "$port"
  --stage1_adapter_path "$stage1"
  --save_strategy steps
  --save_steps "$max_steps"
  --max_steps "$max_steps"
  --warm_up_steps 10
  --seed 42
  "${objective_arguments[@]}"
)
[[ $launch == true ]] || coordinator_arguments+=(--dry-run)

export STAGE1_CHECKPOINT=$base
export DATA_PATH=$repo/data/TRAIN.jsonl
export IMAGE_ROOT=$repo/data/images_comt
export TUNING_METHOD=dora
export LORA_R=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05
export LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export LATENT_SIZE=8
export SUPERVISE_LATENT_BOUNDARIES=true
export LEARNING_RATE=5e-5
export GRADIENT_ACCUMULATION_STEPS=4
export ATTENTION_IMPLEMENTATION=sdpa
export VISION_ATTENTION_IMPLEMENTATION=flash_attention_2
export STAGE2_LATENT_ACTIVATION_CHECKPOINTING=true
export GRADIENT_CHECKPOINTING=true
export MAX_SEQUENCE_LENGTH=8192

echo "arm=$arm run_id=$run_id launch=$launch" >&2
exec "$script_dir/run_multinode_stage2_training.sh" "${coordinator_arguments[@]}"
