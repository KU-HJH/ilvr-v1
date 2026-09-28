#!/usr/bin/env bash
# Source after any old run.env, then set the experiment's DIL selector/variants.
# This file only sets environment; it never starts a process or creates a run.
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  printf 'Fresh DIL profile refuses RESUME_CHECKPOINT; use saved resume settings\n' >&2
  return 2 2>/dev/null || exit 2
fi
_dil_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
export STAGE1_CHECKPOINT="$_dil_repo_root/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct"
export STAGE1_ADAPTER_PATH="$_dil_repo_root/checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431"
export DATA_PATH="$_dil_repo_root/data/TRAIN.jsonl"
export IMAGE_ROOT="$_dil_repo_root/data"
export TUNING_METHOD=dora LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.05
export LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export LATENT_SIZE=8 SUPERVISE_LATENT_BOUNDARIES=true
export LEARNING_RATE=5e-5 GRADIENT_ACCUMULATION_STEPS=4 SEED=42
export CE_WEIGHT=1.0 MAX_SEQUENCE_LENGTH=8192 DATALOADER_NUM_WORKERS=0
export ATTENTION_IMPLEMENTATION=sdpa VISION_ATTENTION_IMPLEMENTATION=flash_attention_2
export GRADIENT_CHECKPOINTING=false STAGE2_LATENT_ACTIVATION_CHECKPOINTING=true
export STEP_PREFERENCE_ENABLED=true STEP_PREFERENCE_MODE=stage2b
export STEP_PREFERENCE_OBJECTIVE=context_preference STEP_PREFERENCE_TARGET_SCOPE=immediate_text
export STEP_PREFERENCE_PREFIX_MODE=clean_prefix STEP_PREFERENCE_BRANCH_MODE=independent
export STEP_PREFERENCE_BLOCK_ORIGINAL_IMAGE=false
export STEP_VISUAL_ONLY_CONTEXT_PREFERENCE=true STEP_VISUAL_OFF_SUPPRESS_LATENT_BOUNDARIES=false
export STEP_SHARED_PREFIX_ENABLED=true STEP_PREFERENCE_PACKING=false
export STEP_PREFERENCE_BETA=0.5 STEP_PREFERENCE_WEIGHT=1.0 STEP_PREFERENCE_MARGIN=0
export STEP_DPO_BETA=0.5 STEP_DPO_WEIGHT=1.0 STEP_ENTROPY_WEIGHT=0 STEP_ENTROPY_MARGIN=0.1
export STEP_SIDECAR_ATTENTION_BACKEND=flex STEP_SIDECAR_QUERY_BUDGET=1024
export STAGE2_SAMPLER_POLICY=reference_random STAGE2_ASSISTANT_PLAN_RENDERING=concatenated-v1
export STAGE2B_DIRECT_FROM_STAGE1=true STAGE2A_RATIO=0.5 STAGE2_TOTAL_STEPS=212
export STEP_PREFERENCE_LOSS_TYPE=dpo STEP_PREFERENCE_TOPK=''
export STEP_PREFERENCE_LAMBDA_TOPK_MASS=0 STEP_PREFERENCE_ENTROPY_TOP_PERCENT=''
export STEP_PREFERENCE_ENTROPY_SOURCE=clean
unset STAGE2A_CHECKPOINT_PATH STEP_NEGATIVE_PATH STEP_NEGATIVE_MANIFEST
unset _dil_repo_root
