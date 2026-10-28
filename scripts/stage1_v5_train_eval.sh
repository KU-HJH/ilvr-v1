#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$repo_root"
run_id= port= remote_root= remote_host_root= max_steps= settings=
dry_run=false
query_mode=legacy history_text_gamma=0.9 history_attention_layers=last
voff_mode=attention_mask
query_arguments=()
while (( $# )); do
    case $1 in
        --run-id|--main-process-port|--remote-repo-root|--remote-host-repo-root|--max-steps|--settings)
            (( $# >= 2 )) && [[ -n $2 ]] || { printf 'error: %s requires a value\n' "$1" >&2; exit 2; }
            case $1 in
                --run-id) run_id=$2 ;; --main-process-port) port=$2 ;;
                --remote-repo-root) remote_root=$2 ;; --remote-host-repo-root) remote_host_root=$2 ;;
                --max-steps) max_steps=$2 ;; --settings) settings=$2 ;;
            esac
            shift 2 ;;
        --query_mode|--history_text_gamma|--history_attention_layers)
            (( $# >= 2 )) && [[ -n $2 ]] || { printf 'error: %s requires a value\n' "$1" >&2; exit 2; }
            query_arguments+=("$1" "$2")
            shift 2 ;;
        --stage1-v5-voff-mode)
            (( $# >= 2 )) && [[ -n $2 ]] || { printf 'error: %s requires a value\n' "$1" >&2; exit 2; }
            voff_mode=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ $run_id =~ ^[a-z0-9][a-z0-9-]{2,119}$ ]] || { printf 'error: explicit valid run ID required\n' >&2; exit 2; }
[[ $max_steps =~ ^[1-9][0-9]*$ ]] || { printf 'error: --max-steps must come from the prepared-loader audit\n' >&2; exit 2; }
[[ $port =~ ^[0-9]+$ ]] && (( 10#$port >= 1024 && 10#$port <= 65535 )) || { printf 'error: valid main process port required\n' >&2; exit 2; }
[[ $remote_root == /* && $remote_host_root == /* ]] || { printf 'error: explicit absolute remote roots required\n' >&2; exit 2; }
# Preserve caller query choices across sourced stale settings.
readonly -a requested_query_arguments=("${query_arguments[@]}")
readonly requested_voff_mode=$voff_mode
if [[ -n $settings ]]; then source "$settings"; fi
voff_mode=$requested_voff_mode
case $voff_mode in
    attention_mask|cross_helper_reverse_topk_independent|cross_helper_reverse_topk_cumulative) ;;
    *) printf 'error: invalid Stage1-v5 V-off mode\n' >&2; exit 2 ;;
esac
query_mode=legacy history_text_gamma=0.9 history_attention_layers=last
history_requested=false
for (( i=0; i<${#requested_query_arguments[@]}; i+=2 )); do
    case ${requested_query_arguments[i]} in
        --query_mode) query_mode=${requested_query_arguments[i+1]} ;;
        --history_text_gamma) history_text_gamma=${requested_query_arguments[i+1]}; history_requested=true ;;
        --history_attention_layers) history_attention_layers=${requested_query_arguments[i+1]}; history_requested=true ;;
    esac
done
if [[ $query_mode != legacy && $query_mode != adaptive ]]; then
    printf 'error: query mode must be legacy or adaptive\n' >&2; exit 2
fi
if [[ $query_mode == legacy && $history_requested == true ]]; then
    printf 'error: legacy query does not accept history controls\n' >&2; exit 2
fi
# All selected settings below are explicit after sourcing. The original FULL
# base is shared external pretrained input, never a previous trained checkpoint.
base=/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
python="$repo_root/venv/ilvr-venv/bin/python"
export PYTHONPATH="$repo_root/transformers/src:$repo_root:$repo_root/src"
export STAGE1_V5_DEBUG=false
export STAGE1_NUM_PROCESSES=4 STAGE1_CE_WEIGHT=0.1 CE_WEIGHT=0.1 STAGE1_GRADIENT_ACCUMULATION=4 STAGE1_SEED=42
export STAGE1_MODEL="$base" STAGE1_MICRO_BATCH=1 STAGE1_LEARNING_RATE=5e-5 STAGE1_EMA_TAU=1.0
# Clear optional sourced invocation state so fresh runs cannot accidentally resume.
unset STAGE1_RESUME_PATH STAGE1_BENCHMARK_RUN_CONFIG STAGE1_BENCHMARK_EVIDENCE_PATH STAGE1_BENCHMARK_SMOKE_RESUME_WITNESS
roots=(--local-repo-root "$repo_root" --remote-repo-root "$remote_root" --remote-host-repo-root "$remote_host_root")
eval_id="$run_id-test400"
prepared="$repo_root/checkpoints/evaluation-staging/$run_id/prepared-model"
train=("$repo_root/run_multinode_stage1_training.sh" --run-id "$run_id" --zero-stage 1 --main-process-port "$port" "${roots[@]}"
    --method dora --model "$base" --data-path "$repo_root/data/TRAIN.jsonl" --image-root "$repo_root/data/images_comt"
    --stage1-variant v5 --no-stage1-v5-debug --stage1-loss-type dil-bce-orig --ce-weight 0.1 --sim-weight 0.1
    --lambda-preference 1 --lambda-topk-entropy 0.1 --preference-topk-k 50 --preference-topk-mode independent
    --no-topk-include-gt --stage1-v5-selection all --stage1-v5-text-kl-weight 0.01
    --loss-chunk-positions 128 --step-preference-beta 1 --step-preference-margin 0
    --micro-batch 1 --gradient-accumulation 4 --epochs 1 --max-steps "$max_steps" --learning-rate 5e-5
    --seed 42 --warm-up-steps 10 --ema-tau 1.0 --latent-size 8 --attention-implementation sdpa
    --vision-attention-implementation flash_attention_2 --max-seq-length-train 32768 --dataloader-num-workers 0
    --save-strategy steps --save-steps "$max_steps" --final-save)
train+=(--query_mode "$query_mode")
train+=(--stage1-v5-voff-mode "$voff_mode")
if [[ $query_mode == adaptive ]]; then
    unset STAGE1_TEXTUAL_STATES
    train+=(--history_text_gamma "$history_text_gamma" --history_attention_layers "$history_attention_layers")
else
    train+=(--textual_states previous)
fi
prepare=("$python" -m src.multinode prepare-inference --stage stage1 --tuning-method dora --zero-stage 1
    --run-id "$run_id" --main-process-port "$port" "${roots[@]}" --checkpoint-step "$max_steps" --destination "$prepared")
evaluate=("$python" -m src.multinode evaluate --run-id "$eval_id" "${roots[@]}" --model-dir "$prepared" --base-model-dir "$base"
    --test-data-path "$repo_root/data/TEST.jsonl" --task-name zebra-cot --cache-dir "$repo_root/data/cache" --output-dir "$repo_root/results/$eval_id"
    --max-new-tokens 4096 --max-sequence-length 128000 --repetition-penalty 1 --temperature 0 --top-p 1
    --vision-ablation none --generation-limit-policy external_full_latent_block)
if [[ $dry_run == true ]]; then
    for name in train prepare evaluate; do declare -n command=$name; printf '%q ' "${command[@]}"; printf '\n'; done
    exit 0
fi
if ! [[ -f "$repo_root/data/TEST.jsonl" ]] || ! awk 'NF { count++ } END { exit count != 400 }' "$repo_root/data/TEST.jsonl"; then
    printf 'error: TEST400 requires exactly 400 nonempty records in data/TEST.jsonl\n' >&2
    exit 2
fi
"${train[@]}"
"${prepare[@]}"
"${evaluate[@]}"
