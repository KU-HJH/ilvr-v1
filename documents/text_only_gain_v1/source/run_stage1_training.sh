#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$script_dir"
repository_python="$script_dir/venv/ilvr-venv/bin/python"
repository_accelerate="$script_dir/venv/ilvr-venv/bin/accelerate"
[[ -x $repository_python ]] || {
    printf 'error: repository Python is not executable: %s\n' "$repository_python" >&2
    exit 2
}
[[ -x $repository_accelerate ]] || {
    printf 'error: repository Accelerate is not executable: %s\n' "$repository_accelerate" >&2
    exit 2
}

manifest_mode=false
manifest_path=
if (( $# )) && [[ $1 == --manifest ]]; then
    if (( $# != 2 && $# != 3 )) || [[ -z ${2:-} ]] || { (( $# == 3 )) && [[ $3 != --dry-run ]]; }; then
        printf 'error: --manifest requires one path and optionally --dry-run\n' >&2
        exit 2
    fi
    manifest_mode=true
    manifest_path=$2
    manifest_dry_run=false
    if (( $# == 3 )); then
        manifest_dry_run=true
    fi
    manifest_arguments=()
    mapfile -d '' -t manifest_arguments < <(
        "$repository_python" -m src.stage1.benchmark launcher-args \
            --manifest "$manifest_path"
    )
    if (( ${#manifest_arguments[@]} == 0 )); then
        printf 'error: manifest did not produce launcher arguments: %s\n' "$manifest_path" >&2
        exit 2
    fi
    set -- "${manifest_arguments[@]}"
    if [[ $manifest_dry_run == true ]]; then
        set -- "$@" --dry-run
    fi
fi

method=${STAGE1_METHOD:-}
model=${STAGE1_MODEL:-}
data_path=${STAGE1_DATA_PATH:-}
source_data_path=${STAGE1_SOURCE_DATA_PATH:-}
image_root=${STAGE1_IMAGE_ROOT:-}
output_dir=${STAGE1_OUTPUT_DIR:-}
task=${STAGE1_TASK:-}
cache_dir=${STAGE1_CACHE_DIR:-}
max_seq_length=${STAGE1_MAX_SEQ_LENGTH:-}
micro_batch=${STAGE1_MICRO_BATCH:-8}
epochs=${STAGE1_EPOCHS:-3}
learning_rate=${STAGE1_LEARNING_RATE:-1e-5}
gradient_accumulation=${STAGE1_GRADIENT_ACCUMULATION:-2}
seed=${STAGE1_SEED:-42}
deepspeed_config=${STAGE1_DEEPSPEED_CONFIG:-"$script_dir/configs/deepspeed_zero3.json"}
stage1_num_processes=${STAGE1_NUM_PROCESSES:-2}
attention_implementation=${STAGE1_ATTENTION_IMPLEMENTATION:-flash_attention_2}
# EMA modes: moving updates the teacher; frozen keeps its parameters fixed.
ema_setting=${STAGE1_EMA_SETTING:-moving}
preference_mode=${STAGE1_PREFERENCE_MODE:-none}
preference_weight=${STAGE1_PREFERENCE_WEIGHT:-1.0}
preference_beta=${STAGE1_PREFERENCE_BETA:-0.5}
save_strategy=${STAGE1_SAVE_STRATEGY:-no}
save_steps=${STAGE1_SAVE_STEPS:-612}
resume_path=${STAGE1_RESUME_PATH:-}
log_file=${STAGE1_LOG_FILE:-}
max_steps=${STAGE1_MAX_STEPS:-}
benchmark_run_config=${STAGE1_BENCHMARK_RUN_CONFIG:-}
benchmark_evidence_path=${STAGE1_BENCHMARK_EVIDENCE_PATH:-}
benchmark_smoke_resume_witness=${STAGE1_BENCHMARK_SMOKE_RESUME_WITNESS:-}
prefix_conditioning_config=${STAGE1_PREFIX_CONDITIONING_CONFIG:-}
clean_kl_beta=${STAGE1_CLEAN_KL_BETA:-0.0}
clean_kl_warmup_ratio=${STAGE1_CLEAN_KL_WARMUP_RATIO:-0.05}
text_kl_beta=${STAGE1_TEXT_KL_BETA:-0.0}
text_kl_warmup_ratio=${STAGE1_TEXT_KL_WARMUP_RATIO:-0.05}
gain_alpha=${STAGE1_VISION_GAIN_ALPHA:-0.0}
gain_min_h_norm_sq=${STAGE1_VISION_GAIN_MIN_H_NORM_SQ:-1e-30}
gain_execution=${STAGE1_VISION_GAIN_EXECUTION:-independent}
gain_branch_mode=${STAGE1_VISION_GAIN_BRANCH_MODE:-independent}
dry_run=false
final_save=true
legacy_use_lora=false

usage() {
    printf '%s\n' \
        "Usage: $0 --method full|dora --model MODEL --data-path FILE" \
        "  --image-root DIR --output-dir DIR [options]" \
        "  --preference-mode none|iier --preference-weight FLOAT --preference-beta FLOAT" \
        "  --vision-gain-alpha FLOAT --vision-gain-min-h-norm-sq FLOAT" \
        "  --clean-kl-beta FLOAT --clean-kl-warmup-ratio FLOAT" \
        "  --text-kl-beta FLOAT --text-kl-warmup-ratio FLOAT" \
        "  --prefix-conditioning-config JSON --vision-gain-execution packed"
}

require_value() {
    if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'error: %s requires a value\n' "$1" >&2
        exit 2
    fi
}

while (( $# )); do
    case $1 in
        --prefix-conditioning-config) require_value "$@"; prefix_conditioning_config=$2; shift 2 ;;
        --clean-kl-beta) require_value "$@"; clean_kl_beta=$2; shift 2 ;;
        --clean-kl-warmup-ratio) require_value "$@"; clean_kl_warmup_ratio=$2; shift 2 ;;
        --text-kl-beta) require_value "$@"; text_kl_beta=$2; shift 2 ;;
        --text-kl-warmup-ratio) require_value "$@"; text_kl_warmup_ratio=$2; shift 2 ;;
        --vision-gain-alpha) require_value "$@"; gain_alpha=$2; shift 2 ;;
        --preference-mode) require_value "$@"; preference_mode=$2; shift 2 ;;
        --preference-weight) require_value "$@"; preference_weight=$2; shift 2 ;;
        --preference-beta) require_value "$@"; preference_beta=$2; shift 2 ;;
        --vision-gain-min-h-norm-sq) require_value "$@"; gain_min_h_norm_sq=$2; shift 2 ;;
        --vision-gain-execution) require_value "$@"; gain_execution=$2; shift 2 ;;
        --vision-gain-branch-mode) require_value "$@"; gain_branch_mode=$2; shift 2 ;;
        --method) require_value "$@"; method=$2; shift 2 ;;
        --model) require_value "$@"; model=$2; shift 2 ;;
        --data-path) require_value "$@"; data_path=$2; shift 2 ;;
        --source-data-path) require_value "$@"; source_data_path=$2; shift 2 ;;
        --image-root) require_value "$@"; image_root=$2; shift 2 ;;
        --output-dir) require_value "$@"; output_dir=$2; shift 2 ;;
        --task) require_value "$@"; task=$2; shift 2 ;;
        --cache-dir) require_value "$@"; cache_dir=$2; shift 2 ;;
        --max-seq-length) require_value "$@"; max_seq_length=$2; shift 2 ;;
        --micro-batch) require_value "$@"; micro_batch=$2; shift 2 ;;
        --epochs) require_value "$@"; epochs=$2; shift 2 ;;
        --learning-rate) require_value "$@"; learning_rate=$2; shift 2 ;;
        --gradient-accumulation) require_value "$@"; gradient_accumulation=$2; shift 2 ;;
        --seed) require_value "$@"; seed=$2; shift 2 ;;
        --deepspeed-config) require_value "$@"; deepspeed_config=$2; shift 2 ;;
        --attention-implementation) require_value "$@"; attention_implementation=$2; shift 2 ;;
        --ema-setting) require_value "$@"; ema_setting=$2; shift 2 ;;
        --save-strategy) require_value "$@"; save_strategy=$2; shift 2 ;;
        --save-steps) require_value "$@"; save_steps=$2; shift 2 ;;
        --resume) require_value "$@"; resume_path=$2; shift 2 ;;
        --log-file) require_value "$@"; log_file=$2; shift 2 ;;
        --max-steps) require_value "$@"; max_steps=$2; shift 2 ;;
        --benchmark-run-config) require_value "$@"; benchmark_run_config=$2; shift 2 ;;
        --producer-evidence) require_value "$@"; benchmark_evidence_path=$2; shift 2 ;;
        --smoke-resume-witness) require_value "$@"; benchmark_smoke_resume_witness=$2; shift 2 ;;
        --final-save) final_save=true; shift ;;
        --no-final-save) final_save=false; shift ;;
        --use_lora) legacy_use_lora=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in method model data_path image_root output_dir; do
    value=${!required}
    if [[ -z $value ]]; then
        flag=${required//_/-}
        printf 'error: --%s is required\n' "$flag" >&2
        exit 2
    fi
done

case $method in
    full|dora) ;;
    lora) legacy_use_lora=true ;;
    *) printf 'error: --method must be full or dora (lora is legacy-only)\n' >&2; exit 2 ;;
esac
if [[ $legacy_use_lora == true && $method != lora ]]; then
    printf 'error: --use_lora requires legacy --method lora\n' >&2
    exit 2
fi
case $save_strategy in
    no|steps|epoch) ;;
    *) printf 'error: --save-strategy must be no, steps, or epoch\n' >&2; exit 2 ;;
esac
case $attention_implementation in
    eager|sdpa|flash_attention_2) ;;
    *) printf 'error: unsupported attention implementation\n' >&2; exit 2 ;;
esac
case $ema_setting in
    moving|frozen) ;;
    *) printf 'error: --ema-setting must be moving or frozen\n' >&2; exit 2 ;;
esac
for integer_name in micro_batch gradient_accumulation seed save_steps stage1_num_processes; do
    value=${!integer_name}
    if [[ ! $value =~ ^[0-9]+$ ]] || [[ $integer_name != seed && $value -lt 1 ]]; then
        printf 'error: %s must be a valid integer\n' "$integer_name" >&2
        exit 2
    fi
done
if [[ -n $max_steps && ( ! $max_steps =~ ^[0-9]+$ || $max_steps -lt 1 ) ]]; then
    printf 'error: max_steps must be a positive integer\n' >&2
    exit 2
fi
if [[ -n $max_seq_length && ( ! $max_seq_length =~ ^[0-9]+$ || $max_seq_length -lt 1 ) ]]; then
    printf 'error: max_seq_length must be a positive integer\n' >&2
    exit 2
fi

is_positive_finite() {
    local value=$1
    [[ $value =~ ^\+?(([0-9]+(\.[0-9]*)?)|(\.[0-9]+))([eE][+-]?[0-9]+)?$ ]] || return 1
    awk -v value="$value" \
        'BEGIN { number = value + 0; exit !(number > 0 && number <= 1.7976931348623157e308) }'
}

for scalar_name in epochs learning_rate; do
    value=${!scalar_name}
    if ! is_positive_finite "$value"; then
        printf 'error: %s must be a positive finite number\n' "$scalar_name" >&2
        exit 2
    fi
done

experiment_id=${output_dir##*/}
canonical_log_root="$script_dir/logs"
if [[ -z $log_file ]]; then
    if [[ -n $resume_path ]]; then
        log_file="$canonical_log_root/$experiment_id/resume-$(date +%Y%m%dT%H%M%S)-$$.log"
    else
        log_file="$canonical_log_root/$experiment_id/train.log"
    fi
fi

if [[ $dry_run != true ]]; then
    artifact_preflight=(
        "$repository_python" -m src.artifact_paths validate-run
        --checkpoint "$output_dir" --log "$log_file"
    )
    [[ -z $benchmark_run_config ]] || artifact_preflight+=(--benchmark-run-config "$benchmark_run_config")
    [[ -z $benchmark_evidence_path ]] || artifact_preflight+=(--benchmark-evidence-path "$benchmark_evidence_path")
    [[ -z $benchmark_smoke_resume_witness ]] || artifact_preflight+=(--benchmark-smoke-resume-witness "$benchmark_smoke_resume_witness")
    "${artifact_preflight[@]}" >/dev/null
    if [[ $manifest_mode == true ]]; then
        resolved_manifest=$(readlink -f -- "$manifest_path") || {
            printf 'error: manifest path cannot be resolved: %s\n' "$manifest_path" >&2
            exit 2
        }
        expected_manifest=$(readlink -m -- "$output_dir/launch-manifest.json")
        if [[ $resolved_manifest != "$expected_manifest" ]]; then
            printf 'error: live manifest must be the output launch-manifest.json\n' >&2
            exit 2
        fi
    fi
    [[ -f $data_path ]] || { printf 'error: data path is not a file: %s\n' "$data_path" >&2; exit 2; }
    [[ -d $image_root ]] || { printf 'error: image root is not a directory: %s\n' "$image_root" >&2; exit 2; }
    [[ -f $deepspeed_config ]] || { printf 'error: DeepSpeed config is not a file: %s\n' "$deepspeed_config" >&2; exit 2; }
    if [[ -n $resume_path && ! -d $resume_path ]]; then
        printf 'error: resume path is not a directory: %s\n' "$resume_path" >&2
        exit 2
    fi
    "$repository_python" -m src.deepspeed_runtime validate \
        --config "$deepspeed_config" --world-size "$stage1_num_processes" >/dev/null
    artifact_dirs=("$output_dir" "$(dirname -- "$log_file")")
    for result_path in "$benchmark_run_config" "$benchmark_evidence_path" "$benchmark_smoke_resume_witness"; do
        if [[ -n $result_path ]]; then
            artifact_dirs+=("$(dirname -- "$result_path")")
            break
        fi
    done
    if [[ -z $resume_path && ${STAGE1_ARTIFACTS_PRECLAIMED:-false} != true ]]; then
        mkdir -p -- "$(dirname -- "$output_dir")" "$(dirname -- "$(dirname -- "$log_file")")"
        if (( ${#artifact_dirs[@]} == 3 )); then
            mkdir -p -- "$(dirname -- "${artifact_dirs[2]}")"
        fi
        for target_dir in "${artifact_dirs[@]}"; do
            if ! mkdir -- "$target_dir"; then
                printf 'error: fresh run artifact directory already exists: %s\n' "$target_dir" >&2
                exit 2
            fi
        done
    elif [[ -e $log_file ]]; then
        printf 'error: resume log file must be distinct: %s\n' "$log_file" >&2
        exit 2
    elif [[ ${STAGE1_ARTIFACTS_PRECLAIMED:-false} == true ]]; then
        artifact_claim_token=${STAGE1_ARTIFACT_CLAIM_TOKEN:-}
        if [[ ! $artifact_claim_token =~ ^[0-9a-f]{64}$ ]]; then
            printf 'error: preclaimed artifacts require a valid claim token\n' >&2
            exit 2
        fi
        for target_dir in "${artifact_dirs[@]}"; do
            [[ -d $target_dir ]] || {
                printf 'error: preclaimed artifact directory is missing: %s\n' "$target_dir" >&2
                exit 2
            }
            claim_marker="$target_dir/.stage1-artifact-claim"
            if [[ -L $claim_marker || ! -f $claim_marker ]] || \
                [[ $(stat -c '%a' -- "$claim_marker") != 600 ]] || \
                [[ $(<"$claim_marker") != "$artifact_claim_token" ]]; then
                printf 'error: preclaimed artifact claim token is invalid: %s\n' "$target_dir" >&2
                exit 2
            fi
        done
        for target_dir in "${artifact_dirs[@]}"; do
            rm -- "$target_dir/.stage1-artifact-claim"
        done
    else
        mkdir -p -- "$output_dir" "$(dirname -- "$log_file")"
    fi
fi

command=(
    "$repository_accelerate" launch
    --num_processes "$stage1_num_processes"
    --main_process_port 29501
    src/main.py
    --stage stage1
    --model "$model"
    --data_path "$data_path"
    --image_root "$image_root"
    --save_model_path "$output_dir"
    --log_file "$log_file"
    --per_device_train_batch_size "$micro_batch"
    --epochs "$epochs"
    --learning_rate "$learning_rate"
    --gradient_accumulation_steps "$gradient_accumulation"
    --seed "$seed"
    --deepspeed_config "$deepspeed_config"
    --attention_implementation "$attention_implementation"
    --ema_setting "$ema_setting"
    --stage1_vision_gain_alpha "$gain_alpha"
    --stage1_clean_kl_beta "$clean_kl_beta"
    --stage1_clean_kl_warmup_ratio "$clean_kl_warmup_ratio"
    --stage1_text_kl_beta "$text_kl_beta"
    --stage1_text_kl_warmup_ratio "$text_kl_warmup_ratio"
    --stage1_preference_mode "$preference_mode"
    --stage1_preference_weight "$preference_weight"
    --stage1_preference_beta "$preference_beta"
    --stage1_vision_gain_min_h_norm_sq "$gain_min_h_norm_sq"
    --stage1_vision_gain_execution "$gain_execution"
    --stage1_vision_gain_branch_mode "$gain_branch_mode"
    --save_strategy "$save_strategy"
    --save_steps "$save_steps"
    --gradient_checkpointing
)

if [[ $method == dora ]]; then
    command+=(
        --tuning_method dora
        --lora_r 32
        --lora_alpha 64
        --lora_dropout 0.05
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    )
elif [[ $method == full ]]; then
    command+=(--tuning_method full)
else
    command+=(
        --use_lora
        --lora_r 32
        --lora_alpha 64
        --lora_dropout 0.05
        --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
    )
fi

if [[ -n $task ]]; then
    command+=(--task "$task")
fi
if [[ -n $cache_dir ]]; then
    command+=(--cache_dir "$cache_dir")
fi
if [[ -n $max_seq_length ]]; then
    command+=(--max_seq_length_train "$max_seq_length")
fi

if [[ -n $prefix_conditioning_config ]]; then
    command+=(--stage1_prefix_conditioning_config "$prefix_conditioning_config")
fi
if [[ -n $resume_path ]]; then
    command+=(--resume_from_checkpoint_path "$resume_path")
fi
if [[ -n $max_steps ]]; then
    command+=(--max_steps "$max_steps")
fi
if [[ -n $benchmark_run_config || -n $benchmark_evidence_path ]]; then
    if [[ -z $benchmark_run_config || -z $benchmark_evidence_path || -z $source_data_path ]]; then
        printf 'error: benchmark run config, source data, and producer evidence path are required together\n' >&2
        exit 2
    fi
    command+=(
        --benchmark_source_data_path "$source_data_path"
        --benchmark_run_config "$benchmark_run_config"
        --benchmark_evidence_path "$benchmark_evidence_path"
    )
fi
if [[ -n $benchmark_smoke_resume_witness ]]; then
    if [[ -z $benchmark_run_config || -z $benchmark_evidence_path || -z $resume_path ]]; then
        printf 'error: smoke resume witness requires benchmark config, evidence, and resume path\n' >&2
        exit 2
    fi
    command+=(--benchmark_smoke_resume_witness "$benchmark_smoke_resume_witness")
fi
if [[ $final_save == true ]]; then
    command+=(--final_save)
else
    command+=(--no_final_save)
fi

if [[ $dry_run == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$script_dir/transformers/src:$script_dir/src"
export PYTHON_EXEC="$repository_python"
"${command[@]}"
