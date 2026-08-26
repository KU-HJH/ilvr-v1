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
micro_batch=${STAGE1_MICRO_BATCH:-8}
epochs=${STAGE1_EPOCHS:-3}
learning_rate=${STAGE1_LEARNING_RATE:-1e-5}
gradient_accumulation=${STAGE1_GRADIENT_ACCUMULATION:-2}
seed=${STAGE1_SEED:-42}
deepspeed_config=${STAGE1_DEEPSPEED_CONFIG:-"$script_dir/configs/config_stage3.json"}
attention_implementation=${STAGE1_ATTENTION_IMPLEMENTATION:-flash_attention_2}
save_strategy=${STAGE1_SAVE_STRATEGY:-no}
save_steps=${STAGE1_SAVE_STEPS:-612}
resume_path=${STAGE1_RESUME_PATH:-}
log_file=${STAGE1_LOG_FILE:-}
max_steps=${STAGE1_MAX_STEPS:-}
benchmark_run_config=${STAGE1_BENCHMARK_RUN_CONFIG:-}
benchmark_evidence_path=${STAGE1_BENCHMARK_EVIDENCE_PATH:-}
benchmark_smoke_resume_witness=${STAGE1_BENCHMARK_SMOKE_RESUME_WITNESS:-}
dry_run=false
final_save=true
legacy_use_lora=false

usage() {
    printf '%s\n' \
        "Usage: $0 --method full|dora --model MODEL --data-path FILE" \
        "  --image-root DIR --output-dir DIR [options]"
}

require_value() {
    if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'error: %s requires a value\n' "$1" >&2
        exit 2
    fi
}

while (( $# )); do
    case $1 in
        --method) require_value "$@"; method=$2; shift 2 ;;
        --model) require_value "$@"; model=$2; shift 2 ;;
        --data-path) require_value "$@"; data_path=$2; shift 2 ;;
        --source-data-path) require_value "$@"; source_data_path=$2; shift 2 ;;
        --image-root) require_value "$@"; image_root=$2; shift 2 ;;
        --output-dir) require_value "$@"; output_dir=$2; shift 2 ;;
        --micro-batch) require_value "$@"; micro_batch=$2; shift 2 ;;
        --epochs) require_value "$@"; epochs=$2; shift 2 ;;
        --learning-rate) require_value "$@"; learning_rate=$2; shift 2 ;;
        --gradient-accumulation) require_value "$@"; gradient_accumulation=$2; shift 2 ;;
        --seed) require_value "$@"; seed=$2; shift 2 ;;
        --deepspeed-config) require_value "$@"; deepspeed_config=$2; shift 2 ;;
        --attention-implementation) require_value "$@"; attention_implementation=$2; shift 2 ;;
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
for integer_name in micro_batch gradient_accumulation seed save_steps; do
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

if [[ $dry_run != true ]]; then
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
    if [[ -L $output_dir ]]; then
        printf 'error: output path must not be a symlink: %s\n' "$output_dir" >&2
        exit 2
    fi
    if [[ -d $output_dir ]] && find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
        manifest_only=false
        if [[ $manifest_mode == true ]]; then
            entry_count=$(find "$output_dir" -mindepth 1 -maxdepth 1 -printf '.' | wc -c)
            if [[ $entry_count == 1 && -f $output_dir/launch-manifest.json ]]; then
                manifest_only=true
            fi
        fi
        if [[ -z $resume_path && $manifest_only != true ]]; then
            printf 'error: output directory is nonempty; pass --resume explicitly: %s\n' "$output_dir" >&2
            exit 2
        fi
    elif [[ -e $output_dir && ! -d $output_dir ]]; then
        printf 'error: output path exists and is not a directory: %s\n' "$output_dir" >&2
        exit 2
    fi
    [[ -f $data_path ]] || { printf 'error: data path is not a file: %s\n' "$data_path" >&2; exit 2; }
    [[ -d $image_root ]] || { printf 'error: image root is not a directory: %s\n' "$image_root" >&2; exit 2; }
    [[ -f $deepspeed_config ]] || { printf 'error: DeepSpeed config is not a file: %s\n' "$deepspeed_config" >&2; exit 2; }
    if [[ -n $resume_path && ! -d $resume_path ]]; then
        printf 'error: resume path is not a directory: %s\n' "$resume_path" >&2
        exit 2
    fi
    mkdir -p -- "$output_dir"
fi

if [[ -z $log_file ]]; then
    log_file=$output_dir/train.log
fi

command=(
    "$repository_accelerate" launch
    --num_processes 2
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
export PYTHON_EXEC="$repository_python"
"${command[@]}"
