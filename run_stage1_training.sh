#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$script_dir"

method=${STAGE1_METHOD:-}
model=${STAGE1_MODEL:-}
data_path=${STAGE1_DATA_PATH:-}
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
resume_path=${STAGE1_RESUME_PATH:-}
log_file=${STAGE1_LOG_FILE:-}
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
        --resume) require_value "$@"; resume_path=$2; shift 2 ;;
        --log-file) require_value "$@"; log_file=$2; shift 2 ;;
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
for integer_name in micro_batch gradient_accumulation seed; do
    value=${!integer_name}
    if [[ ! $value =~ ^[0-9]+$ ]] || [[ $integer_name != seed && $value -lt 1 ]]; then
        printf 'error: %s must be a valid integer\n' "$integer_name" >&2
        exit 2
    fi
done

if [[ -d $output_dir ]] && find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    if [[ -z $resume_path ]]; then
        printf 'error: output directory is nonempty; pass --resume explicitly: %s\n' "$output_dir" >&2
        exit 2
    fi
elif [[ -e $output_dir && ! -d $output_dir ]]; then
    printf 'error: output path exists and is not a directory: %s\n' "$output_dir" >&2
    exit 2
fi

if [[ $dry_run != true ]]; then
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
    accelerate launch
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
"${command[@]}"
