#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$script_dir"
repository_python="$script_dir/venv/ilvr-venv/bin/python"
repository_accelerate="$script_dir/venv/ilvr-venv/bin/accelerate"
if [[ ${ILVR_STAGE1_RUNTIME_PROFILE:-default} == native-ema-0180 ]]; then
    native_stage1_venv=/root/workspace/ILVR/.worktrees/stage1-visual-history-only-20261019/venv/ilvr-venv
    repository_python="$native_stage1_venv/bin/python"
    repository_accelerate="$native_stage1_venv/bin/accelerate"
fi
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
micro_batch=${STAGE1_MICRO_BATCH:-1}
epochs=${STAGE1_EPOCHS:-1}
learning_rate=${STAGE1_LEARNING_RATE:-5e-5}
ce_weight=${STAGE1_CE_WEIGHT:-${CE_WEIGHT:-}}
gradient_accumulation=${STAGE1_GRADIENT_ACCUMULATION:-2}
ema_tau=${STAGE1_EMA_TAU:-1.0}
seed=${STAGE1_SEED:-42}
deepspeed_config=${STAGE1_DEEPSPEED_CONFIG:-"$script_dir/configs/deepspeed_zero3.json"}
stage1_num_processes=${STAGE1_NUM_PROCESSES:-2}
attention_implementation=${STAGE1_ATTENTION_IMPLEMENTATION:-flash_attention_2}
save_strategy=${STAGE1_SAVE_STRATEGY:-no}
save_steps=${STAGE1_SAVE_STEPS:-612}
textual_states=${STAGE1_TEXTUAL_STATES:-previous}
resume_path=${STAGE1_RESUME_PATH:-}
log_file=${STAGE1_LOG_FILE:-}
max_steps=${STAGE1_MAX_STEPS:-}
benchmark_run_config=${STAGE1_BENCHMARK_RUN_CONFIG:-}
benchmark_evidence_path=${STAGE1_BENCHMARK_EVIDENCE_PATH:-}
benchmark_smoke_resume_witness=${STAGE1_BENCHMARK_SMOKE_RESUME_WITNESS:-}
training_evidence_path=
extra_worker_arguments=()
query_mode=legacy
declare -A requested_options=()
dry_run=false
final_save=true
legacy_use_lora=false

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
if [[ $ilvr_multinode == true ]]; then
    for topology_variable in "${ilvr_topology_variables[@]}"; do
        if [[ ! -v $topology_variable || -z ${!topology_variable} ]]; then
            topology_label=$topology_variable
            if [[ $topology_variable == ILVR_SOCKET_IFNAME ]]; then
                topology_label=interface
            fi
            printf 'error: all distributed variables must be explicitly set and non-empty: %s\n' "$topology_label" >&2
            exit 2
        fi
    done
    ilvr_num_machines=$ILVR_NUM_MACHINES
    ilvr_num_processes=$ILVR_NUM_PROCESSES
    ilvr_machine_rank=$ILVR_MACHINE_RANK
    ilvr_main_process_ip=$ILVR_MAIN_PROCESS_IP
    ilvr_main_process_port=$ILVR_MAIN_PROCESS_PORT
    ilvr_socket_ifname=$ILVR_SOCKET_IFNAME
    [[ $ilvr_num_machines == 2 ]] || { printf 'error: ILVR_NUM_MACHINES must be exactly 2\n' >&2; exit 2; }
    [[ $ilvr_num_processes == 4 ]] || { printf 'error: ILVR_NUM_PROCESSES must be exactly 4\n' >&2; exit 2; }
    [[ $ilvr_machine_rank == 0 || $ilvr_machine_rank == 1 ]] || { printf 'error: ILVR_MACHINE_RANK must be a machine rank of 0 or 1\n' >&2; exit 2; }
    [[ $ilvr_main_process_ip == 172.16.11.41 ]] || { printf 'error: ILVR_MAIN_PROCESS_IP must be 172.16.11.41\n' >&2; exit 2; }
    [[ $ilvr_socket_ifname == bond0.11 ]] || { printf 'error: ILVR_SOCKET_IFNAME must be interface bond0.11\n' >&2; exit 2; }
    if [[ ! $ilvr_main_process_port =~ ^[0-9]+$ ]] || \
        (( 10#$ilvr_main_process_port < 1024 || 10#$ilvr_main_process_port > 65535 )); then
        printf 'error: ILVR_MAIN_PROCESS_PORT must be an integer between 1024 and 65535\n' >&2
        exit 2
    fi
    if [[ -v STAGE1_NUM_PROCESSES && $STAGE1_NUM_PROCESSES != "$ilvr_num_processes" ]]; then
        printf 'error: STAGE1_NUM_PROCESSES must match ILVR_NUM_PROCESSES\n' >&2
        exit 2
    fi
else
    ilvr_num_machines=1
    ilvr_num_processes=$stage1_num_processes
    ilvr_machine_rank=0
    ilvr_main_process_ip=
    ilvr_main_process_port=29501
    ilvr_socket_ifname=
fi

accelerate_topology=(
    --num_processes "$ilvr_num_processes"
    --main_process_port "$ilvr_main_process_port"
)
if [[ $ilvr_num_machines == 2 ]]; then
    accelerate_topology+=(
        --num_machines 2
        --machine_rank "$ilvr_machine_rank"
        --main_process_ip "$ilvr_main_process_ip"
    )
    export MASTER_ADDR="$ilvr_main_process_ip"
    export GLOO_SOCKET_IFNAME="$ilvr_socket_ifname"
    export NCCL_SOCKET_IFNAME="$ilvr_socket_ifname"
fi

usage() {
    printf '%s\n' \
        "Usage: $0 --method full|lora|dora --model MODEL --data-path FILE" \
        "  --image-root DIR --output-dir DIR [options]"
}

require_value() {
    if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'error: %s requires a value\n' "$1" >&2
        exit 2
    fi
}

# Keep environment knobs as explicit requests; unset/empty values are fallbacks.
for binding in STAGE1_MICRO_BATCH:--micro-batch STAGE1_EPOCHS:--epochs STAGE1_LEARNING_RATE:--learning-rate \
    STAGE1_CE_WEIGHT:--ce-weight CE_WEIGHT:--ce-weight STAGE1_GRADIENT_ACCUMULATION:--gradient-accumulation \
    STAGE1_EMA_TAU:--ema-tau STAGE1_SEED:--seed STAGE1_ATTENTION_IMPLEMENTATION:--attention-implementation \
    STAGE1_SAVE_STRATEGY:--save-strategy STAGE1_SAVE_STEPS:--save-steps STAGE1_TEXTUAL_STATES:--textual_states \
    STAGE1_MAX_STEPS:--max-steps; do
    env_name=${binding%%:*}
    if [[ -v $env_name && -n ${!env_name} ]]; then requested_options[${binding#*:}]=1; fi
done

while (( $# )); do
    requested_options[$1]=1
    case $1 in
        --query_mode) require_value "$@"; query_mode=$2; extra_worker_arguments+=(--query_mode "$2"); shift 2 ;;
        --history_text_gamma|--history_attention_layers) require_value "$@"; extra_worker_arguments+=("$1" "$2"); shift 2 ;;
        --sim-weight) require_value "$@"; extra_worker_arguments+=(--sim_weight "$2"); shift 2 ;;
        --stage1-match-v5-execution) extra_worker_arguments+=(--stage1_match_v5_execution); shift ;;
        --stop-after-checkpoint) require_value "$@"; extra_worker_arguments+=(--stage1_stop_after_checkpoint "$2"); shift 2 ;;
        --stage1-ce-entropy-top-percent) extra_worker_arguments+=(--stage1_ce_entropy_top_percent "$2"); shift 2 ;;
        --stage1-v5-debug) extra_worker_arguments+=(--stage1_v5_debug); shift ;;
        --no-stage1-v5-debug) extra_worker_arguments+=(--no-stage1_v5_debug); shift ;;
        --stage1-variant) require_value "$@"; extra_worker_arguments+=(--stage1_variant "$2"); shift 2 ;;
        --stage1-cache) require_value "$@"; extra_worker_arguments+=(--stage1_cache "$2"); shift 2 ;;
        --stage1-cache-completion) require_value "$@"; extra_worker_arguments+=(--stage1_cache_completion "$2"); shift 2 ;;
        --stage1-runtime-profile) require_value "$@"; extra_worker_arguments+=(--stage1_runtime_profile "$2"); shift 2 ;;
        --stage1-cache-validation) require_value "$@"; extra_worker_arguments+=(--stage1_cache_validation "$2"); shift 2 ;;
        --stage1-cache-validation-stop) require_value "$@"; extra_worker_arguments+=(--stage1_cache_validation_stop "$2"); shift 2 ;;
        --stage1-check-policy) require_value "$@"; extra_worker_arguments+=(--stage1_check_policy "$2"); shift 2 ;;
        --stage1-loss-type) require_value "$@"; extra_worker_arguments+=(--stage1_loss_type "$2"); shift 2 ;;
        --lambda-preference) require_value "$@"; extra_worker_arguments+=(--lambda_preference "$2"); shift 2 ;;
        --lambda-topk-entropy) require_value "$@"; extra_worker_arguments+=(--lambda_topk_entropy "$2"); shift 2 ;;
        --lambda-entropy) require_value "$@"; extra_worker_arguments+=(--lambda_entropy "$2"); shift 2 ;;
        --preference-topk-k) require_value "$@"; extra_worker_arguments+=(--preference_topk_k "$2"); shift 2 ;;
        --preference-topk-mode) require_value "$@"; extra_worker_arguments+=(--preference_topk_mode "$2"); shift 2 ;;
        --lambda-topk-mass) require_value "$@"; extra_worker_arguments+=(--lambda_topk_mass "$2"); shift 2 ;;
        --lambda-tail-kl) require_value "$@"; extra_worker_arguments+=(--lambda_tail_kl "$2"); shift 2 ;;
        --tail-kl-temperature) require_value "$@"; extra_worker_arguments+=(--tail_kl_temperature "$2"); shift 2 ;;
        --lambda-topk-chosen-mass) require_value "$@"; extra_worker_arguments+=(--lambda_topk_chosen_mass "$2"); shift 2 ;;
        --preference-entropy-top-percent) require_value "$@"; extra_worker_arguments+=(--preference_entropy_top_percent "$2"); shift 2 ;;
        --stage1-v5-selection) require_value "$@"; extra_worker_arguments+=(--stage1_v5_selection "$2"); shift 2 ;;
        --stage1-v5-text-kl-weight) require_value "$@"; extra_worker_arguments+=(--stage1_v5_text_kl_weight "$2"); shift 2 ;;
        --stage1-v5-voff-mode) require_value "$@"; extra_worker_arguments+=(--stage1_v5_voff_mode "$2"); shift 2 ;;
        --loss-chunk-positions) require_value "$@"; extra_worker_arguments+=(--loss_chunk_positions "$2"); shift 2 ;;
        --vision-attention-implementation) require_value "$@"; extra_worker_arguments+=(--vision_attention_implementation "$2"); shift 2 ;;
        --warm-up-steps) require_value "$@"; extra_worker_arguments+=(--warm_up_steps "$2"); shift 2 ;;
        --latent-size) require_value "$@"; extra_worker_arguments+=(--latent_size "$2"); shift 2 ;;
        --max-seq-length-train) require_value "$@"; extra_worker_arguments+=(--max_seq_length_train "$2"); shift 2 ;;
        --dataloader-num-workers) require_value "$@"; extra_worker_arguments+=(--dataloader_num_workers "$2"); shift 2 ;;
        --step-preference-beta) require_value "$@"; extra_worker_arguments+=(--step_preference_beta "$2"); shift 2 ;;
        --step-preference-margin) require_value "$@"; extra_worker_arguments+=(--step_preference_margin "$2"); shift 2 ;;
        --topk-include-gt) extra_worker_arguments+=(--topk_include_gt); shift ;;
        --no-topk-include-gt) extra_worker_arguments+=(--no-topk_include_gt); shift ;;
        --method) require_value "$@"; method=$2; shift 2 ;;
        --model) require_value "$@"; model=$2; shift 2 ;;
        --data-path) require_value "$@"; data_path=$2; shift 2 ;;
        --source-data-path) require_value "$@"; source_data_path=$2; shift 2 ;;
        --image-root) require_value "$@"; image_root=$2; shift 2 ;;
        --output-dir) require_value "$@"; output_dir=$2; shift 2 ;;
        --micro-batch) require_value "$@"; micro_batch=$2; shift 2 ;;
        --epochs) require_value "$@"; epochs=$2; shift 2 ;;
        --learning-rate) require_value "$@"; learning_rate=$2; shift 2 ;;
        --ce-weight) require_value "$@"; ce_weight=$2; shift 2 ;;
        --gradient-accumulation) require_value "$@"; gradient_accumulation=$2; shift 2 ;;
        --ema-tau) require_value "$@"; ema_tau=$2; shift 2 ;;
        --seed) require_value "$@"; seed=$2; shift 2 ;;
        --deepspeed-config) require_value "$@"; deepspeed_config=$2; shift 2 ;;
        --attention-implementation) require_value "$@"; attention_implementation=$2; shift 2 ;;
        --save-strategy) require_value "$@"; save_strategy=$2; shift 2 ;;
        --save-steps) require_value "$@"; save_steps=$2; shift 2 ;;
        --save-total-limit)
            require_value "$@"
            [[ $2 =~ ^[1-9][0-9]*$ ]] || { printf 'error: save_total_limit must be a positive integer\n' >&2; exit 2; }
            extra_worker_arguments+=(--save_total_limit "$2"); shift 2 ;;
        --textual_states) require_value "$@"; textual_states=$2; shift 2 ;;
        --resume) require_value "$@"; resume_path=$2; shift 2 ;;
        --log-file) require_value "$@"; log_file=$2; shift 2 ;;
        --max-steps) require_value "$@"; max_steps=$2; shift 2 ;;
        --benchmark-run-config) require_value "$@"; benchmark_run_config=$2; shift 2 ;;
        --producer-evidence) require_value "$@"; benchmark_evidence_path=$2; shift 2 ;;
        --smoke-resume-witness) require_value "$@"; benchmark_smoke_resume_witness=$2; shift 2 ;;
        --training-evidence) require_value "$@"; training_evidence_path=$2; shift 2 ;;
        --final-save) final_save=true; shift ;;
        --no-final-save) final_save=false; shift ;;
        --use_lora) legacy_use_lora=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

v5_resume=false
if [[ -n $resume_path && -f $resume_path/stage1_v5.json ]]; then v5_resume=true; fi

for required in method model data_path image_root output_dir; do
    value=${!required}
    if [[ -z $value ]]; then
        flag=${required//_/-}
        printf 'error: --%s is required\n' "$flag" >&2
        exit 2
    fi
done

case $method in
    full|lora|dora) ;;
    *) printf 'error: --method must be full, lora, or dora\n' >&2; exit 2 ;;
esac
if [[ $legacy_use_lora == true && $method != lora ]]; then
    printf 'error: --use_lora requires legacy --method lora\n' >&2
    exit 2
fi
case $save_strategy in
    no|steps|epoch) ;;
    *) printf 'error: --save-strategy must be no, steps, or epoch\n' >&2; exit 2 ;;
esac
case $textual_states in
    all|previous) ;;
    *) printf 'error: --textual_states must be all or previous\n' >&2; exit 2 ;;
esac
case $attention_implementation in
    eager|sdpa|flash_attention_2) ;;
    *) printf 'error: unsupported attention implementation\n' >&2; exit 2 ;;
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

if [[ ! $ema_tau =~ ^\+?(([0-9]+(\.[0-9]*)?)|(\.[0-9]+))([eE][+-]?[0-9]+)?$ ]] || \
    ! awk -v value="$ema_tau" \
        'BEGIN { number = value + 0; exit !(number >= 0 && number <= 1) }'; then
    printf 'error: ema_tau must be a finite number in [0, 1]\n' >&2
    exit 2
fi

experiment_id=${output_dir##*/}
canonical_log_root="$script_dir/logs/zero-stage-compatibility"
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
    [[ -z $training_evidence_path ]] || artifact_preflight+=(--training-evidence-path "$training_evidence_path")
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
        --config "$deepspeed_config" --world-size "$ilvr_num_processes" >/dev/null
    artifact_dirs=("$output_dir" "$(dirname -- "$log_file")")
    for result_path in "$benchmark_run_config" "$benchmark_evidence_path" "$benchmark_smoke_resume_witness" "$training_evidence_path"; do
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

accelerate_deepspeed=()
if [[ $ilvr_num_machines == 2 ]]; then
    accelerate_deepspeed=(
        --use_deepspeed
        --deepspeed_multinode_launcher standard
        --deepspeed_config_file "$deepspeed_config"
    )
fi

command=(
    "$repository_accelerate" launch
    "${accelerate_topology[@]}"
    "${accelerate_deepspeed[@]}"
    src/main.py
    --stage stage1
    --model "$model"
    --data_path "$data_path"
    --image_root "$image_root"
    --save_model_path "$output_dir"
    --log_file "$log_file"
    --deepspeed_config "$deepspeed_config"
)

append_execution_option() {
    local shell_option=$1 worker_option=$2 value=$3
    if [[ $v5_resume != true || -v requested_options[$shell_option] ]]; then
        command+=("$worker_option" "$value")
    fi
}
append_execution_option --micro-batch --per_device_train_batch_size "$micro_batch"
append_execution_option --epochs --epochs "$epochs"
append_execution_option --learning-rate --learning_rate "$learning_rate"
append_execution_option --gradient-accumulation --gradient_accumulation_steps "$gradient_accumulation"
append_execution_option --ema-tau --ema_tau "$ema_tau"
append_execution_option --seed --seed "$seed"
append_execution_option --attention-implementation --attention_implementation "$attention_implementation"
append_execution_option --save-strategy --save_strategy "$save_strategy"
append_execution_option --save-steps --save_steps "$save_steps"
if [[ $query_mode != adaptive || -v requested_options[--textual_states] ]]; then
    append_execution_option --textual_states --textual_states "$textual_states"
fi
if [[ $v5_resume != true ]]; then command+=(--gradient_checkpointing); fi

command+=(--tuning_method "$method" "${extra_worker_arguments[@]}")
if [[ -n $ce_weight ]]; then
    append_execution_option --ce-weight --ce_weight "$ce_weight"
fi

if [[ $v5_resume != true && ( $method == lora || $method == dora ) ]]; then
    command+=(
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
    append_execution_option --max-steps --max_steps "$max_steps"
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
if [[ -n $training_evidence_path ]]; then
    if [[ -n $benchmark_run_config || -n $benchmark_evidence_path ]]; then
        printf 'error: ordinary training evidence cannot be combined with benchmark evidence\n' >&2
        exit 2
    fi
    command+=(--training_evidence_path "$training_evidence_path")
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
