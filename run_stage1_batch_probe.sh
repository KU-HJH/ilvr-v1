#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$script_dir"

method=
model=
data_path=
image_root=
output_root=
probe_rows=
start=8
max_per_gpu=64
fallback=
descending=false
gradient_accumulation=2
learning_rate=1e-5
seed=42
dora_rank=32
dora_alpha=64
dora_dropout=0.05
deepspeed_config="$script_dir/configs/config_stage3.json"
attention_implementation=flash_attention_2
launcher=${STAGE1_LAUNCHER:-"$script_dir/run_stage1_training.sh"}
benchmark_python=${STAGE1_BENCHMARK_PYTHON:-"$script_dir/venv/ilvr-venv/bin/python"}

usage() {
    printf '%s\n' \
        "Usage: $0 --method full|dora --model MODEL --data-path FILE" \
        "  --image-root DIR --output-root DIR --probe-rows AUDIT [options]"
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
        --output-root) require_value "$@"; output_root=$2; shift 2 ;;
        --probe-rows) require_value "$@"; probe_rows=$2; shift 2 ;;
        --start) require_value "$@"; start=$2; shift 2 ;;
        --max-per-gpu) require_value "$@"; max_per_gpu=$2; shift 2 ;;
        --fallback) require_value "$@"; fallback=$2; shift 2 ;;
        --descending) descending=true; shift ;;
        --gradient-accumulation) require_value "$@"; gradient_accumulation=$2; shift 2 ;;
        --learning-rate) require_value "$@"; learning_rate=$2; shift 2 ;;
        --seed) require_value "$@"; seed=$2; shift 2 ;;
        --dora-rank) require_value "$@"; dora_rank=$2; shift 2 ;;
        --dora-alpha) require_value "$@"; dora_alpha=$2; shift 2 ;;
        --dora-dropout) require_value "$@"; dora_dropout=$2; shift 2 ;;
        --deepspeed-config) require_value "$@"; deepspeed_config=$2; shift 2 ;;
        --attention-implementation) require_value "$@"; attention_implementation=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in method model data_path image_root output_root probe_rows; do
    value=${!required}
    if [[ -z $value ]]; then
        printf 'error: --%s is required\n' "${required//_/-}" >&2
        exit 2
    fi
done
[[ $method == full || $method == dora ]] || { printf 'error: method must be full or dora\n' >&2; exit 2; }
[[ -x $launcher ]] || { printf 'error: launcher is not executable: %s\n' "$launcher" >&2; exit 2; }
[[ -x $benchmark_python ]] || { printf 'error: benchmark Python is not executable: %s\n' "$benchmark_python" >&2; exit 2; }
[[ -f $data_path ]] || { printf 'error: data path is not a file: %s\n' "$data_path" >&2; exit 2; }
[[ -d $image_root ]] || { printf 'error: image root is not a directory: %s\n' "$image_root" >&2; exit 2; }
[[ -f $probe_rows ]] || { printf 'error: probe row audit is not a file: %s\n' "$probe_rows" >&2; exit 2; }
for integer_name in start max_per_gpu gradient_accumulation seed dora_rank dora_alpha; do
    value=${!integer_name}
    if [[ ! $value =~ ^[0-9]+$ ]] || [[ $integer_name != seed && $value -lt 1 ]]; then
        printf 'error: %s must be a valid integer\n' "$integer_name" >&2
        exit 2
    fi
done
if [[ -z $fallback && $max_per_gpu -lt 8 ]]; then
    printf 'error: max_per_gpu must be at least the initial batch 8\n' >&2
    exit 2
fi
if [[ $gradient_accumulation != 2 || $seed != 42 ]]; then
    printf 'error: official probes require accumulation 2 and seed 42\n' >&2
    exit 2
fi
if ! awk -v value="$learning_rate" 'BEGIN { number = value + 0; exit !(number == 0.00001) }'; then
    printf 'error: official probes require learning rate 1e-5\n' >&2
    exit 2
fi
if [[ $attention_implementation != flash_attention_2 ]]; then
    printf 'error: official probes require FlashAttention 2\n' >&2
    exit 2
fi
if [[ $method == dora && ( $dora_rank != 32 || $dora_alpha != 64 ) ]]; then
    printf 'error: official DoRA probes require rank 32, alpha 64, dropout 0.05\n' >&2
    exit 2
fi
if [[ $method == dora ]] && ! awk -v value="$dora_dropout" 'BEGIN { number = value + 0; exit !(number == 0.05) }'; then
    printf 'error: official DoRA probes require rank 32, alpha 64, dropout 0.05\n' >&2
    exit 2
fi

if [[ $descending == true ]]; then
    if [[ -n $fallback && $fallback != "$start" ]]; then
        printf 'error: --fallback and --start disagree\n' >&2
        exit 2
    fi
    fallback=$start
fi

if [[ -n $fallback ]]; then
    case $fallback in 4|2|1) ;; *) printf 'error: fallback must be 4, 2, or 1\n' >&2; exit 2 ;; esac
    if ! "$benchmark_python" -m src.stage1.benchmark require-fallback \
        --probe-dir "$output_root" --method "$method" \
        --candidate "$fallback" >/dev/null; then
        printf 'error: fallback requires every prior candidate to be a confirmed capacity OOM\n' >&2
        exit 2
    fi
    candidates=("$fallback")
else
    if [[ $start != 8 ]]; then
        printf 'error: the primary capacity search must start at 8\n' >&2
        exit 2
    fi
    mapfile -t candidates < <(
        "$benchmark_python" -m src.stage1.benchmark capacity-candidates \
            --start "$start" --max-per-gpu "$max_per_gpu" | tr ' ' '\n'
    )
fi

probe_data="$output_root/probe-input.jsonl"
"$benchmark_python" -m src.stage1.benchmark probe-jsonl \
    --audit "$probe_rows" --data-path "$data_path" \
    --image-root "$image_root" --output "$probe_data" >/dev/null

counter=0
for micro_batch in "${candidates[@]}"; do
    "$benchmark_python" -m src.stage1.benchmark require-candidate \
        --probe-dir "$output_root" --method "$method" \
        --candidate "$micro_batch" >/dev/null
    counter=$((counter + 1))
    attempt_id="$(date +%Y%m%dT%H%M%S%N)-$$-$counter"
    attempt_dir="$output_root/attempts/$method/micro-batch-$micro_batch/attempt-$attempt_id"
    logs_dir="$attempt_dir/logs"
    evidence_dir="$attempt_dir/evidence"
    candidate_output="$attempt_dir/output"
    stdout_path="$logs_dir/stdout.log"
    stderr_path="$logs_dir/stderr.log"
    run_config="$evidence_dir/run-config.json"
    producer_evidence="$evidence_dir/producer-evidence.json"
    mkdir -p -- "$logs_dir" "$evidence_dir"

    "$benchmark_python" -m src.stage1.benchmark prepare-probe-run \
        --run-config "$run_config" --method "$method" --model "$model" \
        --source-data-path "$data_path" --data-path "$probe_data" \
        --image-root "$image_root" --output-dir "$candidate_output" \
        --micro-batch "$micro_batch" --world-size 2 \
        --gradient-accumulation "$gradient_accumulation" --max-steps 3 \
        --deepspeed-config "$deepspeed_config" --probe-rows "$probe_rows" >/dev/null

    launch_command=(
        "$launcher"
        --method "$method"
        --model "$model"
        --data-path "$probe_data"
        --image-root "$image_root"
        --output-dir "$candidate_output"
        --micro-batch "$micro_batch"
        --epochs 3
        --learning-rate "$learning_rate"
        --gradient-accumulation "$gradient_accumulation"
        --seed "$seed"
        --deepspeed-config "$deepspeed_config"
        --attention-implementation "$attention_implementation"
        --save-strategy no
        --log-file "$logs_dir/train.log"
        --max-steps 3
        --no-final-save
        --benchmark-run-config "$run_config"
        --producer-evidence "$producer_evidence"
    )

    set +e
    "${launch_command[@]}" >"$stdout_path" 2>"$stderr_path" &
    child_pid=$!
    wait "$child_pid"
    exit_code=$?
    set -e

    "$benchmark_python" -m src.stage1.benchmark record-attempt \
        --attempt-dir "$attempt_dir" \
        --attempt-id "$attempt_id" \
        --method "$method" \
        --micro-batch "$micro_batch" \
        --world-size 2 \
        --gradient-accumulation "$gradient_accumulation" \
        --learning-rate "$learning_rate" \
        --seed "$seed" \
        --model "$model" \
        --source-data-path "$data_path" \
        --image-root "$image_root" \
        --deepspeed-config "$deepspeed_config" \
        --attention-implementation "$attention_implementation" \
        --process-id "$child_pid" \
        --exit-code "$exit_code" \
        --max-steps 3 \
        --output-dir "$candidate_output" \
        --stdout-path "$stdout_path" \
        --stderr-path "$stderr_path" \
        --probe-rows "$probe_rows" \
        -- "${launch_command[@]}" >/dev/null

    if (( exit_code != 0 )); then
        printf 'requires-classification: method=%s micro_batch=%s attempt=%s\n' \
            "$method" "$micro_batch" "$attempt_dir" >&2
        exit "$exit_code"
    fi
done
