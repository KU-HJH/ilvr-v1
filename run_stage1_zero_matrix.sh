#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd -- "$script_dir"
repository_python="$script_dir/venv/ilvr-venv/bin/python"
launcher=${STAGE1_LAUNCHER:-"$script_dir/run_stage1_training.sh"}

canonical_checkpoint_root='/root/workspace/ILVR/checkpoints'
canonical_result_root='/root/workspace/ILVR/results'
canonical_log_root='/root/workspace/ILVR/logs/zero-stage-compatibility'
matrix_result_dir="$canonical_result_root/stage1-zero-matrix"

dry_run=false
model=
source_data_path=
probe_data=
image_root=
sample_audits=()
prerequisite_chain=${STAGE1_SMOKE_PREREQUISITE:-"$script_dir/outputs/stage1-smoke/prerequisite-chain.json"}
seed=42

usage() {
    printf '%s\n' \
        "Usage: $0 --model MODEL --source-data-path FILE --data-path FILE" \
        "  --image-root DIR --sample-audit FILE [--sample-audit LEGACY] [--dry-run]"
}

require_value() {
    if (( $# < 2 )) || [[ -z $2 ]]; then
        printf 'error: %s requires a value\n' "$1" >&2
        exit 2
    fi
}

while (( $# )); do
    case $1 in
        --dry-run) dry_run=true; shift ;;
        --model) require_value "$@"; model=$2; shift 2 ;;
        --source-data-path) require_value "$@"; source_data_path=$2; shift 2 ;;
        --data-path) require_value "$@"; probe_data=$2; shift 2 ;;
        --image-root) require_value "$@"; image_root=$2; shift 2 ;;
        --sample-audit) require_value "$@"; sample_audits+=("$2"); shift 2 ;;
        --prerequisite-chain) require_value "$@"; prerequisite_chain=$2; shift 2 ;;
        --matrix-result-dir) require_value "$@"; matrix_result_dir=$2; shift 2 ;;
        --seed) require_value "$@"; seed=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

for required in model source_data_path probe_data image_root; do
    if [[ -z ${!required} ]]; then
        printf 'error: --%s is required\n' "${required//_/-}" >&2
        exit 2
    fi
done
if (( ${#sample_audits[@]} == 0 )); then
    printf 'error: --sample-audit is required\n' >&2
    exit 2
fi
if [[ ! $seed =~ ^[0-9]+$ ]]; then
    printf 'error: seed must be a non-negative integer\n' >&2
    exit 2
fi
if [[ $(readlink -m -- "$matrix_result_dir") != "$canonical_result_root/stage1-zero-matrix" ]]; then
    printf 'error: matrix result directory must be the canonical Stage 1 matrix directory\n' >&2
    exit 2
fi

stages=(3 2 1)
methods=(full dora)
micro_batches=(2 4 8 1)
accumulations=(4 2 1 8)
counter=0

next_experiment_id() {
    local zero_stage=$1 method=$2 micro_batch=$3 accumulation=$4 run_kind=$5
    counter=$((counter + 1))
    local suffix
    if [[ $dry_run == true ]]; then
        suffix="dryrun-$$-$counter"
    else
        suffix="$(date -u +%Y%m%dt%H%M%S%N)-$$-$counter"
    fi
    printf 'stage1-zero%s-%s-mb%s-ga%s-%s-%s\n' \
        "$zero_stage" "$method" "$micro_batch" "$accumulation" "$run_kind" "$suffix"
}

emit_plan() {
    local zero_stage=$1 method=$2 micro_batch=$3 accumulation=$4 run_kind=$5
    local experiment_id checkpoint_dir result_dir log_dir max_steps final_save
    local resume_from_kind save_strategy save_steps checkpoint_path resume_from
    experiment_id=$(next_experiment_id \
        "$zero_stage" "$method" "$micro_batch" "$accumulation" "$run_kind")
    checkpoint_dir="$canonical_checkpoint_root/$experiment_id"
    result_dir="$canonical_result_root/$experiment_id"
    log_dir="$canonical_log_root/$experiment_id"
    max_steps=3
    final_save=false
    resume_from_kind=null
    save_strategy=no
    save_steps=null
    checkpoint_path=null
    resume_from=null
    if [[ $run_kind == selected-save ]]; then
        save_strategy=steps
        save_steps=3
        checkpoint_path="\"$checkpoint_dir/checkpoint-3\""
        planned_selected_checkpoint="$checkpoint_dir/checkpoint-3"
    elif [[ $run_kind == selected-resume ]]; then
        max_steps=4
        save_strategy=steps
        save_steps=1
        resume_from_kind='"selected-save"'
        resume_from="\"$planned_selected_checkpoint\""
        checkpoint_path="\"$checkpoint_dir/checkpoint-4\""
    fi
    printf '{"zero_stage":%s,"method":"%s","micro_batch":%s,"gradient_accumulation":%s,"run_kind":"%s","max_steps":%s,"save_strategy":"%s","save_steps":%s,"final_save":%s,"resume_from_kind":%s,"resume_from":%s,"checkpoint_path":%s,"experiment_id":"%s","checkpoint_dir":"%s","result_dir":"%s","log_dir":"%s"}\n' \
        "$zero_stage" "$method" "$micro_batch" "$accumulation" "$run_kind" \
        "$max_steps" "$save_strategy" "$save_steps" "$final_save" \
        "$resume_from_kind" "$resume_from" "$checkpoint_path" \
        "$experiment_id" "$checkpoint_dir" "$result_dir" "$log_dir"
}

if [[ $dry_run == true ]]; then
    for zero_stage in "${stages[@]}"; do
        for method in "${methods[@]}"; do
            for index in "${!micro_batches[@]}"; do
                emit_plan "$zero_stage" "$method" \
                    "${micro_batches[$index]}" "${accumulations[$index]}" candidate
            done
            # Selection is data-dependent.  The smallest official row is used only as
            # a shape placeholder in dry-run output; no selection or GPU work occurs.
            emit_plan "$zero_stage" "$method" 1 8 selected-save
            emit_plan "$zero_stage" "$method" 1 8 selected-resume
        done
    done
    exit 0
fi

[[ -x $repository_python ]] || { printf 'error: repository Python is not executable\n' >&2; exit 2; }
[[ -x $launcher ]] || { printf 'error: launcher is not executable: %s\n' "$launcher" >&2; exit 2; }
[[ -f $source_data_path ]] || { printf 'error: source data path is not a file: %s\n' "$source_data_path" >&2; exit 2; }
[[ -f $probe_data ]] || { printf 'error: probe data path is not a file: %s\n' "$probe_data" >&2; exit 2; }
[[ -d $image_root ]] || { printf 'error: image root is not a directory: %s\n' "$image_root" >&2; exit 2; }
[[ -f $prerequisite_chain ]] || { printf 'error: prerequisite chain is not a file: %s\n' "$prerequisite_chain" >&2; exit 2; }

audit_command=(
    "$repository_python" -m src.stage1.zero_matrix audit-status
    --data-path "$source_data_path" --image-root "$image_root" --count 64
)
for candidate_audit in "${sample_audits[@]}"; do
    audit_command+=(--candidate-audit "$candidate_audit")
done
audit_status=$("${audit_command[@]}")
sample_audit=$("$repository_python" -c \
    'import json,sys; print(json.loads(sys.argv[1])["path"])' "$audit_status")

mkdir -p -- "$canonical_checkpoint_root" "$canonical_result_root" \
    "$canonical_log_root" "$matrix_result_dir"

# One full-matrix status call loads each canonical runtime profile once and passes
# the resulting immutable objects through all methods and candidate rows.
reuse_status=$("$repository_python" -m src.stage1.zero_matrix reuse-status \
    --matrix-result-dir "$matrix_result_dir" --method all --model "$model" \
    --seed "$seed" --world-size 2 --sample-audit "$sample_audit" \
    --deepspeed-config configs/deepspeed_zero3.json \
    --deepspeed-config configs/deepspeed_zero2.json \
    --deepspeed-config configs/deepspeed_zero1.json)

reuse_field() {
    local zero_stage=$1 method=$2 micro_batch=$3 field=$4
    "$repository_python" -c '
import json, sys
rows = json.loads(sys.argv[1])
stage, method, micro, field = int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5]
row = next(item for item in rows if item["identity"]["zero_stage"] == stage and item["identity"]["method"] == method and item["identity"]["micro_batch"] == micro)
value = row[field]
print("" if value is None else str(value).lower() if isinstance(value, bool) else value)
' "$reuse_status" "$zero_stage" "$method" "$micro_batch" "$field"
}

profile_digest() {
    local zero_stage=$1
    "$repository_python" -c '
import json, sys
rows = json.loads(sys.argv[1])
stage = int(sys.argv[2])
print(next(item for item in rows if item["identity"]["zero_stage"] == stage)["identity"]["deepspeed_config_digest"])
' "$reuse_status" "$zero_stage"
}

claim_run_directories() {
    local experiment_id=$1
    checkpoint_dir="$canonical_checkpoint_root/$experiment_id"
    result_dir="$canonical_result_root/$experiment_id"
    log_dir="$canonical_log_root/$experiment_id"
    "$repository_python" -m src.artifact_paths resolve \
        --experiment-id "$experiment_id" >/dev/null
    mkdir -- "$checkpoint_dir" "$result_dir" "$log_dir"
    artifact_claim_token=$("$repository_python" -c 'import secrets; print(secrets.token_hex(32))')
    for target_dir in "$checkpoint_dir" "$result_dir" "$log_dir"; do
        (
            umask 077
            set -C
            printf '%s\n' "$artifact_claim_token" > "$target_dir/.stage1-artifact-claim"
        ) || {
            printf 'error: cannot create preclaimed artifact marker: %s\n' "$target_dir" >&2
            exit 2
        }
    done
}

launch_candidate() {
    local zero_stage=$1 method=$2 micro_batch=$3 accumulation=$4 run_kind=$5
    local resume_path=${6:-}
    local experiment_id run_config producer_evidence stdout_path stderr_path
    local exit_code max_steps save_strategy save_steps
    experiment_id=$(next_experiment_id \
        "$zero_stage" "$method" "$micro_batch" "$accumulation" "$run_kind")
    claim_run_directories "$experiment_id"
    run_config="$result_dir/run-config.json"
    producer_evidence="$result_dir/producer-evidence.json"
    stdout_path="$log_dir/stdout.log"
    stderr_path="$log_dir/stderr.log"
    max_steps=3
    save_strategy=no
    save_steps=
    selected_checkpoint_path=
    selected_initial_step=0
    selected_final_step=3
    if [[ $run_kind == selected-save ]]; then
        save_strategy=steps
        save_steps=3
        selected_checkpoint_path="$checkpoint_dir/checkpoint-3"
    fi
    if [[ $run_kind == selected-resume ]]; then
        max_steps=4
        save_strategy=steps
        save_steps=1
        selected_checkpoint_path="$checkpoint_dir/checkpoint-4"
        selected_initial_step=3
        selected_final_step=4
    fi

    if [[ $run_kind == candidate ]]; then
        deepspeed_config_digest=$(profile_digest "$zero_stage")
        "$repository_python" -m src.stage1.zero_matrix prepare \
            --run-config "$run_config" --method "$method" --model "$model" \
            --source-data-path "$source_data_path" --data-path "$probe_data" \
            --image-root "$image_root" --output-dir "$checkpoint_dir" \
            --micro-batch "$micro_batch" --world-size 2 --seed "$seed" \
            --deepspeed-config "configs/deepspeed_zero${zero_stage}.json" \
            --deepspeed-config-digest "$deepspeed_config_digest" \
            --zero-stage "$zero_stage" \
            --sample-audit "$sample_audit" \
            --prerequisite-chain "$prerequisite_chain" >/dev/null
    fi

    launch_command=(
        "$launcher" --method "$method" --model "$model"
        --source-data-path "$source_data_path" --data-path "$probe_data"
        --image-root "$image_root"
        --deepspeed-config "configs/deepspeed_zero${zero_stage}.json"
        --micro-batch "$micro_batch" --gradient-accumulation "$accumulation"
        --max-steps "$max_steps" --save-strategy "$save_strategy"
        --no-final-save --output-dir "$checkpoint_dir"
        --log-file "$log_dir/train.log"
    )
    if [[ -n $save_steps ]]; then
        launch_command+=(--save-steps "$save_steps")
    fi
    if [[ $run_kind == candidate ]]; then
        launch_command+=(
            --benchmark-run-config "$run_config"
            --producer-evidence "$producer_evidence"
        )
    fi
    if [[ -n $resume_path ]]; then
        launch_command+=(--resume "$resume_path")
    fi

    set +e
    STAGE1_ARTIFACTS_PRECLAIMED=true STAGE1_ARTIFACT_CLAIM_TOKEN="$artifact_claim_token" \
        "${launch_command[@]}" >"$stdout_path" 2>"$stderr_path"
    exit_code=$?
    set -e

    if [[ $run_kind == candidate ]]; then
        "$repository_python" -m src.stage1.zero_matrix classify \
            --returncode "$exit_code" --stderr-file "$stderr_path" \
            --run-config "$run_config" --producer-evidence "$producer_evidence" \
            --matrix-result-dir "$matrix_result_dir"
    else
        rm -- "$result_dir/.stage1-artifact-claim"
        if (( exit_code != 0 )); then
            printf 'error: %s launch failed: zero=%s method=%s\n' \
                "$run_kind" "$zero_stage" "$method" >&2
            return "$exit_code"
        fi
    fi
}

for zero_stage in "${stages[@]}"; do
    for method in "${methods[@]}"; do
        for index in "${!micro_batches[@]}"; do
            micro_batch=${micro_batches[$index]}
            accumulation=${accumulations[$index]}
            if [[ $(reuse_field "$zero_stage" "$method" "$micro_batch" reuse) == true ]]; then
                reused_status=$(reuse_field "$zero_stage" "$method" "$micro_batch" status)
                if [[ $reused_status == integration_failure ]]; then
                    printf 'error: refusing to continue after reusable integration failure: zero=%s method=%s batch=%s\n' \
                        "$zero_stage" "$method" "$micro_batch" >&2
                    exit 1
                fi
                continue
            fi
            classification=$(launch_candidate "$zero_stage" "$method" \
                "$micro_batch" "$accumulation" candidate)
            status=$("$repository_python" -c \
                'import json,sys; print(json.loads(sys.argv[1])["status"])' "$classification")
            if [[ $status == integration_failure ]]; then
                printf 'error: matrix producer integration failure: zero=%s method=%s batch=%s\n' \
                    "$zero_stage" "$method" "$micro_batch" >&2
                exit 1
            fi
        done

        report=$("$repository_python" -m src.stage1.zero_matrix select \
            --matrix-result-dir "$matrix_result_dir" --method "$method" \
            --model "$model" --seed "$seed" --world-size 2 \
            --deepspeed-config "configs/deepspeed_zero${zero_stage}.json" \
            --deepspeed-config-digest "$(profile_digest "$zero_stage")" \
            --zero-stage "$zero_stage" \
            --sample-audit "$sample_audit")
        selected_micro=$("$repository_python" -c \
            'import json,sys; print(json.loads(sys.argv[1])["selected"]["identity"]["micro_batch"])' "$report")
        selected_identity=$("$repository_python" -c \
            'import json,sys; print(json.dumps(json.loads(sys.argv[1])["selected"]["identity"], separators=(",", ":")))' "$report")
        selected_accumulation=$((8 / selected_micro))
        selected_registry_dir="$matrix_result_dir/selected"
        mkdir -p -- "$selected_registry_dir"
        selected_save_registry="$selected_registry_dir/zero${zero_stage}-${method}-selected-save.json"
        selected_save_status=$("$repository_python" -m src.stage1.zero_matrix selected-status \
            --witness "$selected_save_registry" --run-kind selected-save \
            --identity-json "$selected_identity")
        if [[ $("$repository_python" -c \
            'import json,sys; print(str(json.loads(sys.argv[1])["reuse"]).lower())' \
            "$selected_save_status") == true ]]; then
            selected_checkpoint=$("$repository_python" -c \
                'import json,sys; print(json.loads(sys.argv[1])["witness"]["checkpoint_path"])' \
                "$selected_save_status")
        else
            launch_candidate "$zero_stage" "$method" "$selected_micro" \
                "$selected_accumulation" selected-save >/dev/null
            selected_checkpoint=$selected_checkpoint_path
            for selected_witness in \
                "$result_dir/selected-run-witness.json" "$selected_save_registry"; do
                "$repository_python" -m src.stage1.zero_matrix record-selected \
                    --witness "$selected_witness" --run-kind selected-save \
                    --identity-json "$selected_identity" --experiment-id "${checkpoint_dir##*/}" \
                    --checkpoint-dir "$checkpoint_dir" --result-dir "$result_dir" \
                    --checkpoint-path "$selected_checkpoint_path" \
                    --log-dir "$log_dir" --initial-step "$selected_initial_step" \
                    --final-step "$selected_final_step" >/dev/null
            done
        fi
        # The resume run has a distinct experiment ID and output roots.  It resumes
        # only from the selected winner; losing candidates are never serialized.
        selected_resume_registry="$selected_registry_dir/zero${zero_stage}-${method}-selected-resume.json"
        selected_resume_status=$("$repository_python" -m src.stage1.zero_matrix selected-status \
            --witness "$selected_resume_registry" --run-kind selected-resume \
            --identity-json "$selected_identity")
        if [[ $("$repository_python" -c \
            'import json,sys; print(str(json.loads(sys.argv[1])["reuse"]).lower())' \
            "$selected_resume_status") != true ]]; then
            launch_candidate "$zero_stage" "$method" "$selected_micro" \
                "$selected_accumulation" selected-resume "$selected_checkpoint" >/dev/null
            for selected_witness in \
                "$result_dir/selected-run-witness.json" "$selected_resume_registry"; do
                "$repository_python" -m src.stage1.zero_matrix record-selected \
                    --witness "$selected_witness" --run-kind selected-resume \
                    --identity-json "$selected_identity" --experiment-id "${checkpoint_dir##*/}" \
                    --checkpoint-dir "$checkpoint_dir" --result-dir "$result_dir" \
                    --checkpoint-path "$selected_checkpoint_path" \
                    --log-dir "$log_dir" --initial-step "$selected_initial_step" \
                    --final-step "$selected_final_step" \
                    --resume-from "$selected_checkpoint" >/dev/null
            done
        fi
    done
done
