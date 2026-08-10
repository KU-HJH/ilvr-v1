#!/bin/bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VENV_BIN="${REPO_ROOT}/venv/ilvr-venv/bin"

export HF_HUB_OFFLINE=1
MODEL_DIR="${MODEL_DIR:-}"
TEST_DATA_PATH="${TEST_DATA_PATH:-${REPO_ROOT}/data/TEST.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/data/images_comt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results}"
EVAL_SCRIPT_PATH="${EVAL_SCRIPT_PATH:-${REPO_ROOT}/src/evaluate_deepseed.py}"
CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/cache}"
TASK_NAME="zebra-cot"
MAX_NEW_TOKENS=4096
TEMPERATURE=0.0
TOP_P=1.0
GPU_ID="${GPU_ID:-0}"

if [ -z "${MODEL_DIR}" ] || [ ! -d "${MODEL_DIR}" ]; then
    echo "error: MODEL_DIR must name an existing trained-model directory." >&2
    exit 1
fi

if [ ! -f "${TEST_DATA_PATH}" ]; then
    echo "error: test data file '${TEST_DATA_PATH}' is missing." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
OUTPUT_FILE="${OUTPUT_DIR}/TEST_results.jsonl"
echo "save to: ${OUTPUT_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${VENV_BIN}/python" "${EVAL_SCRIPT_PATH}" \
    --model_dir "${MODEL_DIR}" \
    --test_data_path "${TEST_DATA_PATH}" \
    --image-root "${IMAGE_ROOT}" \
    --task_name "${TASK_NAME}" \
    --output_json_path "${OUTPUT_FILE}" \
    --cache_dir "${CACHE_DIR}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}"

echo "========================================================"
echo "finished"
echo "========================================================"
