#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
arguments=("$@")
legacy_lora=false
method_supplied=false
for argument in "${arguments[@]}"; do
    [[ $argument == --use_lora ]] && legacy_lora=true
    [[ $argument == --method ]] && method_supplied=true
done
if [[ $legacy_lora == true && $method_supplied == false ]]; then
    arguments=(--method lora "${arguments[@]}")
fi

exec "$script_dir/run_stage1_training.sh" "${arguments[@]}"
