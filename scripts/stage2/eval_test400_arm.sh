#!/usr/bin/env bash
# TEST400 evaluation of one Stage 2 adapter with the CE baseline's exact
# evaluation code (voff-reproduction f8f80be) and settings. Shards 0/1 run on the
# remote node (rank 0 host), shards 2/3 locally, like the CE run. Merges with the
# same merge_shards. Usage: eval_test400_arm.sh <label> <adapter-dir>
#
# EVAL_PROFILE=paper switches to the repository's paper-aligned generation
# (upstream eval.py semantics): bounded limit, repetition penalty 1.05, max
# sequence 32768; it also pins Transformers to the f8f80be tree on both nodes
# and pushes the staged adapter to the remote node (for adapters that exist
# only locally). The default reproduces the CE-matched external profile.
set -euo pipefail
profile=${EVAL_PROFILE:-ce_matched}
case $profile in
  ce_matched) policy=external_full_latent_block; repetition=1.0; max_sequence=128000; suffix= ;;
  paper) policy=bounded; repetition=1.05; max_sequence=32768; suffix=-paper ;;
  *) echo "unknown EVAL_PROFILE=$profile" >&2; exit 2 ;;
esac
label=${1:?label}; adapter=${2:?adapter dir under checkpoints/evaluation-staging}
repo=/root/workspace/ILVR
eval_commit=f8f80be2
remote=korea@10.1.102.41
base=$repo/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
run=eval-stage2-$label$suffix-test400-$(date -u +%Y%m%d-%H%M%S)
out=$repo/results/$run
local_tree=$repo/.worktrees/eval-f8f80be
[[ $adapter == $repo/checkpoints/evaluation-staging/* ]] || { echo "adapter must be under evaluation-staging" >&2; exit 2; }
mkdir -p "$out" "$repo/logs/$run"
if [[ ! -d $local_tree ]]; then git -C "$repo" worktree add --detach "$local_tree" "$eval_commit" -q; fi
[[ $(git -C "$local_tree" rev-parse --short=8 HEAD) == "$eval_commit" ]]
ssh -o BatchMode=yes "$remote" "docker exec llm_kd bash -lc 'git -C $repo/.worktrees/voff-reproduction-20260917 rev-parse --short=8 HEAD'" | grep -qx "$eval_commit"
# The remote node already holds the source checkpoint; copy the same allowlisted
# files inside the container (root) and verify byte identity with the local copy.
step=$(basename "$adapter"); step=${step##*-}
source_ckpt=$repo/checkpoints/$(basename "$(dirname "$adapter")")/checkpoint-$step
files=$(cd "$adapter" && ls | tr "\n" " ")
if [[ $profile == paper ]]; then
  host_adapter=/raid/workspace/ILVR/${adapter#$repo/}
  ssh -o BatchMode=yes "$remote" "mkdir -p $host_adapter"
  rsync -a "$adapter/" "$remote:$host_adapter/"
  ssh -o BatchMode=yes "$remote" "docker exec llm_kd bash -lc 'cd $adapter && sha256sum $files'" > "$repo/logs/$run/remote-adapter.sha256"
else
  ssh -o BatchMode=yes "$remote" "docker exec llm_kd bash -lc 'mkdir -p $adapter && cd $source_ckpt && cp -n $files $adapter/ && cd $adapter && sha256sum $files'" > "$repo/logs/$run/remote-adapter.sha256"
fi
(cd "$adapter" && sha256sum $files) | diff - "$repo/logs/$run/remote-adapter.sha256"
ssh -o BatchMode=yes "$remote" "docker exec llm_kd bash -lc 'mkdir -p $out $repo/logs/$run'"

worker() {  # tree shard gpu
  local path_prefix=
  [[ $profile == paper ]] && path_prefix="$1/transformers/src:"
  printf 'cd %s && CUDA_VISIBLE_DEVICES=%s PYTHONPATH=%s%s:%s/src TOKENIZERS_PARALLELISM=false %s/venv/ilvr-venv/bin/python src/evaluate_deepseed.py --model_dir %s --base_model_dir %s --test_data_path %s/data/TEST.jsonl --task_name zebra-cot --output_json_path %s/shard-%s.jsonl --max_new_tokens 4096 --max_sequence_length %s --repetition_penalty %s --generation-limit-policy %s --temperature 0 --top_p 1 --num-shards 4 --shard-index %s --model-identity %s --vision-ablation none > %s/logs/%s/shard-%s.log 2>&1; echo $? > %s/logs/%s/shard-%s.exit' \
    "$1" "$3" "$path_prefix" "$1" "$1" "$repo" "$adapter" "$base" "$repo" "$out" "$2" "$max_sequence" "$repetition" "$policy" "$2" "$label" "$repo" "$run" "$2" "$repo" "$run" "$2"
}
for shard in 0 1; do
  cmd=$(worker "$repo/.worktrees/voff-reproduction-20260917" "$shard" "$shard")
  ssh -o BatchMode=yes "$remote" "docker exec -d llm_kd bash -lc $(printf '%q' "$cmd")"
done
for shard in 2 3; do
  cmd=$(worker "$local_tree" "$shard" "$((shard - 2))")
  setsid nohup bash -lc "$cmd" >/dev/null 2>&1 < /dev/null &
done
echo "$run"
until [[ -f $repo/logs/$run/shard-2.exit && -f $repo/logs/$run/shard-3.exit ]] &&
  ssh -o BatchMode=yes "$remote" "docker exec llm_kd bash -lc 'test -f $repo/logs/$run/shard-0.exit && test -f $repo/logs/$run/shard-1.exit'"; do
  sleep 60
done
for shard in 0 1; do
  rsync -a "$remote:/home/korea/Workspace/hjh/workspace/ILVR/results/$run/shard-$shard.jsonl" "$remote:/home/korea/Workspace/hjh/workspace/ILVR/results/$run/shard-$shard.jsonl.manifest.json" "$out/"
  rsync -a "$remote:/home/korea/Workspace/hjh/workspace/ILVR/logs/$run/shard-$shard.log" "$remote:/home/korea/Workspace/hjh/workspace/ILVR/logs/$run/shard-$shard.exit" "$repo/logs/$run/"
done
cat "$repo"/logs/"$run"/shard-*.exit
cd "$local_tree" && PYTHONPATH=$local_tree:$local_tree/src "$repo/venv/ilvr-venv/bin/python" - "$out" <<'PY'
import json, sys
from pathlib import Path
from multinode_evaluation import merge_shards
out = Path(sys.argv[1])
summary = merge_shards([out / f"shard-{k}.jsonl.manifest.json" for k in range(4)], expected_sample_count=400,
                       output_jsonl=out / "merged.jsonl", summary_json=out / "summary.json")
print(json.dumps({k: summary[k] for k in summary if not isinstance(summary[k], (list, dict))}))
PY
