# DIL-BCE Stage 2B dry-run recipes

These commands use the two-node coordinator (rank 0 on `korea@10.1.102.41`, rank 1 locally), ZeRO-2, and four GPU processes. Run them from this reviewed worktree after both nodes have compatible code and environments. A `--dry-run` makes no training run directory; inspect both node commands and the resolved settings before removing that flag for a separately authorized training launch. A current retained set of 3,379 examples gives 212 updates at global batch 16; recompute the budget from the actual retained set and `reference_random` remainder behavior if inputs change. Stage 2A/B combined runs retain their existing phase allocation; these commands start Stage 2B directly from the reproduced Stage 1 adapter.

Source any old environment first, then source the fresh profile. The profile rejects `RESUME_CHECKPOINT`, resets stale DIL options, and creates no run directories. The underlying optimizer is AdamW with linear schedule, weight decay 0.01, max grad norm 1.0, warmup 10 updates and BF16. This profile sets microbatch 1, accumulation 4, LR 5e-5, seed 42, DoRA r32/alpha64/dropout0.05 with latent 8 and the validated decoder modules, SDPA language attention, FlashAttention2 vision, Flex sidecars with query budget 1024, native checkpointing off and latent/sidecar checkpointing on, max train sequence 8192, data workers 0, CE 1, entropy weight 0, beta 0.5 and preference weight 1.0. It uses V-only `context_preference`, `immediate_text`, `clean_prefix`, independent branches, shared prefix, visible and supervised latent boundaries, original image access, `reference_random`, and `concatenated-v1` with the approved crossing-token exclusion. Use `STEP_PREFERENCE_TARGET_SCOPE=until_next_latent` only as an explicit override.

```bash
# common setup: source once before one of the named variant commands
source ./scripts/stage2/dil_bce_profile.sh
export PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"
run_stamp=$(date -u +%Y%m%d-%H%M%S)
dil_vocab_size=$(./venv/ilvr-venv/bin/python - "$STAGE1_ADAPTER_PATH" <<'PY'
from pathlib import Path
import sys
from stage1.checkpointing import read_stage1_peft_metadata_snapshot, decode_stage1_peft_metadata_snapshot
metadata = decode_stage1_peft_metadata_snapshot(read_stage1_peft_metadata_snapshot(Path(sys.argv[1])))
print(metadata.resized_vocab_size)
PY
)
k_pct1=$((dil_vocab_size / 100))
k_pct10=$((dil_vocab_size / 10))
printf 'Stage1 resized_vocab_size=%s; floor(1%%)=%s; floor(10%%)=%s\n' "$dil_vocab_size" "$k_pct1" "$k_pct10" >&2
```

The strict Stage 1 metadata loader validates provenance and reads `resized_vocab_size`; the original-base `pretrained_model_vocab_size` and tokenizer `base_vocab_size` are different. For the reproduced Stage 1 checkpoint, these are 151668, 151936 and 151665 respectively, so the percentage variants resolve to 1516 and 15166. The training model preparation separately checks tokenizer and LM-head vocabulary alignment before forward. `k` counts vocabulary entries, not probability mass. The `--dry-run` JSON and the printed ratio resolution are the prelaunch record; retain them for both nodes.

### bce

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-bce-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

### topk-k8

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK=8 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-k8-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

### topk-k5-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK=5 STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-k5-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

### topk-k8-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK=8 STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-k8-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

### topk-k50-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK=50 STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-k50-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

### topk-pct1-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK="$k_pct1" STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-pct1-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

### topk-pct10-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK="$k_pct10" STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-pct10-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

### topk-pct1-mass-entropy20

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK="$k_pct1" STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 STEP_PREFERENCE_ENTROPY_TOP_PERCENT=20 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-pct1-mass-entropy20-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

For the explicit visual-off entropy source ablation, use the same final command with `STEP_PREFERENCE_ENTROPY_SOURCE=voff` and a fresh run ID:

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK="$k_pct1" STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 STEP_PREFERENCE_ENTROPY_TOP_PERCENT=20 STEP_PREFERENCE_ENTROPY_SOURCE=voff \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-pct1-mass-entropy20-voff-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --max_steps "$STAGE2_TOTAL_STEPS" --warm_up_steps 10
```

For inference, prepare the completed PEFT artifact with the existing `src.multinode prepare-inference` path under `checkpoints/evaluation-staging/<run-id>/`; use the original Qwen2.5-VL-3B base. Then run the existing `run_multinode_evaluation.sh` entrypoint. The default TEST400 evaluation is greedy (`temperature=0`, `top_p=1`), max new tokens 4096, max sequence 128000, repetition penalty 1, no vision ablation, and `external_full_latent_block`. Follow [the operator guide](multinode-training.md) for the exact preparation and evaluation command arguments. A prepared inference copy is not a training resume checkpoint.

### Reversed chosen/rejected reductions

Select `STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk_reverse` (CLI: `--step_preference_loss_type dil_bce_rf_topk_reverse`) for the separately versioned reduction variant. For each original step with selected positions `T`, its chosen loss is `mean_t softplus(-beta * clean_GT_logp_t)` and its rejected loss is `softplus(beta * mean_t sum_v q_t(v) * log(q_t(v)))`. The OFF distribution `q` is renormalized over the detached OFF top-k support; its probabilities and expectation remain differentiable. Both the candidate expectation and selected-position mean precede the single rejected softplus. The existing live top-k entropy statistic supplies the expectation as `-H(q)`.

Sum each component over steps and divide by the original step count. A step with no selected positions contributes zero, including no `softplus(0)`. Metrics `loss/dil_positive` and `loss/dil_negative` report these actual chosen and rejected losses; log-probability metrics keep their existing meaning. CE, clean-detached group-mass KL, entropy selection and step weighting follow the existing top-k path.

For the approved experiment, keep top-k 1516 (1%), sample-level clean full-vocabulary entropy top20, beta 0.5, preference/CE/mass weights 1, entropy weight 0, and `until_next_latent`. Change only the loss selector in the approved saved fresh-run profile and use a fresh run ID with the existing coordinator. The original `dil_bce_rf_topk` formulas and `dpo` default are preserved. Checkpoints save formula version `dil_bce_rf_topk_reverse_v1`; resuming an original top-k run with this selector, or the reverse, is rejected.

### Planned checkpoint stop for a bounded resume probe

For an independently authorized three-update DIL smoke, one fresh run can stop after a complete checkpoint 2 and resume to the original budget 3. Keep the approved dataset and profile. This optional diagnostic requires enabled DIL `context_preference` Stage 2B and a positive integer strictly below the resolved total budget. Both coordinator and workers receive the option; the coordinator also requires explicit `--max_steps`.

```bash
# After the fresh profile, use the separately approved smoke input and run ID.
: "${run_id:?set a fresh smoke run ID}"
: "${smoke_data:?set the approved smoke dataset}"
export DATA_PATH="$smoke_data" STAGE2_TOTAL_STEPS=3
export STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf
./run_multinode_stage2_training.sh --dry-run --run-id "$run_id" \
  --zero-stage 2 --main-process-port 29631 --max_steps 3 --save_steps 3 \
  --warm_up_steps 10 --stage2_stop_after_checkpoint 2
```

After a separately authorized launch, every rank requests a normal full checkpoint at optimizer step 2. The Trainer stops after saving and the existing sidecar publication barriers; `trainer_state.json` retains `global_step=2` and `max_steps=3`. The coordinator waits for both node exits to be zero, then applies its unchanged four-rank collector to checkpoint 2. The attempt witness records `planned_stop` with `budget_complete=false`. Missing/incomplete checkpoint state or mismatched producer steps fail validation.

Use canonical checkpoint 2 in the same-run resume recipe below, with a fresh attempt ID, saved `max_steps=3`, and `saved_cli=(--save_steps 3 --warm_up_steps 10)`. Omit `--stage2_stop_after_checkpoint` from resume settings: it is fresh-run only. The ordinary resume restores state and performs update 3. Do not edit trainer state, change the budget, signal workers, or collect after nonzero exits. CPU lifecycle tests cover full save/restore and another update; actual four-rank ZeRO restoration remains a separately authorized GPU smoke claim.

### Same-run DIL resume

Use the interrupted run's saved environment and CLI settings, with the same run ID, original base, data, length manifest, objective, sampler, checkpointing, optimizer and schedule settings on both nodes. Do not source `dil_bce_profile.sh`: it is for fresh runs. Resume requires the complete training checkpoint (including trainer, optimizer, scheduler, RNG, sampler/runtime sidecars and ZeRO state), not the final adapter or inference copy. The coordinator requires a new attempt ID and the same run's immediate `checkpoint-N` path.

The initialization flags and `STAGE2_TOTAL_STEPS` are fresh-run selectors. Remove them after sourcing saved settings. Pass the already-resolved budget from `trainer_state.json` as `--max_steps`; this preserves both direct Stage 1 and phased Stage 2A/B allocations. DIL resume rejects an ambiguous total-step allocation or a changed resolved budget before model loading. Keep all remaining saved CLI options, including warmup and save cadence. In the example below, `saved_cli` must contain those original non-initialization CLI options (for the fresh recipes above, `--warm_up_steps 10`).

```bash
# Set these to the interrupted run and its saved settings before executing.
: "${run_id:?set the original run ID}"
: "${saved_run_env:?set the saved environment file for that run}"
: "${resume_step:?set the saved checkpoint step}"
set -a
source "$saved_run_env"
set +a
export PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD"
: "${STAGE1_CHECKPOINT:?saved environment must identify the original Qwen base}"
export RESUME_CHECKPOINT="$PWD/checkpoints/$run_id/checkpoint-$resume_step"
unset STAGE2B_DIRECT_FROM_STAGE1 STAGE1_ADAPTER_PATH STAGE2A_CHECKPOINT_PATH STAGE2_TOTAL_STEPS
saved_max_steps=$(./venv/ilvr-venv/bin/python - "$RESUME_CHECKPOINT/trainer_state.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    budget = json.load(handle)["max_steps"]
if type(budget) is not int or budget <= 0:
    raise ValueError("saved trainer_state.max_steps must be a positive integer")
print(budget)
PY
)
# Restore the original additional CLI settings; remove any fresh-init/total-step flags.
saved_cli=(--warm_up_steps 10)
./run_multinode_stage2_training.sh --dry-run --run-id "$run_id" \
  --attempt-id "resume-$(date -u +%Y%m%d-%H%M%S)" \
  --zero-stage 2 --main-process-port 29631 \
  --max_steps "$saved_max_steps" "${saved_cli[@]}"
```

Inspect both node commands and worker dry-run JSON before an independently authorized launch. The actual entry path verifies the saved DIL identity and budget, original-base/PEFT provenance, sparse token rows and vocabulary/boundary contract. The trainer repeats the identity check before delegating state restoration to the existing Trainer/DeepSpeed path. Missing or changed DIL identity is rejected; DPO Stage 2B resume acceptance remains unchanged.
