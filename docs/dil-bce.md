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
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

### topk-k8

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK=8 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-k8-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

### topk-k5-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK=5 STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-k5-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

### topk-k8-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK=8 STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-k8-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

### topk-k50-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK=50 STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-k50-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

### topk-pct1-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK="$k_pct1" STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-pct1-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

### topk-pct10-mass

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK="$k_pct10" STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-pct10-mass-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

### topk-pct1-mass-entropy20

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK="$k_pct1" STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 STEP_PREFERENCE_ENTROPY_TOP_PERCENT=20 \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-pct1-mass-entropy20-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

For the explicit visual-off entropy source ablation, use the same final command with `STEP_PREFERENCE_ENTROPY_SOURCE=voff` and a fresh run ID:

```bash
env STEP_PREFERENCE_LOSS_TYPE=dil_bce_rf_topk STEP_PREFERENCE_TOPK="$k_pct1" STEP_PREFERENCE_LAMBDA_TOPK_MASS=1 STEP_PREFERENCE_ENTROPY_TOP_PERCENT=20 STEP_PREFERENCE_ENTROPY_SOURCE=voff \
  ./run_multinode_stage2_training.sh --dry-run --run-id "dil-topk-pct1-mass-entropy20-voff-$run_stamp" \
  --zero-stage 2 --main-process-port 29631 --warm_up_steps 10
```

For inference, prepare the completed PEFT artifact with the existing `src.multinode prepare-inference` path under `checkpoints/evaluation-staging/<run-id>/`; use the original Qwen2.5-VL-3B base. Then run the existing `run_multinode_evaluation.sh` entrypoint. The default TEST400 evaluation is greedy (`temperature=0`, `top_p=1`), max new tokens 4096, max sequence 128000, repetition penalty 1, no vision ablation, and `external_full_latent_block`. Follow [the operator guide](multinode-training.md) for the exact preparation and evaluation command arguments. A prepared inference copy is not a training resume checkpoint.
