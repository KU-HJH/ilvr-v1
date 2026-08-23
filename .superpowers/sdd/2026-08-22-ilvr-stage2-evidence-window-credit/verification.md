# Task 14 verification — Stage-2 evidence-window credit

Verified implementation HEAD: `e5c1012` (documentation-only commits may follow).

Date: 2026-08-23

Environment:

- Python: `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python`
- Model: `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/comt_ckpt`
- Dataset: `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl`, sample 920
- Hardware: NVIDIA B200; one GPU for the consolidated ZeRO-3 matrix and two GPUs for ZeRO-1/2 online-reference runs
- Reference artifact reused without recomputation: `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/reference-sample-920-hidden-003`

## Final passing commands

Focused Stage-2 paths:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest -q tests/evidence_window/test_stage2_qwen_attention.py tests/evidence_window/test_stage2_engine.py tests/evidence_window/test_stage2_baseline.py tests/evidence_window/test_stage2_smoke.py
```

Result: `119 passed in 67.32s`.

Online-reference tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest -q tests/evidence_window/test_stage2_online_reference.py
```

Result: `14 passed in 15.32s`.

Smoke-harness tests after the final gradient-witness fix:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest -q tests/evidence_window/test_stage2_smoke.py
```

Result: `70 passed in 8.78s`.

Two-rank ZeRO-3 latent-prefix ledger with LoRA and gradient checkpointing:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 tests/evidence_window/stage2_zero3_latent_ledger_worker.py
```

Result: exit 0. Rank records were `(1, 8, (3,), True, True)` and `(3, 8, (3,), True, True)`, proving unequal local latent counts, exactly eight decoder calls on both ranks, coordinated three-step replay, finite backward, and a LoRA-B update.

Consolidated real-7B Stage-1/Stage-2 ZeRO-3 matrix:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/stage2_large_model_smoke.py --case all-single-gpu --model /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/models/Qwen2.5-VL-7B-Instruct --stage1-checkpoint /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/comt_ckpt --data /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data --sample-index 920 --attempt-id task14-single-009 --deepspeed-config configs/config_stage3.json --reference-artifact-dir /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/reference-sample-920-hidden-003 --reference-dataset-sha256 5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8 --reference-checkpoint-id ilvr-comt-stage1-global_step810 --reference-checkpoint-witness /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/comt_ckpt@global_step810 --reference-tokenizer-id /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/comt_ckpt --reference-parser-version stage2-marker-v1 --reference-model-config-id qwen2.5-vl-stage2-self-latent-v1 --output-dir /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/large-results/single-009
```

Result: exit 0 and completion manifest `complete`, world size 1, six authenticated results: Stage-1 baseline, Stage-2 baseline, reference-free EW, precomputed EW-DPO, relation KD, and hidden-residual KD. Every loss and policy global gradient norm was finite; all active Stage-2 cases selected six evidence-window edges.

Two-rank real-7B online frozen reference under ZeRO-1:

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/stage2_large_model_smoke.py --case online-frozen-zero1 --model /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/models/Qwen2.5-VL-7B-Instruct --stage1-checkpoint /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/comt_ckpt --data /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data --sample-index 920 --attempt-id task14-zero1-011 --deepspeed-config configs/config_stage1.json --output-dir /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/large-results/zero1-011
```

Result: exit 0 and completion manifest `complete`, world size 2, two authenticated rank results. `loss/ew_pref=0.6931471824645996`, policy `grad_norm=0.007402094546705484`, six edges, and no reference gradients.

Two-rank real-7B online frozen reference under ZeRO-2:

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/stage2_large_model_smoke.py --case online-frozen-zero2 --model /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/models/Qwen2.5-VL-7B-Instruct --stage1-checkpoint /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/comt_ckpt --data /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data --sample-index 920 --attempt-id task14-zero2-013 --deepspeed-config .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/zero2_smoke.json --output-dir /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/large-results/zero2-013
```

Result: exit 0 and completion manifest `complete`, world size 2, two authenticated rank results. `loss/ew_pref=0.6931471824645996`, policy `grad_norm=0.007455034181475639`, six edges, and no reference gradients. The smoke-specific config omits the production `config_stage2.json` 8-rank sequence/data-parallel mesh and changes no objective semantics.

Final complete CPU/synthetic suite:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window -q
```

Result: `442 passed in 262.99s`.

Static compilation:

```bash
PYTHONPYCACHEPREFIX=<fresh-temporary-directory> /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m compileall -q src transformers/src/transformers/models/qwen2_5_vl
```

Result: exit 0. `git diff --check` also exited 0.

## Failures encountered and resolved

1. Stage-2 latent placeholders initially reused `<|image_pad|>` and over-consumed image-grid rows. `bedcc2e` renders latent interiors as `<|latent_pad|>` and preserves plan normalization.
2. Precomputed reference replay inherited mixed FP32/BF16 checkpoint tensors. `cff0422` normalizes the frozen scorer execution dtype.
3. The smoke harness pre-wrapped LoRA without the input-gradient hook required by reentrant checkpointing. `00dde2c` matches production PEFT setup.
4. PyTorch non-reentrant latent-prefix checkpoints observed ZeRO-3 full/partitioned metadata inconsistently during recomputation. `79ad791` uses DeepSpeed's ZeRO-aware non-reentrant checkpoint only for `ds_id` parameters; non-ZeRO behavior is unchanged.
5. Autograd parameter hooks were not a reliable post-step ZeRO-3 witness. `67a55ba` authenticates the finite nonzero DeepSpeed global norm already computed before optimizer clearing.
6. Online frozen references had the same mixed-checkpoint dtype issue as precompute. `0d97866` normalizes them to BF16 CUDA or FP32 CPU before eval/freeze/Accelerator preparation.
7. Production `configs/config_stage2.json` describes an 8-rank (`2 x 4`) mesh and cannot run on this two-GPU host. `e5c1012` adds a smoke-only two-rank ZeRO-2 config; the online-reference objective and ZeRO stage are unchanged.

Failed immutable attempt directories contain no `completion.json` and are diagnostic only. Successful attempts are `single-009`, `zero1-011`, and `zero2-013`; they must be reused rather than rerun.

## Verified limitations

- The exact implementation is one full auxiliary replay plus one blocked replay per selected evidence-window edge: `O(1 + |E|)`, not the proposed `O(1 + J)` optimization.
- Online frozen reference is supported only for ZeRO-1/2 and owns one full frozen 7B replica per rank. Precomputed targets are recommended for ZeRO-3.
- The current CUDA reference execution dtype is BF16, matching the validated ILVR runtime; FP16/FP32 CUDA online-reference operation is not established.
- Hidden-state residual KD is implemented; logit-residual KD is not implemented in the revised Stage-2 design.
- Stage-2 model batch size greater than one is deliberately rejected, and active auxiliary paths require zero dropout.
- LOO measures a fixed-history conditional/direct dependency, not total causal effect, and single-evidence LOO cannot resolve every synergy/redundancy pattern.
