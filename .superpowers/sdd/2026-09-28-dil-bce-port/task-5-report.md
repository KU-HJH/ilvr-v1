# Task 5 implementation evidence

Worktree: `/root/workspace/ILVR/.worktrees/dil-bce-port`. Implementation: coordinator allowlist forwards five DIL variables; worker shell forwards all five CLI values and omits empty `k`/percent. The shell rejects DIL-specific settings paired with `legacy_dpo`; context-preference validation stays with the existing Python parser/runtime validator. No coordinator rank ownership, topology, remote path, or preflight code changed. The fresh profile is source-only, rejects nonempty `RESUME_CHECKPOINT`, and resets all baseline values and DIL variant fields. The docs contain eight named dry-run commands and one `voff` override. Worktree AGENTS historical batch-8 wording now points to batch 16 and 212 current Stage 2 updates.

## TDD witnesses

All pytest commands used `PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and `./venv/ilvr-venv/bin/python -m pytest -p no:cacheprovider`. Logs are under `logs/dil-bce-port-workflow/`.

| Phase | Command suffix | Log | Exit / result |
|---|---|---|---|
| RED forwarding | `-q tests/multinode/test_dil_forwarding.py` | `task-5-red.log` | 1; two failures at missing `STEP_PREFERENCE_LOSS_TYPE` in coordinator-owned env |
| GREEN forwarding | same | `task-5-green-1.log` | 0; 2 passed, both owned worker JSON commands |
| RED profile | `-q tests/step_preference/test_dil_examples.py` | `task-5-red-profile.log` | 1; stale values and missing profile |
| GREEN profile | same | `task-5-green-profile.log` | 0; 2 passed |
| RED legacy guard | `-q tests/multinode/test_dil_forwarding.py -k invalid_dil_combination` | `task-5-red-legacy.log` | 1; four negative settings rejected, but valid `legacy_dpo` ignored DIL selector |
| GREEN examples | `-q tests/step_preference/test_dil_examples.py --basetemp="$PWD/checkpoints/dil-bce-port-cpu/task-5"` | `task-5-doc-green.log` | 0; 11 passed |
| Scoped integration | `-q tests/multinode/test_dil_forwarding.py tests/step_preference/test_dil_examples.py tests/multinode/test_stage2b_forwarding.py tests/multinode/test_evidence_gain_forwarding.py tests/multinode/test_insertion_forwarding.py --basetemp="$PWD/checkpoints/dil-bce-port-cpu/task-5"` | `task-5-scoped-green.log` | 0; 34 passed in 218.47s |
| Final new tests | `-q tests/multinode/test_dil_forwarding.py tests/step_preference/test_dil_examples.py --basetemp="$PWD/checkpoints/dil-bce-port-cpu/task-5"` | `task-5-new-final.log` | 0; 18 passed in 130.09s after offline/GPU env assertions and shell validation cleanup |

`bash -n run_stage2_training.sh scripts/stage2/dil_bce_profile.sh` and `git diff --check` both exited 0. Project full-suite and independent review are owned by the controller.

## Actual documented commands

The test extracts the common bash source block and each of the eight named variant blocks plus the `voff` block from `docs/dil-bce.md` verbatim. It runs each as `bash -c` with `set -euo pipefail` and offline/GPU-disabled environment. The coordinator's `--dry-run` JSON gives local rank 1 and remote rank 0 owned commands. The test executes each embedded worker shell with `STAGE2_DRY_RUN=true`, `STAGE2_DRY_RUN_FORMAT=json`, empty `CUDA_VISIBLE_DEVICES`, and offline Hugging Face flags, then checks its resolved argv. The Stage 1 metadata fixture is a strict complete sidecar at a checkpoint path containing spaces, with `resized_vocab_size=151668`, `pretrained_model_vocab_size=151936`, and `base_vocab_size=151665`. The actual loader yields `k=1516` and `15166` from the resized field. The test confirms no corresponding checkpoint or result run directory was made.

| Recipe | Selector | k | mass | percent | source |
|---|---|---:|---:|---:|---|
| bce | dil_bce_rf | omitted | 0 | omitted | clean |
| topk-k8 | dil_bce_rf_topk | 8 | 0 | omitted | clean |
| topk-k5-mass | dil_bce_rf_topk | 5 | 1 | omitted | clean |
| topk-k8-mass | dil_bce_rf_topk | 8 | 1 | omitted | clean |
| topk-k50-mass | dil_bce_rf_topk | 50 | 1 | omitted | clean |
| topk-pct1-mass | dil_bce_rf_topk | 1516 | 1 | omitted | clean |
| topk-pct10-mass | dil_bce_rf_topk | 15166 | 1 | omitted | clean |
| topk-pct1-mass-entropy20 | dil_bce_rf_topk | 1516 | 1 | 20 | clean |
| entropy20-voff | dil_bce_rf_topk | 1516 | 1 | 20 | voff |

For every row and both machine ranks, worker argv contains two machines/four processes, microbatch 1/GAS4, LR5e-5/seed42/warmup10, beta0.5/preference1, entropy-loss0, 212 steps, `reference_random`, `concatenated-v1`, direct Stage 1 path, shared prefix, V-only and supervised boundaries. The source profile sets r32/alpha64/dropout0.05, latent8, SDPA/FlashAttention2/Flex1024, native gradient checkpoint off, latent checkpoint on, 8192 sequence, workers0, CE1, zero margin, direct Stage1 original-base path, and A/B ratio0.5. AdamW/linear, weight decay0.01, max-grad-norm1, BF16 and existing sidecar checkpoint behavior are training implementation defaults noted in the docs.

## Review and concerns

Self-review found that an initial negative test failed due to unrelated Stage 2B admission rules, so its fixture was corrected to provide a valid legacy DPO initialization and negative inputs. The final RED then established silent acceptance of DIL selector with legacy objective; the targeted guard fixes it. The shell relies on existing Python DIL validation for numeric and mode checks, avoiding a second incompatible validator. Full training, remote-node compatibility, PEFT preparation, and evaluation were not run or authorized in this task. No run directories or model artifacts were created by the profile or dry-run tests. Existing untracked `src/__pycache__/` directories were left unstaged.

Commit: `feat: expose DIL-BCE variants through multinode launchers` (this commit).
