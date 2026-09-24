# Shared ILVR training policy

Read `/root/workspace/ILVR/AGENTS.md` and `/root/workspace/ILVR/documents/training-defaults.md` before training in this worktree. Their standing multi-node, storage, and reproduced-baseline policies apply here and take precedence over historical launch examples.

New Stage 1/2/2A/2B, CE-only, preference and ablation runs default to two nodes/four ranks, microbatch 1, accumulation 2, global batch 8, LR 5e-5, seed 42 and warmup 10. Verify each stage's retained-data update budget; do not copy Stage 1's 431 or the old Stage 2's 212 indiscriminately. Resume retains its saved settings. Preserve the stage-specific loss, backend and checkpoint implementation.
