# Shared ILVR training policy

Read `/root/workspace/ILVR/AGENTS.md` and `/root/workspace/ILVR/documents/training-defaults.md` before training in this worktree. Their standing multi-node, storage, and reproduced-baseline policies apply here and take precedence over historical launch examples.

New Stage 1/2/2A/2B, CE-only, preference and ablation runs default to two nodes/four ranks, microbatch 1, accumulation 4, global batch 16, LR 5e-5, seed 42 and warmup 10. Derive each stage's update budget from retained samples and actual sampler/Trainer remainder behavior; current Stage 2 with 3,379 retained samples targets 212 updates. The former global-batch-8 Stage 1 431 and Stage 2 423 values are historical. Resume retains its saved settings. Preserve the stage-specific loss, backend and checkpoint implementation.
