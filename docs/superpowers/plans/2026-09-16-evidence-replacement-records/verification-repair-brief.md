# Same-image Evidence Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Repository policy requires delegated implementation; executing-plans/inline implementation is not an alternative. Steps use checkbox syntax for tracking.

**Goal:** Add the approved opt-in same-image evidence-replacement comparison to ours1 CE+Preference training, with deterministic selection and detached replacement copies.

**Architecture:** A pure planner selects earlier latent donors or image-feature fallback. A Qwen scorer replays original/replacement prefixes with native visibility and paired RNG. Trainer, configuration and launch paths expose the comparison and source-weighted diagnostics without changing old modes.

**Tech Stack:** Repository Python venv, patched Transformers/Qwen2.5-VL, PyTorch, pytest, shell worker launcher and existing multi-node coordinator.

**Spec:** `docs/superpowers/specs/2026-09-16-evidence-replacement-design.md`, design commitment v1. Approved by user reply "진헹하자" after presentation; approved bytes SHA256 `8fa8e36fb433ebe2e7563852ee309ce1ab2e97fda6f6a30bc6841ab1983e1f4e` at source commit `5557bdd1ef41ea8e5354a2ac1990b636f0d5cf19`.

## Global Constraints

- Work only in `/root/workspace/ILVR/.worktrees/ours1-evidence-replacement-20260916`, branch `feat/ours1-evidence-replacement-20260916`. Source baseline is `ae5a1cd`, a faithful snapshot of current dirty ours1 plus relevant untracked files. Base branch for later human integration choice is `ours1`.
- Use `./venv/ilvr-venv/bin/python` with `PYTHONPATH=src:transformers/src`; reuse the installed environment. No dependency reinstall, pretrained-model loading, training launch or external reference code.
- Opt-in mode string is `evidence_replacement`; original image access remains, boundaries remain, per-device microbatch one, one image, independent visual-only context preference, entropy weight zero. Do not change old defaults.
- Donors are earlier validated scored steps with exactly configured latent size. First/single steps use same-image encoder features; M >= L samples without replacement, 0 < M < L with replacement, sorted source indices. Both replacement copies are detached; original donor occurrence remains differentiable.
- Sampling version is `ilvr-evidence-replacement-v1`. Seed uses SHA256 of UTF-8 compact JSON `[version, training_seed, canonical_sample_id, step_id, source_kind]`, first eight bytes big-endian masked to 63 bits. A private CPU generator chooses donor via `randint(N,(1,))`, image via `randperm(M)[:L]` or `randint(M,(L,))`; no global RNG consumption.
- Scores gather `input_ids[:, target_start:target_end]` from predictors `target_start-1 .. target_end-2`. Full-prefix original/replacement replays share positions/native visibility/dropout draws; no clean/donor KV reuse. Dense uses native 2D mask, Flex matches actual native causal/sliding rule. No new public Stage2B resume support.
- Preserve code/data/checkpoint symlinks; outputs under this worktree's results/, logs under logs/. No generated model weights. CPU tests and tiny CUDA forward/backward verification are allowed; optimizer/training jobs are not.
- Implementers must read TDD plus writing-good-tests, show RED before implementation and GREEN afterward, self-review and commit. No subagents or paperthin hooks from implementers/reviewers. Root alone coordinates review/fix gates and writes workflow control records.

## Handoff clarifications

`ae5a1cd` snapshots existing ours1 changes on top of the approved-spec commit `5557bdd`; it does not change the bound spec. Tiny models are randomly initialized local fixtures, never pretrained downloads. Invalid planner metadata raises `ValueError` with a useful field-specific message. Existing `SidecarPlan` records define scored/active branches; empty branches mean CE-only. Consult their definitions for shapes and the existing target-scope implementation, not a new interpretation. Paired RNG covers CPU plus the active embedding CUDA device; one process uses one GPU. The durable workflow record is the sibling `2026-09-16-evidence-replacement-evidence.md`. Exact new diagnostics may follow existing naming style but must name original/replacement and keep donor/image sum/count weighting explicit. The TDD skill and `writing-good-tests.md` are under `/root/.codex/plugins/cache/openai-curated-remote/superpowers/6.3.0/skills/test-driven-development/`.


## Verification-discovered regression, separate from final-review fix wave
BASE3112e9677771269ddd38a1a1e2337840e66d3458. Final project command returned6failed3297passed6skipped5deselected11warnings. Full log logs/evidence-replacement-20260916/final-cpu.log. Only assigned new failure: tests/step_preference/test_stage2b_trainer.py::test_capacity_final_adapter_save_defers_processor_but_numbered_save_does_not. New unconditional attach_evidence_replacement_identity(model,self.step_preference_config) at trainer.py:1608 raises AttributeError on existing object.__new__ fixture lacking config.
Use systematic-debugging to trace actual Trainer initialization/save and determine whether test fixture should supply the real config invariant or saving legitimately permits absent config. Make smallest compatibility correction, justified by actual runtime contract, not a guessed getattr workaround. Reproduce RED, apply minimal fix, focused GREEN with test_stage2b_trainer.py and replacement_trainer/config tests. Add/retain one causal regression as needed; existing failing test can be witness. No unrelated failed-suite fixes: root handles baseline diagnosis. No whole suite; root completion witness after scoped review. No new scope, no training/weights/optimizer, no subagents/paperthin. Report exact commands/full output/diagnosis/self-review to .superpowers/sdd/2026-09-16-evidence-replacement/verification-repair-report.md; commit only task files.
