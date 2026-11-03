# Centralize Model Weights Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every local model weight, including linked-worktree training artifacts and downloaded base models, beneath the canonical repository `checkpoints/` directory and make both checkouts resolve the same artifact roots.

**Architecture:** Preserve each existing model or run directory as one atomic tree beneath the canonical `/root/workspace/ILVR/checkpoints` path, whose symlink target is the external checkpoint storage. Keep canonical result and log roots separate and retain their lexical identity checks.

**Tech Stack:** POSIX filesystem operations, Git worktrees, Bash, pytest

**Spec:** `AGENTS.md`

## Global Constraints

- Use `./venv/ilvr-venv` for Python and tests.
- Store weights under `/root/workspace/ILVR/checkpoints/`.
- Store results under `/root/workspace/ILVR/results/` and logs under `/root/workspace/ILVR/logs/`.
- Preserve complete checkpoint directories and do not overwrite any existing destination.

---

### Task 1: Preflight and canonical roots

**Files:**
- Inspect: `/root/workspace/ILVR/checkpoints`
- Inspect: `/root/workspace/ILVR/models`
- Inspect: `/root/workspace/ILVR/.worktrees/vcsd-ilvr-stage2/outputs`
- Inspect: `/root/workspace/ILVR/.worktrees/vcsd-ilvr-stage2/cache`

**Interfaces:**
- Consumes: Existing primary checkout and linked-worktree artifact trees.
- Produces: Exact, conflict-free migration source and destination map.

- [ ] **Step 1: Check for active ILVR training or evaluation processes**

Run: `ps -eo pid,etimes,args | rg '/root/workspace/ILVR|accelerate|src/main.py'`

Expected: no active training or evaluation process referencing the migration sources.

- [ ] **Step 2: Verify sources and destination conflicts**

Run read-only `find`, `du`, and `test -e` checks for each exact source and destination.

Expected: all intended sources exist and no destination would be overwritten.

### Task 2: Move model-bearing trees and add compatibility links

**Files:**
- Move: `/root/workspace/ILVR/checkpoints` to canonical `checkpoints/`
- Keep: primary models within canonical `checkpoints/pretrained/models`
- Keep: linked-worktree cache within canonical `checkpoints/pretrained/huggingface-cache-vcsd-ilvr-stage2`
- Keep: linked-worktree outputs within canonical `checkpoints/imported-worktrees/vcsd-ilvr-stage2/outputs`
- Create: compatibility symlinks in the primary checkout and linked worktree

**Interfaces:**
- Consumes: Conflict-free map from Task 1.
- Produces: One physical checkpoint hierarchy with stable legacy paths.

- [ ] **Step 1: Create physical checkpoint subdirectories and canonical parents**

Use `mkdir -p` only for the exact canonical repository parent from `AGENTS.md`.

- [ ] **Step 2: Move each complete source tree once**

Use explicit `mv -- SOURCE DESTINATION` operations; do not use globs or separately delete sources. Verify free space before the cross-overlay checkpoint move.

- [ ] **Step 3: Add compatibility symlinks**

Link primary `checkpoints`, `models`, `results`, and `logs`, plus worktree `checkpoints`, `models`, `cache`, `results`, `logs`, and the historical `outputs` archive, to the corresponding canonical targets.

### Task 3: Synchronize settings and verify ownership

**Files:**
- Modify: `/root/workspace/ILVR/.gitignore`
- Modify: `/root/workspace/ILVR/.worktrees/vcsd-ilvr-stage2/.gitignore`
- Verify: `/root/workspace/ILVR/.worktrees/vcsd-ilvr-stage2/tests/test_artifact_paths.py`

**Interfaces:**
- Consumes: Centralized filesystem layout from Task 2.
- Produces: Both checkouts ignore the same artifact links and runtime validation continues to enforce canonical destinations.

- [ ] **Step 1: Run a failing repository-layout audit**

Scan for real model-weight files outside the canonical checkpoint root and fail when any are found. This directly exercises the filesystem behavior without adding a host-specific unit test to the repository.

- [ ] **Step 2: Run the test and verify the expected failure**

Run: `find /root/workspace/ILVR ...` with an explicit nonzero exit when matches are found.

Expected: failure before model-bearing trees and compatibility paths are synchronized.

- [ ] **Step 3: Synchronize ignore settings**

Ignore artifact links/directories (`checkpoints/`, `models/`, `cache/`, `outputs/`, `results/`, and `logs/`) without changing source-code ignores.

- [ ] **Step 4: Repair linked-worktree Git metadata for the current checkout location**

Use `git worktree repair` with the exact `/root/workspace/ILVR/.worktrees/vcsd-ilvr-stage2` path and command-local safe-directory configuration.

- [ ] **Step 5: Run focused tests and repeat the filesystem audit**

Run the artifact-path test file through the repository virtual environment and scan for non-symlink model weight files outside canonical `checkpoints/`.

Expected: tests pass and the audit prints no model weight files outside the canonical hierarchy.
