# Matched Stage1 periodic resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. User selected Native; root implements, one fresh independent reviewer checks the final change.

**Goal:** Make retained10% matched-clean20 Stage1 checkpoints actually resumable with full saved state and preserved source outputs.

**Architecture:** Keep native Trainer/DeepSpeed continuation. Bind the legacy objective/execution using source run records, assemble existing four-rank shards after completion, and prepare a fresh canonical destination before the existing coordinator resumes it.

**Tech Stack:** Python3.11, repository venv, local patched Transformers, Accelerate1.10.1/DeepSpeed0.18.7, pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-stage1-periodic-resume-design.md`

## Global Constraints

- Original run remains source86237ee, uninterrupted train->automatic TEST400.
- Current full schedule2160/save216/keep10, micro1/GAS4/world4, CE1+alignment1/clean20, savedworkers0, EMA1, DoRA/ZeRO1 and all original settings retained. Fresh-runworkers4 is separate.
- Models only `checkpoints/`, results only `results/`, logs only `logs/`; new resume destination must be fresh.
- Existing multi-node coordinator only; two nodes/four ranks for GPU proof, no arbitrary process signals.
- Preserve source bytes/markers; fail closed on missing or conflicting evidence and artifacts.

## Review Focus

- A v1 marker alone does not establish a complete resume profile; missing source evidence must reject (Task1).
- Both composition and direct Trainer entrypoints enforce objective/execution, including omission/drift of clean20 or workers (Task1).
- Epoch-tail accumulation and random sampler continuation must match actual saved position, not only globalstep (Task1).
- Split periodic versus already-complete final sources, partial copies and destination collisions must not corrupt history (Task2).
- Terminal2160 must not train beyond the budget; bounded GPU proof must not lower configured schedule (Task3).

### Task 1: Validate and restore matched resume state

**Files:** Create `src/stage1/matched_resume.py`, `tests/stage1/test_matched_resume.py`; modify `src/stage1/matched_execution.py`, `src/stage1/composition.py`, `src/stage1/trainer.py` and necessary worker arguments.

**Interfaces:** Produces saved matched resume contract, legacy-profile import from original launch/resolved records, strict profile validation and canonical inheritance arguments. Task2 consumes these for preparing the copied checkpoint.

- [ ] Write causal tests for currently rejected compatible matched resume, legacy/profile drift refusal, preserved source markers, and actual uninterrupted-vs-split native clean20/EMA continuation across a non-divisible accumulation epoch.
- [ ] Run focused tests, observe missing-feature failures (not fixture/import errors).
- [ ] Implement the smallest contract and use it at both entrypoints while retaining native state restoration.
- [ ] Run `PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_matched_resume.py tests/stage1/test_matched_execution.py tests/stage1/test_entropy_ce.py`; expected all pass.
- [ ] Commit Task1 with evidence ledger.

### Task 2: Prepare selected periodic source into a fresh output

**Files:** Create `src/stage1/resume_preparation.py`, `tests/stage1/test_resume_preparation.py`; modify bounded coordinator routing in `src/multinode.py` only as needed.

**Interfaces:** Consumes Task1 profile/contract validation. Produces a complete fresh destination checkpoint and explicit source/destination identity plus ready existing-coordinator arguments; preparation itself does not train.

- [ ] Write failing real-file tests for split-rank assembly, already assembled source, successful supervisor requirement, complete-state validation, path/collision rejection and source byte preservation.
- [ ] Implement using existing manifest/transfer/classification primitives; retain truthful source commit distinct from reader/destination identity.
- [ ] Run preparation tests and relevant existing multinode artifact tests; expected all pass.
- [ ] Commit Task2 with evidence ledger.

### Task 3: Expose and verify the usable resume path

**Files:** Modify existing coordinator/worker CLI and operator documentation; add focused integration tests and the minimal fixed-budget stop control for a bounded continuation witness.

**Interfaces:** Consumes Tasks1/2 ready checkpoint and argument profile. Produces actual existing-coordinator resume, terminal no-op handling and a bounded GPU proof invocation.

- [ ] Write failing CLI/integration tests including epoch-boundary continuation, budget preservation, terminal no-op and stop-after-save semantics.
- [ ] Implement minimal wiring and concise operator instructions. Keep original source/runtime untouched.
- [ ] Run focused covering tests; commit changes and request one independent specification/quality review of full diff.
- [ ] Resolve critical/important findings in one Native causal RED->GREEN pass; record deferred minors/rulings.
- [ ] Use fresh full project suite once for verification-before-completion and applicable branch finish gate; additional claims get their own bounded witnesses.
- [ ] When current train/eval releases GPUs, dry-run and preflight both nodes, prepare copied selected checkpoint, and run one existing-coordinator two-node/four-rank fixed2160-budget continuation witness. Validate full state restore/globalstep advance and original source preservation; no full duplicate training.
