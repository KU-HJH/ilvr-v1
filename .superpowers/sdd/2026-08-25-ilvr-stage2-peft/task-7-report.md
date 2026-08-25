# Task 7 Report: Launcher and User Documentation

## Status

Complete. The Stage 2 launcher now owns, validates, and conditionally forwards
the full/LoRA/DoRA tuning contract. The README documents fresh full tuning,
fresh LoRA/DoRA, PEFT resume, adapter artifacts, and base-plus-adapter
evaluation without claiming the Task 8 GPU validations have run.

Implementation commit: `e0451dfd289dd7fa487734ecf2f4a40d199580d6`

## TDD Evidence

### Baseline

Command:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py -q
```

Result before Task 7 tests:

```text
57 passed in 4.26s
```

### RED

The new tests were added before production changes and exercised launcher
dry-runs and exit behavior rather than inspecting shell source.

Command:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py -q
```

Result:

```text
38 failed, 56 passed in 4.81s
```

The expected failures showed missing fresh defaults and explicit PEFT values,
missing validation, an explicit resume method not being forwarded, and all ten
new managed-option syntax cases being accepted. The resume-with-omitted-method
test already passed because the old launcher emitted no tuning arguments; its
stronger assertion includes valid stray `LORA_*` variables and is protected by
the fresh/explicit tests that require tuning arguments in the other branches.

### GREEN

First implementation verification:

```text
94 passed in 4.93s
```

Final fresh verification command:

```bash
set -e
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py -q
bash -n run_stage2_training.sh
git diff --check
```

Result:

```text
94 passed in 4.70s
```

`bash -n` and `git diff --check` both exited 0 with no output.

## Implemented Behavior

- Fresh runs explicitly forward `full` plus rank 32, alpha 64, dropout 0.05,
  and the complete seven-projection decoder allowlist.
- Fresh DoRA settings and explicit LoRA resume settings are forwarded exactly,
  including safe target subsets and shell-preserved values.
- Resume with `TUNING_METHOD` truly absent omits all five tuning arguments, even
  when valid `LORA_*` variables are present, leaving the sidecar authoritative.
- Resume with an explicit method forwards the complete five-option set so
  checkpoint compatibility validation can reject any mismatch.
- Method, positive integer rank/alpha, finite `[0, 1)` dropout, nonempty exact
  decoder target names, and visual/out-of-scope targets fail before startup.
  Numeric validation invokes the repository venv Python by absolute repo path.
- All five tuning options are launcher-owned under separate and equals syntax;
  unmanaged quoted passthrough arguments remain unchanged.
- README examples distinguish fresh full, fresh LoRA, fresh DoRA, adapter
  resume, full evaluation, and base-plus-adapter evaluation. It documents
  numbered Trainer state, relocation-safe structural validation, frozen PEFT
  boundaries, the full-weight Stage 1 base requirement, no merge/export, and
  the rank-local `B == 1` restriction.
- LoRA and DoRA two-GPU ZeRO-3 save/resume are each reported as not yet run.

## Files Changed

- `run_stage2_training.sh`
- `tests/stage2/test_launcher.py`
- `tests/stage2/test_comt_readiness.py`
- `README.md`
- `.superpowers/sdd/2026-08-25-ilvr-stage2-peft/task-7-report.md`

## Self-Review

- Reviewed the complete scoped diff against every Task 7 brief item and the
  binding launcher policy.
- Confirmed explicitness is captured before defaulting and that the same
  `TUNING_ARGS` array controls dry-run and real execution.
- Confirmed every realistic launcher mutation is covered: missing/wrong
  defaults, dropping explicit settings, forwarding tuning on inherited resume,
  omitting tuning on explicit resume, accepting invalid values, or allowing a
  managed passthrough spelling.
- Confirmed documentation separates final adapter files from numbered Trainer
  checkpoint state and does not imply adapter merge or GPU verification.
- Confirmed only the four approved implementation files were included in the
  required implementation commit.
- External reviewer dispatch was skipped because this task explicitly forbids
  subagents; review was performed locally.

## Systematic Debugging Note

A cleanup attempt used `rm -rf` on two exact untracked test-generated
`__pycache__` directories and was rejected by command safety policy. Inspection
confirmed the directories contained only the listed bytecode files. They were
then cleaned with narrowly scoped `find -delete` and `rmdir`; no source or
tracked file was removed.

## Concerns

- Per the task boundary, no GPU, DeepSpeed, training, or evaluation job ran.
  Dedicated LoRA and DoRA ZeRO-3 save/resume verification remains Task 8 work.
- On an inherited-method resume, valid stray `LORA_*` values are ignored while
  malformed values still fail the launcher's global early validation. This
  preserves sidecar authority without allowing malformed environment input to
  pass silently.

## Review Fix Round 1

Review-fix commit: `61b3ea0c075b3a1ceac086164ab39b1cec09a5d3`

### Findings and root causes

- The launcher target loop checked every token against the decoder allowlist
  but did not retain a seen-set. Consequently `q_proj,q_proj` passed fresh
  launcher validation and, on inherited-method resume, was silently omitted
  with the rest of the tuning arguments instead of failing global validation.
- The documented LoRA/DoRA fresh commands enabled ZeRO-3, while the PEFT resume
  example omitted `DEEPSPEED_CONFIG`. That would construct a different Trainer
  backend and could not restore DeepSpeed optimizer/scheduler state.
- The checkpoint layout described ordinary Trainer root files as universal.
  Transformers delegates optimizer/scheduler saving to
  `DeepSpeedEngine.save_checkpoint` when DeepSpeed is enabled, producing
  backend-owned `global_step*` rank state instead.

### RED

Added a parameterized regression that executes the real launcher for both a
fresh explicit LoRA run and a resume with `TUNING_METHOD` absent:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_launcher.py::test_stage2_launcher_rejects_duplicate_lora_targets \
  -q
```

Result before the fix:

```text
2 failed in 0.10s
```

Both failures returned exit code 0 and demonstrated the exact missing
uniqueness guard.

### GREEN and final verification

The focused duplicate regression passed first:

```text
2 passed in 0.05s
```

Final focused command:

```bash
set -e
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py -q
bash -n run_stage2_training.sh
git diff --check
```

Result:

```text
96 passed in 4.77s
```

`bash -n` and `git diff --check` exited 0 without output.

### Corrections

- Added an associative seen-set after exact allowlist validation. Duplicate
  targets now fail with a clear launcher error while the original target order
  remains unchanged.
- Added the same `DEEPSPEED_CONFIG=configs/config_stage3.json` to the documented
  PEFT resume command and require the resume backend/configuration to match the
  checkpoint-producing run.
- Distinguished ordinary Trainer `optimizer.pt`/`scheduler.pt` files from
  DeepSpeed-owned `global_step*`, `*_model_states.pt`, and
  `*_optim_states.pt` layout, and instruct users to resume the complete
  numbered checkpoint directory.
- No GPU, DeepSpeed, training, or evaluation job was run for this correction.

### Self-review

- Both duplicate cases exercise launcher behavior rather than source text.
- Removing the seen-set or moving it behind tuning-argument omission makes the
  inherited-resume regression fail.
- Existing safe subsets still preserve caller order, and allowlist/error
  behavior is unchanged for empty, visual, and out-of-scope targets.
- Documentation now matches the Transformers DeepSpeed save branch inspected
  in this repository and does not promise root optimizer/scheduler files for
  ZeRO-3.
