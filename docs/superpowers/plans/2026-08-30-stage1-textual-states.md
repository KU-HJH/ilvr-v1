# Stage 1 Textual-State History Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage 1 query history selectable between the repository-compatible `previous` behavior and paper-faithful equation (4) `all` behavior.

**Architecture:** Add one validated mode to `Stage1TeacherTargetBuilder`. Keep Git's original per-block query calculation as the `previous` branch and use running sums/counts for the paper's separately averaged textual and latent histories in the `all` branch; thread the mode through the existing CLI, launcher, composition, and trainer path without changing model execution.

**Tech Stack:** Python 3.11, PyTorch, pytest, argparse, Bash, Transformers/TRL trainer composition

**Spec:** `docs/superpowers/specs/2026-08-30-stage1-textual-states-design.md`

## Global Constraints

- Run every Python and pytest command through `/root/workspace/ILVR/venv/ilvr-venv`.
- Run from `/root/workspace/ILVR`, addressing the target as `.worktrees/vcsd-ilvr-stage2`.
- Preserve all unrelated dirty-worktree changes; never restore or stage a whole overlapping file blindly.
- Do not launch long training.
- Do not change teacher/student visual or language forward structure.
- Do not create model artifacts; any future model state belongs below `./checkpoints/<run-id>/`, results below `./results/<run-id>/`, and logs below `./logs/<run-id>/` in the worktree.

---

### Task 1: Preserve repository queries and add the paper-faithful query mode

**Files:**
- Modify: `src/stage1/targets.py`
- Modify: `tests/stage1/test_targets.py`

**Interfaces:**
- Consumes: `Stage1TeacherTargetBuilder(tokenizer, coverage_p=0.9, image_pool_k=8, helper_group_L=256)` and Git's pre-change query loop.
- Produces: `Stage1TeacherTargetBuilder(..., textual_states: str = "previous")`, with `textual_states` restricted to `"all"` or `"previous"`.

- [ ] **Step 1: Make both mode witnesses fail against the current production API**

  Update the existing three-block tests so construction is explicit:

  ```python
  builder = PreviousQueryBuilder(textual_states="previous")
  builder = QueryRecordingBuilder(textual_states="all")
  ```

  Have the test subclasses accept `textual_states` and pass it into
  `super().__init__`. Keep literal expected query rows. The `previous` test must
  expect Git's original local-text and immediately-previous-selected-mean
  values; the `all` test must expect ordered component rows
  `[global_intent, cumulative_text_mean, cumulative_latent_mean]`, a final query
  shape of `(1, 2)`, and the hand-derived third query. Both tests retain exact
  teacher language/original visual/helper visual call-count assertions.

- [ ] **Step 2: Run the two tests and verify RED**

  Run:

  ```bash
  cd .worktrees/vcsd-ilvr-stage2 && PYTHONPATH=transformers/src:src /root/workspace/ILVR/venv/ilvr-venv/bin/pytest -q \
    tests/stage1/test_targets.py::test_previous_textual_states_matches_repository_queries \
    tests/stage1/test_targets.py::test_all_textual_states_averages_all_text_and_prior_latent_rows_in_order
  ```

  Expected: both fail because the constructor does not accept
  `textual_states`; a failure caused only by a misspelled test or fixture is not
  valid RED.

- [ ] **Step 3: Implement the smallest two-mode query branch**

  In `Stage1TeacherTargetBuilder.__init__`, validate and store the mode:

  ```python
  if textual_states not in ("all", "previous"):
      raise ValueError("textual_states must be 'all' or 'previous'")
  self.textual_states = textual_states
  ```

  Restore Git's `_assistant_start` behavior for the compatible path. Compute the
  longest marker end only for `all`, then choose the initial text-span boundary
  with:

  ```python
  rationale_start = (
      assistant_end if self.textual_states == "all" else assistant_start
  )
  ```

  In the block loop, `previous` uses the current local span mean and
  `previous_selected_mean`. `all` updates running text sums/counts before query
  construction and uses the running latent sums/counts from completed blocks.
  Both branches call `_reasoning_query(global_intent, textual_history,
  latent_history)`, which averages its two or three `[1, H]` components. Update
  `previous_selected_mean` after every non-empty selection; update cumulative
  latent sums/counts only in `all`.

- [ ] **Step 4: Run the focused target suite and verify GREEN**

  Run:

  ```bash
  cd .worktrees/vcsd-ilvr-stage2 && PYTHONPATH=transformers/src:src /root/workspace/ILVR/venv/ilvr-venv/bin/pytest -q tests/stage1/test_targets.py
  ```

  Expected: all target tests pass, including the two runtime query witnesses and
  the unchanged one-language/one-helper-visual call-count test.

- [ ] **Step 5: Commit only Task 1-owned hunks**

  Stage only the plan/spec plus Task 1-owned hunks in `targets.py` and
  `test_targets.py`; inspect `git diff --cached` before committing. Do not stage
  any unrelated dirty file or hunk.

  ```bash
  git commit -m "feat(stage1): select textual query history"
  ```

### Task 2: Thread the mode through every supported Stage 1 launch path

**Files:**
- Modify: `src/utils_deepseed.py`
- Modify: `src/stage1/composition.py`
- Modify: `src/stage1/trainer.py`
- Modify: `run_stage1_training.sh`
- Modify: `tests/stage1/test_cli.py`
- Modify: `tests/stage1/test_composition.py`
- Modify: `tests/stage1/test_integration.py`
- Modify: `tests/stage1/test_launcher.py`

**Interfaces:**
- Consumes: `Stage1TeacherTargetBuilder(..., textual_states="previous")` from Task 1.
- Produces: CLI option `--textual_states {all,previous}`, launcher forwarding, and `CustomTrainerStage1(..., textual_states="previous")`.

- [ ] **Step 1: Add failing CLI, composition, and launcher routing tests**

  Add a parser test that asserts omission yields `previous`, explicit `all`
  yields `all`, and an invalid value raises `SystemExit`. Extend a composition
  test to build with `--textual_states all` and assert
  `bundle.trainer.kwargs["textual_states"] == "all"`. Extend the launcher dry-run
  test to assert its default command includes `--textual_states previous`, and
  add an explicit `--textual_states all` dry-run assertion. Add a tiny
  composition test using the real `CustomTrainerStage1` and existing tiny
  dependencies, without calling `train()`, and assert
  `components.trainer.target_builder.textual_states == "all"`.

- [ ] **Step 2: Run routing tests and verify RED**

  Run:

  ```bash
  cd .worktrees/vcsd-ilvr-stage2 && PYTHONPATH=transformers/src:src /root/workspace/ILVR/venv/ilvr-venv/bin/pytest -q \
    tests/stage1/test_cli.py \
    tests/stage1/test_composition.py::test_textual_states_reaches_stage1_trainer \
    tests/stage1/test_integration.py::test_textual_states_reaches_actual_target_builder \
    tests/stage1/test_launcher.py::test_dora_dry_run_emits_exact_safe_two_process_command_without_mutation \
    tests/stage1/test_launcher.py::test_launcher_forwards_explicit_textual_states
  ```

  Expected: failures identify the missing parser argument, missing trainer
  kwarg, and missing launcher forwarding.

- [ ] **Step 3: Add exact runtime plumbing**

  Add to `parse_args`:

  ```python
  parser.add_argument(
      "--textual_states",
      choices=("all", "previous"),
      default="previous",
  )
  ```

  Add `textual_states: str = "previous"` to `CustomTrainerStage1`, store it, and
  pass it into `Stage1TeacherTargetBuilder`. Pass `args.textual_states` from
  composition. In `run_stage1_training.sh`, read
  `STAGE1_TEXTUAL_STATES` with default `previous`, accept the exact
  `--textual_states` option, validate `all|previous`, and append
  `--textual_states "$textual_states"` to the Python command.

- [ ] **Step 4: Run the routing tests and verify GREEN**

  Re-run exactly the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Run one minimal runtime routing probe**

  Run the launcher in `--dry-run` mode with `--textual_states all`, parse the
  emitted command and parse that command's Stage 1 arguments. Pair this with the
  real tiny-trainer integration witness from Step 1 to prove the parsed value
  reaches an actual target builder. This probe performs no large model load and
  writes no checkpoint, result, or log artifact.

- [ ] **Step 6: Commit only Task 2-owned hunks**

  Stage clean files directly and stage only the new `textual_states` hunks from
  the already-dirty launcher. Inspect `git diff --cached` before committing.

  ```bash
  git commit -m "feat(stage1): route textual history mode"
  ```

### Task 3: Verify compatibility and execution-count invariants

**Files:**
- Verify only: `src/stage1/targets.py`
- Verify only: `src/stage1/trainer.py`
- Verify only: focused Stage 1 tests

**Interfaces:**
- Consumes: completed Task 1 and Task 2 behavior.
- Produces: fresh evidence for the completion report; no production changes unless a causal regression test first exposes a defect.

- [ ] **Step 1: Run the focused combined suite once**

  Run:

  ```bash
  cd .worktrees/vcsd-ilvr-stage2 && PYTHONPATH=transformers/src:src /root/workspace/ILVR/venv/ilvr-venv/bin/pytest -q \
    tests/stage1/test_targets.py tests/stage1/test_cli.py \
    tests/stage1/test_composition.py tests/stage1/test_launcher.py \
    tests/stage1/test_integration.py
  ```

  Expected: zero failures. Do not repeat this broad focused suite.

- [ ] **Step 2: Inspect the actual diff and runtime call path**

  Confirm from runtime witnesses, not declarations, that each mode creates the
  expected third query and that both modes retain one teacher language call,
  one original visual call, and one helper visual call. Confirm the trainer
  still removes helper tensors before its sole student model forward. Confirm
  no touched code branches on Full/LoRA/DoRA or ZeRO stage.

- [ ] **Step 3: Record verification without committing unrelated state**

  Record test counts, exact commands, observed query tensors, and remaining
  uncertainty in the SDD ledger. No 3-epoch run is authorized or required.
