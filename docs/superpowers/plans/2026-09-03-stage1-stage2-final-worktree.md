# Stage 1/2 Final Worktree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one compact `feat/stage1-stage2-final` worktree that preserves the final Stage 1 and Stage 2 training paths, runs without reading source or documentation from another worktree, and carries a self-contained runbook and provenance record.

**Architecture:** Start from the final `main` code baseline plus the approved design/plan commits, retain `main` as the authority for Stage 1 and shared multi-node behavior, and port only the missing Stage 1 textual-history and Stage 2 no-cache/manifest behavior from the pinned Stage 2 lineage. Keep runtime assets outside Git behind worktree-local links, make active runtime roots derive from the checkout, and prove the selective integration with focused tests before documentation and the full suite.

**Tech Stack:** Bash, Python 3.11, PyTorch 2.7.1 + CUDA 12.8, FlashAttention 2.8.3, Hugging Face Transformers/Trainer, PEFT, Accelerate, DeepSpeed ZeRO-1/2/3, pytest, Git worktrees.

**Spec:** `docs/superpowers/specs/2026-09-03-stage1-stage2-final-worktree-design.md`

## Global Constraints

- Work only on branch `feat/stage1-stage2-final` in `/root/workspace/ILVR/.worktrees/stage1-stage2-final` after Task 1 creates it.
- The code baseline is `85e74e97de3773375ff037690511153c8d1f40cc`; the branch also carries approved spec commit `108fd12` and this plan commit.
- The Stage 2 source is `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0`, with merge base `82d10d8d9ae26a288da42df18b0c5c655efb04ff`.
- The approved three-file dirty PEFT patch SHA-256 is `0729ecd05ed8950a7c1c8d2ca74b2ba64a48172484e4a7e0ce3570cabe636557`.
- Preserve the existing Stage 1 `main` behavior and all 18 Stage × Full/LoRA/DoRA × ZeRO-1/2/3 routing contracts.
- Do not merge the complete Stage 2 branch or copy a broad directory over `main`; integrate the pinned files and semantics listed below.
- Use only `./venv/ilvr-venv/bin/python` for Python and pytest in the new worktree.
- Model-bearing artifacts stay below `./checkpoints`; generated results stay below `./results`; logs stay below `./logs`.
- Do not add data, virtualenv contents, checkpoints, caches, logs, results, bytecode, editor settings, or unrelated untracked files to Git.
- Do not delete, move, overwrite, stage, or restore user-owned files in the primary or Stage 2 worktrees.
- Do not launch full training or evaluation. This plan verifies integration and runtime contracts, not numerical trajectory equivalence.
- If the integration first nail fails, stop later documentation work, identify the missing dependency, add one causal regression test, and retry with the smallest correction.
- If a pinned identity, target nonexistence check, or existing `/raid` asset
  validation fails, make no replacement or overwrite. Stop and report the exact
  command, observed value, and expected value to the human.
- All implementation edits use `apply_patch`; the fixed `git diff | git apply --3way` commands below are permitted only for the listed mechanical source-port patches that have already passed `--check`.

## File Structure and Responsibilities

- `run_stage1_training.sh`: canonical Stage 1 Full/LoRA/DoRA launcher, including textual-history routing and repository-local paths.
- `run_stage2_training.sh`: canonical Stage 2 launcher, preserving `main` multi-node topology while adding the immutable length-manifest and diagnostics controls.
- `src/stage1/targets.py`: Stage 1 `previous` versus `all` textual/latent query-history construction.
- `src/stage1/composition.py`, `src/stage1/trainer.py`, `src/utils_deepseed.py`: carry the selected Stage 1 history mode and Stage 2 manifest/diagnostics arguments through the runtime.
- `src/stage2/data.py`, `src/stage2/length_manifest.py`: prepare Stage 2 examples once and validate an immutable retained-index snapshot.
- `src/stage2/efficiency_probe.py`: bounded, non-training evidence for the no-cache execution profile.
- `src/stage2/trainer.py`: combine `main` checkpoint/evidence behavior with opt-in diagnostics and loss-only Stage 2 forwards.
- `src/main.py`: consume the sealed Stage 2 manifest without losing `main` PEFT, ZeRO-3, evidence, or evaluation behavior.
- `src/artifact_paths.py`: validate the Stage 2 manifest inside the same run result directory while preserving `main` nested evidence support.
- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`: project only supervised loss rows for Stage 2 loss-only forwards.
- `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`: optimized no-cache recurrence, prefix-sized embedding assembly, and opt-in diagnostics.
- `src/multinode*.py`, `run_stage1_batch_probe.sh`, `run_stage1_zero_matrix.sh`: derive active repository/artifact paths from the consolidated checkout.
- `docs/final-stage1-stage2/README.md`: sole operator runbook from Stage 1 through Stage 2 and evaluation.
- `docs/final-stage1-stage2/PROVENANCE.md`: source/blob inclusion manifest, selected/superseded document index, and evidence confidence levels.

---

### Task 1: Create the Isolated Worktree and Validate Runtime Assets

**Files:**
- Create worktree: `/root/workspace/ILVR/.worktrees/stage1-stage2-final`
- Create branch: `feat/stage1-stage2-final`
- Create ignored links: `data`, `venv`
- Validate tracked link: `checkpoints`
- Create ignored directories: `logs`, `results`

**Interfaces:**
- Consumes: baseline/spec/plan commits on `main`; pinned Stage 2 branch and dirty PEFT source.
- Produces: an isolated checkout whose `./venv/ilvr-venv/bin/python`, `./data`, and `./checkpoints` paths are usable by every later task.

- [ ] **Step 1: Verify the pinned source identities and dirty patch before creating anything**

Run from `/root/workspace/ILVR`:

```bash
test "$(git rev-parse 108fd12^1)" = 85e74e97de3773375ff037690511153c8d1f40cc
test "$(git rev-parse feat/vcsd-ilvr-stage2)" = b0ce19b086cd46961fa9dcc12aa3844f659ff0b0
test "$(git merge-base 85e74e97de3773375ff037690511153c8d1f40cc b0ce19b086cd46961fa9dcc12aa3844f659ff0b0)" = 82d10d8d9ae26a288da42df18b0c5c655efb04ff
test "$(git rev-list --count 85e74e97de3773375ff037690511153c8d1f40cc..main)" = 2
test "$(git diff --name-only 85e74e97de3773375ff037690511153c8d1f40cc...main | sort)" = "$(printf '%s\n' \
  docs/superpowers/plans/2026-09-03-stage1-stage2-final-worktree.md \
  docs/superpowers/specs/2026-09-03-stage1-stage2-final-worktree-design.md | sort)"
test "$(git log -1 --format=%s main)" = \
  "docs: plan final stage1 stage2 worktree"
git -C .worktrees/vcsd-ilvr-stage2 diff --binary -- \
  src/stage2/checkpointing.py \
  src/stage2/trainer.py \
  tests/stage2/test_peft_checkpointing.py | sha256sum
git -C .worktrees/vcsd-ilvr-stage2 hash-object \
  src/stage2/checkpointing.py \
  src/stage2/trainer.py \
  tests/stage2/test_peft_checkpointing.py
```

Expected: the patch command prints
`0729ecd05ed8950a7c1c8d2ca74b2ba64a48172484e4a7e0ce3570cabe636557  -`,
and the final command prints, in order:

```text
ff90d2bd18a6fb4484516b3aede242d5e2e22246
c0c66059a56d75010dfb35ec47bd230952bffd5e
8c2d2164ad3c066465793663d358a5df3d5dcc32
```

The two-commit count and two-path diff are the immutable starting-tree witness:
baseline code, then exactly the approved spec and plan commits. If any identity
differs, stop without creating the worktree.

- [ ] **Step 2: Confirm the target branch and worktree do not already exist**

```bash
test ! -e /root/workspace/ILVR/.worktrees/stage1-stage2-final
! git show-ref --verify --quiet refs/heads/feat/stage1-stage2-final
```

Expected: both commands return zero.

- [ ] **Step 3: Create the branch and worktree through the worktree skill**

Invoke `superpowers:using-git-worktrees`, then run:

```bash
git worktree add \
  /root/workspace/ILVR/.worktrees/stage1-stage2-final \
  -b feat/stage1-stage2-final \
  main
```

Expected: the new branch contains spec commit `108fd12` and this plan commit.
All remaining steps run from the new worktree.

- [ ] **Step 4: Prepare the direct `/raid` data target without changing the source**

```bash
install -d /raid/workspace/ILVR
if [[ ! -e /raid/workspace/ILVR/data ]]; then
  cp -a --reflink=auto /root/workspace/ILVR/data /raid/workspace/ILVR/data
fi
test -f /raid/workspace/ILVR/data/TRAIN.jsonl
test -f /raid/workspace/ILVR/data/TEST.jsonl
test -d /raid/workspace/ILVR/data/images_comt
test "$(stat -c %s /raid/workspace/ILVR/data/TRAIN.jsonl)" = \
  "$(stat -c %s /root/workspace/ILVR/data/TRAIN.jsonl)"
test "$(stat -c %s /raid/workspace/ILVR/data/TEST.jsonl)" = \
  "$(stat -c %s /root/workspace/ILVR/data/TEST.jsonl)"
test "$(find /raid/workspace/ILVR/data/images_comt -type f -printf '%s\n' | \
  awk '{total += $1} END {print total + 0}')" = \
  "$(find /root/workspace/ILVR/data/images_comt -type f -printf '%s\n' | \
  awk '{total += $1} END {print total + 0}')"
```

Expected: all validations return zero; `/root/workspace/ILVR/data` remains intact.

- [ ] **Step 5: Prepare the direct `/raid` virtualenv target without changing the source**

```bash
install -d /raid/workspace/ILVR/venv
if [[ ! -e /raid/workspace/ILVR/venv/ilvr-venv ]]; then
  cp -a --reflink=auto \
    /root/workspace/ILVR/venv/ilvr-venv \
    /raid/workspace/ILVR/venv/ilvr-venv
fi
test -x /raid/workspace/ILVR/venv/ilvr-venv/bin/python
```

Expected: the copied Python exists and the original virtualenv remains intact.

- [ ] **Step 6: Create the worktree-local runtime layout**

```bash
ln -s /raid/workspace/ILVR/data data
ln -s /raid/workspace/ILVR/venv venv
test "$(readlink checkpoints)" = /raid/workspace/ILVR/checkpoints/
test -d "$(readlink -f checkpoints)"
checkpoint_probe=$(realpath -m checkpoints/final-worktree-preflight)
[[ "$checkpoint_probe" == /raid/workspace/ILVR/checkpoints/* ]]
mkdir -p logs results
```

Expected:

```text
data -> /raid/workspace/ILVR/data
venv -> /raid/workspace/ILVR/venv
checkpoints -> /raid/workspace/ILVR/checkpoints/
```

- [ ] **Step 7: Validate the repository environment through the required logical path**

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python - <<'PY'
import platform
import torch
import deepspeed
import flash_attn
import transformers

print(platform.python_version())
print(torch.__version__, torch.version.cuda)
print(deepspeed.__version__)
print(flash_attn.__version__)
print(transformers.__version__)
assert platform.python_version().startswith("3.11.")
assert torch.__version__.startswith("2.7.1")
assert torch.version.cuda == "12.8"
assert flash_attn.__version__ == "2.8.3"
assert "stage1-stage2-final/transformers/src" in transformers.__file__
PY
```

Expected: every assertion passes and local Transformers resolves inside this worktree.

---

### Task 2: Preserve the Final Stage 1 Textual-History Mode on the Main Baseline

**Files:**
- Modify: `run_stage1_training.sh`
- Modify: `src/stage1/targets.py`
- Modify: `src/stage1/composition.py`
- Modify: `src/stage1/trainer.py`
- Modify: `src/utils_deepseed.py`
- Modify: `tests/stage1/test_targets.py`
- Modify: `tests/stage1/test_cli.py`
- Modify: `tests/stage1/test_composition.py`
- Modify: `tests/stage1/test_integration.py`
- Modify: `tests/stage1/test_launcher.py`

**Interfaces:**
- Consumes: Stage 1 `main` true-batch/EMA code and the behavior specified by commits `ee16f81` and `ca30ab9`.
- Produces: `Stage1TeacherTargetBuilder(..., textual_states: Literal["previous", "all"] = "previous")`; CLI/launcher option `--textual_states`; unchanged `previous` default behavior.

- [ ] **Step 1: Record and inspect the two fixed Stage 1 source patches**

```bash
git diff --binary \
  82d10d8d9ae26a288da42df18b0c5c655efb04ff..ca30ab92e54e0c41fafaacc221a5120e0be8e6a5 \
  -- tests/stage1/test_targets.py tests/stage1/test_cli.py \
     tests/stage1/test_composition.py tests/stage1/test_integration.py \
     tests/stage1/test_launcher.py | sha256sum
git diff --binary \
  82d10d8d9ae26a288da42df18b0c5c655efb04ff..ca30ab92e54e0c41fafaacc221a5120e0be8e6a5 \
  -- run_stage1_training.sh src/stage1/targets.py \
     src/stage1/composition.py src/stage1/trainer.py src/utils_deepseed.py | sha256sum
```

Expected hashes, in order:

```text
511796ae812d6df0a07495fbbaa95d6cd7efc30e6d20da5f3bae51c05dbb3a89  -
92611aeaafe873f7fb10b6a9dba439157334ab2ca289e0e497dd3ee6d70b6ec6  -
```

- [ ] **Step 2: Add the focused Stage 1 tests before production changes**

Use `apply_patch` to port these exact behaviors from the pinned commits while retaining every existing `main` test:

```python
# tests/stage1/test_cli.py
def test_textual_states_parser_defaults_accepts_all_and_rejects_invalid_values():
    assert parse_args(["--stage", "stage1"]).textual_states == "previous"
    assert parse_args(
        ["--stage", "stage1", "--textual_states", "all"]
    ).textual_states == "all"
    with pytest.raises(SystemExit):
        parse_args(["--stage", "stage1", "--textual_states", "invalid"])
```

Also port the named tests without replacing nearby final-main coverage:

```text
test_text_history_starts_after_the_longest_assistant_marker
test_previous_textual_states_matches_repository_queries
test_all_textual_states_averages_all_text_and_prior_latent_rows_in_order
test_textual_states_reaches_stage1_trainer
test_textual_states_reaches_actual_target_builder
test_launcher_forwards_explicit_textual_states
```

- [ ] **Step 3: Run the new tests and verify the baseline fails for the intended reason**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/stage1/test_cli.py::test_textual_states_parser_defaults_accepts_all_and_rejects_invalid_values \
  tests/stage1/test_targets.py::test_text_history_starts_after_the_longest_assistant_marker \
  tests/stage1/test_targets.py::test_previous_textual_states_matches_repository_queries \
  tests/stage1/test_targets.py::test_all_textual_states_averages_all_text_and_prior_latent_rows_in_order \
  tests/stage1/test_composition.py::test_textual_states_reaches_stage1_trainer \
  tests/stage1/test_integration.py::test_textual_states_reaches_actual_target_builder \
  tests/stage1/test_launcher.py::test_launcher_forwards_explicit_textual_states
```

Expected: FAIL because `textual_states`, `_assistant_bounds`, or the launcher option is absent, not because of an import or fixture error.

- [ ] **Step 4: Implement the Stage 1 history mode without replacing main behavior**

Use `apply_patch` to make these exact interface changes:

```python
# src/utils_deepseed.py
parser.add_argument(
    "--textual_states",
    choices=("all", "previous"),
    default="previous",
)
```

```python
# src/stage1/targets.py
def __init__(
    self,
    tokenizer,
    coverage_p: float = 0.9,
    image_pool_k: int = 8,
    helper_group_L: int | None = 256,
    textual_states: str = "previous",
) -> None:
    if textual_states not in ("all", "previous"):
        raise ValueError("textual_states must be 'all' or 'previous'")
    self.textual_states = textual_states

@staticmethod
def _reasoning_query(global_intent, textual_history=None, latent_history=None):
    parts = [global_intent]
    if textual_history is not None:
        parts.append(textual_history)
    if latent_history is not None:
        parts.append(latent_history)
    return torch.cat(parts, dim=0).mean(dim=0, keepdim=True)
```

For `previous`, keep the final-main query exactly as global intent plus current
inter-block text plus the immediately previous selected-latent mean. For `all`,
start after the longest assistant marker, maintain cumulative text and selected-
latent sums/counts, and pass their means to `_reasoning_query`. Thread the value
through `build_stage1_training_components` and `CustomTrainerStage1`.

In `run_stage1_training.sh`, add:

```bash
textual_states=${STAGE1_TEXTUAL_STATES:-previous}
```

Accept `--textual_states all|previous`, reject other values, and append
`--textual_states "$textual_states"` to the Python command.

- [ ] **Step 5: Run the focused tests and the existing true-batch regression set**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/stage1/test_cli.py \
  tests/stage1/test_targets.py \
  tests/stage1/test_composition.py \
  tests/stage1/test_integration.py \
  tests/stage1/test_launcher.py \
  tests/stage1/test_trainer.py
```

Expected: PASS. The default remains `previous`; no true-batch or EMA regression fails.

- [ ] **Step 6: Commit the Stage 1 preservation unit**

```bash
git add run_stage1_training.sh src/stage1/targets.py \
  src/stage1/composition.py src/stage1/trainer.py src/utils_deepseed.py \
  tests/stage1/test_targets.py tests/stage1/test_cli.py \
  tests/stage1/test_composition.py tests/stage1/test_integration.py \
  tests/stage1/test_launcher.py
git commit -m "feat(stage1): preserve final textual history modes"
```

---

### Task 3: Port the Conflict-Free Stage 2 No-Cache Runtime and Tests

**Files:**
- Modify: `src/stage2/data.py`
- Create: `src/stage2/efficiency_probe.py`
- Create: `src/stage2/length_manifest.py`
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Modify: `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`
- Modify: `tests/stage2/test_comt_readiness.py`
- Create: `tests/stage2/test_efficiency_probe.py`
- Create: `tests/stage2/test_length_manifest.py`
- Modify: `tests/stage2/test_qwen_forward.py`

**Interfaces:**
- Consumes: fixed merge-base-to-Stage-2 test patch SHA-256 `86601e93...f4cbe1` and production patch SHA-256 `72aa6d04...fe745f`.
- Produces: `validate_stage2_length_manifest(...) -> Stage2LengthManifestSnapshot`,
  where `snapshot.manifest.retained_indices: tuple[int, ...]` and
  `snapshot.prepared_examples: tuple[Stage2PreparedExample, ...]`; the
  representative efficiency probe CLI; and Qwen Stage 2
  `stage2_loss_only`/opt-in-diagnostics behavior.

- [ ] **Step 1: Verify the exact test patch and apply only those tests**

```bash
git diff --binary \
  82d10d8d9ae26a288da42df18b0c5c655efb04ff..b0ce19b086cd46961fa9dcc12aa3844f659ff0b0 \
  -- tests/stage2/test_comt_readiness.py \
     tests/stage2/test_efficiency_probe.py \
     tests/stage2/test_length_manifest.py \
     tests/stage2/test_qwen_forward.py | sha256sum
git diff --binary \
  82d10d8d9ae26a288da42df18b0c5c655efb04ff..b0ce19b086cd46961fa9dcc12aa3844f659ff0b0 \
  -- tests/stage2/test_comt_readiness.py \
     tests/stage2/test_efficiency_probe.py \
     tests/stage2/test_length_manifest.py \
     tests/stage2/test_qwen_forward.py | git apply --3way -
```

Expected hash:
`86601e93f202a3679b87e59e8a115b2dfd9306640743df3d8a80b288d8f4cbe1  -`.
The apply must finish without conflict markers.

- [ ] **Step 2: Run the ported tests and verify the intended red state**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/stage2/test_comt_readiness.py \
  tests/stage2/test_efficiency_probe.py \
  tests/stage2/test_length_manifest.py \
  tests/stage2/test_qwen_forward.py
```

Expected: collection or assertion failures specifically for missing
`stage2.efficiency_probe`, `stage2.length_manifest`, `stage2_loss_only`, or the
optimized rollout contract.

- [ ] **Step 3: Verify and apply the exact production patch**

```bash
git diff --binary \
  82d10d8d9ae26a288da42df18b0c5c655efb04ff..b0ce19b086cd46961fa9dcc12aa3844f659ff0b0 \
  -- src/stage2/data.py src/stage2/efficiency_probe.py \
     src/stage2/length_manifest.py \
     transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
     transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py | sha256sum
git diff --binary \
  82d10d8d9ae26a288da42df18b0c5c655efb04ff..b0ce19b086cd46961fa9dcc12aa3844f659ff0b0 \
  -- src/stage2/data.py src/stage2/efficiency_probe.py \
     src/stage2/length_manifest.py \
     transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
     transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py | \
  git apply --3way -
```

Expected hash:
`72aa6d04660f2fbf0371cb203dfd622649a46a7378e8807e0027d79b25fe745f  -`.
The apply must retain the later `main` image-root fixes in `src/stage2/data.py`.

- [ ] **Step 4: Run the Stage 2 no-cache test unit**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/stage2/test_comt_readiness.py \
  tests/stage2/test_efficiency_probe.py \
  tests/stage2/test_length_manifest.py \
  tests/stage2/test_qwen_forward.py
```

Expected: PASS.

- [ ] **Step 5: Commit the conflict-free Stage 2 unit**

```bash
git add src/stage2/data.py src/stage2/efficiency_probe.py \
  src/stage2/length_manifest.py \
  transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py \
  tests/stage2/test_comt_readiness.py tests/stage2/test_efficiency_probe.py \
  tests/stage2/test_length_manifest.py tests/stage2/test_qwen_forward.py
git commit -m "perf(stage2): integrate final no-cache runtime"
```

---

### Task 4: Merge Stage 2 Manifest, Trainer, Launcher, and PEFT Contracts

**Files:**
- Modify: `run_stage2_training.sh`
- Modify: `src/artifact_paths.py`
- Modify: `src/main.py`
- Modify: `src/stage2/trainer.py`
- Modify: `src/utils_deepseed.py`
- Modify: `tests/stage2/test_launcher.py`
- Modify: `tests/stage2/test_routing.py`
- Modify: `tests/test_artifact_paths.py`
- Verify unchanged/subsuming: `src/stage2/peft.py`
- Verify unchanged/subsuming: `src/stage2/checkpointing.py`
- Verify unchanged/subsuming: `tests/stage2/test_peft.py`
- Verify unchanged/subsuming: `tests/stage2/test_peft_checkpointing.py`

**Interfaces:**
- Consumes: Task 3 `validate_stage2_length_manifest`, Stage 2 loss-only model API,
  fixed shared-test patch SHA-256 `44ae09c3...f0799b`, fixed shared-production
  patch SHA-256 `018c1a43...47d505`, and `main` multi-node/PEFT/evidence code.
- Produces: paired CLI fields `stage2_length_manifest`/`stage2_length_manifest_sha256`, `stage2_diagnostics_interval: int`, `dataloader_num_workers: int`, run-local manifest creation, and a Trainer that preserves both no-cache efficiency and adapter-only/evidence behavior.

- [ ] **Step 1: Add the shared integration tests before changing production**

First verify that the complete merge-base-to-Stage-2 diffs for this manual unit
are the reviewed inputs:

```bash
git diff --binary \
  82d10d8d9ae26a288da42df18b0c5c655efb04ff..b0ce19b086cd46961fa9dcc12aa3844f659ff0b0 \
  -- tests/stage2/test_launcher.py tests/stage2/test_routing.py \
     tests/test_artifact_paths.py | sha256sum
git diff --binary \
  82d10d8d9ae26a288da42df18b0c5c655efb04ff..b0ce19b086cd46961fa9dcc12aa3844f659ff0b0 \
  -- run_stage2_training.sh src/artifact_paths.py src/main.py \
     src/stage2/trainer.py src/utils_deepseed.py | sha256sum
```

Expected, in order:

```text
44ae09c3181751fea2ccaeeccf3e316094b93e050c53def92f92bb91b7f0799b  -
018c1a43e78136b7f36728c0d5ee0a0a183269c7a7cb5cf50b7f6b891747d505  -
```

Use `apply_patch` to add or update these exact contracts while retaining every
later-main test:

```python
# tests/test_artifact_paths.py
def test_generated_paths_accept_only_the_run_stage2_length_manifest(stage1_args):
    paths = resolve_run_artifact_paths("stage2-zero2-dora-final")
    stage1_args.save_model_path = str(paths.checkpoint_dir)
    stage1_args.log_file = str(paths.log_dir / "train.log")
    stage1_args.stage2_length_manifest = str(
        paths.result_dir / "stage2-length-manifest.json"
    )
    assert validate_generated_artifact_paths(stage1_args) == paths
    stage1_args.stage2_length_manifest = str(paths.result_dir / "other.json")
    with pytest.raises(ValueError, match="stage2_length_manifest"):
        validate_generated_artifact_paths(stage1_args)
```

Port the merge-base-to-Stage-2 additions in `tests/stage2/test_routing.py`,
including these named tests:

```text
test_stage2_trainer_requests_loss_only_when_outputs_are_not_returned
test_stage2_trainer_diagnostics_interval_schedules_microbatches
test_stage2_trainer_diagnostics_interval_zero_disables_diagnostics
test_stage2_diagnostics_interval_defaults_to_disabled
test_stage2_rejects_negative_diagnostics_interval
test_stage2_cli_rejects_incomplete_length_manifest_or_negative_dataloader_num_workers
test_stage2_length_manifest_rejects_wrong_sha_or_configuration
test_stage2_components_select_length_manifest_indices_without_remeasuring_images
test_stage2_manifest_indices_respect_max_train_samples
test_stage2_training_arguments_set_requested_dataloader_workers
```

Extend `test_stage2_launcher_resolves_repo_runtime_and_propagates_arguments` to
require `--stage2_length_manifest`, its 64-character digest,
`--stage2_diagnostics_interval 0`, and `--dataloader_num_workers 0` while retaining
the four-process multi-node assertions from `main`.

- [ ] **Step 2: Run the shared tests and verify the intended red state**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/test_artifact_paths.py::test_generated_paths_accept_only_the_run_stage2_length_manifest \
  tests/stage2/test_routing.py \
  tests/stage2/test_launcher.py::test_stage2_launcher_resolves_repo_runtime_and_propagates_arguments
```

Expected: FAIL because `main` does not yet validate/consume the Stage 2 manifest
or expose diagnostics and worker controls.

- [ ] **Step 3: Add only the missing parser fields**

Use `apply_patch` in `src/utils_deepseed.py`:

```python
parser.add_argument("--stage2_diagnostics_interval", type=int, default=0)
parser.add_argument("--stage2_length_manifest")
parser.add_argument("--stage2_length_manifest_sha256")
parser.add_argument("--dataloader_num_workers", type=int, default=0)
```

Reject a negative diagnostics interval or worker count during argument
normalization. Require the two manifest fields together, leaving both absent for
legacy/direct unit construction.

- [ ] **Step 4: Merge manifest ownership into `src/artifact_paths.py`**

Keep `main`'s checkpoint root, evaluation staging root, nested Stage 1 evidence,
and Stage 2 evidence behavior. Add:

```python
manifest = getattr(args, "stage2_length_manifest", None)
if manifest is not None:
    manifest_path = require_canonical_artifact_path(manifest, kind="result")
    expected_manifest = paths.result_dir / "stage2-length-manifest.json"
    if manifest_path != expected_manifest:
        raise ValueError(
            "stage2_length_manifest must use the run stage2-length-manifest.json"
        )
```

Add `--stage2-length-manifest` to the `validate-run` subcommand and forward it as
`stage2_length_manifest`. Do not replace the canonical roots with the older
branch's `zero-stage-compatibility` nesting.

- [ ] **Step 5: Merge immutable-manifest consumption into `src/main.py`**

Preserve `main`'s platform compatibility wrapper, Stage 2 PEFT ZeRO-3 guard,
producer evidence, inference, and evaluation imports. Add the branch behavior:

```python
manifest_path = getattr(args, "stage2_length_manifest", None)
manifest_sha256 = getattr(args, "stage2_length_manifest_sha256", None)
if bool(manifest_path) != bool(manifest_sha256):
    raise ValueError(
        "--stage2_length_manifest and --stage2_length_manifest_sha256 "
        "must be provided together"
    )
manifest_snapshot = (
    validate_stage2_length_manifest(
        path=manifest_path,
        expected_sha256=manifest_sha256,
        data_path=args.data_path,
        image_root=args.image_root,
        processor=processor,
        latent_size=decision.latent_size,
        supervise_latent_boundaries=decision.supervise_latent_boundaries,
        max_sequence_length=args.max_seq_length_train,
    )
    if manifest_path is not None
    else None
)
```

When the snapshot exists, select `manifest_snapshot.prepared_examples` by
`manifest_snapshot.manifest.retained_indices` without remeasuring image lengths.
Apply `max_train_samples` to `prepared_examples` before filtering indices, exactly
as follows, so retained indices always remain in the original manifest index
domain:

```python
prepared = (
    manifest_snapshot.prepared_examples
    if args.max_train_samples is None
    else manifest_snapshot.prepared_examples[: args.max_train_samples]
)
kept = [
    prepared[index]
    for index in manifest_snapshot.manifest.retained_indices
    if index < len(prepared)
]
```

Otherwise retain the existing direct prepare/filter path. Pass non-negative
`dataloader_num_workers` into `SFTConfig`, set
`dataloader_prefetch_factor=2` only when workers are positive, and pass
`stage2_diagnostics_interval` into `Stage2Trainer`.

- [ ] **Step 6: Merge loss-only diagnostics into `src/stage2/trainer.py`**

Preserve all current `main` callbacks, representative gradient evidence,
DeepSpeed checkpoint identity, sparse-row saves, and `_save(...,
save_embedding_layers=False)`. Add only:

```python
self.stage2_diagnostics_interval = stage2_diagnostics_interval
self._stage2_microbatch_index = 0

interval = self.stage2_diagnostics_interval
collect_diagnostics = (
    interval > 0 and self._stage2_microbatch_index % interval == 0
)
self._stage2_microbatch_index += 1
outputs = model(
    **inputs,
    stage2_loss_only=not return_outputs,
    collect_stage2_diagnostics=collect_diagnostics,
)
```

The constructor parameter is `stage2_diagnostics_interval: int = 0`. Do not
replace the final-main `_save`, `training_step`, or evidence callbacks with their
older Stage 2 branch versions.

- [ ] **Step 7: Merge manifest generation into `run_stage2_training.sh`**

Preserve all final-main `ILVR_NUM_MACHINES`, `ILVR_NUM_PROCESSES`, machine-rank,
rendezvous, socket-interface, DeepSpeed, and `BENCHMARK_EVIDENCE_PATH` behavior.
Use repository-local roots:

```bash
canonical_log_root="$REPO_ROOT/logs/zero-stage-compatibility"
result_dir="$REPO_ROOT/results/$experiment_id"
length_manifest="$result_dir/stage2-length-manifest.json"
processor_source=${RESUME_CHECKPOINT:-$STAGE1_CHECKPOINT}
STAGE2_DIAGNOSTICS_INTERVAL=${STAGE2_DIAGNOSTICS_INTERVAL:-0}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
```

For a fresh non-dry run, exclusively create `result_dir`, invoke
`python -m stage2.length_manifest` with the selected data, image root, processor,
latent size, boundary flag, and max sequence length, then SHA-256 the result. For
resume, require and validate the existing manifest. For dry-run, pass the intended
manifest path and 64 zeroes without filesystem mutation. Append both manifest
arguments, diagnostics interval, and worker count to `TRAINING_ARGS` while retaining
the final-main topology array.

- [ ] **Step 8: Prove that main already subsumes the dirty PEFT patch**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/stage2/test_peft.py::test_stage2_peft_accepts_optimized_target_selectors_for_multilayer_decoder \
  tests/stage2/test_peft.py::test_stage2_peft_rejects_unmatched_compact_target_selector \
  tests/stage2/test_peft_checkpointing.py::test_relocated_stage1_base_accepts_compact_saved_target_selectors \
  tests/stage2/test_peft_checkpointing.py::test_stage2_trainer_peft_save_excludes_resized_base_embeddings
```

Expected: PASS without applying the older dirty code over `main`. This is the
semantic preservation witness for the dirty patch; record it as “subsumed by the
main baseline” in provenance.

- [ ] **Step 9: Run the integration first nail**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/stage1/test_cli.py tests/stage1/test_targets.py \
  tests/stage1/test_composition.py tests/stage1/test_integration.py \
  tests/stage1/test_launcher.py tests/stage1/test_trainer.py \
  tests/stage2/test_comt_readiness.py tests/stage2/test_data.py \
  tests/stage2/test_efficiency_probe.py tests/stage2/test_length_manifest.py \
  tests/stage2/test_rollout.py tests/stage2/test_checkpointing.py \
  tests/stage2/test_peft.py tests/stage2/test_peft_checkpointing.py \
  tests/stage2/test_qwen_forward.py tests/stage2/test_routing.py \
  tests/stage2/test_launcher.py tests/test_artifact_paths.py
```

Expected: PASS. This proves only the selected integration surface, not full-run
numerical equivalence. If it fails, do not start Task 5 or Task 6.

- [ ] **Step 10: Commit the Stage 2 shared integration unit**

```bash
git add run_stage2_training.sh src/artifact_paths.py src/main.py \
  src/stage2/trainer.py src/utils_deepseed.py \
  tests/stage2/test_launcher.py tests/stage2/test_routing.py \
  tests/test_artifact_paths.py
git commit -m "feat(stage2): integrate final manifest training path"
```

---

### Task 5: Make Active Training and Multi-Node Paths Worktree-Local

**Files:**
- Modify: `run_stage1_training.sh`
- Modify: `run_stage2_training.sh`
- Modify: `run_stage1_batch_probe.sh`
- Modify: `run_stage1_zero_matrix.sh`
- Modify: `src/artifact_paths.py`
- Modify: `src/multinode.py`
- Modify: `src/multinode_artifacts.py`
- Modify: `src/multinode_capacity.py`
- Modify: `src/multinode_evidence.py`
- Modify: `src/multinode_results.py`
- Modify: `src/multinode_runtime.py`
- Modify: `tests/test_artifact_paths.py`
- Modify: `tests/stage1/test_launcher.py`
- Modify: `tests/stage2/test_launcher.py`
- Modify: `tests/multinode/test_topology.py`
- Modify: `tests/multinode/test_preflight.py`
- Modify: `tests/multinode/test_coordinator.py`
- Modify: `tests/multinode/test_lifecycle_matrix.py`
- Modify: `tests/multinode/test_artifacts.py`
- Modify: `tests/multinode/test_capacity.py`
- Modify: `tests/multinode/test_result_validation.py`

**Interfaces:**
- Consumes: current file location and the fixed remote base mapping `/root/workspace/ILVR` ↔ `/home/korea/Workspace/hjh/workspace/ILVR`.
- Produces: `Path(__file__).resolve().parents[1]` repository roots in Python and script-directory roots in Bash; remote host paths preserve the relative `.worktrees/stage1-stage2-final` suffix.

- [ ] **Step 1: Add worktree-root assertions before production changes**

In every listed test module that validates a default repository path, define:

```python
REPO_ROOT = Path(__file__).resolve().parents[2]  # tests/multinode/*.py
```

Use `Path(__file__).resolve().parents[1]` in `tests/test_artifact_paths.py` and
`tests/stage1|stage2/test_launcher.py`. Add these focused assertions:

```python
def test_canonical_repository_root_is_the_active_checkout():
    assert artifact_paths.CANONICAL_REPOSITORY_ROOT == REPO_ROOT

def test_default_topology_preserves_the_worktree_suffix_on_remote_host():
    spec = ClusterSpec.default()
    assert spec.remote_repo_root == REPO_ROOT
    relative = REPO_ROOT.relative_to(Path("/root/workspace/ILVR"))
    assert spec.remote_host_repo_root == (
        Path("/home/korea/Workspace/hjh/workspace/ILVR") / relative
    )
```

Replace literal-root expectations only where they assert a module-generated
default. Keep literal paths that intentionally represent malformed, foreign, or
user-supplied fixtures.

- [ ] **Step 2: Run the path-focused tests and verify they fail against hardcoded roots**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/test_artifact_paths.py::test_canonical_repository_root_is_the_active_checkout \
  tests/multinode/test_topology.py::test_default_topology_preserves_the_worktree_suffix_on_remote_host \
  tests/stage1/test_launcher.py::test_dry_run_derives_canonical_log_from_checkpoint_experiment_id \
  tests/stage2/test_launcher.py::test_stage2_launcher_resolves_repo_runtime_and_propagates_arguments
```

Expected: FAIL with paths rooted at `/root/workspace/ILVR` instead of the final worktree.

- [ ] **Step 3: Derive Python repository roots from each production module**

Use `apply_patch` to replace only active fixed roots:

```python
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
```

In `src/artifact_paths.py`, keep the public name:

```python
CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
```

Derive checkpoint, result, log, evaluation-staging, capacity, and evidence roots
from those repository constants. Replace fixed-path error messages with f-strings
that print the computed root. In `src/multinode_runtime.py`, construct the minimal
PATH as:

```python
"PATH": (
    f"{_REPOSITORY_ROOT / 'venv/ilvr-venv/bin'}:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
),
```

- [ ] **Step 4: Preserve the worktree suffix in the fixed remote topology**

In `src/multinode.py`, define:

```python
_REMOTE_CONTAINER_REPOSITORY_BASE = Path("/root/workspace/ILVR")
_REMOTE_HOST_REPOSITORY_BASE = Path(
    "/home/korea/Workspace/hjh/workspace/ILVR"
)

def _remote_host_repository_root(repository_root: Path) -> Path:
    relative = repository_root.relative_to(_REMOTE_CONTAINER_REPOSITORY_BASE)
    return _REMOTE_HOST_REPOSITORY_BASE / relative
```

Then make `ClusterSpec.default()` use `_REPOSITORY_ROOT` for
`remote_repo_root` and `_remote_host_repository_root(_REPOSITORY_ROOT)` for
`remote_host_repo_root`. The fixed host, container, `172.16.11.41`, and
`bond0.11` values do not change.

- [ ] **Step 5: Derive every active Bash artifact root from its script root**

In the four listed scripts, use the already computed `REPO_ROOT` or add:

```bash
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
canonical_checkpoint_root="$REPO_ROOT/checkpoints"
canonical_result_root="$REPO_ROOT/results"
canonical_log_root="$REPO_ROOT/logs/zero-stage-compatibility"
```

Do not change experiment IDs, ownership checks, DeepSpeed arguments, or output
collision behavior.

- [ ] **Step 6: Run all path and multi-node unit tests**

```bash
./venv/ilvr-venv/bin/python -m pytest -q \
  tests/test_artifact_paths.py \
  tests/stage1/test_launcher.py tests/stage2/test_launcher.py \
  tests/multinode/test_topology.py tests/multinode/test_preflight.py \
  tests/multinode/test_coordinator.py tests/multinode/test_lifecycle_matrix.py \
  tests/multinode/test_artifacts.py tests/multinode/test_capacity.py \
  tests/multinode/test_result_validation.py
```

Expected: PASS. All default-generated paths use the consolidated worktree; fixed
foreign-input fixtures keep their literal values.

- [ ] **Step 7: Run a direct active-source independence scan**

```bash
! rg -n '/root/workspace/ILVR/\.worktrees/(?!stage1-stage2-final)' --pcre2 \
  run_stage1_training.sh run_stage2_training.sh \
  run_stage1_batch_probe.sh run_stage1_zero_matrix.sh src tests
! rg -n "(canonical_(checkpoint|result|log)_root|_REPOSITORY_ROOT)=(['\"])?/root/workspace/ILVR" \
  run_stage1_training.sh run_stage2_training.sh \
  run_stage1_batch_probe.sh run_stage1_zero_matrix.sh src
```

Expected: both negated searches return zero.

- [ ] **Step 8: Commit the worktree-local path unit**

```bash
git add run_stage1_training.sh run_stage2_training.sh \
  run_stage1_batch_probe.sh run_stage1_zero_matrix.sh \
  src/artifact_paths.py src/multinode.py src/multinode_artifacts.py \
  src/multinode_capacity.py src/multinode_evidence.py \
  src/multinode_results.py src/multinode_runtime.py \
  tests/test_artifact_paths.py tests/stage1/test_launcher.py \
  tests/stage2/test_launcher.py tests/multinode/test_topology.py \
  tests/multinode/test_preflight.py tests/multinode/test_coordinator.py \
  tests/multinode/test_lifecycle_matrix.py tests/multinode/test_artifacts.py \
  tests/multinode/test_capacity.py tests/multinode/test_result_validation.py
git commit -m "fix: derive training paths from active worktree"
```

---

### Task 6: Curate the Canonical Runbook and Provenance Manifest

**Files:**
- Modify: `README.md`
- Create: `docs/final-stage1-stage2/README.md`
- Create: `docs/final-stage1-stage2/PROVENANCE.md`
- Create from pinned source: `docs/superpowers/specs/2026-08-30-stage1-textual-states-design.md`
- Create from pinned source: `docs/superpowers/plans/2026-08-30-stage1-textual-states.md`
- Create from pinned source: `docs/superpowers/specs/2026-09-02-stage2-efficiency-design.md`
- Create from pinned source: `docs/superpowers/plans/2026-09-02-stage2-no-cache-efficiency.md`

**Interfaces:**
- Consumes: verified Tasks 2–5 behavior, final Git blobs, existing local training artifacts, and the approved spec.
- Produces: one entry-point runbook and one auditable source/evidence manifest; no link to another worktree.

- [ ] **Step 1: Restore only the four implemented lineage documents absent from main**

```bash
git restore --source=b0ce19b086cd46961fa9dcc12aa3844f659ff0b0 -- \
  docs/superpowers/specs/2026-08-30-stage1-textual-states-design.md \
  docs/superpowers/plans/2026-08-30-stage1-textual-states.md \
  docs/superpowers/specs/2026-09-02-stage2-efficiency-design.md \
  docs/superpowers/plans/2026-09-02-stage2-no-cache-efficiency.md
```

Expected: only those four documents are added. Do not restore the segmented-KV
plan or the unimplemented multi-node sampler proposal.

- [ ] **Step 2: Write the canonical runbook with executable repository-local commands**

Create `docs/final-stage1-stage2/README.md` with these sections in this order:

```text
Scope and confidence
Repository/runtime layout
Supported Full/LoRA/DoRA and ZeRO routing
Stage 1 role and outputs
Stage 1 single-node launch
Stage 1 two-node/four-rank launch
Stage 1 checkpoint handoff
Stage 2 role and outputs
Stage 2 single-node launch
Stage 2 two-node/four-rank ZeRO-2 launch at lr=5e-5
Resume
Evaluation
Checkpoint contract
Selected design and implementation documents
Known limits
```

The routing section must name all 18 supported Stage × method × ZeRO
combinations as the Cartesian product of `stage1|stage2`, `full|lora|dora`, and
`zero1|zero2|zero3`. It must map ZeRO-1/2/3 to
`configs/deepspeed_zero1.json`, `configs/deepspeed_zero2.json`, and
`configs/deepspeed_zero3.json`, and state the existing Stage 2 PEFT ZeRO-3
specialized config where the launcher selects it. It must explain that support
means routing/checkpoint-contract preservation, not 18 new full training runs.

The Stage 1 examples must explicitly set `--method`, `--model`, `--data-path`,
`--image-root`, `--output-dir`, `--learning-rate`, `--deepspeed-config`, and
`--log-file`. The two-node example must set all six `ILVR_*` topology variables,
use rank 0 on `korea@172.16.11.41` and rank 1 on this host, and use the same run ID,
port, final commit, and logical worktree path on both nodes. The single-node
example must omit every `ILVR_*` variable and set `STAGE1_NUM_PROCESSES` to the
number of visible local GPUs.

Use this complete single-node Stage 1 command shape; the multi-node rank blocks
repeat it with the six topology variables and the same `run_id`:

```bash
cd "$(git rev-parse --show-toplevel)"
run_id="stage1-dora-z1-$(date +%Y%m%d-%H%M%S)"
STAGE1_NUM_PROCESSES=2 ./run_stage1_training.sh \
  --method dora \
  --model "$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct" \
  --data-path "$PWD/data/TRAIN.jsonl" \
  --image-root "$PWD/data" \
  --output-dir "$PWD/checkpoints/$run_id" \
  --learning-rate 2.5e-5 \
  --deepspeed-config "$PWD/configs/deepspeed_zero1.json" \
  --log-file "$PWD/logs/zero-stage-compatibility/$run_id/train.log"
```

The handoff section must distinguish a complete numbered Stage 1 checkpoint
from an evaluation-prepared copy, require identity/sidecar validation before
resume, and pass the selected Stage 1 artifact to Stage 2 only through
`STAGE1_CHECKPOINT`. The resume section must repeat the appropriate launcher
command with `STAGE1_RESUME_PATH` for Stage 1 or `RESUME_CHECKPOINT` for Stage 2;
it must state that ZeRO stage, world size, optimizer/scheduler/global step,
rank-local RNG state, and Stage-specific sidecars must match and are never
converted across ZeRO stages.

The canonical Stage 2 example must be directly runnable after assigning a fresh
run ID:

```bash
cd "$(git rev-parse --show-toplevel)"
run_id="stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-$(date +%Y%m%d-%H%M%S)"
STAGE1_CHECKPOINT="$PWD/checkpoints/evaluation-staging/stage1-dora-z1-ema-frozen-b1-ga2-e1-lr2p5e5-egb8-20260902-054900/prepared-checkpoint-431" \
DATA_PATH="$PWD/data/TRAIN.jsonl" \
IMAGE_ROOT="$PWD/data" \
OUTPUT_DIR="$PWD/checkpoints/$run_id" \
LOG_FILE="$PWD/logs/zero-stage-compatibility/$run_id/train.log" \
TUNING_METHOD=dora \
DEEPSPEED_CONFIG="$PWD/configs/deepspeed_zero2.json" \
LEARNING_RATE=5e-5 \
GRADIENT_ACCUMULATION_STEPS=2 \
GRADIENT_CHECKPOINTING=true \
ATTENTION_IMPLEMENTATION=flash_attention_2 \
ILVR_NUM_MACHINES=2 \
ILVR_NUM_PROCESSES=4 \
ILVR_MACHINE_RANK=0 \
ILVR_MAIN_PROCESS_IP=172.16.11.41 \
ILVR_MAIN_PROCESS_PORT=29520 \
ILVR_SOCKET_IFNAME=bond0.11 \
./run_stage2_training.sh
```

Document the local rank-1 command by repeating the complete block with only
`ILVR_MACHINE_RANK=1`; do not replace the block with “same as above.” Label the
rank-0 block for `korea@172.16.11.41` in container `llm_kd`, and label the rank-1
block for this host. State that both hosts must check out the same final commit at
the same logical container worktree path; this task does not deploy the remote
checkout.

Do not embed a future literal “final commit” SHA in the runbook. Give both
operators this preflight and require the printed values to match before launch:

```bash
git rev-parse HEAD
git status --short --untracked-files=no
```

The evaluation section must use only an evaluation-ready model under
`./checkpoints/evaluation-staging/<evaluation-id>/`, never copy a model into
`results/`, and show the checked-in `eval.py` interface with all paths rooted at
the active checkout:

```bash
cd "$(git rev-parse --show-toplevel)"
evaluation_id="stage2-evaluation-$(date +%Y%m%d-%H%M%S)"
model_dir="$PWD/checkpoints/evaluation-staging/$evaluation_id/local-model"
base_model_dir="$PWD/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct"
mkdir -p "$PWD/results/$evaluation_id" "$PWD/logs/$evaluation_id"
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python eval.py \
    --model_dir "$model_dir" \
    --base_model_dir "$base_model_dir" \
    --test_data_path "$PWD/data/TEST.jsonl" \
    --task_name comt \
    --output_json_path "$PWD/results/$evaluation_id/predictions.jsonl" \
    >"$PWD/logs/$evaluation_id/eval.log" 2>&1
```

State that `model_dir` must be populated by the repository's validated
evaluation-preparation path before this command; the locally partial
`checkpoint-846` inventory is historical evidence and is not advertised as an
evaluation-ready artifact.

- [ ] **Step 3: Add the root README entry point**

At the top of `README.md`, add a short “Final Stage 1/2 training” section linking
only by relative path:

```markdown
For the consolidated Stage 1 → Stage 2 training and evaluation workflow, start
with [the final training runbook](docs/final-stage1-stage2/README.md). Its
[provenance manifest](docs/final-stage1-stage2/PROVENANCE.md) records the exact
source and evidence boundaries.
```

Do not rewrite the historical README sections in this task.

- [ ] **Step 4: Write the file-by-file source provenance table**

Create `docs/final-stage1-stage2/PROVENANCE.md`. Record the fixed baseline,
Stage 2 source, merge base, Stage 1 test/production patch hashes, Stage 2
test/production patch hashes, and dirty PEFT patch hash from this plan.

Build the exact runtime inclusion list with:

```bash
{
  printf '%s\n' \
    run_stage1_training.sh run_stage2_training.sh \
    run_stage1_batch_probe.sh run_stage1_zero_matrix.sh \
    src/main.py src/artifact_paths.py src/utils_deepseed.py \
    src/multinode.py src/multinode_artifacts.py \
    src/multinode_capacity.py src/multinode_evidence.py \
    src/multinode_results.py src/multinode_runtime.py \
    configs/deepspeed_zero1.json configs/deepspeed_zero2.json \
    configs/deepspeed_zero3.json configs/deepspeed_zero3_stage2_peft.json
  git ls-files src/stage1 src/stage2 \
    transformers/src/transformers/models/qwen2_5_vl \
    tests/stage1 tests/stage2 tests/multinode tests/test_artifact_paths.py
} | sort -u > /tmp/ilvr-final-runtime-paths.txt
test -s /tmp/ilvr-final-runtime-paths.txt
```

For every path in `/tmp/ilvr-final-runtime-paths.txt`, including unchanged
baseline files and configs, record:

```text
Path | Source class | Source commit/patch | Source blob SHA-1 | Final blob SHA-1 | Reason
```

Compute source and final blobs with `git rev-parse <commit>:<path>` and
`git hash-object <path>`. Use these exact source assignments:

- `ca30ab92e54e0c41fafaacc221a5120e0be8e6a5` for the five Stage 1
  textual-history production paths and five Stage 1 textual-history test paths
  listed in Task 2.
- `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0` for the five Stage 2 no-cache
  production paths and four test paths listed in Task 3.
- both `85e74e97de3773375ff037690511153c8d1f40cc` and
  `b0ce19b086cd46961fa9dcc12aa3844f659ff0b0`, as two source rows with the same
  final blob, for the shared manual-merge paths listed in Task 4.
- `85e74e97de3773375ff037690511153c8d1f40cc` plus the final path-fix commit for
  paths changed in Task 5; retain any earlier Task 2–4 source row for overlapping
  launchers or shared modules.
- `85e74e97de3773375ff037690511153c8d1f40cc` for every remaining runtime-list
  path.

“Shared manual-merge paths” means exactly the eight modified paths in Task 4:
`run_stage2_training.sh`, `src/artifact_paths.py`, `src/main.py`,
`src/stage2/trainer.py`, `src/utils_deepseed.py`,
`tests/stage2/test_launcher.py`, `tests/stage2/test_routing.py`, and
`tests/test_artifact_paths.py`. The four `Verify unchanged/subsuming` paths are
not members of that group unless they also receive an implementation edit.

Use this loop shape so a missing source or final blob fails the step instead of
silently producing an incomplete table:

```bash
while IFS= read -r path; do
  test -f "$path"
  git hash-object "$path"
done < /tmp/ilvr-final-runtime-paths.txt
```

Use these source classes exactly: `main-baseline`, `stage1-textual-port`,
`stage2-no-cache-port`, `manual-three-way-merge`, `worktree-path-fix`, and
`main-baseline/subsumes-dirty-patch`. Repeated path rows are required when more
than one source class contributed. Mark the three dirty PEFT files with the last
class, record patch hash
`0729ecd05ed8950a7c1c8d2ca74b2ba64a48172484e4a7e0ce3570cabe636557`, and cite
the four passing test nodes from Task 4 Step 8.

For the uncommitted dirty source, `Source blob SHA-1` means `git hash-object` of
the three working-tree files at the moment Task 1 validates the patch. Record
these expected values beside the patch hash:

```text
src/stage2/checkpointing.py             ff90d2bd18a6fb4484516b3aede242d5e2e22246
src/stage2/trainer.py                   c0c66059a56d75010dfb35ec47bd230952bffd5e
tests/stage2/test_peft_checkpointing.py 8c2d2164ad3c066465793663d358a5df3d5dcc32
```

The corresponding tracked Stage 2 source blobs are, in the same order,
`f3b9fcea067d72f6ff86cfcbf351d7da85bdb6bf`,
`b41de51ad93cd8a951e28ca2c43bde868908a9ea`, and
`df023efefb76d83ece8e41211e45bf859703f3cd`. Use the working-tree blob as the
dirty-patch source blob and the tracked blob as a separate Stage 2 lineage row;
never write `N/A` for an available blob.

- [ ] **Step 5: Record evidence at its verified confidence level**

Hash the directly available files:

```bash
sha256sum \
  /root/workspace/ILVR/logs/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/train-node1.log \
  /root/workspace/ILVR/results/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/stage2-length-manifest.json
find /raid/workspace/ILVR/checkpoints/zero-stage-compatibility/stage2-dora-z2-b1-ga2-e2-lr5e5-egb8-20260902-161341-r3/checkpoint-846 \
  -maxdepth 2 -type f -printf '%P\t%s\n' | sort | sha256sum
```

If any one of these three evidence paths is absent, do not invent a digest and
do not replace the asset. Record its exact path as `not present at provenance
capture`, downgrade claims that depended on it to `historical-report`, and
continue only the source/documentation verification.

Label `learning_rate=5e-5`, ZeRO-2, world size 4, and the Stage 1 handoff as
`direct-local-log`. Label checkpoint 846 as `local-partial-inventory`. Label loss
`0.24126`, evaluation `201/400 (50.25%)`, non-overlap `149/286 (52.10%)`, and the
Stage 1 `45.0%` comparison as `historical-report` unless an authoritative result
file is located and hashed during this step. Describe `+5.25pp` only as an observed
cross-report difference because protocols are not proven identical.
For the four metrics without an authoritative local result file, write their
source as `prior operator-session report; no canonical result artifact located`
rather than implying a file citation.

- [ ] **Step 6: Record selected and excluded documentation**

The selected index must link these implemented lineages by the following exact
relative paths:

```text
docs/superpowers/specs/2026-08-24-ilvr-stage2-reimplementation-design.md
docs/superpowers/plans/2026-08-24-ilvr-stage2-reimplementation.md
docs/superpowers/specs/2026-08-25-ilvr-stage2-peft-design.md
docs/superpowers/plans/2026-08-25-ilvr-stage2-peft.md
docs/superpowers/specs/2026-08-26-ilvr-stage1-dora-batched-training-design.md
docs/superpowers/plans/2026-08-26-ilvr-stage1-dora-batched-training.md
docs/superpowers/specs/2026-08-27-zero-stage-compatibility-design.md
docs/superpowers/plans/2026-08-27-zero-stage-compatibility.md
docs/superpowers/specs/2026-08-28-ilvr-multinode-four-gpu-design.md
docs/superpowers/plans/2026-08-28-ilvr-multinode-four-gpu.md
docs/superpowers/specs/2026-08-28-stage1-stage2-zero123-lora-integration-design.md
docs/superpowers/plans/2026-08-28-stage1-stage2-zero123-lora-integration.md
docs/superpowers/specs/2026-08-30-stage1-textual-states-design.md
docs/superpowers/plans/2026-08-30-stage1-textual-states.md
docs/superpowers/specs/2026-09-01-stage1-batch-equivalence-design.md
docs/superpowers/plans/2026-09-01-stage1-batch-equivalence-v26.md
docs/superpowers/plans/2026-09-01-stage1-evaluation-repair.md
docs/superpowers/plans/2026-09-01-stage1-true-batch-controlled-comparison.md
docs/superpowers/specs/2026-09-02-stage2-efficiency-design.md
docs/superpowers/plans/2026-09-02-stage2-no-cache-efficiency.md
docs/superpowers/specs/2026-09-03-stage1-stage2-final-worktree-design.md
docs/superpowers/plans/2026-09-03-stage1-stage2-final-worktree.md
```

The excluded index must name these exact non-copied source artifacts:

```text
docs/superpowers/plans/2026-09-02-stage2-segmented-kv.md
documents/multi-node-stage2-training-20260920.md
documents/stage-2-kv-cache-20260902.md
documents/TODO-ours1.md
documents/vcsd-ilvr-stage2/guideline-stage2_start.md
```

Record reason `not part of the verified final runtime`; do not link the first two
or the last file through another worktree.

- [ ] **Step 7: Validate documentation self-containment**

```bash
! rg -n '/root/workspace/ILVR/\.worktrees/(?!stage1-stage2-final)' --pcre2 \
  README.md docs/final-stage1-stage2
rg -n 'historical-report|direct-local-log|local-partial-inventory' \
  docs/final-stage1-stage2/PROVENANCE.md
test "$(find docs/final-stage1-stage2 -maxdepth 1 -type f | wc -l)" = 2
```

Expected: no foreign worktree reference, all evidence grades present, exactly two
canonical final documents.

- [ ] **Step 8: Commit the documentation and provenance unit**

```bash
git add README.md docs/final-stage1-stage2 \
  docs/superpowers/specs/2026-08-30-stage1-textual-states-design.md \
  docs/superpowers/plans/2026-08-30-stage1-textual-states.md \
  docs/superpowers/specs/2026-09-02-stage2-efficiency-design.md \
  docs/superpowers/plans/2026-09-02-stage2-no-cache-efficiency.md
git commit -m "docs: publish final stage1 stage2 runbook"
```

---

### Task 7: Review, Verify, and Hand Off the Final Worktree

**Files:**
- Verify: all files committed by Tasks 2–6
- Do not create new runtime artifacts beyond ignored `logs/` and `results/`

**Interfaces:**
- Consumes: the complete consolidated branch.
- Produces: reviewed, verified, clean tracked worktree and a precise handoff report; no merge or deletion.

- [ ] **Step 1: Request code review before completion**

Invoke `superpowers:requesting-code-review`. Review must compare the implementation
against the approved spec and this plan, with special attention to:

```text
Stage 1 true-batch/EMA regressions
Stage 2 no-cache autograd and loss-only forward behavior
main evidence/checkpoint behavior retained during three-way merge
dirty PEFT behavior semantically subsumed, not overwritten
active worktree path independence
provenance confidence labels
```

Apply only verified review fixes, each with its causal focused test, and commit
each accepted fix separately.

- [ ] **Step 2: Select the single full-suite witness for both completion gates**

Invoke `superpowers:verification-before-completion` and
`superpowers:finishing-a-development-branch`. Use this exact full-suite command
once for both overlapping test-pass claims:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python -m pytest -q
```

Expected: exit code 0 with no failed or errored tests. Read the complete output.
Do not rerun the identical suite merely to produce a second witness.

- [ ] **Step 3: Run the distinct shell/config/path witnesses once**

```bash
bash -n run_stage1_training.sh run_stage2_training.sh \
  run_stage1_batch_probe.sh run_stage1_zero_matrix.sh
./venv/ilvr-venv/bin/python -m src.deepspeed_runtime validate \
  --config "$PWD/configs/deepspeed_zero1.json" --world-size 4
./venv/ilvr-venv/bin/python -m src.deepspeed_runtime validate \
  --config "$PWD/configs/deepspeed_zero2.json" --world-size 4
./venv/ilvr-venv/bin/python -m src.deepspeed_runtime validate \
  --config "$PWD/configs/deepspeed_zero3.json" --world-size 4
test "$(readlink data)" = /raid/workspace/ILVR/data
test "$(readlink venv)" = /raid/workspace/ILVR/venv
test "$(readlink checkpoints)" = /raid/workspace/ILVR/checkpoints/
! rg -n '/root/workspace/ILVR/\.worktrees/(?!stage1-stage2-final)' --pcre2 \
  run_stage1_training.sh run_stage2_training.sh \
  run_stage1_batch_probe.sh run_stage1_zero_matrix.sh src \
  README.md docs/final-stage1-stage2
git diff --check 85e74e97de3773375ff037690511153c8d1f40cc...HEAD
test -z "$(git status --short --untracked-files=no)"
```

Expected: every command returns zero. Report ignored runtime links/directories
separately from tracked cleanliness.

- [ ] **Step 4: Confirm the branch/worktree handoff without merging or deleting**

```bash
git worktree list --porcelain
git log --oneline --decorate \
  85e74e97de3773375ff037690511153c8d1f40cc..HEAD
git status --short
```

Expected: `/root/workspace/ILVR/.worktrees/stage1-stage2-final` is attached to
`feat/stage1-stage2-final`; tracked state is clean; only intentional ignored or
pre-existing untracked runtime entries may appear. Present the human with the
verified worktree path, branch, commit list, exact verification evidence, and
remaining no-retraining uncertainty. Keep the worktree and branch as-is unless
the human later requests merge or cleanup.
