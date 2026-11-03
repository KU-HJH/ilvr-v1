# Qwen2.5-VL 3B One-Epoch Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the local Qwen2.5-VL 3B model on all 3,450 CoMT training records for one epoch and compare the resulting runtime and training metrics with the completed 7B run.

**Architecture:** Create an isolated `train-3b` linked worktree from the same CoMT baseline as the 7B experiment, then reproduce the already-validated launcher behavior with the 3B model as its default. Run one focused RED/GREEN launcher cycle before starting the real two-GPU DeepSpeed job, and use the training process exit status, log, and small checkpoint metadata as the validation witness without hashing or consolidating the large checkpoint.

**Tech Stack:** Bash, Python 3.11, unittest, PyTorch 2.7.1 CUDA 12.8, Accelerate, DeepSpeed ZeRO-3, Qwen2.5-VL-3B-Instruct, NVIDIA B200 x2

## Global Constraints

- Use `/root/workspace/ILVR/venv/ilvr-venv/bin` for every Python, test, and training command.
- Keep all source, test, design, and plan changes uncommitted.
- Use the existing local `models/Qwen2.5-VL-3B-Instruct`; do not download model weights.
- Use all 3,450 records from `data/TRAIN.jsonl` for exactly one epoch.
- Set `HF_HOME` equal to `XDG_CACHE_HOME=/root/.cache`.
- Write only to `.worktrees/train-3b/ckpts-3b` and `.worktrees/train-3b/logs/train-3b.log`; do not modify the 7B outputs.
- Do not consolidate, recursively hash, or reread the completed checkpoint tree for postprocessing.

---

### Task 1: Isolated 3B launcher and training run

**Files:**
- Create: `.worktrees/train-3b/` linked worktree on branch `feat/train-3b`
- Modify: `.worktrees/train-3b/tests/test_launcher_defaults.py`
- Modify: `.worktrees/train-3b/run_training.sh`
- Create at runtime: `.worktrees/train-3b/logs/train-3b.log`
- Create at runtime: `.worktrees/train-3b/ckpts-3b/`

**Interfaces:**
- Consumes: local model directory `models/Qwen2.5-VL-3B-Instruct`, dataset `data/TRAIN.jsonl`, image references rooted at `data`, and venv executables under `venv/ilvr-venv/bin`
- Produces: DeepSpeed checkpoint selected by `ckpts-3b/latest`, training log `logs/train-3b.log`, and a 3B-versus-7B metric comparison

- [ ] **Step 1: Create the isolated worktree from the matching baseline**

Run from the repository root:

```bash
git check-ignore -q .worktrees
git worktree add .worktrees/train-3b -b feat/train-3b 24d35ba
git -C .worktrees/train-3b status --short
```

Expected: `.worktrees` is ignored, the new branch points to `24d35ba`, and the new worktree has no tracked modifications.

- [ ] **Step 2: Write the failing 3B launcher test**

In `.worktrees/train-3b/tests/test_launcher_defaults.py`, update the training fixture and default-launch assertion to require:

```python
(self.sandbox / "models/Qwen2.5-VL-3B-Instruct").mkdir(parents=True)

self.assertEqual(
    arguments[model_index + 1],
    str(self.sandbox / "models/Qwen2.5-VL-3B-Instruct"),
)
self.assertEqual(arguments[arguments.index("--epochs") + 1], "1")
self.assertEqual(arguments[arguments.index("--save_model_path") + 1], "./checkpoints/3b-one-epoch")
self.assertEqual(arguments[arguments.index("--log_file") + 1], "./logs/train-3b.log")
```

Also retain the validated assertions that `HF_HOME` equals `XDG_CACHE_HOME`, `VENV_BIN` can point to the repository venv, and missing model/data/image inputs fail before launch.

- [ ] **Step 3: Run the focused test and verify RED**

Run:

```bash
/root/workspace/ILVR/venv/ilvr-venv/bin/python -m unittest tests.test_launcher_defaults.LauncherDefaultsTests.test_training_launcher_passes_local_3b_model_and_requested_paths -v
```

Expected: FAIL because the baseline launcher still uses its old epoch/output/cache behavior and does not yet meet the 3B experiment contract. A test import/name error does not count as RED and must be corrected before proceeding.

- [ ] **Step 4: Implement the minimal 3B launcher configuration**

Update `.worktrees/train-3b/run_training.sh` to preserve the validated preflight behavior while using these effective definitions:

```bash
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${REPO_ROOT}"
VENV_BIN="${VENV_BIN:-${REPO_ROOT}/venv/ilvr-venv/bin}"

: "${XDG_CACHE_HOME:?XDG_CACHE_HOME must be set and non-empty}"
HF_HOME="${XDG_CACHE_HOME}"
DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/TRAIN.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-${REPO_ROOT}/data/images_comt}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-./checkpoints/3b-one-epoch}"
LOG_FILE="${LOG_FILE:-./logs/train-3b.log}"
MODEL_NAME="${MODEL_NAME:-${REPO_ROOT}/models/Qwen2.5-VL-3B-Instruct}"
EPOCHS="${EPOCHS:-1}"
```

Keep these preflight checks before `accelerate launch`:

```bash
[[ -x "${VENV_BIN}/python" ]] || { echo "Missing venv Python: ${VENV_BIN}/python" >&2; exit 1; }
[[ -x "${VENV_BIN}/accelerate" ]] || { echo "Missing Accelerate: ${VENV_BIN}/accelerate" >&2; exit 1; }
[[ -d "${MODEL_NAME}" ]] || { echo "Missing model directory: ${MODEL_NAME}" >&2; exit 1; }
[[ -f "${DATA_PATH}" ]] || { echo "Missing training data: ${DATA_PATH}" >&2; exit 1; }
[[ -d "${IMAGE_ROOT}" ]] || { echo "Missing image root: ${IMAGE_ROOT}" >&2; exit 1; }
mkdir -p "${SAVE_MODEL_PATH}" "$(dirname "${LOG_FILE}")"
```

- [ ] **Step 5: Run the focused launcher suite and verify GREEN**

Run from `.worktrees/train-3b`:

```bash
/root/workspace/ILVR/venv/ilvr-venv/bin/python -m unittest tests.test_launcher_defaults -v
bash -n run_training.sh
```

Expected: all launcher tests pass and Bash syntax validation exits 0. Stop after this proportionate suite; do not rerun the repository-wide suite already validated on the same source baseline.

- [ ] **Step 6: Confirm small immutable inputs and launch one epoch**

Run lightweight checks only:

```bash
test -d /root/workspace/ILVR/models/Qwen2.5-VL-3B-Instruct
test -f /root/workspace/ILVR/data/TRAIN.jsonl
test "$(wc -l < /root/workspace/ILVR/data/TRAIN.jsonl)" -eq 3450
test ! -e ckpts-3b/latest
```

Then launch from `.worktrees/train-3b`:

```bash
XDG_CACHE_HOME=/root/.cache \
VENV_BIN=/root/workspace/ILVR/venv/ilvr-venv/bin \
MODEL_NAME=/root/workspace/ILVR/models/Qwen2.5-VL-3B-Instruct \
DATA_PATH=/root/workspace/ILVR/data/TRAIN.jsonl \
IMAGE_ROOT=/root/workspace/ILVR/data \
SAVE_MODEL_PATH=./checkpoints/3b-one-epoch \
LOG_FILE=./logs/train-3b.log \
EPOCHS=1 \
NUM_PROCESSES=2 \
bash ./run_training.sh
```

Expected: the two-process job loads the local 3B model, runs 215 optimizer steps over 3,450 samples, and exits 0.

- [ ] **Step 7: Validate the training witness and report the comparison**

After the launch exits, inspect only the log tail and shallow checkpoint metadata:

```bash
tail -n 120 logs/train-3b.log
test -s ckpts-3b/latest
checkpoint_name=$(<ckpts-3b/latest)
test -d "ckpts-3b/${checkpoint_name}"
find ckpts-3b -maxdepth 2 -type f -printf '%P\t%s bytes\n' | sort
du -sh ckpts-3b
git status --short
```

Expected: the log includes `training finish`, final `train_runtime`, `train_loss`, `train_samples_per_second`, `train_steps_per_second`, and `mean_token_accuracy`; `latest` names a present DeepSpeed checkpoint; source/test changes remain uncommitted. Compare those values with the prior 7B witness: 3,477.9937 seconds, 215 steps, loss 0.8988884415737418, token accuracy 0.9243825947244962, 0.992 samples/second, and 0.062 steps/second.
