# Qwen2.5-VL 3B One-Epoch Training Design

## Goal

Run one full-data training epoch with the locally installed
`Qwen2.5-VL-3B-Instruct` model so its speed and training metrics can be
compared directly with the completed 7B run.

## Isolation and inputs

- Create `.worktrees/train-3b` on an uncommitted `feat/train-3b` branch from
  the same `24d35ba` CoMT dataset baseline used for the 7B experiment.
- Use the repository environment at `./venv/ilvr-venv`; from the linked
  worktree, pass its absolute `bin` directory through `VENV_BIN`.
- Load the existing model from `models/Qwen2.5-VL-3B-Instruct`; do not
  download another model.
- Train on all 3,450 records in `data/TRAIN.jsonl` for exactly one epoch.
- Resolve dataset image paths relative to the main repository `data`
  directory, because records already contain the `images_comt/` prefix.
- Use two visible B200 GPUs through the launcher's two-process DeepSpeed
  configuration, matching the 7B experiment.

## Launcher configuration

The worktree launcher keeps the validated 7B experiment settings for cache,
paths, preflight checks, and numeric overrides. Its default model changes to
the local 3B directory and its default epoch count remains `1`.

Runtime settings:

- `XDG_CACHE_HOME=/root/.cache`
- `HF_HOME=$XDG_CACHE_HOME`
- `SAVE_MODEL_PATH=./checkpoints/3b-one-epoch`
- `LOG_FILE=./logs/train-3b.log`
- `EPOCHS=1`
- `NUM_PROCESSES=2`

The new output names ensure that `.worktrees/train-7b/ckpts` and its log are
never modified.

## Verification and reporting

Before training, a focused launcher test must fail against the old launcher
expectation and pass after the minimal 3B configuration change. Run the
existing launcher test suite once as the proportionate regression check.

Training succeeds when the process exits with status 0, the log records the
final trainer metrics, and the 3B output contains a published DeepSpeed
checkpoint plus its `latest` pointer. Report runtime, optimizer steps, loss,
token accuracy, sample throughput, step throughput, and output size alongside
the prior 7B values when available.

Do not consolidate or hash the checkpoint tree: that would reread tens of
gigabytes without improving the completed training witness. Keep all source,
test, design, and plan changes uncommitted, as requested.
