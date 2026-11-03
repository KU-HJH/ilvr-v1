# Training Output and Cache Path Design

## Goal

Give `run_training.sh` usable local defaults for checkpoints and logs, and make
the Hugging Face cache follow the host's existing `XDG_CACHE_HOME` setting.

## Selected configuration

- Save model checkpoints under `./checkpoints/model_stage1`.
- Write the training log to `./logs/train.log`. `LOG_FILE` must name a file,
  because `src/main.py` passes it directly to `logging.FileHandler`.
- Set `HF_HOME` from the existing `XDG_CACHE_HOME` environment variable.
- Fail before launching training when `XDG_CACHE_HOME` is unset or empty rather
  than silently selecting a different cache location.
- Create `./checkpoints/model_stage1` and the parent directory of `./logs/train.log` before launch.

All relative paths remain relative to the repository directory when the launcher
is run from the documented repository root.

## Alternatives considered

1. Use `LOG_FILE=./logs`. This is rejected because the application expects a
   file path; it conflicts with treating `logs` as a directory.
2. Hard-code the current absolute cache directory. This is rejected because it
   makes the launcher host-specific.
3. Fall back to a repository cache when `XDG_CACHE_HOME` is missing. This is
   rejected because the requested cache relationship would become implicit and
   environment-dependent.

## Validation

A lightweight launcher test will inspect the effective arguments without
starting model training. It will verify the three configured paths, directory
creation, and early failure when `XDG_CACHE_HOME` is absent. A Bash syntax check
will provide the final end-to-end validation for the changed launcher.
