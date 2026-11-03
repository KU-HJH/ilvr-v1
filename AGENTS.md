# Repository Environment

- Always use the repository virtual environment at `./venv/ilvr-venv` for Python, pip, tests, training, and evaluation.
- Activate it with `source ./venv/ilvr-venv/bin/activate`, or invoke tools through `./venv/ilvr-venv/bin/` explicitly.
- Do not use the system Python or the legacy `./ilvr-venv` directory for this repository.
- The validated baseline on this host is Python 3.11, PyTorch 2.7.1 with CUDA 12.8, and FlashAttention 2.8.3 for NVIDIA B200 GPUs.
- If `requirements.txt` is reinstalled, reinstall the CUDA 12.8 PyTorch packages afterward as documented in `README.md`; the frozen dependency list contains CUDA 12.4-era transitive pins.

# Dataset Storage

- Store all datasets under `./data/`, resolved relative to the current repository directory. This includes downloaded dataset archives, source data, images, extracted samples, transformed datasets, and evaluation manifests.
- Use the current repository's `./data/` path even when it is a symbolic link; preserve the existing link and do not replace it with a physical directory.
- Keep dataset-specific caches under `./data/` when downloading or preparing datasets.

# Checkpoint Storage

- Store every model-bearing artifact under `./checkpoints/`, resolved relative to the current repository directory. This includes final and intermediate weights, adapter weights, optimizer and scheduler state, RNG state, DeepSpeed shards, prepared inference models, and local or remote evaluation model copies.
- The `./checkpoints/` directory is intentionally a symbolic link to the checkpoint storage location; keep and use that symbolic link.
- Use a distinct subdirectory of `./checkpoints/` for each experiment or run.
- Store evaluation-ready and evaluation-local model copies only under `./checkpoints/evaluation-staging/<run-id>/`; never create them under `./results/`.
- Validate every training output, resume checkpoint, inference preparation destination, and evaluation model path against the canonical `./checkpoints/` tree before launching work that can write model state.
- Do not copy model-bearing artifacts into another repository directory or leave compatibility symlinks to them under `./results/`. Move an existing misplaced artifact into `./checkpoints/` instead of duplicating it.
- Keep the repository filesystem small: do not replace the `./checkpoints/` symlink with a physical directory and do not create model, optimizer, or checkpoint files elsewhere in this repository.
- Unless a task explicitly says otherwise, this checkpoint policy does not apply to datasets, caches, or downloaded pretrained weights.

# Result and Log Storage

- Store every generated result, including evaluation metrics, generated samples, tables, and plots, under `./results/`, resolved relative to the current repository directory.
- The `./results/` tree is metadata/output-only and must not contain model weights, adapters, optimizer/scheduler/RNG state, DeepSpeed shards, prepared inference models, or symlinks to those artifacts.
- Store every training, evaluation, and execution log under `./logs/`, resolved relative to the current repository directory.
- Use the `results/` and `logs/` directories in the current repository directory, including when working from a linked worktree.
- Use a distinct subdirectory for each experiment or run and, when practical, use the same experiment identifier across `checkpoints/`, `results/`, and `logs/`.

# Evaluation Migration

- Read `documents/migration_plan_codex_reference_ko.md`, `docs/evaluation/audit.md`, and the approval status in `docs/superpowers/specs/2026-09-10-ilvr-eval-foundation-vsp-design.md` before continuing the evaluation migration.
- Treat `Reference/` as read-only upstream evidence. Verify independent repository roots, commits, clean state, and file hashes against `references.lock.json` before source comparisons. Do not import or install reference repositories as runtime dependencies.
- Preserve the existing patched Transformers, latent generation, prompts, image ordering, and evaluation entrypoints. Implement offline scoring before adding inference connections.
- New offline scorer CPU tests must not require model loading, torch, CUDA, or `Reference/`. Keep answer extraction independent of gold targets; benchmark-specific comparison may read targets.
- Use explicit benchmark/profile selection and fail on unknown values in the new path. Do not silently change dependencies, paper splits, judges, or existing CLI behavior.
- Report CPU contract tests, upstream-environment comparisons, GPU smoke, full benchmark evaluation, and paper-protocol verification separately. Missing evidence is not a pass.
