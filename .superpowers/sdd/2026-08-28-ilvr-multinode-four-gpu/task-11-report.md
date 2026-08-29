# Task 11 implementation report

Date: 2026-08-28

## Result

Task 11 is implemented. The operator guide covers the fixed two-node/four-GPU
topology and exact current launcher interfaces for preflight, all Stage 1 and
Stage 2 Full/LoRA/DoRA × ZeRO 1/2/3 cells, same-identity resumes, dual-node
inference preparation, evaluation, capacity, the 18-port lifecycle matrix,
atomic validation publication, inspection, and pidfd-only owned cleanup.

The Task 10 breaker and the Task 11 production blockers found while checking
the documented commands were fixed before hardware execution:

- every `_rename_noreplace` exception preserves both namespace names and the
  retained staging descriptor without cleanup, and reports the exact output
  root and orphan staging path;
- `stop-owned --run-id RUN_ID --machine-rank RANK` validates every canonical
  ownership record first and then signals only exact live owners through
  pidfds;
- Stage 2 forwards a canonical positive `--save_steps`, with documented values
  equal to the intended terminal `--max_steps`;
- `prepare-inference` reserves and mirrors the exact inference allowlist plus
  its manifest to the same canonical results path on the remote node and
  validates the remote artifact before returning;
- Stage 2 inference preparation is transfer-only and no longer tries to build
  an unused Stage 2 worker command that requires training environment values;
- evaluation's local inference copy is a retained sibling of the strict final
  result directory; and
- `merged.jsonl` is a required direct evaluation child whose source-order rows
  and coverage must exactly equal the four validated shards.

Task 12 commands were updated to make a real Stage 2 Full/ZeRO-1 checkpoint-1,
prepare it on both nodes, evaluate only that results artifact, pass
`--save_steps 1` for one-step jobs, isolate the exact capacity tree, and publish
the final report bundle below a fresh absent validation parent.

## TDD evidence

All implementation changes were preceded by focused failing causals. The
observed red/green results were:

- report publication ambiguity and same-UID staging replacement: the focused
  selectors first produced two assertion failures plus the propagated
  `KeyboardInterrupt`; after the fix, `3 passed`;
- public owned cleanup: the four new selectors initially failed because the
  CLI/runtime entry points were absent; after the pidfd-only implementation,
  `4 passed`;
- Stage 2 `--save_steps`: the forward test first failed on the allowlist and
  the six strict-value tests then failed before validation was added; after the
  fix, `7 passed`;
- prepared inference remote mirror: the production runner test first failed
  because the mirror entry point was absent, then passed after the exact
  reserve/rsync/show sequence was implemented;
- evaluation sibling staging and merged validation:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src:$PWD/tests/stage2" \
    ./venv/ilvr-venv/bin/pytest \
    tests/multinode/test_evaluation_coordinator.py::test_local_inference_staging_is_a_sibling_of_the_final_evaluation_result \
    tests/multinode/test_result_validation.py::test_validate_results_requires_matrix_eval_and_all_capacity_cells \
    tests/multinode/test_result_validation.py::test_validate_results_rejects_merged_rows_that_differ_from_ordered_shards -q
  ```

  Red: `3 failed`. Green: `3 passed`. The two complete affected files then
  reported `64 passed`.

- Stage 2 transfer-only preparation:

  ```bash
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src:$PWD/tests/stage2" \
    ./venv/ilvr-venv/bin/pytest \
    tests/multinode/test_artifacts.py::test_prepare_inference_cli_dispatches_without_building_training_commands -q
  ```

  Red: `2 failed`, including the causal missing Stage 2 environment failure.
  Green: `2 passed`.

No SSH, GPU, Docker, tmux, rsync, or synchronization command was executed.

## Required focused verification

The required implementation suite was run exactly once:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/transformers/src:$PWD:$PWD/src:$PWD/tests/stage2" \
  ./venv/ilvr-venv/bin/pytest \
  tests/multinode \
  tests/stage1/test_launcher.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage2/test_launcher.py \
  tests/stage2/test_eval_integration.py \
  tests/stage2/integration/test_distributed_contract.py \
  tests/test_artifact_paths.py \
  tests/test_deepspeed_runtime.py -q
```

Result: `708 passed, 6 skipped, 8 warnings in 15.59s`. The warnings are the
existing PEFT vocabulary/config warnings in Stage 2 evaluation integration
tests. The host reports Python `3.11.15+`; the two distributed modules used
their established process-local `platform.python_version()` suffix
normalization. No global or persistent platform change was made.

Shell verification:

```bash
bash -n run_stage1_training.sh run_stage2_training.sh \
  run_multinode_stage1_training.sh run_multinode_stage2_training.sh \
  run_multinode_evaluation.sh run_multinode_stage1_batch_probe.sh \
  run_multinode_lifecycle_matrix.sh
```

Result: exit 0 with no output. Extracting every `bash` block from
`docs/multinode-training.md` and passing it to `bash -n` also exited 0.

## Path and integration checks

The repository venv, all seven documented launchers, and all three ZeRO
configuration files exist in this worktree. The canonical existing input paths
were checked read-only and exist:

```text
/root/workspace/ILVR/checkpoints/comt_ckpt
/root/workspace/ILVR/checkpoints/pretrained/models/Qwen2.5-VL-3B-Instruct
/root/workspace/ILVR/data/TRAIN.jsonl
/root/workspace/ILVR/data/TEST.jsonl
/root/workspace/ILVR/data/images_comt
```

Every run-specific checkpoint/result/log, prepared inference model, local
evaluation inference sibling, capacity subset, matrix witness, and validation
bundle is identified in the guide as a fresh one-time-created output. Remote
paths were not inspected because this task expressly forbids SSH; the exact
live preflight remains the Task 12 proof for the remote checkout, venv, inputs,
interface, GPUs, and capacity.

## Remaining boundary

The canonical local checkout does not yet contain the new tracked launchers and
ZeRO configs from this feature worktree. This commit must be integrated into
the clean canonical checkout on both nodes before Task 12 preflight. Hardware,
container, tmux, SSH, rsync, and real pidfd behavior remain intentionally
unexecuted here and are Task 12 live-validation concerns.
