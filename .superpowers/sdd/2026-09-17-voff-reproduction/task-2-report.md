# Task 2 implementation report

Status: implementation complete for selective latent-prefix replay, CLI, launcher,
coordinator, model config, and full/PEFT checkpoint resume identity. Independent
review is pending.

## Commits

- `b92bdb0 feat: checkpoint Stage 2 latent prefixes selectively`
- `5f9d3ba docs: record latent checkpointing evidence`
- Follow-up PEFT resume commit recorded after final verification.

## TDD evidence

RED command:

```bash
./venv/ilvr-venv/bin/python -m pytest -q tests/stage2/test_reproduction_checkpointing.py
```

Recorded at `logs/task-2-checkpointing/red.log`: 2 failed because
`LatentRolloutEngine.rollout()` did not accept
`latent_activation_checkpointing`.

Focused GREEN command on the reviewed revision:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD" \
  ./venv/ilvr-venv/bin/python -m pytest -q \
  tests/stage2/test_reproduction_checkpointing.py \
  tests/stage2/test_rollout.py \
  tests/stage2/test_launcher.py::test_stage2_launcher_resolves_repo_runtime_and_propagates_arguments \
  tests/stage2/test_checkpointing.py::test_configure_stage2_metadata_writes_canonical_and_legacy_aliases
```

Recorded at `logs/task-2-checkpointing/final-focused-reviewed.log`: 16 passed in
8.83 seconds.

PEFT resume RED recorded at `logs/task-2-checkpointing/red-peft-resume.log`: four
tests failed because the resolver did not accept the requested setting. Focused
PEFT/routing GREEN recorded at
`logs/task-2-checkpointing/final-peft-regression.log`: 84 passed in 17.91 seconds.

The broader existing launcher/checkpoint regression run recorded 109 passed and
one exact-token fixture failure at
`logs/task-2-checkpointing/green-launcher-checkpoint.log`; the fixture lacked the
new default-off worker argument. The fixture was updated and its focused test is
included in the final green command above.

## Implementation evidence

- Each cache-free decoder prefix is assembled with the existing allocation-efficient
  helper. Its input, mask, and position tensors are explicit checkpoint operands.
- The opt-in path uses non-reentrant checkpointing, preserved RNG, and disabled
  early stop. It runs only while gradients are enabled.
- The test uses trainable linear parameters, dropout, four latent positions in two
  blocks, and compares final embeddings, CE-derived scalar loss, input gradients,
  parameter gradients, retained latent gradients, and observed decoder replay.
- The default remains false. The launcher accepts and validates
  `STAGE2_LATENT_ACTIVATION_CHECKPOINTING`, emits an explicit positive or negative
  CLI flag, and the multinode environment allowlist preserves it.
- Model forwarding reads the saved config field independently of global gradient
  checkpointing. Full Stage 2 resume rejects a changed saved value.

## Self-review and concerns

- No cache state is checkpointed, latent state is not detached, and loss/dropout
  behavior is unchanged in the parity test.
- Task 1 rendering arguments and exact launcher ordering are preserved, with the
  new default-off flag added to the existing fixture.
- Full and PEFT resumes now read the authoritative saved checkpoint model config
  through the existing load-decision identity path. Both mismatch directions are
  rejected. Old checkpoints without the field resolve to false, and explicit
  false remains accepted. No sidecar schema or provenance system was added.
- No GPU training or distributed launch was performed.
