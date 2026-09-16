# Task 3 implementation report

Base: `de160103fef3e19e69a1b335303a7c3c9a46f308`

## RED

Before production edits, the focused configuration witness failed with the expected missing public mode:

```text
8 failed, 49 deselected
unknown preference prefix_mode
invalid choice: 'evidence_replacement'
```

Command and full captured output: `logs/evidence-replacement-20260916/task-3-red.log`.

## Implementation

- Added the opt-in `evidence_replacement` public configuration/CLI/worker mode, tied its replacement seed to public `--seed`, and enforce its fixed CE+V-only, original-image, kept-boundary, independent-branch, entropy-zero constraints.
- Routed canonical sample ID, seed and fixed sampling version to the native scorer. The Trainer now combines clean CE with the mean paired original-minus-replacement preference loss.
- Added raw all/donor/image margin sum/count diagnostics. Logging reduces each raw pair before dividing, including zero-count outputs.
- Saves the replacement objective identity in the model config for both full and DoRA save paths.
- Added dry-run worker forwarding, real tiny vision Trainer, zero-step, source-weighted logging, config reload, configuration, and launcher contracts. Documented the coordinator invocation and constraints.

## GREEN

Focused commands and full captured output are in `logs/evidence-replacement-20260916/task-3-green.log`:

```bash
bash -n run_stage2_training.sh
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q \
  tests/step_preference/test_config.py \
  tests/step_preference/test_context_preference.py \
  tests/step_preference/test_replacement_trainer.py \
  tests/step_preference/test_replacement_runtime.py \
  tests/stage2/test_launcher.py \
  tests/multinode/test_evidence_gain_forwarding.py
```

Result: `193 passed, 1 skipped in 166.77s`; after adding the one-encoder/one-clean-rollout assertion, the dedicated Trainer witness also passed: `5 passed in 9.77s`. `bash -n run_stage2_training.sh` and `git diff --check` passed. The commit identity is included in the Task 3 handoff to the root controller.

## Limits

The allowed CPU tests directly witness dense native replay and Trainer wiring. The save/reload contract uses a full tiny model config; the DoRA `_save` path is wired to the same attachment helper but has no PEFT save fixture in this task. Source logging calls the real `accelerator.reduce` branch, exercised with a single-rank accelerator test double; it establishes raw-pair-before-division behavior but is not a live multi-rank collective witness. The separate CUDA Flex witness remains for the root's final reviewed revision; no training, optimizer step, pretrained load, dependency change, or Stage2B resume path was added.

## Self-review

Reviewed the staged integration diff and ran `git diff --check`. The public route has no independent sampling-version knob, old-mode required metrics and logging retain their existing branch, and only replacement source means use the new cross-rank raw sum/count reduction.

## Controller audit correction after task review
The initially observed config-only RED was8 failed49 deselected. Its path was later reused for aggregate RED: the retained task-3-red.log is17 failed147 passed in155.27s, covering config/Trainer/launcher/forwarding. Do not treat the retained aggregate output as the config-only output. The retained GREEN log records intermediate9passed49deselected;1failed3passed;8passed95deselected;5passed;5passed98deselected; final193passed1skipped166.77s; then dedicated5passed9.77s. Controller read these actual summaries. Exact aggregate RED invocation was not retained in the report and is not reconstructed here.
