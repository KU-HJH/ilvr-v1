# Task 3 report: Stage 1 continuous evaluation and auditable rows

## Scope

- Added a shared prompt-preparation helper that excludes gold data, computes a
  canonical prompt SHA-256, and retains prompt token length/tensors.
- Routed `eval.py` Stage 1 generation through `ContinuousLatentController` on
  the resolved Qwen model. Stage 2 continues to use the strict Stage 2 resolver
  and compatibility controller.
- Persisted prompt, suffix, raw decode, assistant-only, latent, parse, and gold
  audit fields; shard summaries and strict result validation now carry both
  denominator conventions and `{index, reason}` invalid-gold records.
- Preserved the VSP branch and did not run training.

## RED evidence

Focused first RED command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_eval_integration.py::test_stage1_eval_uses_continuous_latent_controller_without_model_generate -q
```

Output: `1 failed in 8.30s`. The failure was the expected
`AssertionError: legacy model.generate was called` from `eval.py:422`.

Required RED suite command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_eval_integration.py \
  tests/multinode/test_evaluation.py \
  tests/multinode/test_result_validation.py -q
```

Output: `21 failed, 172 passed, 8 warnings in 11.64s`. The intentional
failures covered missing audit row fields, legacy Stage 1 `model.generate`,
missing `evaluation.runtime`, absent gold summary denominators/records, and
the strict validator rejecting the new summary contract.

## GREEN evidence

Focused implementation probes:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_eval_integration.py::test_stage1_eval_uses_continuous_latent_controller_without_model_generate \
  tests/stage2/test_eval_integration.py::test_prompt_preparation_excludes_gold_from_tensors_and_identity \
  tests/multinode/test_evaluation.py::test_merge_preserves_gold_denominators_and_invalid_gold_records \
  tests/multinode/test_result_validation.py::test_validate_results_requires_matrix_eval_and_all_capacity_cells -q
```

Output: `4 passed in 8.18s`.

Required focused evaluation command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/src:$PWD/transformers/src" \
./venv/ilvr-venv/bin/pytest \
  tests/evaluation \
  tests/stage2/test_eval_integration.py \
  tests/multinode/test_evaluation.py \
  tests/multinode/test_result_validation.py -q
```

Output: `210 passed, 8 warnings in 10.88s`.

## Warning accounting

All eight warnings are pre-existing `peft.utils.save_and_load` `UserWarning`s:
`Could not find a config file in  - will assume that the vocabulary was not
modified.` They arise in these eight Stage 2 adapter fixture parametrizations:

1. `test_peft_eval_loads_processor_from_adapter_and_model_from_stage1_base` (`eval`)
2. the same test (`evaluate_deepseed`)
3. `test_eval_rejects_saved_adapter_after_sidecar_is_removed` (`eval`)
4. the same test (`evaluate_deepseed`)
5. `test_peft_eval_requires_base_before_loading_or_iterating` (`eval`)
6. the same test (`evaluate_deepseed`)
7. `test_peft_eval_token_mismatch_fails_during_artifact_loading` (`eval`)
8. the same test (`evaluate_deepseed`)

No new warning category was introduced.

## Files changed

- `src/evaluation/runtime.py`
- `eval.py`
- `src/evaluate_deepseed.py`
- `src/evaluation/continuous_latent.py`
- `src/evaluation/scoring.py`
- `src/multinode_evaluation.py`
- `src/multinode_results.py`
- `tests/evaluation/test_scoring.py`
- `tests/stage2/test_eval_integration.py`
- `tests/multinode/test_evaluation.py`
- `tests/multinode/test_result_validation.py`

## Self-review and concerns

- Independent commit review against `97323daf27956201d721768d46bd03ab8334f40b`
  found no actionable Critical or Important issue.
- Confirmed `raw_output` decodes the full sequence, while assistant extraction
  and scoring decode only the suffix sliced at `prompt_token_length`.
- Prompt identity hashes only ordered prompt tensors using the existing
  canonical tensor hash and canonical JSON encoder; gold is not read during
  prompt preparation.
- The Stage 1 protocol check is structural (`@runtime_checkable`) and occurs on
  the resolved Qwen model before controller invocation.
- `git diff --check` was clean before commit. Untracked `.superpowers/shower/`
  and `__pycache__/` paths predate/are outside Task 3 and are intentionally
  untouched.
- No remaining functional concern from the focused suites. Full repository
  tests and a real checkpoint evaluation were out of Task 3 scope.
