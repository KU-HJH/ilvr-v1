# Task 2 report: native paired Qwen replay and dispatch

## Scope and changed interfaces

- Added `score_evidence_replacement(model, options, input_ids,
  base_inputs_embeds, clean_inputs_embeds, clean_positions, image_grid_thw)`.
  It builds Task 1's immutable `ReplacementPlan` before any checkpointed
  replay, then returns `(results, stats)`.
- Each result has `step_id`, `kind` (`visual_original` or
  `visual_replacement`), `mean_logp`, target-length `entropies`,
  `source_kind`, `donor_step_id`, and `source_positions`.
- Dense replays use the Qwen decoder's native 2-D mask path.  Flex replays use
  a full-prefix BlockMask matching Qwen's actual uncached causal/sliding rule.
  Neither branch accepts a clean/donor decoder KV cache.
- `validate_shared_inputs` accepts `prefix_mode=evidence_replacement` only with
  the canonical sample ID, integer replacement seed, and exact sampling
  version.  It keeps V-only context, unsuppressed boundaries, independent
  branches, and original-image access requirements.
- Stage 2 dispatch requires `pixel_values` and exactly one image grid for an
  active evidence-replacement plan, then sends base encoder embeddings, clean
  rollout embeddings, and mRoPE positions to the new scorer.

## TDD record

The initial test file was written before the scorer.  The first collection
command exposed an incorrect test-only fixture import; after correcting that
test import, the required RED command was:

```bash
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_replacement_runtime.py
```

It failed `2/2` because the required
`transformers.models.qwen2_5_vl.stage2_evidence_replacement` module did not
exist.  Production code was then added.  The final focused CPU GREEN command
was:

```bash
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q \
  tests/step_preference/test_replacement_runtime.py \
  tests/step_preference/test_replacement_plan.py \
  tests/step_preference/test_shared_prefix_model.py \
  tests/step_preference/test_context_sidecars.py \
  tests/step_preference/test_evidence_gain.py -m 'not gpu'
```

Result: `93 passed, 1 skipped, 2 deselected in 15.18s`.
The preserved command and complete pytest output are in
`logs/evidence-replacement-20260916/task-2-cpu.log`.

The focused CUDA witness was:

```bash
RUN_REPLACEMENT_CUDA=1 PYTHONPATH=src:transformers/src \
  ./venv/ilvr-venv/bin/python -m pytest -q \
  tests/step_preference/test_replacement_runtime.py::test_cuda_flex_dispatch_matches_dense_outputs_and_gradients
```

Result: `1 passed in 12.27s`.  This test observed a positive actual
`_flex_attention_forward_count`, compared output scores and entropy vectors to
dense replay, and compared parameter gradients.  It used
`sliding_window=4` with `use_sliding_window=False` so this is an observed
native Flex dispatch, not inferred support configuration.  `git diff --check`
also passed.
The preserved command and complete pytest output are in
`logs/evidence-replacement-20260916/task-2-cuda.log`; the pre-implementation
RED witness is in `logs/evidence-replacement-20260916/task-2-red.log`.

## Runtime witnesses

`test_replacement_runtime.py` provides these direct witnesses:

- literal independent full-prefix replacement through `model.model` with a
  native 2-D attention mask; its expected score selects exact predictor rows
  and target tokens;
- first-predictor perturbation, variable target lengths (`1` and `3`), and a
  zero-step CE-only plan;
- no current-latent influence in replacement, detached copied donor gradients,
  live donor occurrence gradients, nonzero decoder and clean-input gradients,
  source-specific image perturbation, and original branch image access;
- checkpoint on/off equality for loss and input/model gradients with fixed
  deterministic source metadata;
- paired attention dropout equality for self-substitution, native original
  scores, and post-pair CPU RNG state; both checkpoint settings run;
- native Flex visibility compared with Qwen's actual causal-mask helper below
  and above the sliding window while `use_sliding_window=False`;
- actual Stage 2 dispatch rejection for missing pixels or non-single grid and
  successful real tiny pixel/grid routing;
- CUDA dense/Flex output and backward equivalence with a positive runtime Flex
  call count.

## Self-review

Reviewed the changed scorer and dispatch paths for: frozen planner metadata in
the checkpoint closure; detached replacement-only source copies; no K/V or
generation route; predictor-row/next-token alignment; paired CPU/CUDA RNG
restoration; native dense/Flex masking; active-image preconditions; and
preservation of the old sidecar modes.  The focused tests above exercise each
of these paths.  No concern remains for Task 2's bounded scope.

## Commit

`feat: add native evidence replacement replay` (the task implementation
commit; its exact identity is reported with the handoff).
