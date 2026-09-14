# Task 2 report: fixed math, vocabulary, windows, and opt-in settings

Status: DONE_WITH_CONCERNS
Implementation commit: `ae458a8` (`feat: add fixed Stage 2 objective math and windows`)

## Implemented

- Added frozen, JSON-serializable `Stage2ObjectiveConfig`, opt-in CLI defaults, validation for finite/nonnegative settings, positive integer windows/epsilon, Stage-2-only enablement, marker supervision, and the dynamic-reference requirement. `active` is true only when enabled and at least one auxiliary weight is positive.
- Added the exact per-block entropy-transition and counterfactual Vision-gain formulas. Reference entropy and M score detach at the math boundary; student inputs retain gradient. Selected-logit entropy and full-vocabulary target log likelihood operate in FP32 without internal block reduction or clipping and reject nonfinite inputs/results and `-100` gathers.
- Added tokenizer-resolved text vocabulary construction. Actual special/padding/config rows, Qwen chat/modality tokens, non-special tool/FIM structural tokens, latent tokens, and unmapped output rows are enumerated with reasons; ordinary added text tokens remain. `identity` is deterministic JSON-safe data.
- Added immutable per-block window records and B>1 selection. The helper validates strict latent structure and assistant-segment adjacency, uses pre text states independent of next-label validity, aligns post targets to logits at `i-1`, scopes overlap to each batch row, removes overlap once from both sides without refill, and counts exclusions.
- Added opt-in collator metadata based on complete-render tokenizer offsets. It verifies exact assistant content bytes, excludes the template newline/chat termination, detects boundary ambiguity, verifies the complete final assistant suffix after processor image expansion, and splits the verified range at strict latent markers. Feature-off batch contents and legacy labels are unchanged.

## RED evidence

Initial required command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_objective_math.py \
  tests/stage2/test_objective_windows.py \
  tests/stage2/test_objective_config.py
```

Observed: `40 failed in 2.33s`, all for the missing objective/window modules, objective CLI options, or collator opt-in interface; collection completed. Full log: `logs/stage2-objective-dev/task2-red.log`.

Later independent RED checks caught three implementation defects before their fixes:

- a config control ID inside output width but absent from tokenizer mapping did not raise (`DID NOT RAISE`);
- overlap positions were keyed only by sequence index, so equal indices in different batch rows removed two valid blocks (`expected 2, got 0`);
- an incomplete block was discarded before overlap calculation, so its surviving post side failed to trim the next block (`expected pre (5,), got (4, 5)`).

Each received one targeted implementation change and a focused GREEN rerun. The first GREEN attempt also produced `38 passed, 2 failed`: a real fast tokenizer returns flat offset pairs for a single string, but generic normalization treated them as a batch. Systematic tracing confirmed the tokenizer output shape; separate fixes accepted flat pairs and wrapped collator metadata at the required batch dimension. Log: `logs/stage2-objective-dev/task2-green-attempt1.log`.

## GREEN and verification evidence

Fresh completion command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_objective_math.py \
  tests/stage2/test_objective_windows.py \
  tests/stage2/test_objective_config.py \
  tests/stage2/test_data.py
```

Observed: `68 passed in 1.91s`. `py_compile` on all changed Python/test files and `git diff --check` also exited 0. Full log: `logs/stage2-objective-dev/task2-verification.log`.

The scalar tests independently check `loss_dyn=-0.5`, derivative signs, `loss_vis=log(2)`, `dS_F=-0.5`, reference/M detachment, per-block nonlinearity, text-only normalization invariant to control logits, untouched CE logits, full-vocabulary score gathers, and finite FP32/BF16 extreme-value gradients. Window tests use literal IDs/positions, including `pre=(1,)`, `post=(5,)`, and `target=(12,)` for the eight-token example. The real collator tests compare every tensor with legacy output after removing only the new metadata.

## Local asset probes (not self-contained test dependencies)

- Local `comt_ckpt` tokenizer/config vocabulary probe: output size `151668`, text IDs `151643`, exclusions `25`; `<tool_call>` and `<|fim_prefix|>` were excluded as non-special structural tokens. Log: `logs/stage2-objective-dev/task2-local-tokenizer-probe.log`.
- Local actual Qwen processor/tokenizer collator probe, using one disposable 56x56 image under `data/` and no model: normal P/latent/Y decoded segments were `['Before', 'After']`; latent-first content produced an exactly empty pre range and `['', 'After']` after real image expansion. The image/directory were removed. Log: `logs/stage2-objective-dev/task2-local-qwen-collator-probe.log`.

## Files in implementation commit

- `src/stage2/objective.py`
- `src/stage2/objective_windows.py`
- `src/stage2/data.py`
- `src/utils_deepseed.py`
- `tests/stage2/test_objective_math.py`
- `tests/stage2/test_objective_windows.py`
- `tests/stage2/test_objective_config.py`

## Concerns and unverified scope

- The brief says “ten options,” but the binding Global Constraints, approved spec, normative prompt, and the brief's defaults enumerate nine. All nine explicitly named settings were implemented; no unspecified tenth setting was invented.
- Per dispatch, no model forward, large checkpoint load/download, GPU, training, DDP, DeepSpeed, or ZeRO execution was run. Task 2 supplies math/window/collator/config producers only; later tasks still own reference/runtime/trainer/main wiring and distributed reduction.
- The actual local processor reports that it uses the saved slow image processor and warns that a future Transformers default will select `use_fast=True`; this probe establishes current checked-in/local behavior only.
- Concurrent root-owned plan changes and copied untracked `AGENTS.md`/documents were not edited or staged.
