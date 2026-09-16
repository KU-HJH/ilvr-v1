# Evidence replacement final-fix report

## Scope and findings

Implemented the one assigned final-review fix wave from `606c854706f4a37723aa027f6c5f3a2c2bfba247`.
Assigned product/test changes are committed as `3112e9677771269ddd38a1a1e2337840e66d3458` (`fix: strengthen evidence replacement review coverage`).

1. The replacement runtime now compares production donor-copy gradients with an independently hand-assembled detached replay. It preserves the live original-donor and decoder-gradient assertions. It also evaluates the fallback image copy against its literal detached replay. An attached-copy mutation fails both the donor literal-gradient and fallback-detach checks.
2. `sample_id`, `replacement_seed`, and `replacement_version` are now accepted only when `prefix_mode` is `evidence_replacement`. The old-mode model route rejects them.
3. The Trainer objective fixture now uses distinct immediate/full target endpoints for two consecutive post-latent text entries. The planner receives literal `(13,)` for immediate scope and `(13, 14)` for the full scope; each then runs through the actual Trainer/model loss route.
4. The fixed-contract table directly rejects `visual_only_context_preference=False`.

## RED and mutation evidence

The mode-validation RED command was:

```bash
set -o pipefail
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_replacement_runtime.py::test_old_shared_prefix_modes_reject_evidence_replacement_metadata -q 2>&1 | tee logs/evidence-replacement-20260916/final-fix-mode-validation-red.log
exit ${PIPESTATUS[0]}
```

It exited 1 with `Failed: DID NOT RAISE ValueError`; full output is `logs/evidence-replacement-20260916/final-fix-mode-validation-red.log`.

The exact in-memory mutation script is `results/evidence-replacement-20260916/final-fix-detach-mutation.py`. It obtains the production `_paired_score` source in memory, changes only `source.index_select(1, indices).detach()` to `source.index_select(1, indices)`, runs the runtime gradient witness, and restores the original callable in `finally`. It does not write production files.

Its command was:

```bash
set -o pipefail
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python results/evidence-replacement-20260916/final-fix-detach-mutation.py 2>&1 | tee logs/evidence-replacement-20260916/final-fix-detach-mutation-v3.log
exit ${PIPESTATUS[0]}
```

The wrapper exited 0 after observing the required pytest exit 1. The captured assertion reports `donor_gradient_match=False` and `fallback_copy_detached=False`, so removal of the production detach is causally caught for both source kinds. Complete output is `logs/evidence-replacement-20260916/final-fix-detach-mutation-v3.log`.

## Fixture-diagnosis note

The first focused GREEN exposed two fixture errors. The fallback source is intentionally detached from `base`, so a positional `base` gradient assertion was invalid; the literal detached-reference comparison correctly expects both `base` gradients to be `None` and the mutation makes only the production result non-`None`. Also, the immediate target endpoint initially overlapped the next span's previous-text range. The checked `sidecar._validate_spans` trajectory rules require that the next previous-text interval follow the prior immediate target, or be an allowed suffix for the full target. The fixture now uses `10:11` for that range, satisfying both scopes. No production correction was made for either test defect.

## GREEN evidence

Mode validation after the production fix:

```bash
set -o pipefail
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_replacement_runtime.py::test_old_shared_prefix_modes_reject_evidence_replacement_metadata tests/step_preference/test_config.py::test_evidence_replacement_accepts_only_its_fixed_context_contract -q 2>&1 | tee logs/evidence-replacement-20260916/final-fix-mode-validation-green.log
exit ${PIPESTATUS[0]}
```

Exit 0: `8 passed in 8.85s`. Full output is `logs/evidence-replacement-20260916/final-fix-mode-validation-green.log`.

Focused final command:

```bash
set -o pipefail
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest tests/step_preference/test_config.py tests/step_preference/test_replacement_runtime.py tests/step_preference/test_replacement_trainer.py tests/step_preference/test_shared_prefix_model.py -q 2>&1 | tee logs/evidence-replacement-20260916/final-fix-focused-green-v2.log
exit ${PIPESTATUS[0]}
```

Exit 0: `86 passed, 1 skipped in 12.53s`. Full output is `logs/evidence-replacement-20260916/final-fix-focused-green-v2.log`.

`git diff --check` exited 0 before commit. No broad CPU suite, GPU run, training, optimizer, download, dependency, or checkpoint work was performed.

## Concerns

No unresolved assigned finding. The requested full CPU/GPU completion verification remains the root controller's final gate.
