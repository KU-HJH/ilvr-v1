# Task 1 implementation report

## Status and identity

- Status: DONE
- Task base: `ae3ab4fb1a980295f85ad7213b604f5ad98eff9b`
- Implementation commit: `6e49a5e` (`feat: add deterministic evidence replacement planner`)
- Approved specification identity from the task brief: design commitment v1, SHA256 `8fa8e36fb433ebe2e7563852ee309ce1ab2e97fda6f6a30bc6841ab1983e1f4e`, source commit `5557bdd1ef41ea8e5354a2ac1990b636f0d5cf19`
- Skills read and followed: `superpowers:test-driven-development` and its `writing-good-tests.md` reference.

## Implementation

Created `src/step_preference/replacement.py` with frozen `ReplacementBranch` and `ReplacementPlan` records and the public keyword-only `build_replacement_plan` function. The planner:

- accepts only an active visual-only context `SidecarPlan` with unsuppressed latent boundaries;
- returns an empty plan for a valid CE-only sidecar without requiring image metadata;
- validates batch/root bounds, sample/seed types, image cardinality, branch order, latent block length/range, replay IDs and positions, score targets, and root predictor agreement before sampling;
- uses same-image token positions for the first/single scored step and only earlier validated latent blocks for later donor selection;
- implements the exact `ilvr-evidence-replacement-v1` compact-JSON SHA256 seed derivation with the first eight bytes, big-endian, masked to 63 bits;
- samples through a private CPU `torch.Generator`, with `randperm` without replacement for `M >= L`, `randint` with replacement for `M < L`, and sorted absolute source positions;
- preserves target IDs and logical predictor positions and computes each replay prefix from its last predictor.

Created `tests/step_preference/replacement_helpers.py` for real context-sidecar fixtures and `tests/step_preference/test_replacement_plan.py` with literal expectations covering first/image and later/donor planning, one-step and CE-only plans, multiple donor candidates, local determinism and global RNG isolation, replacement sampling, invalid image metadata, malformed donors, suppressed boundaries, textual/rejected plans, invalid identity/seed values, root/batch bounds, malformed targets/predictors/order, replay IDs, and frozen records.

## TDD evidence

Initial RED command:

```text
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_replacement_plan.py
```

Result: exit 2, expected collection failure `ModuleNotFoundError: No module named 'step_preference.replacement'`; `1 error in 0.94s`. Preserved at `logs/evidence-replacement-20260916/task-1-red.log`.

After the initial implementation, the focused planner suite passed: `23 passed in 2.86s`.

During self-review, two missing malformed-metadata cases were added before their production checks. The focused suite then produced the intended RED result: two failures because overlapping latent/predictor ranges and mismatched replay token IDs did not raise. Minimal validation for those two cases was added.

Final GREEN command:

```text
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_replacement_plan.py tests/step_preference/test_sidecar.py tests/step_preference/test_context_sidecars.py -m 'not gpu and not distributed'
```

Result: exit 0, `66 passed, 2 deselected in 10.63s`. Preserved at `logs/evidence-replacement-20260916/task-1-green.log`.

## Self-review

- Ran `git diff --cached --check`: clean.
- Compared every brief requirement against the staged implementation and tests.
- Confirmed all expectations are literal or invariant based; tests call the real sidecar builder and planner without mocks.
- Confirmed the commit contains only the three assigned files. Ignored RED/GREEN logs and this control report remain outside the implementation commit as required by “commit the planner and tests only.”
- Confirmed no training, dependency, model-loading, configuration, or unrelated repository changes were made.

## Concerns

None. Independent specification and quality review remains root controller responsibility.

## Fix round 1

- Fix base: `6e49a5efd471c602658b83e737c25a7ea5d15a50`
- Fix commit: `25536e94b8070f189308118d389703d5771ae8c7` (`fix: validate replacement plan causality`)
- Review verdict entering the round: Needs fixes.

### RED

Added literal regression tests for an in-range predictor gap, latent blocks ordered contrary to their step IDs, `None` target IDs and clean embedding sources, a `None` CE-only visual group, and `None` CE-only root predictors.

Command:

```text
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_replacement_plan.py
```

Result: exit 1, `5 failed, 25 passed in 2.80s`. The first two cases did not raise, both sequence-field cases escaped as `TypeError`, and the malformed CE-only group returned successfully. A sixth root-predictor validation case was added within the same RED phase before production changes.

After the minimal fix, the first focused rerun had `1 failed, 30 passed in 2.77s`: the sparse predictor was rejected, but its message did not yet match the required field-specific contract. The message was corrected before final GREEN.

### Findings resolution

1. Sparse predictors: active visual-off branches now require every selected position and logical predictor position to form the exact full contiguous target range. The regression uses valid in-range indices `(0, 2)` over three replay rows with literal matching targets, so ordinary range and target checks cannot satisfy it.
2. Donor causality: validated latent blocks must occur in non-overlapping absolute causal order. Donor selection also explicitly rejects a donor whose block ends after the current block begins. The regression reverses the two latent blocks while keeping each block individually in bounds and before its own target.
3. Malformed metadata: sidecar groups and root predictors are validated before the CE-only early return. Every branch sequence field is validated as a tuple before `len`, indexing, iteration or conversion, producing field-specific `ValueError` messages for `target_token_ids` and `clean_embedding_sources` instead of incidental `TypeError`.

### GREEN

Command:

```text
PYTHONPATH=src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q tests/step_preference/test_replacement_plan.py tests/step_preference/test_sidecar.py tests/step_preference/test_context_sidecars.py -m 'not gpu and not distributed'
```

Result: exit 0, `72 passed, 2 deselected in 10.69s`.

### Fix self-review and concerns

- `git diff --check` and staged `git diff --cached --check` were clean.
- The fix commit contains only `src/step_preference/replacement.py` and `tests/step_preference/test_replacement_plan.py`.
- No training, dependency, model-loading, configuration or unrelated changes were made.
- Concerns: none. Scoped independent rereview remains root controller responsibility.
