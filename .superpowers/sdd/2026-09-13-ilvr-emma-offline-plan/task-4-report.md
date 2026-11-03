# Task 4 implementation report

## Result

Implemented pinned EMMA-mini preparation and CLI dispatch in the assigned files. The preparation path validates all metadata, parquet facts, row fields, subject/type populations, IDs, choices, referenced image slots, decoded image formats, and the exact 614-image population before creating the output. It preserves raw IDs, case, source order, sparse slots such as `phy_6` slots 2 and 3, and original PNG/JPEG/WEBP bytes. It writes the canonical manifest, source-selection audit, flattened SOURCE record, and requested summary fields.

The fixture creates independent synthetic parquet and image inputs under `data/`, with 400 rows, the pinned subject/type distribution, 614 images, six choice cardinalities, lowercase Physics answers, null and empty open-ended options, sparse `phy_6` slots, all three accepted image formats, context, and a correct choice longer than 512 characters.

The CLI keeps existing benchmark dispatch and exit behavior, adds the EMMA branch, and labels rejected CoMT-only arguments as EMMA. Pipeline coverage exercises preparation success/input error, offline scoring incomplete/complete exit codes, scorer/report identity and diagnostics, an isolated copied source tree that blocks torch/transformers/Reference imports, and rejection by the VSP-only generation path before runtime loading.

## RED evidence

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' \
./venv/ilvr-venv/bin/pytest -q tests/evaluation/test_emma_prepare.py \
  --basetemp=data/evaluation/p5-t4-red-20260913-01
```

Output: `25 failed in 1.20s`. Failures were caused by the missing `evaluation.emma_prepare` module/import, as expected. Full output: `logs/evaluation/p5-t4-red-20260913-01/pytest.log`.

## GREEN evidence

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' \
./venv/ilvr-venv/bin/pytest -q \
  tests/evaluation/test_emma_prepare.py \
  tests/evaluation/test_emma_pipeline.py \
  tests/evaluation/test_visulogic_pipeline.py \
  tests/evaluation/test_comt_pipeline.py \
  tests/evaluation/test_vsp_pipeline.py \
  --basetemp=data/evaluation/p5-t4-green-20260913-03
```

Output: `45 passed in 2.92s`, exit 0. Full output: `logs/evaluation/p5-t4-green-20260913-03/pytest.log`.

## Self-review and concerns

- Production hash and size pins exactly match the approved brief. Tests monkeypatch only `PARQUET_FILES`; there is no production override or test mode.
- The reviewed Task 2 runtime contract uses `open-ended` in `EmmaTarget.question_type`, while preparation summary keys use the requested canonical `open_ended`. The Task 4 pseudocode spells the target value `open_ended`; following that spelling would fail the existing reviewed validator and manifest contract, so preparation uses `open-ended` for the target and `open_ended` only for aggregate summary keys.
- The actual pinned dataset was deliberately not prepared. Root reserved the production run after review.
- No package installation, model artifact, Reference dependency, commit, push, or merge was performed.

## Fix round 1: canonical open-ended type

Review found that the preparer and Task 2 validator retained the raw source spelling `open-ended` in the normalized `EmmaTarget.question_type`. The approved canonical target spelling is `open_ended`; only the source-selection audit retains raw `row["type"] == "open-ended"`.

The causal regression changes the target contract to accept `open_ended` and reject `open-ended`, checks all prepared target type values, confirms the raw audit spelling remains unchanged, and sends prepared synthetic row `che_78` through the real score/report path. That row is observed and correct in the canonical `question_types.open_ended` bucket.

RED command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' \
./venv/ilvr-venv/bin/pytest -q \
  tests/evaluation/test_emma_contract.py \
  tests/evaluation/test_emma_prepare.py \
  tests/evaluation/test_emma_pipeline.py \
  --basetemp=data/evaluation/p5-t4-fix1-red-20260913-01
```

Output: `3 failed, 42 passed in 1.63s`. The three failures were the expected rejection of canonical `open_ended` and the two observations of the incorrectly emitted `open-ended` value. Full output: `logs/evaluation/p5-t4-fix1-red-20260913-01/pytest.log`.

GREEN command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' \
./venv/ilvr-venv/bin/pytest -q \
  tests/evaluation/test_emma_contract.py \
  tests/evaluation/test_emma_offline.py \
  tests/evaluation/test_emma_prepare.py \
  tests/evaluation/test_emma_pipeline.py \
  tests/evaluation/test_emma_report.py \
  --basetemp=data/evaluation/p5-t4-fix1-green-20260913-01
```

Output: `70 passed in 1.75s`, exit 0. Full output: `logs/evaluation/p5-t4-fix1-green-20260913-01/pytest.log`.

The prior concern about the approved spelling is resolved. Actual pinned-source compatibility remains reserved for Task 5; no production preparation was run.

## Fix round 2: nullable source solution

Root's first production preparation attempt failed during preflight with `solution must be a string`. The preserved failure log is `logs/evaluation/emma-mini400-v1/prepare.log`; no dataset or result directory was created. Independent inventory found 350 string solutions and 50 null solutions, including `Math_54`.

The causal regression sets one synthetic source solution to null, requires preparation to preserve that null in `source-selection.json`, and verifies that solution data is absent from `model_input`. The implementation now accepts only strings or null for the raw audit-only solution field; other types still fail. The scoring recipe now points at the root-reserved retry destination `data/evaluation/emma-mini400-v1-r02`.

RED command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' \
./venv/ilvr-venv/bin/pytest -q \
  tests/evaluation/test_emma_prepare.py::test_null_solution_is_preserved_only_in_source_selection \
  --basetemp=data/evaluation/p5-t4-fix2-red-20260913-01
```

Output: `1 failed in 0.38s`. The test reached `_string(row["solution"], "solution", empty=True)` and failed with the observed `solution must be a string` contract error. Full output: `logs/evaluation/p5-t4-fix2-red-20260913-01/pytest.log`.

GREEN command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' \
./venv/ilvr-venv/bin/pytest -q \
  tests/evaluation/test_emma_prepare.py \
  tests/evaluation/test_emma_pipeline.py \
  --basetemp=data/evaluation/p5-t4-fix2-green-20260913-01
```

Output: `33 passed in 1.74s`, exit 0. Full output: `logs/evaluation/p5-t4-fix2-green-20260913-01/pytest.log`.

The actual `emma-mini400-v1-r02` retry remains reserved for root and was not run here.
