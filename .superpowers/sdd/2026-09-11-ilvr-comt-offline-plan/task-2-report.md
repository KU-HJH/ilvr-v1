# Task 2 implementation report

## Status

Implemented the pinned CoMT TEST400 preparation recipe in the assigned worktree. The recipe validates the canonical dataset identity and revision, pinned source and selection hashes, strict JSON, unique source and selection joins, the 100/100/100/100 population, exact legacy text/gold agreement, the two approved repairs, and every selected PNG before it creates the fresh output directory.

No real dataset preparation, download, GPU work, model invocation, or commit was performed.

## TDD evidence

### Initial focused RED

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_comt_prepare.py
```

Output:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
_________ test_prepare_preserves_population_and_applies_exact_repairs __________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7c3378482d50>

    def test_prepare_preserves_population_and_applies_exact_repairs(monkeypatch):
        from comt_source_fixtures import make_comt_source, pin_fixture
>       from evaluation.comt_prepare import prepare_comt
E       ModuleNotFoundError: No module named 'evaluation.comt_prepare'

tests/evaluation/test_comt_prepare.py:7: ModuleNotFoundError
=========================== short test summary info ============================
FAILED tests/evaluation/test_comt_prepare.py::test_prepare_preserves_population_and_applies_exact_repairs
1 failed in 0.02s
```

This failed for the intended missing production module. After adding the complete negative fixture matrix, the same command reported `30 failed in 0.52s`, all because `evaluation.comt_prepare` was still absent.

### Causal self-review RED: repository data symlink

The fixture was changed to mirror this repository's symlinked `data/` layout before changing production code.

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_comt_prepare.py::test_prepare_preserves_population_and_applies_exact_repairs
```

Output excerpt containing the complete failure cause:

```text
F                                                                        [100%]
E   ValueError: '/raid/workspace/ILVR/data/evaluation/.../repo-data/evaluation/comt-source/comt/data.jsonl' is not in the subpath of '/raid/workspace/ILVR/data/evaluation/.../repo'
1 failed in 0.21s
```

The fix records paths relative to the resolved data root and prefixes them with `data/`, producing stable repository-relative provenance through a symlinked data root.

### Causal self-review RED: input narrowing

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q 'tests/evaluation/test_comt_prepare.py::test_selected_source_rejects_noncanonical_options_or_helper_placeholder' 'tests/evaluation/test_comt_prepare.py::test_selected_image_must_be_one_decodable_file[non_png]'
```

Output:

```text
FFF                                                                      [100%]
FAILED tests/evaluation/test_comt_prepare.py::test_selected_source_rejects_noncanonical_options_or_helper_placeholder[float_option] - Failed: DID NOT RAISE ContractError
FAILED tests/evaluation/test_comt_prepare.py::test_selected_source_rejects_noncanonical_options_or_helper_placeholder[helper_placeholder] - Failed: DID NOT RAISE ContractError
FAILED tests/evaluation/test_comt_prepare.py::test_selected_image_must_be_one_decodable_file[non_png] - Failed: DID NOT RAISE ContractError
3 failed in 0.43s
```

The implementation was narrowed to source option strings/integers, selected questions with no helper image placeholder after IMAGE0 removal, and `.png` selected files that decode successfully.

### Causal self-review RED: nested source symlink escape

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q 'tests/evaluation/test_comt_prepare.py::test_source_files_must_not_escape_source_root_through_symlinks'
```

Output:

```text
FF                                                                       [100%]
FAILED tests/evaluation/test_comt_prepare.py::test_source_files_must_not_escape_source_root_through_symlinks[metadata] - Failed: DID NOT RAISE ContractError
FAILED tests/evaluation/test_comt_prepare.py::test_source_files_must_not_escape_source_root_through_symlinks[source_jsonl] - Failed: DID NOT RAISE ContractError
2 failed in 0.53s
```

The fix resolves `dataset-info.json` and `comt/data.jsonl` strictly and requires both to remain inside the resolved source root.

### Causal self-review RED: preserve the other 398 text strings

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_comt_prepare.py::test_integer_choices_are_stringified_without_rewriting_legacy_text
```

Output:

```text
F                                                                        [100%]
E       assert "Synthetic qu..., '30', '40']" == 'Synthetic qu..., 20, 30, 40]'
E         - 1 option:[10, 20, 30, 40]
E         + 1 option:['10', '20', '30', '40']
1 failed in 0.30s
```

The fix preserves the exact selected `text_input` for ordinary rows while converting their choices to strings. Only the two pinned repair rows rebuild the option suffix.

### Final GREEN gate

Command:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_comt_prepare.py tests/evaluation/test_comt_contract.py
```

Full output:

```text
....................................................................     [100%]
68 passed in 1.88s
```

## Files

- `src/evaluation/comt_prepare.py`: pinned `prepare_comt` implementation and exact two-record correction policy.
- `tests/evaluation/comt_source_fixtures.py`: synthetic 400-row, four-category fixture under active `data/`, including both exact malformed source records and hash pin helper.
- `tests/evaluation/test_comt_prepare.py`: positive, compatibility, integrity, repair, path, image, strict JSON, and negative preflight coverage.
- `.superpowers/sdd/2026-09-11-ilvr-comt-offline-plan/task-2-report.md`: this report.

## Self-review

- The source and selection bytes are read without mutation. Output contains only `manifest.jsonl`, `source-selection.json`, `corrections.json`, and `SOURCE.json`; no images are copied or linked.
- All row and asset validation completes before `output.mkdir`. Input mismatches raise `ContractError`; output directory/write failures are not caught and therefore remain infrastructure errors for the caller.
- Selection joins only on `(category, IMAGE0 token)`. Gold is compared after the join and is never a join key.
- Manifest inputs contain only question text, one IMAGE0 path, and choices. Fixture rationale, annotations, source helper tokens, sequence-plan text, and sequence-plan helper paths are sentinel-tested against leakage.
- Both repairs require the exact source ID, old array, preserved label, pinned revision, and pinned source hash. Any other selected malformed row fails; a malformed unselected source row is not emitted.
- Manifest sample IDs, provenance fields, asset hash coverage, target index/label/content, summary counts, and canonical serialized manifest hash are checked through the shared P1 reader.
- Source metadata and summary identify dataset root `data`, canonical dataset/revision/config/profile, source and selection hashes, modified protocol, and explicit false paper-protocol/OOD verification claims.

## Remaining concerns and out-of-scope checks

- Per the task brief, the controller owns the real pinned 400-row preparation and resulting prepared-data validation. This task used only synthetic fixtures and read-only source-schema inspection.
- No CLI integration, source-function comparison, GPU/model run, download, or benchmark score was run here; those belong to later tasks or controller verification.
- Synthetic fixture tests create uniquely named directories under active `data/evaluation/`; they do not write synthetic images into the repository's real `data/images_comt` tree.
