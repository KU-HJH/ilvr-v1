# P6 Zebra-CoT 2D Offline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the approved fixed 400 Zebra rows and score saved predictions with the approved conservative Jigsaw/Search rules.

**Architecture:** Extend the existing Sample/Prediction and prepare/score CLI path with a Zebra target, scorer and preparer. Source selection and source bytes are already frozen; preparation is local and scoring does not read gold during extraction. Existing category aggregation and incomplete-run semantics remain authoritative.

**Tech Stack:** Python 3.11 in `./venv/ilvr-venv`, standard-library scoring/JSON, installed PyYAML and Pillow, pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-ilvr-zebra-2d-offline-design.md` (design commitment v1, approved by user `진행하자` on 2026-09-14).

## Global Constraints

- Work in `/root/workspace/ILVR/.worktrees/evaluation-p1`, existing branch `codex/evaluation-p1`; do not create a worktree or commit/push/merge/clean/reset. Preserve pre-existing P1–P5 changes and index state. Stage only owned files after review if needed.
- All Python/pip/tests use `./venv/ilvr-venv/bin/`; no dependency installation or environment changes.
- Data/source/images/manifests/fixtures/caches stay under active `data/` symlink. Results stay under `results/evaluation/<run-id>/`, logs under `logs/evaluation/<run-id>/`. Preserve `data/` and `checkpoints/` symlinks. No models, GPU, judge, downloads or remote writes.
- `benchmark=zebra_2d`, `profile=zebra_2d_rules_v1`, categories `Jigsaw`, `Search`; unknown selections fail. Dataset profile `zebra-2d400-v1`, operating preparation requires exactly 200 rows per task.
- Dataset `multimodal-reasoning-lab/Zebra-CoT`, revision `f6d1defc169d19180a69f517925ed6cbb06a1c97`, configs `2D Visual Reasoning - Visual Jigsaw` / `2D Visual Reasoning - Visual Search`, split `train`.
- `selected-ids.json` SHA-256 `92e776e6b7e7e646c7f3b23c05c87b0993af44821faecac789a0e29ef2737927`; `source-records.jsonl` SHA-256 `1a533ce69f426cce9397251df976cf4331118f530b342713747f2ebdae3d4f25`.
- `Sample.id=zebra_2d/{subtask}/{global_row:05d}`; `Provenance.source_id` is the same five-digit row number. ModelInput preserves Question and its image marker, contains exactly one problem image and no choices. Target/reasoning never enter the extraction call.
- `global_row` is unique within each task, not across tasks. Match source and prepared rows by canonical Sample.id, never by list position. Copy each source `problem_image_1.path` unchanged as the prepared manifest-relative image path.
- Preserve `eval.py`, `src/evaluate_deepseed.py`, `src/task_deepseed.py`, `evaluation/{scoring,runtime,continuous_latent}.py`, patched Transformers, P2 bridge/run CLI, prompts/image ordering/generation behavior and other benchmark contracts. Reference is read-only evidence, never a runtime dependency.
- CPU scorer tests load no torch/CUDA/model/Reference. Scorer uses standard library only. Source fixture files are generated under `data/`, not embedded real benchmark rows in tests.
- `protocol_kind=modified`, `ood_verified=unknown`, `paper_protocol_verified=false`; rule scores are not semantic accuracy or paper-judge equivalence. No source-target echo scores are reported as model performance.
- User authorizes continuing the approved design; the standing AGENTS workflow selects SDD, so no execution-choice question is needed. Plan details resolve against the approved spec.

---

### Task 1: Zebra scorer and shared offline contract

**Files:**
- Create `src/evaluation/benchmarks/zebra_2d.py`, `configs/eval/zebra_2d_rules_v1.yaml`.
- Modify `src/evaluation/schema.py`, `io.py`, `registry.py`, `report.py` only for Zebra.
- Create `tests/evaluation/zebra_fixtures.py`, `test_zebra_offline.py`, `test_zebra_contract.py`, `test_zebra_pipeline.py`.

**Interfaces:**
- Consumes existing `ModelInput`, `ParsedAnswer`, `Sample`, `Prediction`, `Score`, `ContractError`, shared reader/report/CLI.
- Produces `ZebraTarget(answer: str)` in schema, added to Sample.target union.
- In zebra_2d: `CATEGORIES=('Jigsaw','Search')`, `PROFILE='zebra_2d_rules_v1'`; `normalize_search(value: str) -> str`; `validate_target(target: ZebraTarget, model_input: ModelInput, subtask: str) -> None`; `extract_answer(response: str, model_input: ModelInput) -> ParsedAnswer`; `validate_candidate(parsed: ParsedAnswer, subtask: str) -> ParsedAnswer`; `compare_answer(parsed: ParsedAnswer, target: ZebraTarget, subtask: str) -> bool`; `score_zebra_2d(sample: Sample, prediction: Prediction) -> Score`.
- Test helper `sample(subtask='Search', answer='knife', source_id='00000') -> Sample` returns a synthetic identity with one problem-image relative path and empty choices. File-based tests construct their own actual tiny image under a unique `data/evaluation/zebra-fixture-<uuid>/` directory. No tests depend on the real 400-row source.

- [x] **Step 1: Write failing scorer tests, including the first-nail target-independence witness.**

```python
import pytest
from dataclasses import replace
from evaluation.schema import ModelInput, Prediction, ZebraTarget
from evaluation.benchmarks.zebra_2d import extract_answer, normalize_search, score_zebra_2d
from tests.evaluation.zebra_fixtures import sample

@pytest.mark.parametrize('value,expected', [
    (' Knife. ', 'knife'), ('3.14', '3.14'), ('word..', 'word..'),
    (' $ 11  billion ', '$ 11 billion'), ('9/25/87', '9/25/87'),
])
def test_search_normalization(value, expected):
    assert normalize_search(value) == expected

def test_changing_gold_does_not_change_extraction():
    first = sample('Jigsaw', 'A')
    second = replace(first, target=ZebraTarget('B'))
    pred = Prediction(first.id, 'm', 'r', r'why\n\boxed{\text{B}}', 'ok')
    a, b = score_zebra_2d(first, pred), score_zebra_2d(second, pred)
    assert (a.parsed_answer, a.extraction_method, a.status) == (b.parsed_answer, b.extraction_method, b.status)
    assert a.correctness is False and b.correctness is True

def test_last_unclosed_box_does_not_rescue_earlier_answer():
    parsed = extract_answer(r'\boxed{A} later \boxed{B', ModelInput('q', ('images/x.jpg',), ()))
    assert parsed.status == 'invalid_format'
```

Add parameter cases for raw/marker/nested box; marker ordering across types; case-insensitive `The` and fullwidth colon; marker after leading tabs; multiline tail retention; decimal preservation; empty raw versus empty extraction; complete box followed by incomplete final box; wrappers only when enclosing the whole selected box; unsupported math wrappers; malformed Jigsaw labels; wrong label versus correct target; Search `A`/`yes`, dates/fractions, currency spacing, article/sentence/substrings and repeated terminal periods. These are contract cases, not model-accuracy evidence. Use the approved spec section 3 grammar verbatim.

- [x] **Step 2: Record RED before writing production code.**

Run `PYTHONPATH=src ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_zebra_offline.py` and save output under `logs/evaluation/p6-task1-20260914-01/red.log`. Expect missing new module/type failure; resolve unrelated collection mistakes before proceeding.

- [x] **Step 3: Implement the scorer and target validation.**

Use the following normalizer, retaining the extracted value for reporting:

```python
def normalize_search(value: str) -> str:
    normalized = ' '.join(value.casefold().split())
    if normalized.endswith('.') and not normalized.endswith('..'):
        normalized = normalized[:-1].rstrip()
    return normalized
```

Use an iterative depth counter for top-level boxes and enclosing wrappers. Marker parsing is line-anchored; choose the last match by location and retain the whole tail. Do not reuse/modify EMMA math or legacy scoring. `validate_candidate` updates only status/error and preserves value/method; Jigsaw label grammar is `[A-Da-d]` or `([A-Da-d])`, each optionally followed by one period. Comparison canonicalizes the accepted label only; Search normalizes each full operand. Direct scorer invocation rejects wrong target/subtask/input shape and non-ok predictions before extraction.

The scorer call order is:

```python
validate_target(sample.target, sample.model_input, sample.subtask)
parsed = extract_answer(prediction.response, sample.model_input)
parsed = validate_candidate(parsed, sample.subtask)
correct = compare_answer(parsed, sample.target, sample.subtask)
return Score(sample.id, sample.subtask, 'zebra_2d', PROFILE, '1',
             parsed.status, parsed.value, parsed.method, correct, None, parsed.error)
```

- [x] **Step 4: Add shared reader/registry/config/report branches with failing integration tests first.**

```python
def test_registry_rejects_unknown_zebra_profile():
    from evaluation.registry import get_scorer
    from evaluation.schema import ContractError
    with pytest.raises(ContractError):
        get_scorer('zebra_2d', 'unknown')

def test_zebra_target_is_not_a_choice_in_model_input():
    s = sample('Jigsaw', 'D')
    assert s.target.answer == 'D'
    assert s.model_input.choices == ()
    assert len(s.model_input.images) == 1
```

Extend actual shared reader tests using JSON roundtrips, so extra target keys, unknown task, non-string/empty target, invalid label/choices/image count and category/max_steps errors are witnessed through entry points. Score synthetic predictions through the real score CLI for complete, missing, infrastructure error, empty/invalid and absent category cases. Inspect JSON result fields and exit codes; don't assert only file existence. Isolated subprocess must load new CPU modules and score without Reference/torch/CUDA; use existing project isolation-test patterns.

Add Zebra report object exactly:

```python
summary['zebra_2d'] = dict(comparison_kind='conservative_rules',
    ood_verified='unknown', paper_sample_identity='unverified',
    paper_judge_equivalent=False)
```

Config uses schema_version 1, the explicit benchmark/profile, both canonical categories, `dataset_root: data/evaluation/zebra-2d400-v1`, and the existing external-model example run_metadata pattern. Document that real scoring requires actual generation identity/settings. Do not add generation support.

- [x] **Step 5: Run focused GREEN and self-review.**

Run `PYTHONPATH=src ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_zebra_offline.py tests/evaluation/test_zebra_contract.py tests/evaluation/test_zebra_pipeline.py tests/evaluation/test_offline_schema.py tests/evaluation/test_offline_io.py tests/evaluation/test_offline_report.py tests/evaluation/test_offline_cli.py`; save complete output in task logs. Report RED/GREEN commands, counts, changed files and concerns. Do not commit. Controller packages task-only before/after diff and runs task review before Task 2.

### Task 2: Fixed source preparation, actual 400-row correspondence and usage documentation

**Files:**
- Create `src/evaluation/zebra_2d_prepare.py`, `tests/evaluation/zebra_source_fixtures.py`, `tests/evaluation/test_zebra_prepare.py`.
- Modify `scripts/prepare_eval.py`, `tests/evaluation/test_zebra_pipeline.py`, `docs/evaluation/protocols.md`, `docs/evaluation/validation.md`.

**Interfaces:**
- Consumes Task 1 `ZebraTarget`, `validate_target(target, model_input, subtask)`, registered scorer and shared Sample/Provenance/canonical_json APIs.
- Produces `prepare_zebra_2d(source_root: Path, output: Path, *, repo_root: Path) -> dict`.
- Constants: `DATASET_ID`, `SOURCE_REVISION`, `DATASET_PROFILE`, `SELECTION_SHA256`, `RECORDS_SHA256`, with exact values in Global Constraints. Synthetic test fixtures monkeypatch only these expected source digests; production defaults remain fixed.
- Test builder `source_fixture(repo_root: Path) -> Path` creates 200 valid synthetic rows per task, selected-ids/source-records with the exact spec keys, one tiny JPEG per record, and returns source root. Files live under repo_root/data. Existing source digest pinning provides population identity; do not add hardening or alternative sample selection.

- [x] **Step 1: Write failing local-source preparation tests.**

```python
def test_prepare_source_preserves_gold_separation(repo_root, monkeypatch):
    import hashlib
    from evaluation import zebra_2d_prepare as module
    from tests.evaluation.zebra_source_fixtures import source_fixture
    source = source_fixture(repo_root)
    monkeypatch.setattr(module, 'SELECTION_SHA256', hashlib.sha256((source/'selected-ids.json').read_bytes()).hexdigest())
    monkeypatch.setattr(module, 'RECORDS_SHA256', hashlib.sha256((source/'source-records.jsonl').read_bytes()).hexdigest())
    output = repo_root/'data/evaluation/zebra-fixture-prepared'
    summary = module.prepare_zebra_2d(source, output, repo_root=repo_root)
    assert summary['N_expected'] == 400
    assert summary['categories'] == {'Jigsaw': 200, 'Search': 200}
    import json
    rows = [json.loads(line) for line in (output/'manifest.jsonl').read_text().splitlines()]
    original = [json.loads(line) for line in (source/'source-records.jsonl').read_text().splitlines()]
    by_id = {f"zebra_2d/{raw['task']}/{raw['global_row']:05d}": raw for raw in original}
    assert {row['id'] for row in rows} == set(by_id)
    for row in rows:
        raw = by_id[row['id']]
        assert row['model_input']['text'] == raw['Question']
        assert row['model_input']['choices'] == []
        assert row['target'] == {'answer': raw['Final Answer']}
        assert set(row['model_input']) == {'text', 'images', 'choices'}
```

Make the fixture `repo_root` resolve to a unique repository data fixture directory using the existing test conventions, not generic system temp storage for dataset files. Add tampered record/selection digest, selected-ID mismatch, wrong task/config/split, invalid bool/int/target, missing/corrupt image, path traversal/symlink escape, duplicate IDs, wrong population and existing-output tests. For structural cases recompute fixture digests deliberately so tests reach the structural guard instead of failing at the hash guard.

- [x] **Step 2: Record focused RED.**

Run `PYTHONPATH=src ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_zebra_prepare.py`; save under `logs/evaluation/p6-task2-20260914-01/red.log` before writing preparer.

- [x] **Step 3: Implement preparation and prepare CLI dispatch.**

Follow the existing EMMA prepare output/path/error pattern with local JSON source, without PyArrow or network. Validate all source/selection keys/types/hash/400-row correspondence and all image bytes before creating output. Enforce root containment on both source and destination and each image reference. Keep original Question, answer and image bytes; zero exclusions. Bind row ID/task/config/split and provenance to the pinned source and selected records.

```python
target = ZebraTarget(raw['Final Answer'])
model_input = ModelInput(raw['Question'], (raw['problem_image_1']['path'],), ())
source_id = f"{raw['global_row']:05d}"
validate_target(target, model_input, raw['task'])
sample = Sample(1, f"zebra_2d/{raw['task']}/{source_id}", 'zebra_2d', raw['task'],
    model_input, target, Provenance(DATASET_ID, SOURCE_REVISION, raw['config'],
    'train', source_id, {model_input.images[0]: raw['problem_image_1']['sha256']}))
```

Write canonical manifest, source-selection audit and SOURCE metadata using existing prepare conventions, summary keys from spec section 2, modified/unknown protocol flags and actual manifest digest. In `prepare_eval.main` add the explicit Zebra branch before VSP fallback and reject CoMT-only paths for Zebra. Extend the actual CLI tests to verify source errors and successful manifest reading by Task 1 reader.

- [x] **Step 4: Run focused GREEN and actual source preparation once.**

Run `PYTHONPATH=src ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_zebra_prepare.py tests/evaluation/test_zebra_pipeline.py`; save complete output. Then run:

```bash
PYTHONPATH=src ./venv/ilvr-venv/bin/python -m scripts.prepare_eval \
  --benchmark zebra_2d --profile zebra_2d_rules_v1 \
  --source-root data/evaluation/zebra-source400-20260914-02 \
  --output-dir data/evaluation/zebra-2d400-v1
```

If the real run fails, preserve logs, diagnose, add one causal regression case, fix narrowly and use a fresh destination/run ID. No generic resume or retry architecture. Tell the controller the actual destination so config/docs match the finished artifact.

- [x] **Step 5: Verify all 400 prepared rows against frozen preimplementation source and document usage.**

Use an independent one-off check under task logs, joining original records and manifest by `(task, global_row)`. For every row assert exact Question/target/one-image/no-choice relation, canonical ID/provenance, source/destination image bytes and SHA-256. Assert 200 per task, 400 images, zero exclusion and prepared manifest digest; write the result under `results/evaluation/p6-implementation-20260914-01/prepared-correspondence.json`. This checks migration correspondence, not label truth/model accuracy.

Update protocols/validation with prepare and `python -m scripts.score_eval --help`-verified scoring invocation; actual dataset path/hash, profile grammar and limits, explicit projection-versus-original-parquet evidence boundary, unknown OOD/paper identity and separate missing evidence. Use no actual-looking fabricated model scores. Include a runnable saved-prediction command with user-supplied real prediction/config paths following the existing CLI. No modifications to inference entrypoints.

- [x] **Step 6: Self-review and report for task review.**

Report files, RED/GREEN commands/output, actual preparation result and all-row correspondence result. Do not rerun broad suites or commit. The controller verifies the final assembled change after task review.

## Controller final verification and completion

- [x] Run the required final whole-change review over the P6-only before/after package, preserving earlier staged work. Apply final-review fixes via one fresh fixer and one scoped re-review if needed. Resolve material findings before completion claims.
- [x] Determine the finishing gate before final tests. This branch continues the migration and remains uncommitted by standing user preference; preserve workspace and do not integrate/clean without user decision. Invoke finishing/verification skills as applicable and reuse one final scoped CPU suite for the shared pass claim.
- [x] Final CPU suite: all `tests/evaluation/test_*.py` except `test_continuous_latent.py` and `test_inference_bridge_runtime.py`, selected explicitly by the controller using a sorted filename list; includes existing P1–P5 CPU/stub contracts and new Zebra tests. The excluded runtime file may only be run as a separately labeled small runtime-stub check if needed by a named concern; it is not GPU evidence. Save exact command, full output and exit code. Do not call this the whole repository suite.
- [x] Recheck protected file hashes, 7 reference roots/33 locked file hashes, source-selection digest, source-record digest, symlinks and `git diff --check` once. Verify config points at the actual successfully prepared dataset. No GPU/full-model/paper claims.
- [x] Update this plan's checkboxes, approved spec/discovery status and durable protocol/validation evidence; retain this plan's SDD ledger while changes are uncommitted, because there is no commit history to replace it.
- [x] Run `/root/.codex/hooks/write_ntfy_summary.sh` with the actual outcome before the Korean final response. Report P6 offline implementation/data completion separately from model evaluation and remaining P7 integration work.

## Plan self-review

Spec sections 1/3/4/5 map to Task 1; sections 2/5/6 map to Task 2 and final checks. Task 2 consumes the exact Task 1 target/validator/scorer interfaces above. Only the shared Zebra pipeline test is touched by both tasks, sequentially. No source/Jigsaw choice text is fabricated, no alternate profile or new inference path is added. Tests are concrete behavioral examples plus specified integration cases. Final commands are rooted in the validated venv and active worktree. Design approval does not establish runtime correctness; TDD, actual preparation, task reviews and final checks were pending at plan creation; their completed evidence is recorded below.

## Handoff clarification after cold read

A blind read of this plan alone found it requires the linked spec and existing repository APIs. The execution artifact is explicitly the **plan plus approved spec**, not a replacement for source/API inspection. Each generated task brief is accompanied by the binding relevant spec sections, so exact extraction grammar and source keys/types travel with the task. Existing APIs are inspected in the named files before changes. The cold read also exposed a positional zip in the preparation test; it now joins by canonical ID, matching the all-row verification requirement. No runtime contract changes were made.

Controller-owned protected-file hashes and the owned-file list are `.superpowers/sdd/2026-09-14-ilvr-zebra-2d-offline-plan/protected-files.json` and `owned-files.json`. Reference identities/files come from `references.lock.json`. Status documents are the Spec path above and `docs/evaluation/p6-zebra-discovery.md`. Actual model identity/settings are deliberately not fabricated; no real model scoring is authorized by this plan.

## Completion evidence

Both implementation tasks and the one final fix wave are complete with clean scoped
re-review. Final CPU run `p6-final-20260914-02`: 633 passed in 8.35s, exit 0.
The preceding run recorded 632 passes and one controller basetemp setup error;
its logs are preserved, the single original symptom passed after a command-path
correction, and the complete scoped set was retried with the fresh ID. No product
code changed for that execution failure. Post-fix actual preparation of all 400
rows passed at `data/evaluation/p6-final-prepared-20260914-01`; its manifest hash
matches the preserved default dataset. Configured reader 400 rows, protected
5,055 files, 7 independent reference roots and 33 hashes, pinned source/proposal
and storage links passed. See `docs/evaluation/validation.md` and
`results/evaluation/p6-final-20260914-02/verification.json` for commands and scope.
Branch integration is outside this continuing migration step; existing uncommitted
work and this plan's ledger are retained. No GPU/model/paper result is claimed.
