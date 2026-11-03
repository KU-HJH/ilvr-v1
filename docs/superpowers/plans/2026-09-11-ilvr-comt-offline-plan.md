# CoMT Offline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare the approved corrected TEST 400 population and score CoMT responses through the existing offline CLI.

**Architecture:** Extend the existing schema/IO with an explicit MCQ target and the registry with one independent CoMT scorer. Add a pinned-source preparation function, then route the existing prepare CLI to it. Reuse existing reporting and scoring CLI; preserve all VSP generation code.

**Tech Stack:** Repository Python 3.11 venv, standard library, existing PyYAML/Pillow/pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-11-ilvr-comt-offline-design.md`, commitment v1 approved by user `ㅇㅋ` on 2026-09-11 including both corrections and implementation.

## Global Constraints

- Work only in `/root/workspace/ILVR/.worktrees/evaluation-p1`; reuse linked branch `codex/evaluation-p1` at HEAD `2497ee7bafad1b51ef6c2aba6f40f084830916c4` and staged P1/P2. No commits, merges, pushes, or cleanup of user files.
- Use `./venv/ilvr-venv/bin/python` and existing dependencies. Every test uses `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:.`.
- Data/assets/manifests/caches and synthetic fixtures stay under active `data/`; results under active `results/evaluation/<run-id>/`; logs under active `logs/evaluation/<run-id>/`. Preserve data/checkpoints symlinks. Never write model state.
- Reference is read-only evidence, not a runtime dependency. No source copying from CoMT. Controller verifies roots/commits/clean/hashes before source comparisons.
- Preserve old eval.py, src/evaluate_deepseed.py, src/task_deepseed.py, scoring/runtime/continuous_latent, P2 inference_bridge.py/run_eval.py, actual editable Transformers, P1 VSP contract and existing CLI arguments/behavior.
- CoMT scorer profile `comt_reference_rules_v1`; dataset profile `ilvr-test400-corrected-v1`. Exactly 400 selected rows, creation/deletion/update/selection each 100; no exclusions.
- Correct only creation-12125 (B→18) and creation-12164 (D→24), with exact approved old/new arrays and source hash. Raw TEST/TRAIN/source remain unchanged. Only two ModelInput.text strings change.
- Extraction sees response and ModelInput only; comparison reads target. IMAGE0 only; rationale/annotations/helper images never enter ModelInput.
- Missing/infra implies incomplete and null accuracy; denominator remains manifest size; unknown inputs/profile fail closed. Synthetic test scores are not model performance.
- No GPU, judge, downloads, dependency changes or broad training/distributed tests. Source data and image bytes already acquired/verified; real prepare is controller-owned after review.
- Workers/reviewers do not spawn agents or invoke paperthin hooks. Task reviews follow sequential implementation; no simultaneous implementation workers.

## Fixed interfaces and task ownership

Task 1 produces `McqTarget(index: int, label: str, content: str)`, Sample.target union, EvalConfig.max_steps `int | None`; existing field order unchanged. VSP still requires positive integer max_steps. CoMT config omits max_steps (explicit null also accepted); nonnull max_steps rejected. Unknown benchmark/profile rejected by the registry and CLI as before, target parser rejects unknown target benchmark.

Task 1 also produces in `evaluation.benchmarks.comt`:

```python
# ParsedAnswer retains the existing value/status/error shape; widen method annotation
# to include CoMT's method names without changing VSP runtime values.
def validate_target(target: McqTarget, model_input: ModelInput) -> None: ...
def extract_answer(response: str, model_input: ModelInput) -> ParsedAnswer: ...
def compare_answer(parsed: ParsedAnswer, target: McqTarget) -> bool: ...
```

Methods: `parenthesized`, `choice_content`, `upper_label`, `lower_label`, `raw` (no match/empty). Status: parsed for A-D; invalid_format for E-Z/no match, empty for whitespace-only. Preserve extracted E-Z value for diagnostics; correctness false. Empty response value is empty string. `score_comt(sample,prediction)` returns existing Score with simulation=None. Final correctness is extracted label == target.label, never an additional content OR.

Task 2 produces:

```python
def prepare_comt(source_root: Path, selection_manifest: Path,
                 image_root: Path, output: Path, *, repo_root: Path) -> dict: ...
```

source_root contains dataset-info.json and comt/data.jsonl. Expected constants in `comt_prepare.py`: `SOURCE_REVISION`, `SOURCE_SHA256`, `SELECTION_SHA256` with approved literal values. This is one pinned data recipe, not a general repair framework. Tests may monkeypatch expected hash constants to their synthetic file hashes; no production injection/test mode. Revision remains pinned, fixture source is visibly synthetic, no real data or network needed.

Production recipe inputs:
- source_root: `data/evaluation/comt-source-discovery-20260911-01`
- source revision: `bde314be94942a7b4ab66581e7f7c4801b84e686`
- source SHA256: `15935e366f3b9998d3fd18661f563d9cd9a248d33a534e38a73d9b5c65310133`
- selection_manifest: `data/TEST.jsonl`, SHA256 `6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`
- image_root: `data/images_comt`
- output: fresh `data/evaluation/comt-ilvr-test400-corrected-v1`

Output files: manifest.jsonl, source-selection.json, corrections.json, SOURCE.json. No new image copies/symlinks. ModelInput images remain data-root-relative `images_comt/<type>/<file>`. Sample ID is `comt/<type>/<original-id>`, source_id is original id (e.g. `creation-12125`); provenance dataset_id=`czh-up/CoMT`, revision pinned, config=`default`, split=`ilvr-test400-corrected-v1`. Asset hashes exactly cover IMAGE0. Target contains index/label/content. SOURCE.json/summary record dataset_root=`data`, source/selection hashes, revision, dataset profile and no-paper/OOD limits. Summary keys include N_expected=400, categories counts, correction_count=2, manifest_sha256.

Task 3 consumes prepare_comt and unchanged scripts.score_eval.main. It adds `--selection-manifest`, `--image-root` optional argparse arguments to prepare_eval, required only for CoMT; VSP rejects CoMT-only values rather than silently ignoring them. Existing required --benchmark/--profile/--source-root/--output-dir remain unchanged. CoMT source root is data, not Reference. Fresh result/log handling and exit codes remain 0 success / 2 input / 1 infrastructure. get_scorer selecting CoMT must route preparation to CoMT, not to the VSP source verifier.

## Task 1: MCQ contracts and CoMT offline scorer

**Files:** Modify `src/evaluation/schema.py`, `src/evaluation/io.py`, `src/evaluation/registry.py`; create `src/evaluation/benchmarks/comt.py`, `tests/evaluation/comt_fixtures.py`, `tests/evaluation/test_comt_offline.py`, `tests/evaluation/test_comt_contract.py`.

**Interfaces:** Consumes existing ModelInput, ParsedAnswer, Score, exact_keys, read_manifest/read_inputs/report. Produces fixed MCQ/scorer interfaces above. Test fixture `make_comt_case(sample_count=4)` mirrors make_case return keys, creates data/results under active repo, supplies choices ['5','10','15','18'], target index1/labelB/content10, rotating four categories, synthetic run identity, default response B. Preserve all old fixtures/tests.

- [x] Write RED tests using imports inside test functions so missing APIs cause an expected assertion/import failure after collection.

```python
def test_choice_mapping_reads_this_question():
    from evaluation.benchmarks.comt import extract_answer, compare_answer
    from evaluation.schema import McqTarget, ModelInput
    parsed = extract_answer('B', ModelInput('Q', (), ('5','10','15','18')))
    assert parsed.value == 'B'
    assert compare_answer(parsed, McqTarget(1,'B','10')) is True
    assert compare_answer(parsed, McqTarget(0,'A','5')) is False
```

Use literal cases: `(A) then (c)`→C; `ANSWER: (A) [Answer] ANSWER: B`→B; `[Answer] (A)`→A; `ANSWER: D`→D via token fallback (with nonnumeric words choices); `15`→C despite containing `5`; `18 then 10`→D (highest choice index, not last textual match); `d a`→D; `A D`→D; `(Z)` invalid; `\\boxed{B}` invalid; `( A )` can fall through to upper token A and must not be called parenthesized; tabs do not act like literal space splitting. Gold mutation must not alter extraction.

- [x] RED contract tests: parse_sample accepts target index/label/content; bool/out-of-range/mismatched/empty target and non-four/empty choices reject; VSP still accepts map-only target without gold path. CoMT config max_steps omitted/null accepted, positive value rejected. read_manifest dispatches max_steps only to VSP. Config categories for CoMT must be a nonempty unique subset of the four canonical names; unknown category rejects. Production prep later requires all four counts.
- [x] Run the two new files with pytest -q -p no:cacheprovider, save RED logs under logs/evaluation/p3-task1-red-<fresh-id>. Confirm actual missing behavior before production edits.
- [x] Implement minimal target dispatch and registry wrapper. Use loops/regex written independently from the explicit behavior; do not paste upstream source. Keep Prediction/Provenance/Score shapes and VSP output values unchanged.

```python
# Dispatcher shape; keep existing VSP validation in its branch.
if benchmark == 'vsp':
    target = VspTarget(map_desc, max_steps)
    normalize_map(target)
elif benchmark == 'comt':
    target = McqTarget(index, label, content)
    validate_target(target, model_input)
else:
    raise ContractError('unsupported sample benchmark')
```

- [x] Test real evaluate/read_inputs with four categories and literal B responses, unknown/duplicate/fingerprint mismatch errors, one missing/one infra producing null accuracy, configured empty category count0/null. Use subprocess import guard to ensure new path does not load torch, transformers, eval or Reference. Add meaningful target mutation and duplicate-content regression.
- [x] GREEN: new two files plus test_offline_schema.py/test_offline_io.py/test_offline_report.py/test_offline_cli.py/test_vsp_offline.py/test_manifest_preflight.py. Read complete output. Stage only owned files, write RED/GREEN report; no commit.

## Task 2: Pinned TEST 400 preparation with two exact corrections

**Files:** Create `src/evaluation/comt_prepare.py`, `tests/evaluation/comt_source_fixtures.py`, `tests/evaluation/test_comt_prepare.py`.

**Interfaces:** Consumes Task1 target/parser validation; produces prepare_comt above. Test utility `make_comt_source()` returns root/source_root/selection_manifest/image_root/output/source_rows/selection_rows plus `source_sha256`/`selection_sha256`; fixture builds 400 rows, four categories100, two exact approved malformed rows. It returns a callable or helper `pin_fixture(monkeypatch, case)` that patches only SOURCE_SHA256/SELECTION_SHA256. All fixture PNG/data live under repo data; results/log paths remain active repo storage. Synthetic fixture may share tiny PNG bytes but each original token filename is explicit. SOURCE_REVISION remains pinned in test metadata. Never import Reference, torch or actual data.

- [x] RED: exercise prepare_comt and assert exactly 400 parsed rows in selection order, corrections2, four categories100, original files unchanged, manifest ModelInput only question/choices/IMAGE0; sentinels in rationale/annotations/helper-image values never appear in model input.

```python
def test_prepare_preserves_population_and_applies_exact_repairs(monkeypatch):
    from comt_source_fixtures import make_comt_source, pin_fixture
    from evaluation.comt_prepare import prepare_comt
    case = make_comt_source()
    pin_fixture(monkeypatch, case)
    summary = prepare_comt(case['source_root'], case['selection_manifest'],
                          case['image_root'], case['output'], repo_root=case['root'])
    assert summary['N_expected'] == 400
    assert summary['correction_count'] == 2
    rows = [json.loads(line) for line in (case['output']/'manifest.jsonl').read_text().splitlines()]
    by_id = {r['provenance']['source_id']:r for r in rows}
    assert by_id['creation-12125']['model_input']['choices'] == ['17','18','22','34']
    assert by_id['creation-12125']['target'] == {'index':1,'label':'B','content':'18'}
    assert by_id['creation-12164']['target'] == {'index':3,'label':'D','content':'24'}
```

- [x] Save focused RED before implementation. Implement all preflight before creating output: paths contained in data, source root metadata id accepts canonical czh-up/CoMT only (stored actual metadata already canonical), revision/hash checks; JSON strict read; unique source id and unique category/IMAGE0 token; exactly400 unique selection IDs, each category100, single existing image from explicit image_root, no escape/foreign category; expected question/option transform exactly matches selection text. Independently compare legacy gold content to pre-correction source content after join, including the two empty legacy values. Never choose a source record by gold.
- [x] Apply only two exact correction records after join. Require both present exactly once and old label/array match; their source rationale evidence is recorded by source hash/ID (no runtime model reasoning). Validate new target/choices. Other selected malformed row rejects; malformed unselected options do not become samples. Validate every selected image SHA/decode, retain relative paths under dataset_root=data and no extra image copies. Metadata paths/line indices go in source-selection.json, not ModelInput.

```python
# Exact policy records, not a generic cleanup.
repairs = {
 'creation-12125': {'old':['17','','18','','22','','34',''], 'new':['17','18','22','34'], 'label':'B'},
 'creation-12164': {'old':['10cm','12cm','16','','24',''], 'new':['10cm','12cm','16','24'], 'label':'D'},
}
# The production table is guarded by the pinned source hash and exact row checks.
```

- [x] Write fresh manifest/corrections/source-selection/SOURCE only after all rows/assets validate. Do not create compatibility symlinks, copies of model state, or fallback source mappings. Existing output rejects. Ordinary infrastructure write errors propagate for CLI classification; input mismatches use ContractError.
- [x] Negative tests: unknown revision, bad hashes, source duplicate/token ambiguity, missing selection/mismatching text/gold, altered repair old array or label, unknown category, extra image or helper injection, outside-data/path traversal, absent/corrupt PNG, existing destination. Mutation tests update fixture expected hashes only when the subject is a deeper validation rule. Count/source identity rejection tests must not pass merely because of an unrelated hash mismatch.
- [x] GREEN: test_comt_prepare.py + test_comt_contract.py. Save command/full output, stage only owned files, report; no real preparation or commit.

## Task 3: CLI integration, configured recipe, usage and regression witnesses

**Files:** Modify `scripts/prepare_eval.py`, `docs/evaluation/protocols.md`, `docs/evaluation/validation.md`; create `configs/eval/comt_reference_rules_v1.yaml`, `tests/evaluation/test_comt_pipeline.py`.

**Interfaces:** Consumes prepare_comt(source_root,selection_manifest,image_root,output,repo_root=...), make_comt_case, make_comt_source/pin_fixture; score_eval unchanged. New recipe uses schema_version1, benchmark comt/profile comt_reference_rules_v1, categories creation/deletion/update/selection, dataset_root data, no max_steps. run_metadata is a clearly labeled template: model_identity='external:model', prompt_template_identity='ilvr-test400-corrected-v1', generation_config={}. Documentation requires replacing all run metadata with actual generation identity/settings and matching P1 Prediction before real scoring; this recipe does not enable run_eval generation.

- [x] RED tests for CoMT prepare CLI routing/required conditional arguments and missing recipe. Use test helper to patch only expected source/selection hashes for synthetic preparation. Assert actual output/exit2 for unknown inputs, no VSP source verification for CoMT, and existing VSP prepare behavior regression.
- [x] Implement one explicit benchmark dispatch in existing CLI with separate source validation contracts; no common adapter framework. Preserve fresh output/result/log handling and file logger cleanup. Validate no unexpected CoMT-only args on VSP. Recipe is added only after its consuming RED test.
- [x] Copied-root subprocess integration copies lightweight source files, scripts, fixture source/selection/PNG only; links fixture data/results/logs/checkpoints to active storage where required. No Reference/eval.py/torch/weight files. Patch fixture hash constants in subprocess setup (not production flags), then run real prepare and score mains. Generate literal synthetic responses independent of targets (fixture normal answer B; two correction answers explicitly B and D). Assert complete 400-row fixture summary and each category100; remove one Prediction and assert incomplete/null accuracy and exit1. Assert IDs/order/hashes/metadata, module origins and no heavy imports. This is fixture integration, not real model400 score.

```python
# Literal expected outcome, not derived from the scorer:
assert exits == [0, 0]
assert summary['N_expected'] == summary['observed'] == 400
assert summary['correct'] == 400
assert summary['categories']['creation']['count'] == 100
# A second score run with one absent ID:
assert incomplete_exit == 1
assert incomplete['N_expected'] == 400
assert incomplete['complete'] is False and incomplete['accuracy'] is None
```

- [x] Add a real run_eval rejection witness with a valid CoMT config/manifest and valid generation kwargs; instrument load_runtime and assert exit2 with zero loader calls. This exercises the actual VSP guard without editing run_eval/bridge or requiring model paths.
- [x] Document exact real prepare/score commands, separate fresh IDs, data-root assets, two corrections/source evidence, corrected data vs reference-rules distinction, no actual model responses/generation support, blank/missing/invalid schema behavior and verification levels. Preserve P1/P2 historical evidence; leave controller final counts/evidence for root to append after execution.
- [x] GREEN: test_comt_pipeline.py + test_vsp_prepare.py/test_offline_cli.py/test_run_eval.py and all new CoMT files. Save logs and report, stage only owned files; no actual source prepare/GPU/commit.

## Controller checks and completion

- [x] Before Task1, create this plan's SDD workspace/ledger, verify linked worktree, snapshot all owned files and protected sources including actual main editable Transformers. Record per-task baseline files for full-context snapshot diffs; no commit-range assumption on staged P1/P2.
- [x] Run sequential task implementer/reviewer gates. Root owns only docs/plan/ledger and actual source comparisons/data runs, not workers' product fixes. One whole-P3 review, at most one final fix wave plus one scoped re-review.
- [x] After Task1 review, verify Reference CoMT root/commit/clean/hash and run isolated in-memory AST extraction of only ALPHA_MAP and judge_answer with re, no top-level Fire/import/network. Compare new extraction+comparison to that function on an explicit literal table spanning rules/duplicates/invalid labels and all four targets. Save every expected/new/upstream judgment and failures; this limited function witness is mandatory and is not upstream_environment pass. No copied upstream code artifact.
- [x] After Task2/3 review, execute actual prepare CLI once using exact pinned paths above with output data/evaluation/comt-ilvr-test400-corrected-v1. Use fresh output; if collision inspect first and choose fresh ID without overwrite. Read resulting 400 rows through read_manifest using recipe, compare selection order/source IDs/category counts/image hashes/target correspondence and exactly two changed text/choice/gold-content values. Save results/evaluation/<dataset-id>/prepared-validation.json. Preserve original TEST/TRAIN/source hashes. Do not fabricate real model predictions.
- [x] Before final verification determine branch-finishing trigger. This is continuing migration with no integration requested, so retain worktree and do not claim project-wide readiness. Use one final P1/P2/P3 lightweight CPU suite and separate existing 7 torch stubs if shared contracts affect runtime. Use fresh --basetemp under data/evaluation/<run-id>, explicitly mkdir its parent before pytest. Logs/results under matching final run ID; no GPU or broad training tests.
- [x] Verify protected source bytes, frozen proposal SHA256 b20cfc5994868dd18d4a75eebaeeecda25951901236f90a90c1a4d663cc76ede, Reference and data/checkpoint links. Record exact tests/commands/exitcodes, limited source-function comparison, actual400 data preparation, GPU/full/paper not_run separately. No fake score promoted to real CoMT performance.
- [x] Append actual execution status to validation/spec/plan/ledger, keep snapshot review artifacts (uncommitted), stage scoped changes, write ntfy summary, deliver P3 outcome and remaining VisuLogic/EMMA/Zebra scope.

## Self-review and execution route

Coverage: Task1 implements MCQ/IO/scorer/denominator contracts; Task2 exact400 join/two corrections/storage/assets; Task3 recipe/CLI/Reference-free integration/legacy rejection/docs; root source comparison/real data/final evidence. Interfaces and owned files are explicit; duplicate numeric choice mapping remains label-based. No new runtime dependencies, inference behavior or plan for uninspected later datasets. User approved design including implementation on 2026-09-11; AGENTS selects SDD, so no further execution-choice question is necessary.

Compatibility constraint: existing copied-root VSP tests intentionally copy only the pre-P3 lightweight file set. Import the new CoMT preparer lazily inside its selected benchmark branch so VSP preparation/scoring does not require the CoMT module file. Shared target/scorer dispatch must likewise preserve VSP-only copied roots.

Fixture layout clarification: make_comt_source creates an isolated repository root under active data/evaluation/<unique-fixture>/repo, with its own data/images_comt and data/evaluation source/output directories. Its results and logs point to the active worktree results/logs (and checkpoints can point to active checkpoints for copied-root integration). Thus image paths remain images_comt/<category>/<file>, all synthetic datasets remain under active data, and result/log output stays in active result/log trees. The returned case['root'] is that isolated repository root. Never add synthetic images to the real data/images_comt directory.
