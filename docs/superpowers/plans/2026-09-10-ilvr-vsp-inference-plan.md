# VSP Data Preparation and ILVR Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare pinned VSP maps under data/, generate through the existing ILVR runtime, and feed preserved responses into the P1 offline scorer.

**Architecture:** Extract the existing manifest reader without changing its contract. Add one VSP preparation module and a thin inference bridge; separate prepare/run CLIs own files and status. Scoring remains the existing P1 CLI.

**Tech Stack:** Repository Python 3.11 venv, standard library, installed PyYAML/Pillow/pytest; existing torch and patched Transformers only after inference preflight.

**Spec:** `docs/superpowers/specs/2026-09-10-ilvr-vsp-inference-design.md` — design commitment v1 approved by the user's `ㅇㅋ` on 2026-09-10.

## Global Constraints

- 모든 Python 실행은 `./venv/ilvr-venv/bin/python`; environment `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:.`.
- Work only in `/root/workspace/ILVR/.worktrees/evaluation-p1`, branch `codex/evaluation-p1`; reuse its links. No commit, merge, push or worktree deletion. Stage only task deliverables when reporting; retain review records because there is no commit history of these changes.
- 데이터는 `data/evaluation/`, dataset cache는 `data/cache/evaluation`; generated model artifacts only in `checkpoints/`, with evaluation model copies only in `checkpoints/evaluation-staging/<run-id>/`. Results and logs use this worktree's `results/evaluation/<run-id>/` and `logs/evaluation/<run-id>/`.
- Preserve P1 schema/profile/score semantics and existing eval.py, src/evaluate_deepseed.py, src/evaluation/runtime.py, continuous_latent.py, controllers and patched Transformers. Do not change dependencies or introduce retries/resume/judges.
- Source is pinned Visual-Spatial-Planning `6d74d7bb46ae71bf41aa4d3d874a4f31fc5f0366`; verify Git root/origin/HEAD/clean and lock hashes before source comparisons. Never import/install/run Reference code. Copy required source data/license into data; prepared run/score must work without Reference.
- Benchmark/profile is `vsp`/`vsp_validated_v1`; categories `level3` through `level8`; max_steps=100; image order legend, worked example, target. Protocol is modified; no paper-protocol claim.
- Prediction schema remains P1. Generation token IDs live in a separate details JSONL. Model input excludes target/gold. Runtime fingerprint is P1 canonical model_identity/prompt_template_identity/generation_config hash, not a weight-content fingerprint.
- Tests use actual behavior and independently derived expected values. Offline tests require no torch/CUDA/model/Reference; bridge runtime tests may use existing torch with mocked weight loading/generation. All test data/images/manifests go beneath this worktree's data/; logs/results beneath their matching roots. Use UUID run IDs for fixture isolation.
- Each code task starts with failing tests, records RED/GREEN logs, self-reviews and reports. Workers and reviewers must not spawn agents or invoke paperthin hooks. Controller dispatches scoped reviews; reviewers do not rerun test suites.
- This is one approved P2 scope. AGENTS.md already chooses SDD execution; no new execution-method question is needed. Route ordinary plan corrections through the ledger with the spec as authority.

## Files and responsibilities

| Task | Files | Responsibility |
|---|---|---|
| 1 | modify src/evaluation/io.py; new tests/evaluation/test_manifest_preflight.py | Manifest-only preflight, P1 reader reuse |
| 2 | new src/evaluation/vsp_prepare.py, scripts/prepare_eval.py, tests/evaluation/vsp_fixtures.py, test_vsp_prepare.py; modify references.lock.json | Read/copy pinned source, ordered inputs, full/smoke manifests |
| 3 | new src/evaluation/inference_bridge.py, tests/evaluation/test_inference_bridge.py, test_inference_bridge_runtime.py | Lightweight validation and exact legacy call adaptation |
| 4 | new scripts/run_eval.py, tests/evaluation/test_run_eval.py | Run output/metadata/failure handling and lazy model loading |
| 5 | new tests/evaluation/test_vsp_pipeline.py, configs/eval/vsp_ilvr_three_images_v1.yaml; modify docs/evaluation/protocols.md, validation.md | Reference-free integration, real data preparation, runnable recipe |

No new plugin, package installer, general run manager, artifact registry or recovery framework is required. Five tasks are sequential because their interfaces are consumed by later tasks.

## Task 1: Manifest-only preflight

**Files:** Modify `src/evaluation/io.py`; create `tests/evaluation/test_manifest_preflight.py`.

**Interfaces:** Consume existing `EvalConfig`, `PreparedEvaluation`, `parse_config`, P1 fixture `make_case`. Produce `read_manifest(manifest_path: Path, config: EvalConfig, repo_root: Path) -> PreparedEvaluation` with empty predictions, validated samples, manifest hash and run fingerprint. Preserve `read_inputs(manifest_path, predictions_path, config, repo_root)` signature and prediction join checks.

- [x] Write tests before editing io.py. The central case must work without any prediction file; malformed maps/assets/IDs still fail. Use existing fixture helpers without generating new fake answers.

```python
def test_manifest_preflight_does_not_need_predictions():
    import evaluation.io as io
    from offline_fixtures import make_case
    case = make_case(1)
    case['prediction_path'].unlink()
    assert hasattr(io, 'read_manifest'), 'manifest-only preflight is missing'
    config = io.parse_config(case['config'], case['root'])
    prepared = io.read_manifest(case['manifest'], config, case['root'])
    assert [s.id for s in prepared.samples] == ['vsp/planning/s0']
    assert prepared.predictions == {}
    assert prepared.manifest_sha256 == __import__('hashlib').sha256(
        case['manifest'].read_bytes()).hexdigest()
```

Also cover duplicate ID, empty manifest, outside-data manifest path and changed image bytes using literal bad inputs. Compare read_manifest's samples/hash/fingerprint with read_inputs for the same actual files; this establishes reader compatibility, not benchmark ground truth.

- [x] RED: run the new test file together with `tests/evaluation/test_offline_io.py`, save combined output/exit code under `logs/evaluation/p2-task1-red-<id>/`. Missing API must be the new test's assertion failure; existing tests remain passing.
- [x] Extract the existing manifest-validation block into the new function, using its current helpers and error semantics. Return:

```python
return PreparedEvaluation(tuple(samples), {}, manifest_hash,
                          run_fingerprint(config.run_metadata))
```

read_inputs retains its input-path validation prelude (dataset root, manifest path, prediction path), calls read_manifest, then validates prediction bytes and joins against `{sample.id for sample in prepared.samples}`. Use `prepared.manifest_sha256` and `prepared.run_fingerprint` instead of recalculating those contracts. Do not accept a missing predictions file in read_inputs, add dummy files in read_manifest, or weaken LF-only JSONL/containment/hash rules.

- [x] GREEN: run `test_manifest_preflight.py test_offline_io.py test_offline_cli.py`; inspect all output and save evidence. Self-review changes to ensure earlier P1 callers still receive the same return shape. Stage only io.py and the new test; do not commit.

## Task 2: Pinned VSP source preparation

**Files:** Create `src/evaluation/vsp_prepare.py`, `scripts/prepare_eval.py`, `tests/evaluation/vsp_fixtures.py`, `tests/evaluation/test_vsp_prepare.py`; modify `references.lock.json` to add exactly the prompt text and two common image evidence hashes.

**Interfaces:** Produce the following names in vsp_prepare.py (standard-library data types unless explicitly noted):

```python
LEVELS = tuple(f'level{n}' for n in range(3, 9))
COMMON_IMAGES = ('prompt-visual-images/system-figure-1.png',
                 'prompt-visual-images/system-figure-2.png')
SOURCE_COMMIT = '6d74d7bb46ae71bf41aa4d3d874a4f31fc5f0366'

def parse_map_pair(table_text: str, description: str, size: int) -> tuple[tuple[int, ...], ...]:
    # Exact table/coordinate contract below; implementation starts after RED.
    raise NotImplementedError

def build_prompt(source_text: str) -> str:
    intro = ('The three images are provided in this order: Image 1 (legend), '
             'Image 2 (worked example), Image 3 (maze to solve).\n\n')
    for marker in ('<IMAGE-1>', '<IMAGE-2>', '<TEST-IMAGE>'):
        if source_text.count(marker) != 1:
            raise ContractError('expected one of each source image marker')
    return intro + source_text.replace('<IMAGE-1>', 'Image 1 (legend)').replace(
        '<IMAGE-2>', 'Image 2 (worked example)').replace(
        '<TEST-IMAGE>', 'Image 3 (maze to solve)')

def prompt_identity(prompt: str) -> str:
    return 'vsp-ilvr-three-images-v1:' + hashlib.sha256(prompt.encode('utf-8')).hexdigest()

def verify_source_reference(source_root: Path, repo_root: Path) -> dict:
    # Return the matching verified lock repository entry, after checks below.
    raise NotImplementedError

def prepare_vsp(source_root: Path, dataset_root: Path, *, source_identity: dict) -> dict:
    # Return {N_expected, categories, manifest_sha256, smoke_manifest_sha256,
    #         prompt_template_identity, protocol_kind, source_commit}.
    raise NotImplementedError
```

The three NotImplementedError declarations specify interfaces, not an instruction to ship stubs. Their complete behavior is below; production implementations must have no unimplemented paths.

Source identity is the verified repository entry with `name`, `upstream_url`, `commit`, `license_files` and `evidence_files`. Unit fixtures supply an explicitly synthetic entry with name/dataset provenance `synthetic:vsp-fixture` and commit `fixture-v1`; production CLI only calls prepare_vsp after real verification.

`scripts.prepare_eval.main(argv: list[str] | None=None, *, repo_root: Path | None=None) -> int` accepts `--benchmark`, `--profile`, `--source-root`, `--output-dir`, all required. Unknown pair fails through existing registry. No subset/resume flags. Output is a fresh direct child of data/evaluation; matching fresh results/log direct children use the dataset directory name.

- [x] Write failing behavior tests. parse_map_pair must enforce the literal mapping and independent descriptions:

```python
TABLE = '| | Col 1 | Col 2 |\n| Row 1 | @ | * |\n| Row 2 | _ | # |'
DESC = ('This is a 2x2 map.\nThe player is at: row 1, column 1;\n'
        'The hole(s) are at: Row 2, Column 2;\n'
        'The goal is at: Row 1, Column 2.')

def test_table_and_description_must_agree():
    from evaluation import vsp_prepare as prep
    assert prep.parse_map_pair(TABLE, DESC, 2) == ((1, 2), (0, -1))
    with pytest.raises(ContractError):
        prep.parse_map_pair(TABLE, DESC.replace('Row 1, Column 2.',
                                             'Row 2, Column 1.'), 2)
```

Add no-hole, duplicate/out-of-range coordinates, dimension mismatch, malformed row/column labels, invalid symbol and multiple starts/goals. Tiny parser fixtures are deliberately not production level sizes.

Fixture utilities must be concrete and reusable by later tasks:

```python
def make_vsp_case(sample_count: int = 2) -> dict:
    # Return root, data_root, result_root, run_id, manifest, samples, config,
    #        model_dir, base_model_dir. Rules in the next paragraph.
    raise NotImplementedError

def write_source_fixture(source_root: Path) -> dict:
    # Write all six levels, ids 0..99, prompt, common images, LICENSE;
    # return the synthetic source_identity described above.
    raise NotImplementedError
```

make_vsp_case uses repository root from the utility file location and UUID storage directories. It writes three visually distinct small PNG assets (red legend, green example, blue target), processed prompt file and a real manifest with IDs `vsp/level3/0`, `vsp/level3/1` onward, map `[[1,2,0],[0,0,0],[0,0,0]]`, max_steps=100, source identity synthetic, exact image hashes and empty choices. Config uses model_identity='synthetic:model', all LEVELS, a matching prompt identity and `{'max_new_tokens': 8, 'max_sequence_length': 32768, 'repetition_penalty': 1.05, 'temperature': 0.0, 'top_p': 1.0, 'do_sample': False}`. Model/base fixture directories contain only tiny config metadata under `checkpoints/evaluation-staging/<run-id>/`; they are fake artifacts for mocked loaders, never reported as real models. The utility can create them lazily for tests that need them. All generated files follow storage rules.

write_source_fixture writes the expected task-main layout, valid rectangular maps with start (1,1), goal (1,2), no holes, matching table/pure_text and tiny PNGs for all 600 IDs. Its prompt includes each marker once. Later tests must not pretend these small colored images independently establish VSP map semantics.

Test the real prepare_vsp against this source: 600 full rows, 6 smoke rows, exact ordering and IDs, raw asset byte preservation, output prompt/hash, no target text in model_input, and license/source information. Delete one source file or mismatch one coordinate to ensure failure occurs before dataset directory creation. Source read tests use files within data/.

- [x] RED: save focused failure output before implementing. Import the new module inside tests so absence is the expected missing-feature failure; once created, individual behavioral tests must demonstrably fail before the corresponding logic is added.
- [x] Implement parsing and source collection. Require exact header `Col 1`..`Col N`, row labels `Row 1`..`Row N`, and `_/#/@/*` cells. Parse the complete pure_text structure including dimension, player, goal, either holes or the exact no-holes sentence; reject leftover malformed coordinate text and duplicate coordinates. Require 1-based bounds and complete equality with the table. Validate the result with existing normalize_map/VspTarget.
- [x] Verify the source Git root/origin/commit/clean and lock evidence/license hashes with subprocess git and hashlib. Ensure source paths resolve inside the source checkout. Add prompt and common-image hashes to the existing VSP lock entry without changing old hashes or other repositories. Do not copy upstream Python.
- [x] Preflight all 600 triplets and common prompt/images/license before creating dataset_root; require exact filename sets. Pillow verifies images without torch. Copy selected raw files with normal file copies, preserving bytes and layout; write processed prompt, LICENSE and SOURCE.json with repository/commit/paths and protocol_kind=modified. Sample creation uses dataclasses.asdict + canonical_json with ID/provenance/model_input/target specified in the spec. Do not attach full manifest hash to its own rows.
- [x] Write both complete manifest payloads only after copies succeed. Full rows are numeric level/id order; smoke selects id 0 per level. Return the seven summary fields above, categories mapping each level to 100. CLI writes prepare.json in results and prepare.log in logs; input/source/path failures return 2, copying/output failures 1, completed preparation 0. Existing output directories are never overwritten.
- [x] GREEN: run test_vsp_prepare.py and Task 1's preflight tests; save output. Stage only the five task files and lock; no commit.

## Task 3: Lightweight bridge and exact runtime boundary

**Files:** Create `src/evaluation/inference_bridge.py`, `tests/evaluation/test_inference_bridge.py`, `tests/evaluation/test_inference_bridge_runtime.py`.

**Interfaces:** Consume P1 `ModelInput`, `PreparedEvaluation`, `EvalConfig`, Task 2 `COMMON_IMAGES`, `LEVELS`, `prompt_identity`. Produce:

```python
def validate_generation_config(raw: dict) -> None: ...
def validate_vsp_inputs(prepared: PreparedEvaluation, config: EvalConfig) -> None: ...
def validate_runtime_paths(model_dir: Path, base_model_dir: Path | None,
                           cache_dir: Path, repo_root: Path) -> tuple[Path, Path | None, Path]: ...
def load_runtime(): ...  # importlib.import_module('eval'), only when called
def generate_one(model, processor, model_input: ModelInput, dataset_root: Path,
                 generation_config: dict, *, run_one_example) -> dict: ...
def runtime_metadata(model, processor, runtime, model_dir: Path,
                     base_model_dir: Path | None, generation_config: dict) -> dict: ...
```

These are signature declarations; no ellipsis body ships. This module has no import-time torch, Transformers, eval, Reference or GPU access. Task 2 import must also remain lightweight (Pillow imports may be inside image validation).

- [x] Write failing lightweight tests for generation validation, prompt/three-image order/choice/category violations and resolved paths outside storage roots. Require exactly the six kwargs in Task 2 config: type(max_new_tokens)=int and >=0; type(max_sequence_length)=int and >0; numeric finite non-bool repetition_penalty>0, temperature>=0, 0<top_p<=1; type(do_sample)=bool and equality to temperature>0. Zero new tokens is allowed because the existing controller explicitly supports it. No defaults or coercions are introduced.

```python
def test_bridge_passes_only_model_input_and_preserves_suffix():
    case = make_vsp_case(1)
    config = parse_config(case['config'], case['root'])
    sample = read_manifest(case['manifest'], config, case['root']).samples[0]
    returned = {'assistant_only': 'Action plan: R', 'generated_suffix_token_ids': [21,22],
                'prompt_identity': 'a'*64, 'prompt_token_length': 7,
                'raw_output': 'prompt Action plan: L\nAction plan: R',
                'extracted_final_answer': 'R', 'latent_start_count': 1,
                'latent_block_count': 1, 'stop_reason': 'eos', 'inference_time': 0.1}
    def recorded(model, processor, payload, kwargs):
        assert set(payload) == {'text_input', 'image_input'}
        assert payload['text_input'] == sample.model_input.text
        assert [Path(p).name for p in payload['image_input']] == [
            'system-figure-1.png', 'system-figure-2.png', '0.png']
        assert kwargs == case['config']['run_metadata']['generation_config']
        return returned
    result = generate_one(object(), object(), sample.model_input, config.dataset_root,
                          config.run_metadata.generation_config, run_one_example=recorded)
    assert result['assistant_only'] == 'Action plan: R'
    assert result['generated_suffix_token_ids'] == [21,22]
```

The recording double is the heavy model-call boundary; assertions target the bridge's emitted arguments. Add absolute/escaping image paths and unexpected legacy return shapes so corrupt tokens/status do not become successful records.

- [x] RED: run only new lightweight tests first. Then add the runtime comparison test before implementing the adapter. For the runtime test reuse the existing FakeProcessor, RecordingController, RecordingStage1Model/ModelWhoseGenerateRaises and `_import_entry_point` via an explicit test-only import of `tests/stage2/test_eval_integration.py`. Keep production runtime/controller dispatch real; replace only expensive model/generation dependencies as the existing tests do. If exact fixture reuse requires local wrappers, put those in this new runtime test file, not production.
- [x] Implement validators. Read the processed prompt from dataset_root, compare its identity and exact text for every sample, require level categories and source-id/path order, no choices, max_steps=100, existing readable PNGs. Store `config.dataset_root` containment as P1 established; validate processed prompt resolution also stays inside it. Model/base directories must exist beneath checkpoints, cache must resolve beneath data (may not yet exist). Base is required when adapter_config.json or Stage 1/2 PEFT metadata is present and rejected for a full artifact; detailed weight/token checks remain the existing loader's job.
- [x] Implement the adapter using only ModelInput. Resolve relative image paths inside dataset_root in order, create `{'text_input': model_input.text, 'image_input': paths}`, pass exact model/processor/kwargs to the supplied callable once, and return its raw result. Validate required string response, nonnegative integer suffix IDs, positive/zero integer prompt length as appropriate, prompt hash string, stop_reason and finite timing/count metadata. Failure raises ContractError and is handled by run CLI. Never construct a second generation config, parse gold or use extracted_final_answer as the response.
- [x] Implement runtime_metadata after load. Record canonical model/base paths and SHA256 for present small config/processor files (`config.json`, `generation_config.json`, `preprocessor_config.json`, `processor_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.json`, `chat_template.jinja`, `adapter_config.json`, Stage 1/2 PEFT metadata). Record resolved qwen config's stage/latent_size, latent token IDs resolved with existing runtime resolver, model/generation config to_dict(), active processor/chat template and kwargs, loaded Transformers __file__/__version__. Use JSON-compatible strings for actual dtype values. Do not read/hash weights or claim a controller executed from config alone.
- [x] Compare direct legacy invocation and bridge invocation with three distinct images and the same controlled returned sequence. Witness actual chat-template messages, processor image pixel ordering, tensors passed to controller, kwargs, prompt length, literal suffix IDs [21,22] and literal assistant response. Mutating only Sample.target while reusing ModelInput must not change messages or output. Cover Stage 1 and Stage 2 entry dispatch using existing small test models. This proves controlled call/data paths, not real GPU numerics.
- [x] GREEN: run both new bridge files plus the four existing targeted runtime tests listed in the discovery log; save separate lightweight and torch stub results. Stage only task files; no commit.

## Task 4: Generation CLI and failure outputs

**Files:** Create `scripts/run_eval.py`, `tests/evaluation/test_run_eval.py`.

**Interfaces:** Consume read_manifest, parse_config, get_scorer, Task 3 validators/load_runtime/generate_one/runtime_metadata. Produce `scripts.run_eval.main(argv: list[str] | None=None, *, repo_root: Path | None=None) -> int`.

Required CLI args: manifest, config, model-dir, model-identity, output-dir. Optional base-model-dir, cache-dir default data/cache/evaluation. Root is explicit repo_root or cwd. There is no resume/limit/judge flag.

- [x] Write RED tests before CLI code. Preflight failure must not call load_runtime at all; successful stub generation must be consumable by real P1 read_inputs/evaluate. Use make_vsp_case and monkeypatch only the loader/model side.

```python
def test_unknown_profile_never_imports_runtime(monkeypatch):
    case = make_vsp_case(1)
    case['config']['profile'] = 'unknown'
    config_file = case['data_root'] / 'config.yaml'
    config_file.write_text(yaml.safe_dump(case['config']))
    import scripts.run_eval as cli
    monkeypatch.setattr(cli, 'load_runtime', lambda: pytest.fail('runtime imported'))
    args = ['--manifest', str(case['manifest']), '--config', str(config_file),
            '--model-dir', str(case['model_dir']), '--model-identity', 'synthetic:model',
            '--output-dir', str(case['result_root'] / 'invalid')]
    # Choose a fresh direct-child output name; fixture result_root itself is a parent
    # only for fixture outputs, not a valid nested generation destination.
    args[-1] = str(case['root'] / 'results/evaluation' / (case['run_id'] + '-invalid'))
    assert cli.main(args, repo_root=case['root']) == 2
```

Add manifest with wrong asset hash, missing PNG, bad generation kwargs, mismatched declared identity, outside-checkpoint model/base, escaped cache, existing results/log, and malformed prompt cases. Use a runtime import blocker subprocess for at least one invalid-manifest case to prove no import-time heavy dependencies. Ensure each test isolates the claimed invalid field with otherwise valid input.

- [x] Implement preflight before load_runtime. Preserve P1 file parsing/YAML errors as exit 2 and unknown benchmark/profile failure. Validate output/log direct-child paths and fresh directories as score_eval does. Log input errors; do not create a successful generation output for them. model_identity must exactly equal config's declared identity.
- [x] After successful preflight, create fresh output/log/cache paths, empty predictions.jsonl and generation-details.jsonl, then write `run.json` with schema_version=1, status='running', protocol_kind='modified', manifest_sha256, run_fingerprint, N_expected, processed=0, declared run_metadata, paths and cache. Only then import/load runtime. Replace status with complete/failed and attach runtime metadata/error on normal finish/failure. Updating this one owned run.json is sufficient; no archival/recovery layer.
- [x] Generate sequentially. Build P1 Prediction using assistant_only, prompt_identity and stop_reason. Sidecar includes schema_version=1, id, hashes, status='ok', generated_suffix_token_ids, prompt_token_length, stop_reason, latent counts, inference_time and raw_output. Write and flush details before prediction. These are append-only during one process; never duplicate an ID.

```python
prediction = Prediction(sample.id, prepared.manifest_sha256,
                        prepared.run_fingerprint, output['assistant_only'], 'ok',
                        termination_reason=output['stop_reason'],
                        prompt_identity=output['prompt_identity'])
details_handle.write(canonical_json(details) + b'\n')
details_handle.flush()
predictions_handle.write(canonical_json(asdict(prediction)) + b'\n')
predictions_handle.flush()
```

On sample inference exception: write matching infrastructure_error detail/Prediction records with error string and null unavailable generation values, set failed run status, stop and return 1. On load failure leave empty prediction/detail files, record failure, return 1. On output write failure log failure, do not mark complete or fabricate new records; retain partial artifacts. On normal 0-token/blank/invalid/token-limit response store status ok, continue and return 0 if all IDs are recorded. Do not score within this CLI.

- [x] Test exact two-row success output and scorer accuracy, first-sample inference failure with remaining missing, model load failure with P1 all-missing score, blank and token-limit usable responses, and a controlled detail write failure that cannot result in complete status. Assert headers/fingerprints are actual config/manifest values, suffix [21,22] is preserved and prompt text never becomes the scoring response.
- [x] GREEN: run test_run_eval.py plus test_offline_cli.py and the lightweight bridge tests. Save logs and stage only task files; no commit.


Runtime-derived compatibility requirement: the existing Stage1 PEFT loader compares requested base_model_reference to metadata by exact string. Canonical path resolution is still mandatory for containment/run metadata, but loader argument must preserve the user-supplied base path as a lexical absolute path without resolving symlinks. Do not modify oldloader/checkpointmetadata or infer alternative paths. Document/use the checkpoint's exact existinglocal base reference. A symlinkalias strict-loader causal regression belongs to Task4. GPU01 failure andGPU02 success verify the rootcause.

## Task 5: Reference-free pipeline and real-data handoff

**Files:** Create `tests/evaluation/test_vsp_pipeline.py`, `configs/eval/vsp_ilvr_three_images_v1.yaml`; update `docs/evaluation/protocols.md`, `docs/evaluation/validation.md`.

**Interfaces:** Consume all earlier task interfaces and the unchanged scripts.score_eval.main CLI. No new production API is introduced.

- [x] Write a failing subprocess integration test using copies of the new lightweight modules/scripts and prepared synthetic assets into a fixture root below data/. Do not copy Reference, legacy eval.py, torch or model weights. Link that fixture root's data/results/logs/checkpoints to the active repository storage trees so generated test artifacts still obey storage policy. Provide a test-only fake `eval` module (via subprocess setup) implementing load_evaluation_artifacts and run_one_example with literal responses/suffixes; record the actual module origins and exact emitted sample keys/ordered images. Patch metadata introspection only at the fake-model boundary if necessary, without bypassing preflight or output handling. The real run CLI and score CLI must execute in this subprocess.
- [x] Require 2 expected/observed samples, manifest/fingerprint consistency, literal 2 correct responses for the synthetic map, preserved suffix IDs, and successful exit codes. Run a second controlled generation failure and assert incomplete P1 summary with accuracy=null. This is fixture integration evidence only.
- [x] Add the real config after its consuming test demonstrates the missing configured recipe. dataset_root is `data/evaluation/vsp-pinned-6d74d7bb46ae-v1`, six categories and max_steps=100. model_identity is `stage1-dora-continuous-v29`; generation kwargs use existing defaults: max_new_tokens=4096, max_sequence_length=32768, repetition_penalty=1.05, temperature=0.0, top_p=1.0, do_sample=false. prompt identity is computed from the pinned source with the exact approved build_prompt text and recorded literally in the YAML. Unit validation can point its dataset_root at a fixture while retaining the actual generation contract; no test may treat the real config as a synthetic benchmark run.
- [x] Run actual prepare CLI once against pinned Reference into the configured fresh data root. If that exact root already exists, do not overwrite: inspect whether it is the matching completed dataset, or choose an explicit new dataset ID and update the active config/documented command consistently. Store real 600/6 verification by reading both outputs with read_manifest and checking ordered IDs, category counts, three asset hashes and no Reference dependencies. Do not use fake model predictions as a 600-example benchmark score.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. \
./venv/ilvr-venv/bin/python -m scripts.prepare_eval \
  --benchmark vsp --profile vsp_validated_v1 \
  --source-root Reference/Visual-Spatial-Planning \
  --output-dir data/evaluation/vsp-pinned-6d74d7bb46ae-v1
```

- [x] Document exact prepare, model/base, run and score commands with separate fresh generation/scoring run IDs; cache and config values; declared identity versus content hashes; modified prompt/parser; exit/failure handling; 600/full versus 6/smoke and actual evidence locations. Keep P1 recipe/history and separate existing stub results from new bridge tests. Record available source adapter/base from the approved spec as GPU candidates, not proof of loadability.
- [x] GREEN: run test_vsp_pipeline.py and all new lightweight test files once, and read prepare output. Report source data and integration evidence. Stage task files; no commit. GPU evaluation is controller-owned below; never launch it from this worker.

## Controller verification and GPU smoke

After task reviews, perform one whole-change code review. Resolve material findings with one focused fix wave and one scoped re-review; preserve the P1 baseline and uncommitted state. Before selecting final tests, determine whether branch finishing applies. No merge/PR/worktree deletion is part of this approved task; do not claim whole-project readiness from evaluation tests.

- [x] Run one final claim-specific CPU suite: all P1 test_offline_*.py and test_vsp_offline.py/test_scoring.py, plus new test_manifest_preflight.py/test_vsp_prepare.py/test_inference_bridge.py/test_run_eval.py/test_vsp_pipeline.py. Run test_inference_bridge_runtime.py separately to label torch stub evidence. Exclude unrelated training/distributed tests from these claims. Save full command/output/exit code in a fresh logs/evaluation/p2-final-<id> and results/evaluation/p2-final-<id>/verification.json.
- [x] Confirm old eval/runtime/controller source bytes and existing CLI interfaces unchanged, reference root/commit/hash/clean still match the expanded lock, data/checkpoint links preserved, and actual prepared manifests have 600/6 rows. No test repetition is needed without a new change or unresolved failure.
- [x] GPU preflight: inspect adapter/base candidate named in the spec for actual weight files and loader metadata. Inspect free GPU memory/foreign processes with nvidia-smi; never signal other processes. Use the existing venv and local checkpoint paths. No new model conversion/copy or dependency installation.
- [x] If resources and candidates are valid, write one small one-shot probe script under the run's logs directory and launch it detached with an explicit run ID. The probe imports the implemented bridge and existing runtime, reads the prepared smoke manifest, loads existing adapter/base once, and compares direct and bridge generation for all six IDs. Override both comparisons to max_new_tokens=128, greedy, seed=1 to bound the smoke, save this exact config under data/evaluation/<gpu-run-id>/config.yaml, and record the distinction from the default 4096-token config. Compare actual prompt_identity, prompt_token_length, suffix IDs, assistant_only and stop_reason; store equality per field and both output records under results. Reset random/NumPy/torch seeds before each paired call; inspect existing controller cache/latent initialization and record the observed scope. Do not patch the generation logic to force equality.
- [x] Run the actual generation CLI on the same six-sample manifest using that bounded smoke config and a separate fresh generation run ID, then the real score CLI with a fresh score run ID. This is the minimal GPU CLI integration check after the direct/bridge witness; retain token-limit outcomes as normal observed model failures. Report its score as six-example smoke only, never full benchmark accuracy. Do not run 600 model examples in this P2 task.
- [x] A runtime failure gets one root-cause diagnosis, smallest causal fix/regression test and fresh run ID per AGENTS.md. Preserve old logs. If resources/model are unavailable, report GPU not_run with the observed reason and finish the CPU implementation without claiming all P2 validation passed.
- [x] Update validation/spec execution status with actual evidence, give the user results promptly, and write ntfy completion summary before final response. Keep SDD workspace because there are no commits replacing its review records.

## Plan self-review and execution status

Spec coverage: Task 1 handles shared preflight; Task 2 source/ID/prompt/copy/license/storage; Task 3 input-only bridge and parity boundaries; Task 4 outputs/status/lazy load; Task 5 config/integration/data recipe; controller handles fresh evidence/GPU and final review. Existing P1 semantics, modified protocol and unverified source-image/paper limits apply to all tasks.

Interfaces are defined once above and consumed by exact names. Source parser and preparation stubs in the interface section are declarations with explicit behavior below, not shippable placeholders. The execution choice is SDD as directed by AGENTS.md. Status: self-review and context-free cold read complete; ready for sequential SDD execution. Cold read found no handoff ambiguity. No implementation task started at plan finalization.

## Final execution status — 2026-09-10

Tasks 1–5 and their reviews are complete. Whole-change review resolved I1/M1
in one bounded fix wave; scoped re-review PASS. Final evidence: CPU 212 passed
in 1.98s in p2-final-20260910-01; torch stubs 7 passed in 8.80s plus source,
Reference and storage preservation in p2-final-20260910-02. The first final
driver omitted the pytest basetemp parent, causing three setup errors. Preserve
that log; the second driver adds parent creation and reruns only the affected
stub set and previously unrun preservation checks. A causal path probe is saved.
No product code changed for this driver fix and no CPU/GPU suite was repeated.

Actual source preparation: 600 full / 6 smoke rows; 602 distinct PNGs decoded.
GPU parity: six direct/bridge pairs match all five compared fields. Actual CLI
generation/scoring exits are 0/0; smoke accuracy is 0/6, with six invalid-format
responses ending at the explicit 128-token cap. Default 4096-token execution,
full benchmark, upstream Gym comparison, paper protocol and whole-project
training/distributed tests remain unrun. See docs/evaluation/validation.md for
commands, evidence links and limits. Keep this worktree and SDD review records;
no commit, merge, push or branch finishing was performed.
