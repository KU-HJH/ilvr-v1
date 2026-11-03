# ILVR Offline Evaluation Foundation and VSP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 고정 manifest와 저장 응답을 CPU에서 검증·재채점하는 VSP 평가 경로를 추가한다.

**Architecture:** 기존 `evaluation` 패키지에 독립 schema/io/registry/report/VSP 모듈을 추가한다. 모든 입력을 먼저 검증한 뒤 gold를 읽지 않는 추출과 target 기반 전이를 실행하며 manifest 분모와 순서를 유지한다. 기존 scorer와 추론 진입점은 변경하지 않는다.

**Tech Stack:** Python 3.11.15 (`./venv/ilvr-venv/bin/python`), 표준 라이브러리, 이미 설치된 PyYAML 6.0.3, pytest 9.1.1.

**Spec:** [승인된 design commitment v2](../specs/2026-09-10-ilvr-eval-foundation-vsp-design.md). 사용자 `Ok`는 설계 및 아래 검증 공백을 유지한 **계획 작성** 승인이다. **2026-09-10 후속 `ㅇㅋ`로 이 계획 및 P1 구현 승인 완료.**

**Evidence:** [P0 감사](../../evaluation/audit.md), [references.lock.json](../../../references.lock.json), [비평/미해결 이견](../../evaluation/p1-design-review.md). ILVR baseline `2497ee7bafad1b51ef6c2aba6f40f084830916c4`; lock SHA-256 `d35e562692d61364eeb1e7195b0d73221b3d1669feb526826f17675da97d4d24`.

**understood as:** 승인된 v2를 다음 구현자가 실행할 수 있는 P1 계획으로 구체화한다. 계획 작성은 완료됐고, 후속 승인에 따라 새 scorer/CLI/test를 구현한다. 실행 위치는 `.worktrees/evaluation-p1`이며 data symlink는 원래 저장 위치를 공유한다.

## Global Constraints

- Python은 `./venv/ilvr-venv/bin/python`, import는 `evaluation.*`, 실행 환경은 `PYTHONPATH=src`를 사용한다.
- P1에서는 dependency 설치/업그레이드나 별도 Python 환경을 만들지 않는다.
- 새 scorer는 표준 라이브러리만 사용하고 CLI YAML에는 이미 설치된 PyYAML을 사용한다.
- torch/Transformers/CUDA/MathRuler/Reference import 또는 원본 source의 런타임 AST 실행은 금지한다.
- `eval.py`, `src/evaluate_deepseed.py`, `src/task_deepseed.py`, 기존 `evaluation/{scoring,runtime,continuous_latent}.py`, patched Transformers와 생성 동작은 유지한다.
- 데이터·이미지·manifest·데이터 fixture/cache는 `data/evaluation/`; 응답·점수·표는 `results/evaluation/<run-id>/`; 실행 로그는 `logs/evaluation/<run-id>/`다.
- 기존 `data/` symlink를 유지한다. P1은 모델 파일을 만들지 않는다.
- 미구현 benchmark를 일반 scorer로 fallback하지 않는다.
- 샘플별 prompt hash는 별도 optional metadata이며 같은 run의 공통 fingerprint를 바꾸지 않는다.
- 실제 Gym runtime과의 대조는 별도다. P1 테스트 통과를 upstream/GPU/full-benchmark/paper 검증으로 올리지 않는다.
- Reference는 개발 근거이며 배포/runtime 의존성이 아니다. 원본 파일을 수정하거나 그 안에 산출물을 쓰지 않는다.
- 자동 commit/push하지 않는다. 아래 작업별 commit은 제안 경계이며, 실제 commit은 별도 사용자 지시가 있을 때만 한다.
- 사용자의 계획 승인 뒤, 현재 isolation을 읽기 전용 확인한다. 이미 격리되어 있으면 중첩 worktree를 만들지 않는다. 이 계획 작성 중에는 현재 루트와 branch를 유지한다.
- 구현은 Task 1→5 순차 SDD로 진행한다. 각 task는 self-review 후 spec 준수와 코드 품질 review를 받는다. reviewer/implementer는 paperthin hook을 호출하지 않는다.

## 산출물과 공통 규약

| Task | Create | Test |
|---|---|---|
| 1 | `src/evaluation/schema.py` | `tests/evaluation/test_offline_schema.py` |
| 2 | `src/evaluation/benchmarks/__init__.py`, `src/evaluation/benchmarks/vsp.py` | `tests/evaluation/test_vsp_offline.py` |
| 3 | `src/evaluation/io.py` | `tests/evaluation/offline_fixtures.py`, `tests/evaluation/test_offline_io.py` |
| 4 | `src/evaluation/registry.py`, `src/evaluation/report.py` | `tests/evaluation/test_offline_report.py` |
| 5 | `scripts/__init__.py`, `scripts/score_eval.py`, `configs/eval/vsp_validated_v1.yaml`, `docs/evaluation/protocols.md`, `docs/evaluation/validation.md` | `tests/evaluation/test_offline_cli.py` |

`src/evaluation/__init__.py`는 현재 docstring-only 상태를 유지한다. 추가 테스트는 아래 정확한 경로로 실행해 Reference/와 기존 GPU/latent 테스트를 수집하지 않는다. `pytest.ini`는 변경하지 않는다.

코드 블록은 **작성할 파일 내용 또는 그 파일의 구현 핵심**이다. 아직 실행된 구현이 아니다. 코드 핵심에 없는 검증은 task별 명시된 입출력/검증 표를 같은 RED→GREEN 주기에 포함한다. 표에서 요구하는 동작을 생략하고 코드 블록만 붙여넣어 완료로 표시하지 않는다.

공통 예외는 `ContractError(ValueError)`다. 모든 JSON 입력은 JSON-compatible 값만 허용하고 non-finite 숫자를 거부한다. `dict` 필드에는 알 수 없는 키를 거부한다. `bool`을 정수로 인정하지 않는다. 허용된 optional 필드만 누락할 수 있다.

### 구체적인 synthetic record

아래는 실제 benchmark 데이터가 아닌 P1 계약 예시다. source ID는 단순한 `s0`이며 실제 원본 복구를 주장하지 않는다.

```json
{"schema_version":1,"id":"vsp/planning/s0","benchmark":"vsp","subtask":"planning","model_input":{"text":"Synthetic map: start immediately left of goal.","images":[],"choices":[]},"target":{"map_desc":[[1,2],[0,0]],"max_steps":100},"provenance":{"dataset_id":"synthetic:vsp-contract","revision":"v1","config":"tiny","split":"fixture","source_id":"s0","asset_sha256":{}}}
```

manifest는 마지막 LF를 포함한 전체 bytes를 hash한다. Prediction의 hash는 실제 manifest와 run metadata에서 생성한다. Task 3의 fixture가 그 생성을 담당한다.

실데이터 후보는 pinned VSP의 `VSP-main/maze/task-main/maps/level5/{img,pure_text,table}/0.*`다. source ID를 `level5/0`, 향후 id를 `vsp/planning/level5/0`로 구성할 수 있다. 원본 table은 `_/#/@/*`, pure_text는 1-based 좌표를 사용한다. P1 Target의 `0/1/2/-1`로 변환하는 작업은 P2 범위다. 전체 600개 이미지의 내용 대응과 기존 prediction과의 join은 미검증이며 P1에서 자동 변환하지 않는다.

## Task 1: Typed contracts and identity

**Files:** Create `src/evaluation/schema.py`; Test `tests/evaluation/test_offline_schema.py`.

**Consumes:** 표준 라이브러리만 사용한다.

**Produces:** 아래 frozen dataclass와 예외 및 `canonical_json(value: object) -> bytes`, `run_fingerprint(run: RunMetadata) -> str`. nested dict까지 immutable인 것은 아니므로 검증된 객체는 호출자가 변경하지 않는다. JSON 경계의 타입/값 검증은 Task 3이 담당한다.

```python
from dataclasses import dataclass, asdict
from typing import Any, Literal
from pathlib import Path
import hashlib
import json

class ContractError(ValueError):
    pass

@dataclass(frozen=True)
class ModelInput:
    text: str
    images: tuple[str, ...]
    choices: tuple[str, ...]

@dataclass(frozen=True)
class VspTarget:
    map_desc: tuple[tuple[int | str, ...], ...]
    max_steps: int

@dataclass(frozen=True)
class Provenance:
    dataset_id: str
    revision: str
    config: str
    split: str
    source_id: str
    asset_sha256: dict[str, str]

@dataclass(frozen=True)
class Sample:
    schema_version: int
    id: str
    benchmark: str
    subtask: str
    model_input: ModelInput
    target: VspTarget
    provenance: Provenance

@dataclass(frozen=True)
class RunMetadata:
    model_identity: str
    prompt_template_identity: str
    generation_config: dict[str, Any]

@dataclass(frozen=True)
class Prediction:
    id: str
    manifest_sha256: str
    run_fingerprint: str
    response: str | None
    status: Literal['ok', 'infrastructure_error']
    error: str | None = None
    termination_reason: str | None = None
    prompt_identity: str | None = None

@dataclass(frozen=True)
class EvalConfig:
    schema_version: int
    benchmark: str
    profile: str
    categories: tuple[str, ...]
    dataset_root: Path
    max_steps: int
    run_metadata: RunMetadata

@dataclass(frozen=True)
class ParsedAnswer:
    value: str
    status: Literal['parsed', 'empty', 'invalid_format']
    method: Literal['raw', 'marker', 'boxed']
    error: str | None = None

@dataclass(frozen=True)
class Simulation:
    success: bool
    position: tuple[int, int]
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    reason: str

@dataclass(frozen=True)
class Score:
    id: str
    category: str
    benchmark: str
    profile: str
    scorer_version: str
    status: str
    parsed_answer: str | None
    extraction_method: str | None
    correctness: bool | None
    simulation: Simulation | None
    error: str | None

@dataclass(frozen=True)
class PreparedEvaluation:
    samples: tuple[Sample, ...]
    predictions: dict[str, Prediction]
    manifest_sha256: str
    run_fingerprint: str

def canonical_json(value: object) -> bytes:
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ContractError('JSON object keys must be strings')
            for child in item.values(): check(child)
        elif isinstance(item, (list, tuple)):
            for child in item: check(child)
    check(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (TypeError, ValueError) as exc:
        raise ContractError('Non-canonical JSON value') from exc

def run_fingerprint(run: RunMetadata) -> str:
    return hashlib.sha256(canonical_json(asdict(run))).hexdigest()
```

- [x] **Step 1 — RED test.** Add the following tests first. Do not add sample-specific prompt identity to RunMetadata.

```python
import pytest
from evaluation.schema import ContractError, RunMetadata, canonical_json, run_fingerprint

def test_fingerprint_ignores_mapping_order_but_changes_with_generation():
    a = RunMetadata('synthetic:model', 'synthetic:template', {'temperature': 0, 'max_new_tokens': 8})
    b = RunMetadata('synthetic:model', 'synthetic:template', {'max_new_tokens': 8, 'temperature': 0})
    c = RunMetadata('synthetic:model', 'synthetic:template', {'max_new_tokens': 9, 'temperature': 0})
    assert run_fingerprint(a) == run_fingerprint(b)
    assert run_fingerprint(a) != run_fingerprint(c)
    assert canonical_json({'b': 2, 'a': 1}) == b'{"a":1,"b":2}'

@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_non_finite_identity_is_rejected(value):
    with pytest.raises(ContractError):
        canonical_json({'temperature': value})
```

- [x] **Step 2 — Run RED.** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_offline_schema.py -q -p no:cacheprovider`. Expected first failure: missing `evaluation.schema`, not an unrelated import/runtime error.
- [x] **Step 3 — Implement** the contracts and identity functions above.
- [x] **Step 4 — Run GREEN** with the same targeted command and read the full output.
- [x] **Step 5 — Review boundary.** Check that neither `Sample` nor `RunMetadata` has a self-referential manifest hash or a sample-specific prompt hash. Suggested commit title, if later authorized: `feat(eval): define offline evaluation contracts`.

## Task 2: VSP extraction, grammar and deterministic transitions

**Files:** Create `src/evaluation/benchmarks/__init__.py` (empty/docstring only), `src/evaluation/benchmarks/vsp.py`; Test `tests/evaluation/test_vsp_offline.py`.

**Consumes:** Task 1 `ModelInput`, `VspTarget`, `ParsedAnswer`, `Simulation`, `ContractError`.

**Produces:** `extract_answer(raw_response: str, model_input: ModelInput) -> ParsedAnswer`, `validate_path(answer: ParsedAnswer) -> ParsedAnswer`, `normalize_map(target: VspTarget) -> tuple[tuple[str, ...], ...]`, `compare_answer(answer: ParsedAnswer, target: VspTarget) -> Simulation`.

Extraction has no Target parameter. `extract_answer` extracts a candidate span; `validate_path` performs whole-candidate validation and returns canonical uppercase actions. `compare_answer` only accepts a parsed, canonical path; its public boundary checks this again so direct callers cannot bypass grammar.

- [x] **Step 1 — RED tests.** Add the following cases before implementation.

```python
import inspect
import pytest
from evaluation.schema import ContractError, ModelInput, VspTarget, ParsedAnswer
from evaluation.benchmarks.vsp import extract_answer, validate_path, normalize_map, compare_answer

INPUT = ModelInput('Synthetic question', (), ())
TARGET = VspTarget(((1, 2), (0, 0)), 100)

def parse(text):
    return validate_path(extract_answer(text, INPUT))

@pytest.mark.parametrize('text', ['R', 'RD', 'final answer: R', r'\boxed{R}', 'Action Plan： r'])
def test_goal_ends_episode(text):
    result = compare_answer(parse(text), TARGET)
    assert (result.success, result.position, result.steps) == (True, (0, 1), 1)
    assert result.terminated and not result.truncated

@pytest.mark.parametrize('text', ['XR', 'RX', 'R.', '"R"', 'RIGHT', ',R', 'R,', 'R,,D', 'R\u00a0'])
def test_unknown_or_empty_action_group_is_not_erased(text):
    assert parse(text).status == 'invalid_format'

def test_extraction_has_no_gold_and_does_not_hide_malformed_last_box():
    assert 'target' not in inspect.signature(extract_answer).parameters
    assert parse('final answer: R').value == 'R'
    assert parse(r'\boxed{R} then \boxed{R').status == 'invalid_format'
    assert parse('Action plan: R\nX').status == 'invalid_format'
    assert parse('Reasoning\nFinal answer: R').value == 'R'
    assert parse('final answer: R\nFinal answer: D').value == 'D'
    assert parse(' \t\n').status == 'empty'
    assert parse('R, d\nL').value == 'RDL'

def test_hole_boundary_and_time_limit():
    hole = VspTarget(((1, -1), (0, 2)), 100)
    result = compare_answer(parse('RD'), hole)
    assert (result.success, result.steps, result.reason) == (False, 1, 'hole')
    result = compare_answer(parse('LR'), VspTarget(((1, 2),), 1))
    assert (result.position, result.steps, result.reason) == ((0, 0), 1, 'step_limit')
    assert result.truncated and not result.terminated
    result = compare_answer(parse('L' * 99 + 'R'), TARGET)
    assert result.success and result.terminated and result.truncated and result.steps == 100
    result = compare_answer(parse('L' * 100 + 'R'), TARGET)
    assert not result.success and result.truncated and result.steps == 100
    result = compare_answer(parse('D'), TARGET)
    assert result.reason == 'path_exhausted' and not result.terminated and not result.truncated

@pytest.mark.parametrize('grid', [(), ((1, 2), (0,)), ((True, 2),), ((1, 1, 2),), ((1, 0),), ((1, 2, 3),)])
def test_invalid_map_is_contract_error(grid):
    with pytest.raises(ContractError):
        normalize_map(VspTarget(grid, 100))

def test_character_map_and_missing_gold_path_are_valid():
    assert normalize_map(TARGET) == (('S', 'G'), ('F', 'F'))
    assert normalize_map(VspTarget((('S', 'G'),), 100)) == (('S', 'G'),)

def test_direct_compare_cannot_bypass_grammar_and_box_is_balanced():
    with pytest.raises(ContractError):
        compare_answer(ParsedAnswer('RX', 'parsed', 'raw'), TARGET)
    assert parse(r'\boxed{}').status == 'empty'
    assert parse(r'\boxed{{R}}').status == 'invalid_format'
    assert parse(r'\boxed{D} then \boxed{R}').value == 'R'

@pytest.mark.parametrize('limit', [0, -1, True, 1.5])
def test_invalid_step_limit(limit):
    with pytest.raises(ContractError):
        normalize_map(VspTarget(((1, 2),), limit))
```

- [x] **Step 2 — Run RED.** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_vsp_offline.py -q -p no:cacheprovider`. Expected missing `evaluation.benchmarks.vsp`.
- [x] **Step 3 — Implement** these algorithms. Only the approved separators are stripped; Unicode whitespace and punctuation are not silently erased.

```python
import re
from evaluation.schema import ContractError, ModelInput, ParsedAnswer, Simulation, VspTarget

WS = ' \t\r\n'
MARKER = re.compile(r'^[ \t]*(?:final[ \t]+answer(?:[ \t]+is)?|action[ \t]+plan)[ \t]*[:：]', re.I | re.M)

def extract_answer(raw_response: str, model_input: ModelInput) -> ParsedAnswer:
    if not isinstance(raw_response, str):
        raise ContractError('response must be a string')
    start = raw_response.rfind('\\boxed{')
    if start >= 0:
        begin, depth = start + len('\\boxed{'), 1
        for pos in range(begin, len(raw_response)):
            depth += (raw_response[pos] == '{') - (raw_response[pos] == '}')
            if depth == 0:
                value = raw_response[begin:pos].strip(WS)
                return ParsedAnswer(value, 'parsed' if value else 'empty', 'boxed')
        return ParsedAnswer('', 'invalid_format', 'boxed', 'unclosed_box')
    markers = list(MARKER.finditer(raw_response))
    value = raw_response[markers[-1].end():] if markers else raw_response
    value = value.strip(WS)
    return ParsedAnswer(value, 'parsed' if value else 'empty', 'marker' if markers else 'raw')

def validate_path(answer: ParsedAnswer) -> ParsedAnswer:
    if answer.status != 'parsed':
        return answer
    value = answer.value.upper()
    if not re.fullmatch(r'[UDLR ,\t\r\n]+', value) or any(not group.strip(WS) for group in value.split(',')):
        return ParsedAnswer(answer.value, 'invalid_format', answer.method, 'invalid_actions')
    return ParsedAnswer(''.join(ch for ch in value if ch in 'UDLR'), 'parsed', answer.method)

def normalize_map(target: VspTarget) -> tuple[tuple[str, ...], ...]:
    if type(target.max_steps) is not int or target.max_steps <= 0:
        raise ContractError('max_steps must be a positive integer')
    grid = target.map_desc
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        raise ContractError('map must be a nonempty rectangle')
    mapping = {0: 'F', 1: 'S', 2: 'G', -1: 'H'}
    result = []
    for row in grid:
        cells = []
        for cell in row:
            if type(cell) is int and cell in mapping:
                cells.append(mapping[cell])
            elif type(cell) is str and cell in ('F', 'S', 'G', 'H'):
                cells.append(cell)
            else:
                raise ContractError('unknown map cell')
        result.append(tuple(cells))
    flat = [cell for row in result for cell in row]
    if flat.count('S') != 1 or flat.count('G') != 1:
        raise ContractError('map requires exactly one start and goal')
    return tuple(result)

def compare_answer(answer: ParsedAnswer, target: VspTarget) -> Simulation:
    if answer.status != 'parsed' or not re.fullmatch('[UDLR]+', answer.value):
        raise ContractError('compare requires canonical parsed actions')
    grid = normalize_map(target)
    row, col = next((r, c) for r, cells in enumerate(grid) for c, cell in enumerate(cells) if cell == 'S')
    moves = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    for step, action in enumerate(answer.value, 1):
        dr, dc = moves[action]
        nr, nc = row + dr, col + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            row, col = nr, nc
        cell = grid[row][col]
        terminated = cell in ('G', 'H')
        truncated = step >= target.max_steps
        reward = 1.0 if cell == 'G' else 0.0
        if terminated or truncated:
            reason = 'goal' if cell == 'G' else 'hole' if cell == 'H' else 'step_limit'
            return Simulation(bool(reward), (row, col), step, reward, terminated, truncated, reason)
    return Simulation(False, (row, col), len(answer.value), 0.0, False, False, 'path_exhausted')
```

- [x] **Step 4 — Run GREEN** for the same VSP test path. Record `RD` as a deliberate difference from P0 legacy behavior.
- [x] **Step 5 — Review boundary.** Check strict candidate validation precedes simulation; boxed content isolation differs intentionally from marker-tail validation. Suggested commit: `feat(eval): add explicit VSP parser and transition contract`.

## Task 3: Parse config, validate manifests, and join predictions

**Files:** Create `src/evaluation/io.py`, `tests/evaluation/offline_fixtures.py`; Test `tests/evaluation/test_offline_io.py`.

**Consumes:** Task 1 types/identity functions; Task 2 `normalize_map`.

**Produces:** `parse_config(raw: dict, repo_root: Path) -> EvalConfig`, `read_inputs(manifest_path: Path, predictions_path: Path, config: EvalConfig, repo_root: Path) -> PreparedEvaluation`, `parse_sample(raw: dict) -> Sample`, `parse_prediction(raw: dict) -> Prediction`.

### JSON boundary requirements

| Boundary | Required validation and conversion |
|---|---|
| EvalConfig | Exact keys equal dataclass fields. schema_version exact int 1; benchmark/profile nonempty strings; categories nonempty unique list of nonempty strings; max_steps positive exact int; dataset_root path resolves inside current repo_root/data; RunMetadata exact keys, two nonempty strings, generation_config dict accepted by canonical_json |
| Sample | Exact keys; schema_version exact int 1; id/benchmark/subtask nonempty str; nested exact-key dictionaries; convert images/choices to tuples of str, map list-of-lists to tuple-of-tuples |
| ModelInput | Exact text/images/choices only, so trace/helper/answer injection through extra fields fails. text is str; image paths are nonempty relative strings; choices are strings. Empty images/choices are allowed for synthetic text-only fixtures |
| Target | Exact map_desc/max_steps only; call normalize_map; max_steps must equal config.max_steps |
| Provenance | Exact fields from dataclass; six named values consist of five nonempty strings plus asset_sha256 dict; its keys exactly equal set(ModelInput.images), values lowercase 64-digit SHA-256 |
| Identity/join | Sample.id exactly `benchmark + '/' + subtask + '/' + provenance.source_id`; sample benchmark equals config; subtask in config.categories; no duplicate sample IDs |
| Assets | Resolve each listed image under config.dataset_root, reject absolute paths or escape through `..`/symlinks, require readable regular file, verify byte SHA-256. Preserve image order. Config root must be under repo_root/data after resolving the existing data symlink |
| Prediction | Required five fields id/manifest_sha256/run_fingerprint/response/status, optional error/termination_reason/prompt_identity only; id nonempty, both hashes lowercase64; status `ok` or `infrastructure_error`; ok requires str response and error=null; error status requires nonempty str error and response str or null; optional identity/reason are str or null |
| Population | Manifest must have at least one row. Existing zero-byte/whitespace-only Prediction JSONL yields empty dict and incomplete output. Nonempty JSONL lines must be objects, not lists/null. Reject duplicate/unknown prediction IDs and mismatched hashes/run fingerprint. A partial known-ID set is allowed |

Do not read gold while parsing a response. IO may validate target fields before extraction. It does not synthesize targets/IDs from questions or inspect model files. JSON parsing rejects NaN/Infinity with `parse_constant` and duplicate object keys with an `object_pairs_hook` that raises `ContractError`.

- [x] **Step 1 — RED fixtures/tests.** Implement only the test helper and tests first. Test fixtures store dataset files under real `data/`, prediction files under real `results/`; tests use fresh identifiers so nothing is overwritten. `repo_root` below is the actual project root from the test file location.

```python
# tests/evaluation/offline_fixtures.py
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import uuid
from evaluation.schema import RunMetadata, run_fingerprint

def make_case(sample_count=2):
    repo_root = Path(__file__).resolve().parents[2]
    run_id = 'p1-test-' + uuid.uuid4().hex
    data_root = repo_root / 'data/evaluation' / run_id
    result_root = repo_root / 'results/evaluation' / run_id
    data_root.mkdir(parents=True, exist_ok=False)
    result_root.mkdir(parents=True, exist_ok=False)
    run = RunMetadata('synthetic:model', 'synthetic:template', {'max_new_tokens': 8})
    samples = []
    for index in range(sample_count):
        source_id = f's{index}'
        samples.append(dict(schema_version=1, id=f'vsp/planning/{source_id}',
            benchmark='vsp', subtask='planning',
            model_input=dict(text='Synthetic question', images=[], choices=[]),
            target=dict(map_desc=[[1, 2], [0, 0]], max_steps=100),
            provenance=dict(dataset_id='synthetic:vsp-contract', revision='v1',
                config='tiny', split='fixture', source_id=source_id, asset_sha256={})))
    manifest = data_root / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in samples))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    predictions = [dict(id=row['id'], manifest_sha256=manifest_hash,
        run_fingerprint=run_fingerprint(run), response='R', status='ok') for row in samples]
    prediction_path = result_root / 'predictions.jsonl'
    prediction_path.write_text(''.join(json.dumps(row) + '\n' for row in predictions))
    config = dict(schema_version=1, benchmark='vsp', profile='vsp_validated_v1',
        categories=['planning', 'planning-empty'], dataset_root=str(data_root),
        max_steps=100, run_metadata=asdict(run))
    return dict(root=repo_root, run_id=run_id, data_root=data_root,
        result_root=result_root, manifest=manifest, prediction_path=prediction_path,
        samples=samples, predictions=predictions, config=config)

def save_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

def save_manifest(case):
    save_rows(case['manifest'], case['samples'])
    digest = hashlib.sha256(case['manifest'].read_bytes()).hexdigest()
    for row in case['predictions']:
        row['manifest_sha256'] = digest
    save_rows(case['prediction_path'], case['predictions'])
```

```python
# tests/evaluation/test_offline_io.py
import copy
import pytest
from offline_fixtures import make_case, save_rows
from evaluation.schema import ContractError
from evaluation.io import parse_config, read_inputs, parse_sample, parse_prediction

def load(case):
    config = parse_config(case['config'], case['root'])
    return read_inputs(case['manifest'], case['prediction_path'], config, case['root'])

def test_map_only_gold_and_partial_population():
    case = make_case()
    save_rows(case['prediction_path'], case['predictions'][:1])
    prepared = load(case)
    assert len(prepared.samples) == 2 and len(prepared.predictions) == 1
    save_rows(case['prediction_path'], [])
    assert load(case).predictions == {}

@pytest.mark.parametrize('kind', ['duplicate', 'unknown', 'manifest', 'run'])
def test_bad_prediction_join_fails_closed(kind):
    case = make_case()
    rows = copy.deepcopy(case['predictions'])
    if kind == 'duplicate': rows.append(rows[0])
    if kind == 'unknown': rows[0]['id'] = 'vsp/planning/not-in-manifest'
    if kind == 'manifest': rows[0]['manifest_sha256'] = '0' * 64
    if kind == 'run': rows[0]['run_fingerprint'] = '0' * 64
    save_rows(case['prediction_path'], rows)
    with pytest.raises(ContractError): load(case)

@pytest.mark.parametrize('field', ['answer', 'Text Reasoning Trace', 'reasoning_image_1'])
def test_gold_or_trace_cannot_enter_model_input(field):
    case = make_case(1)
    row = case['samples'][0]
    row['model_input'][field] = 'leak'
    with pytest.raises(ContractError): parse_sample(row)

def test_infrastructure_status_requires_error_and_preserves_raw_text():
    case = make_case(1)
    row = case['predictions'][0] | dict(status='infrastructure_error', error='worker failed', response='R')
    assert parse_prediction(row).response == 'R'
    row['error'] = ''
    with pytest.raises(ContractError): parse_prediction(row)
```

Append these boundary tests to the same test file. Dummy image bytes test existence/hash, not image decoding. `save_manifest` refreshes prediction hashes so a different invalid field cannot be masked by an incidental hash mismatch.

```python
import hashlib
from offline_fixtures import save_manifest

@pytest.mark.parametrize('kind', ['version', 'target', 'id', 'duplicate', 'limit', 'category'])
def test_invalid_manifest_contract(kind):
    case = make_case(1)
    row = case['samples'][0]
    if kind == 'version': row['schema_version'] = True
    if kind == 'target': del row['target']
    if kind == 'id': row['id'] = 'wrong'
    if kind == 'duplicate': case['samples'].append(copy.deepcopy(row))
    if kind == 'limit': row['target']['max_steps'] = 99
    if kind == 'category': case['config']['categories'] = ['other']
    save_manifest(case)
    with pytest.raises(ContractError): load(case)

@pytest.mark.parametrize('payload', [b'', b'[]\n', b'null\n', b'{"x":NaN}\n', b'{"x":1,"x":2}\n', b'{'])
def test_bad_manifest_json_or_empty_manifest(payload):
    case = make_case(1)
    case['manifest'].write_bytes(payload)
    with pytest.raises(ContractError): load(case)

@pytest.mark.parametrize('key', ['manifest', 'prediction_path'])
def test_missing_input_file(key):
    case = make_case(1)
    case[key].unlink()
    with pytest.raises(ContractError): load(case)

@pytest.mark.parametrize('kind', ['good', 'missing', 'hash', 'absolute', 'parent', 'symlink'])
def test_asset_containment_and_hash(kind):
    case = make_case(1)
    # All asset bytes stay under data/, including the out-of-dataset probe.
    asset = case['data_root'] / 'image.bin'
    asset.write_bytes(b'fixture-image')
    outside = case['data_root'].parent / (case['run_id'] + '-outside.bin')
    outside.write_bytes(b'fixture-image')
    name = 'image.bin'
    if kind == 'missing': name = 'missing.bin'
    if kind == 'absolute': name = str(asset)
    if kind == 'parent': name = '../' + outside.name
    if kind == 'symlink':
        (case['data_root'] / 'escape.bin').symlink_to(outside)
        name = 'escape.bin'
    row = case['samples'][0]
    row['model_input']['images'] = [name]
    row['provenance']['asset_sha256'] = {
        name: '0' * 64 if kind == 'hash' else hashlib.sha256(asset.read_bytes()).hexdigest()}
    save_manifest(case)
    if kind == 'good': assert len(load(case).samples) == 1
    else:
        with pytest.raises(ContractError): load(case)

@pytest.mark.parametrize('kind', ['root', 'identity', 'unknown_field', 'generation_nan'])
def test_config_boundary(kind):
    case = make_case(1)
    if kind == 'root': case['config']['dataset_root'] = str(case['result_root'])
    if kind == 'identity': del case['config']['run_metadata']['model_identity']
    if kind == 'unknown_field': case['config']['fallback'] = True
    if kind == 'generation_nan': case['config']['run_metadata']['generation_config']['temperature'] = float('nan')
    with pytest.raises(ContractError): load(case)

def test_sample_prompt_identity_does_not_change_run_identity():
    case = make_case(2)
    for index, row in enumerate(case['predictions']): row['prompt_identity'] = f'prompt-{index}'
    save_rows(case['prediction_path'], case['predictions'])
    assert len(load(case).predictions) == 2
```

- [x] **Step 2 — Run RED.** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_offline_io.py -q -p no:cacheprovider`. Expected missing `evaluation.io` after Task 1/2 are green.
- [x] **Step 3 — Implement parsing and preflight.** Use these concrete helpers/core, with the table checks at their named boundary. Dataclass nested construction must use already type-checked dictionaries; convert boundary `TypeError`, `ValueError`, `OSError`, `UnicodeError` to contextual ContractError, without a broad catch around scoring.

```python
from dataclasses import fields
from pathlib import Path
import hashlib
import json
import re
from evaluation.schema import (ContractError, EvalConfig, ModelInput, Prediction,
    PreparedEvaluation, Provenance, RunMetadata, Sample, VspTarget,
    canonical_json, run_fingerprint)
from evaluation.benchmarks.vsp import normalize_map

def exact_keys(raw, required, optional=()):
    if not isinstance(raw, dict) or not set(required) <= set(raw) or set(raw) - set(required) - set(optional):
        raise ContractError('unexpected or missing fields')

def nonempty(value):
    if not isinstance(value, str) or not value.strip():
        raise ContractError('expected nonempty string')
    return value

def sha256(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ContractError('expected SHA-256')
    return value

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ContractError('duplicate JSON key')
        result[key] = value
    return result

def reject_constant(value):
    raise ContractError('non-finite JSON number: ' + value)

def read_jsonl_bytes(payload: bytes) -> list[dict]:
    result = []
    for number, line in enumerate(payload.decode('utf-8').split('\n'), 1):
        if not line.strip(): continue
        row = json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_object)
        if not isinstance(row, dict): raise ContractError(f'row {number} is not an object')
        result.append(row)
    return result

def check_assets(sample: Sample, config: EvalConfig):
    if set(sample.provenance.asset_sha256) != set(sample.model_input.images):
        raise ContractError('asset hash keys must match input images')
    for image in sample.model_input.images:
        if Path(image).is_absolute(): raise ContractError('absolute image path')
        path = (config.dataset_root / image).resolve()
        if not path.is_relative_to(config.dataset_root) or not path.is_file():
            raise ContractError('missing or escaped image')
        expected = sha256(sample.provenance.asset_sha256[image])
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ContractError('image hash mismatch')

def read_inputs(manifest_path: Path, predictions_path: Path, config: EvalConfig,
                repo_root: Path) -> PreparedEvaluation:
    try:
        data_tree = (repo_root / 'data').resolve()
        if not manifest_path.resolve().is_relative_to(data_tree):
            raise ContractError('manifest must be under data/')
        if not predictions_path.resolve().is_relative_to((repo_root / 'results').resolve()):
            raise ContractError('predictions must be under results/')
        payload = manifest_path.read_bytes()
        samples = tuple(parse_sample(raw) for raw in read_jsonl_bytes(payload))
        if not samples: raise ContractError('empty manifest')
        known = set()
        for sample in samples:
            expected_id = f'{sample.benchmark}/{sample.subtask}/{sample.provenance.source_id}'
            if sample.id != expected_id or sample.id in known:
                raise ContractError('invalid or duplicate sample ID')
            if sample.benchmark != config.benchmark or sample.subtask not in config.categories:
                raise ContractError('unexpected benchmark/category')
            if sample.target.max_steps != config.max_steps:
                raise ContractError('target/config step limit mismatch')
            normalize_map(sample.target)
            check_assets(sample, config)
            known.add(sample.id)
        manifest_hash = hashlib.sha256(payload).hexdigest()
        fingerprint = run_fingerprint(config.run_metadata)
        predictions = {}
        for raw in read_jsonl_bytes(predictions_path.read_bytes()):
            prediction = parse_prediction(raw)
            if prediction.id not in known or prediction.id in predictions:
                raise ContractError('unknown or duplicate prediction ID')
            if prediction.manifest_sha256 != manifest_hash or prediction.run_fingerprint != fingerprint:
                raise ContractError('prediction identity mismatch')
            predictions[prediction.id] = prediction
        return PreparedEvaluation(samples, predictions, manifest_hash, fingerprint)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        if isinstance(exc, ContractError): raise
        raise ContractError(f'input preflight failed: {exc}') from exc
```

Add these exact boundary constructors to the same io.py file. Registry validates support before CLI read_inputs; IO validates schema and population.

```python
def version_one(value):
    if type(value) is not int or value != 1:
        raise ContractError('unsupported schema version')

def strings(value, *, nonempty_items=False):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ContractError('expected list of strings')
    if nonempty_items:
        for item in value: nonempty(item)
    return tuple(value)

def parse_config(raw: dict, repo_root: Path) -> EvalConfig:
    exact_keys(raw, [field.name for field in fields(EvalConfig)])
    version_one(raw['schema_version'])
    for key in ('benchmark', 'profile', 'dataset_root'): nonempty(raw[key])
    categories = strings(raw['categories'], nonempty_items=True)
    if not categories or len(set(categories)) != len(categories):
        raise ContractError('categories must be nonempty and unique')
    if type(raw['max_steps']) is not int or raw['max_steps'] <= 0:
        raise ContractError('invalid config max_steps')
    meta = raw['run_metadata']
    exact_keys(meta, [field.name for field in fields(RunMetadata)])
    nonempty(meta['model_identity']); nonempty(meta['prompt_template_identity'])
    if not isinstance(meta['generation_config'], dict):
        raise ContractError('generation_config must be an object')
    canonical_json(meta['generation_config'])
    data_root = (repo_root / raw['dataset_root']).resolve()
    if not data_root.is_relative_to((repo_root / 'data').resolve()) or not data_root.is_dir():
        raise ContractError('dataset root must exist under data/')
    return EvalConfig(1, raw['benchmark'], raw['profile'], categories, data_root,
                      raw['max_steps'], RunMetadata(**meta))

def parse_sample(raw: dict) -> Sample:
    exact_keys(raw, [field.name for field in fields(Sample)])
    version_one(raw['schema_version'])
    for key in ('id', 'benchmark', 'subtask'): nonempty(raw[key])
    model = raw['model_input']; target = raw['target']; provenance = raw['provenance']
    exact_keys(model, [field.name for field in fields(ModelInput)])
    if not isinstance(model['text'], str): raise ContractError('text must be a string')
    images = strings(model['images'], nonempty_items=True)
    choices = strings(model['choices'])
    exact_keys(target, [field.name for field in fields(VspTarget)])
    grid = target['map_desc']
    if not isinstance(grid, list) or any(not isinstance(row, list) for row in grid):
        raise ContractError('map_desc must be a list of lists')
    parsed_target = VspTarget(tuple(tuple(row) for row in grid), target['max_steps'])
    normalize_map(parsed_target)
    exact_keys(provenance, [field.name for field in fields(Provenance)])
    for key in ('dataset_id', 'revision', 'config', 'split', 'source_id'):
        nonempty(provenance[key])
    hashes = provenance['asset_sha256']
    if not isinstance(hashes, dict): raise ContractError('asset_sha256 must be an object')
    for key, value in hashes.items(): nonempty(key); sha256(value)
    return Sample(1, raw['id'], raw['benchmark'], raw['subtask'],
        ModelInput(model['text'], images, choices), parsed_target, Provenance(**provenance))

def parse_prediction(raw: dict) -> Prediction:
    exact_keys(raw, ('id', 'manifest_sha256', 'run_fingerprint', 'response', 'status'),
               ('error', 'termination_reason', 'prompt_identity'))
    nonempty(raw['id']); sha256(raw['manifest_sha256']); sha256(raw['run_fingerprint'])
    status = raw['status']; response = raw['response']; error = raw.get('error')
    if status == 'ok':
        if not isinstance(response, str) or error is not None:
            raise ContractError('ok prediction requires string response and no error')
    elif status == 'infrastructure_error':
        nonempty(error)
        if response is not None and not isinstance(response, str):
            raise ContractError('error response must be string or null')
    else:
        raise ContractError('unknown prediction status')
    for key in ('termination_reason', 'prompt_identity'):
        if raw.get(key) is not None and not isinstance(raw[key], str):
            raise ContractError('optional metadata must be string or null')
    return Prediction(**raw)
```

- [x] **Step 4 — Run GREEN** for test_offline_io.py. Then run schema/VSP/IO together once to check the shared contracts.
- [x] **Step 5 — Review boundary.** Review source→ID→manifest bytes→Prediction join, resolved data paths, unknown-key rejection and absent-versus-empty records. Suggested commit: `feat(eval): validate offline manifests and prediction lineage`.

## Task 4: Explicit registry, scoring rows, fixed-denominator summaries

**Files:** Create `src/evaluation/registry.py`, `src/evaluation/report.py`; Test `tests/evaluation/test_offline_report.py`.

**Consumes:** PreparedEvaluation/EvalConfig/Score and Task 2 functions.

**Produces:** `get_scorer(benchmark: str, profile: str) -> Callable[[Sample, Prediction], Score]`; `score_rows(prepared: PreparedEvaluation, config: EvalConfig) -> list[Score]`; `summarize(scores: list[Score], config: EvalConfig) -> dict`; `evaluate(prepared: PreparedEvaluation, config: EvalConfig) -> tuple[list[Score], dict]`.

Only `('vsp','vsp_validated_v1')` is registered. Legacy is an audit name and raises here. No dynamic imports or plugin registry framework.

- [x] **Step 1 — RED tests.**

```python
import pytest
from offline_fixtures import make_case, save_rows
from evaluation.io import parse_config, read_inputs
from evaluation.registry import get_scorer
from evaluation.report import evaluate
from evaluation.schema import ContractError

def run(case):
    config = parse_config(case['config'], case['root'])
    prepared = read_inputs(case['manifest'], case['prediction_path'], config, case['root'])
    return evaluate(prepared, config)

def test_missing_predictions_do_not_shrink_denominator():
    case = make_case(2)
    save_rows(case['prediction_path'], case['predictions'][:1])
    scores, summary = run(case)
    assert [s.id for s in scores] == ['vsp/planning/s0', 'vsp/planning/s1']
    assert summary['N_expected'] == 2 and summary['correct'] == 1
    assert summary['coverage'] == 0.5 and summary['accuracy'] is None
    assert summary['complete'] is False
    assert summary['categories']['planning']['accuracy'] is None
    assert summary['categories']['planning-empty'] == dict(count=0, observed=0, correct=0, complete=True, accuracy=None)

def test_empty_prediction_file_is_incomplete():
    case = make_case(2)
    save_rows(case['prediction_path'], [])
    _, summary = run(case)
    assert summary['N_expected'] == 2 and summary['coverage'] == 0
    assert summary['complete'] is False and summary['accuracy'] is None

def test_empty_and_invalid_are_model_failures_but_infrastructure_is_not():
    case = make_case(3)
    rows = case['predictions']
    rows[0]['response'] = ''
    rows[1]['response'] = 'RX'
    rows[2].update(status='infrastructure_error', error='worker failed', response='R')
    save_rows(case['prediction_path'], rows)
    scores, summary = run(case)
    assert [s.correctness for s in scores] == [False, False, None]
    assert [s.status for s in scores] == ['empty', 'invalid_format', 'infrastructure_error']
    assert summary['coverage'] == pytest.approx(2/3)
    assert summary['accuracy'] is None
    rows[2].update(status='ok', error=None)
    save_rows(case['prediction_path'], rows)
    _, summary = run(case)
    assert summary['complete'] is True and summary['accuracy'] == pytest.approx(1/3)

@pytest.mark.parametrize('benchmark,profile', [('unknown','vsp_validated_v1'), ('vsp','unknown'), ('vsp','vsp_ilvr_legacy')])
def test_unknown_registry_values_fail(benchmark, profile):
    with pytest.raises(ContractError): get_scorer(benchmark, profile)
```

- [x] **Step 2 — Run RED.** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_offline_report.py -q -p no:cacheprovider`. Expected missing registry/report.
- [x] **Step 3 — Implement** this dispatch/aggregation core.

```python
# registry.py
from evaluation.schema import ContractError, Prediction, Sample, Score
from evaluation.benchmarks.vsp import extract_answer, validate_path, compare_answer

def score_vsp(sample: Sample, prediction: Prediction) -> Score:
    if prediction.status != 'ok': raise ContractError('score_vsp requires ok prediction')
    parsed = validate_path(extract_answer(prediction.response, sample.model_input))
    simulation = compare_answer(parsed, sample.target) if parsed.status == 'parsed' else None
    return Score(sample.id, sample.subtask, 'vsp', 'vsp_validated_v1', '1',
        parsed.status, parsed.value, parsed.method,
        simulation.success if simulation else False, simulation, parsed.error)

def get_scorer(benchmark: str, profile: str):
    if (benchmark, profile) != ('vsp', 'vsp_validated_v1'):
        raise ContractError('unsupported benchmark/profile')
    return score_vsp
```

```python
# report.py
from collections import Counter
from evaluation.registry import get_scorer
from evaluation.schema import ContractError, EvalConfig, PreparedEvaluation, Score

def score_rows(prepared: PreparedEvaluation, config: EvalConfig) -> list[Score]:
    scorer = get_scorer(config.benchmark, config.profile)
    result = []
    for sample in prepared.samples:
        prediction = prepared.predictions.get(sample.id)
        if prediction is None or prediction.status == 'infrastructure_error':
            status = 'missing' if prediction is None else 'infrastructure_error'
            error = 'prediction not received' if prediction is None else prediction.error
            result.append(Score(sample.id, sample.subtask, config.benchmark, config.profile,
                '1', status, None, None, None, None, error))
        else:
            result.append(scorer(sample, prediction))
    return result

def category_stats(rows: list[Score]) -> dict:
    observed = sum(row.correctness is not None for row in rows)
    correct = sum(row.correctness is True for row in rows)
    complete = observed == len(rows)
    return dict(count=len(rows), observed=observed, correct=correct, complete=complete,
                accuracy=correct/len(rows) if complete and rows else None)

def summarize(scores: list[Score], config: EvalConfig) -> dict:
    if not scores or len({row.id for row in scores}) != len(scores):
        raise ContractError('nonempty unique score population required')
    if any(row.category not in config.categories for row in scores):
        raise ContractError('unexpected score category')
    total = category_stats(scores)
    return dict(N_expected=total['count'], observed=total['observed'], correct=total['correct'],
        complete=total['complete'], accuracy=total['accuracy'],
        coverage=total['observed']/total['count'], status_counts=dict(Counter(row.status for row in scores)),
        categories={category: category_stats([row for row in scores if row.category == category])
                    for category in config.categories})

def evaluate(prepared: PreparedEvaluation, config: EvalConfig) -> tuple[list[Score], dict]:
    scores = score_rows(prepared, config)
    summary = summarize(scores, config)
    summary.update(schema_version=1, benchmark=config.benchmark, profile=config.profile,
        scorer_version='1', manifest_sha256=prepared.manifest_sha256,
        run_fingerprint=prepared.run_fingerprint, protocol_kind='modified',
        verification=dict(cpu_tests='not_recorded_by_scoring_run', upstream_environment='not_run',
            gpu_smoke='not_run', full_benchmark='not_run', paper_protocol_verified=False))
    return scores, summary
```

Append these cases to the report RED test set:

```python
from offline_fixtures import save_manifest

def test_complete_category_survives_another_incomplete_category():
    case = make_case(2)
    second = case['samples'][1]
    second['subtask'] = 'planning-empty'
    second['id'] = 'vsp/planning-empty/s1'
    case['predictions'][1]['id'] = second['id']
    save_manifest(case)
    save_rows(case['prediction_path'], case['predictions'][:1])
    _, summary = run(case)
    assert summary['categories']['planning']['accuracy'] == 1.0
    assert summary['categories']['planning-empty']['accuracy'] is None
    assert summary['accuracy'] is None and not summary['complete']

def test_prediction_order_does_not_change_population_order():
    case = make_case(2)
    save_rows(case['prediction_path'], list(reversed(case['predictions'])))
    scores, _ = run(case)
    assert [row.id for row in scores] == [row['id'] for row in case['samples']]
```

- [x] **Step 4 — Run GREEN** for the same report tests, including category independence and manifest order.
- [x] **Step 5 — Review boundary.** Ensure failed infrastructure rows with apparently correct text cannot affect correct/observed. No provisional/macro score appears. Suggested commit: `feat(eval): report fixed-population VSP scores and incomplete runs`.

## Task 5: Offline CLI, isolated import smoke and reviewable validation report

**Files:** Create `scripts/__init__.py` (empty/docstring), `scripts/score_eval.py`, `configs/eval/vsp_validated_v1.yaml`, `tests/evaluation/test_offline_cli.py`, `docs/evaluation/protocols.md`, `docs/evaluation/validation.md`.

**Consumes:** parse_config/read_inputs, get_scorer, evaluate, canonical_json; PyYAML only at this CLI boundary.

**Produces:** `main(argv: list[str] | None = None, *, repo_root: Path | None = None) -> int`; `write_reports(output_dir: Path, scores: list[Score], summary: dict) -> None`. CLI contract:

```text
PYTHONPATH=src ./venv/ilvr-venv/bin/python -m scripts.score_eval
  --manifest data/evaluation/<dataset>/manifest.jsonl
  --predictions results/evaluation/<generation-run>/predictions.jsonl
  --config configs/eval/vsp_validated_v1.yaml
  --output-dir results/evaluation/<unique-scoring-run>
```

`repo_root` defaults to CWD (run from the active project root); tests may supply an explicit isolated root. Output directory must be one direct child of resolved repo_root/results/evaluation, with a nonempty unique name. Log directory is repo_root/logs/evaluation/<same-name>. Existing output or log directories are input errors and are preserved. Validate output/log paths before opening files. Every generated result/log stays in those trees. No model path argument, model import, judge call or dataset download is added.

### Configuration example (synthetic-only, not a paper protocol)

```yaml
schema_version: 1
benchmark: vsp
profile: vsp_validated_v1
categories: [planning, planning-empty]
dataset_root: data/evaluation/vsp-synthetic-v1
max_steps: 100
run_metadata:
  model_identity: synthetic:deterministic-responses-v1
  prompt_template_identity: synthetic:vsp-contract-v1
  generation_config:
    do_sample: false
    max_new_tokens: 16
```

This is an explicit synthetic fixture identity, not a missing checkpoint placeholder. Real saved responses require a config with their recorded identities and matching fingerprints. P1 does not create the real VSP dataset manifest. For a smoke use `make_case()` and write its exact config to data_root/config.yaml, rather than mixing fixture metadata with this example's identities.

- [x] **Step 1 — RED tests.** Add CLI tests before the CLI. Test-specific directories are unique, and data/prediction fixtures use Task 3's helper.

```python
import json
from pathlib import Path
import uuid
import yaml
from offline_fixtures import make_case, save_rows
from scripts.score_eval import main

def invoke(case, output_name=None):
    config_path = case['data_root'] / 'config.yaml'
    config_path.write_text(yaml.safe_dump(case['config']))
    out = case['root'] / 'results/evaluation' / (output_name or ('p1-score-' + uuid.uuid4().hex))
    code = main(['--manifest', str(case['manifest']), '--predictions', str(case['prediction_path']),
        '--config', str(config_path), '--output-dir', str(out)], repo_root=case['root'])
    return code, out

def test_complete_cli_and_no_overwrite():
    case = make_case(2)
    code, out = invoke(case)
    assert code == 0
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['N_expected'] == 2 and summary['accuracy'] == 1.0
    before = (out / 'summary.json').read_bytes()
    code, _ = invoke(case, out.name)
    assert code == 2 and (out / 'summary.json').read_bytes() == before

def test_empty_predictions_cli_is_incomplete():
    case = make_case(2)
    save_rows(case['prediction_path'], [])
    code, out = invoke(case)
    assert code == 1
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['complete'] is False and summary['accuracy'] is None

def test_preflight_error_writes_no_scores():
    case = make_case(1)
    case['config']['profile'] = 'unknown'
    code, out = invoke(case)
    assert code == 2 and not out.exists()
```

Append these exact-exit tests. An I/O error while writing results is infrastructure failure (exit 1); preserve partial output and the exception log without retries or cleanup.

```python
import pytest
from offline_fixtures import save_manifest

@pytest.mark.parametrize('kind', ['missing_image', 'bad_hash', 'bad_config', 'fingerprint'])
def test_cli_preflight_failures(kind):
    case = make_case(1)
    if kind in ('missing_image', 'bad_hash'):
        row = case['samples'][0]
        row['model_input']['images'] = ['image.bin']
        row['provenance']['asset_sha256'] = {'image.bin': '0' * 64}
        if kind == 'bad_hash': (case['data_root'] / 'image.bin').write_bytes(b'fixture-image')
        save_manifest(case)
    if kind == 'bad_config': case['config'] = []
    if kind == 'fingerprint':
        case['predictions'][0]['manifest_sha256'] = '0' * 64
        save_rows(case['prediction_path'], case['predictions'])
    code, out = invoke(case)
    assert code == 2 and not out.exists()

def test_argument_error_and_outside_output():
    case = make_case(1)
    args = ['--manifest', str(case['manifest']), '--predictions', str(case['prediction_path']),
            '--config', 'unused.yaml', '--output-dir', str(case['data_root'] / 'forbidden-output')]
    assert main(args, repo_root=case['root']) == 2
    assert not (case['data_root'] / 'forbidden-output').exists()
    with pytest.raises(SystemExit) as exc:
        main(args + ['--unknown'], repo_root=case['root'])
    assert exc.value.code == 2

def test_report_serialization_and_log():
    case = make_case(2)
    code, out = invoke(case)
    assert code == 0
    rows = [json.loads(line) for line in (out / 'scores.jsonl').read_text().splitlines()]
    assert [row['id'] for row in rows] == [row['id'] for row in case['samples']]
    def reject_nonfinite(value): raise AssertionError(value)
    json.loads((out / 'summary.json').read_text(), parse_constant=reject_nonfinite)
    assert (case['root'] / 'logs/evaluation' / out.name / 'score.log').is_file()

def test_output_failure_is_infrastructure_error(monkeypatch):
    import scripts.score_eval as cli
    case = make_case(1)
    def fail_write(out, scores, summary):
        out.mkdir()
        raise OSError('synthetic output failure')
    monkeypatch.setattr(cli, 'write_reports', fail_write)
    code, out = invoke(case)
    assert code == 1 and out.is_dir()
    assert not (out / 'summary.json').exists()
    assert 'synthetic output failure' in (case['root'] / 'logs/evaluation' / out.name / 'score.log').read_text()
```

- [x] **Step 2 — Run RED.** `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_offline_cli.py -q -p no:cacheprovider`. Expected missing `scripts.score_eval`.
- [x] **Step 3 — Implement CLI and outputs.** Use `argparse` for the four required args, `yaml.safe_load` for config, resolve all paths, check output containment/nonexistence, then create a fresh log directory and FileHandler in exclusive mode. Call get_scorer before read_inputs. Input/config failures are caught as ContractError (plus YAML/config read errors) and return 2 before output_dir is created. Unexpected scorer failures are logged and return 1; do not pretend they are model failures.

```python
# score_eval.py: output writer and the core executed after successful preflight
from dataclasses import asdict
from pathlib import Path
from evaluation.schema import Score, canonical_json

def write_reports(output_dir: Path, scores: list[Score], summary: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'scores.jsonl').open('xb') as handle:
        for score in scores:
            handle.write(canonical_json(asdict(score)) + b'\n')
    with (output_dir / 'summary.json').open('xb') as handle:
        handle.write(canonical_json(summary) + b'\n')
```

The full main body follows this fixed order (each named dependency is defined in earlier tasks):

```python
import argparse
import logging
import sys
import yaml
from evaluation.io import parse_config, read_inputs
from evaluation.registry import get_scorer
from evaluation.report import evaluate
from evaluation.schema import ContractError

def main(argv=None, *, repo_root=None):
    parser = argparse.ArgumentParser()
    for option in ('manifest', 'predictions', 'config', 'output-dir'):
        parser.add_argument('--' + option, required=True)
    args = parser.parse_args(argv)
    root = Path.cwd() if repo_root is None else Path(repo_root)
    root = root.resolve()
    resolve = lambda value: (root / value).resolve()
    out = resolve(args.output_dir)
    output_parent = (root / 'results/evaluation').resolve()
    log_dir = (root / 'logs/evaluation' / out.name).resolve()
    log_parent = (root / 'logs/evaluation').resolve()
    if out.parent != output_parent or log_dir.parent != log_parent or out.exists() or log_dir.exists():
        print('invalid or existing output/log directory', file=sys.stderr)
        return 2
    try:
        log_dir.mkdir(parents=True, exist_ok=False)
        handler = logging.FileHandler(log_dir / 'score.log', mode='x', encoding='utf-8')
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    logger = logging.getLogger('evaluation.offline.' + out.name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        try:
            raw = yaml.safe_load(resolve(args.config).read_text())
            config = parse_config(raw, root)
            get_scorer(config.benchmark, config.profile)
            prepared = read_inputs(resolve(args.manifest), resolve(args.predictions), config, root)
        except (ContractError, OSError, UnicodeError, yaml.YAMLError) as exc:
            logger.error('input error: %s', exc)
            print(str(exc), file=sys.stderr)
            return 2
        try:
            scores, summary = evaluate(prepared, config)
            write_reports(out, scores, summary)
        except Exception:
            logger.exception('scoring/output infrastructure failure')
            return 1
        logger.info('complete=%s expected=%s observed=%s', summary['complete'], summary['N_expected'], summary['observed'])
        print(str(out / 'summary.json'))
        return 0 if summary['complete'] else 1
    finally:
        logger.removeHandler(handler)
        handler.close()

if __name__ == '__main__':
    raise SystemExit(main())
```

### Original-source-tree/Reference-free smoke

Add one subprocess test to test_offline_cli.py. Copy only the new schema/io/registry/report/benchmarks modules, docstring-only evaluation/__init__.py, and scripts/__init__.py/score_eval.py to a fresh workspace under results/evaluation/<unique-smoke-run>/workspace. This is a test copy of source code, not a model artifact. Symlink workspace/data, workspace/results, workspace/logs to the active root's corresponding trees so every fixture/result/log remains in its required storage tree. No Reference link is created. Assert copied-tree inventory excludes all legacy inference modules.

Add this full parent test to the same CLI test file. It uses the specified venv with CWD=workspace and replaces inherited PYTHONPATH. The child asserts import origins and rejects heavy dependencies before importing the CLI:

```python
import os
import shutil
import subprocess
import textwrap

def test_copied_workspace_cli_without_reference_or_heavy_imports():
    case = make_case(2)
    workspace = case['result_root'] / 'workspace'
    relative_files = [
        'src/evaluation/__init__.py', 'src/evaluation/schema.py', 'src/evaluation/io.py',
        'src/evaluation/registry.py', 'src/evaluation/report.py',
        'src/evaluation/benchmarks/__init__.py', 'src/evaluation/benchmarks/vsp.py',
        'scripts/__init__.py', 'scripts/score_eval.py']
    for relative in relative_files:
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(case['root'] / relative, destination)
    assert {str(path.relative_to(workspace)) for path in workspace.rglob('*.py')} == set(relative_files)
    for directory in ('data', 'results', 'logs'):
        (workspace / directory).symlink_to((case['root'] / directory).resolve(), target_is_directory=True)
    assert not (workspace / 'Reference').exists()
    config_path = case['data_root'] / 'config.yaml'
    config_path.write_text(yaml.safe_dump(case['config']))
    output = case['root'] / 'results/evaluation' / ('p1-copy-' + uuid.uuid4().hex)
    child = textwrap.dedent('''
        import importlib.abc
        import sys
        from pathlib import Path
        class DenyHeavy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch', 'transformers', 'mathruler', 'gym', 'gymnasium', 'Reference'}:
                    raise RuntimeError('forbidden offline dependency: ' + fullname)
                return None
        sys.meta_path.insert(0, DenyHeavy())
        import scripts.score_eval as cli
        import evaluation
        root = Path.cwd().resolve()
        for name, module in tuple(sys.modules.items()):
            if name == 'evaluation' or name.startswith('evaluation.') or name == 'scripts.score_eval':
                assert Path(module.__file__).resolve().is_relative_to(root), name
        assert not (root / 'Reference').exists()
        raise SystemExit(cli.main(sys.argv[1:]))
    ''')
    env = dict(os.environ, PYTHONPATH=str(workspace / 'src'), PYTHONDONTWRITEBYTECODE='1')
    command = [str(case['root'] / 'venv/ilvr-venv/bin/python'), '-c', child,
        '--manifest', str(case['manifest']), '--predictions', str(case['prediction_path']),
        '--config', str(config_path), '--output-dir', str(output)]
    log_dir = case['root'] / 'logs/evaluation' / case['run_id']
    log_dir.mkdir(parents=True, exist_ok=False)
    with (log_dir / 'copied-workspace.log').open('w') as log:
        result = subprocess.run(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, (log_dir / 'copied-workspace.log').read_text()
    summary = json.loads((output / 'summary.json').read_text())
    assert (summary['N_expected'], summary['correct'], summary['accuracy'], summary['complete']) == (2, 2, 1.0, True)
```

Require exit 0 and independently read the generated summary: N_expected=2, correct=2, accuracy=1.0, complete=true. Also check script/evaluation module `__file__` paths begin with the copied workspace, ensuring the original source was not imported. This is a runtime dependency witness, not just a source grep. It does not prove absence of installed packages; it proves this exercised path does not request the forbidden imports.

- [x] **Step 4 — Run GREEN** for CLI tests, including the copied-workspace smoke. The child stdout/stderr belongs in the test's corresponding logs/evaluation/<run-id>/ directory. No image/model download occurs.
- [x] **Step 5 — Write protocol/validation docs as part of this deliverable.** `protocols.md` records strict parser semantics, R/RD/XR/RX legacy differences, goal/hole/time limit, identity/denominator rules, exit codes, synthetic-only config usage, and pinned source provenance. If implementation actually copies/adapts licensed upstream code rather than independently implementing this contract, add `THIRD_PARTY_NOTICES.md` with exact repo/SHA/file/symbol and applicable license text before publishing; no CoMT/EMMA code is copied in P1. `validation.md` records actual commands/exit codes/test counts, the copied-workspace witness, and the matrix below. Do not prefill a passing result from this plan.
- [x] **Step 6 — Review boundary.** One task spec/quality review covers CLI, storage, docs and independent import smoke. Suggested commit: `feat(eval): expose offline VSP scoring CLI and validation guide`.

Protocol-document evidence is available at `results/evaluation-p0-20260910-01/local-vsp-probe.json` and `docs/evaluation/audit.md`. For map `[[1,2],[0,0]]`, the AST-extracted simulator functions in each of `eval.py`, `src/evaluate_deepseed.py`, and `src/task_deepseed.py` produced the legacy column below. This is a local function observation, not an upstream or whole-CLI witness. The new column is the approved contract and remains pending until its tests execute.

| Raw path | Legacy observation | New contract |
|---|---|---|
| R | success | success at step 1 |
| RD | failure after leaving goal | success at step 1, episode ends |
| XR | success after dropping X | invalid_format before simulation |
| RX | success after dropping X | invalid_format before simulation |

The two existing `extract_path` functions returned the truthy string `"None"` for `final answer: R`; the new contract extracts `R`. Record this separately from simulator observations. Pinned upstream file/symbol/license provenance is in the `repositories[].evidence_files` and `license_files` entries of root `references.lock.json`; consult those entries and the audit when writing citations. The lock is development evidence and must not become a scoring runtime dependency.

## Final verification and handoff

No project-wide training/GPU suite is required to claim this additive offline feature works. Before final verification, determine whether the branch-finishing skill applies to the actual implementation workspace. If it does require a project full-suite command, follow the combined project finish/verification rule and obtain the necessary GPU/resource prerequisites; do not silently substitute this focused suite or mark unavailable full-suite witnesses green. This **plan-writing** turn does not finish a development branch.

For the P1 offline requirements claim, run this focused set once after all tasks/reviews are complete, using a fresh run ID and storing stdout/stderr under logs/evaluation/<run-id>/:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. ./venv/ilvr-venv/bin/python -m pytest \
  tests/evaluation/test_offline_schema.py \
  tests/evaluation/test_vsp_offline.py \
  tests/evaluation/test_offline_io.py \
  tests/evaluation/test_offline_report.py \
  tests/evaluation/test_offline_cli.py \
  tests/evaluation/test_scoring.py -q -p no:cacheprovider
```

The suite includes the critical CLI and dependency smoke; do not repeat them as a second identical verification cycle. The branch handoff decision is deferred to the implementation controller after workspace selection and code review. The repository currently has no documented full-suite command beyond pytest; the focused command above is not evidence of a full-project pass. If the finishing gate applies, inspect project test collection/resource requirements and record the actual full-suite command before executing its Step 1. Do not claim branch completion from this plan's focused command alone.

Run this separate preservation witness once at implementation completion. Its baseline is `logs/evaluation-p0-20260910-01/environment.json`, specifically `source_hashes`. Select the nine recorded `.py` files; AGENTS.md and .gitignore had authorized P0 edits and are deliberately excluded. This records source/reference preservation, not GPU behavior. Run from the active root with the P0 evidence accessible; if using a new worktree, make that read-only evidence available there before starting tasks.

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv/ilvr-venv/bin/python - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path
root = Path.cwd()
baseline = json.loads((root / 'logs/evaluation-p0-20260910-01/environment.json').read_text())
sources = {name: digest for name, digest in baseline['source_hashes'].items() if name.endswith('.py')}
assert len(sources) == 9
for name, expected in sources.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
lock = json.loads((root / 'references.lock.json').read_text())
assert len(lock['repositories']) == 7
for repo in lock['repositories']:
    directory = root / repo['local_dir']
    def git(*args):
        return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()
    assert git('rev-parse', 'HEAD') == repo['commit'], repo['name']
    assert not git('status', '--porcelain', '--untracked-files=all'), repo['name']
    for evidence in repo['evidence_files'] + repo['license_files']:
        assert hashlib.sha256((directory / evidence['path']).read_bytes()).hexdigest() == evidence['sha256'], evidence['path']
print('preserved source files:', len(sources), '; pinned clean references:', len(lock['repositories']))
PY
```

| Verification level | Expected reporting boundary |
|---|---|
| new CPU contract tests + existing scorer compatibility | Record observed pass/fail and actual count after execution |
| copied-workspace CLI/import smoke | Record observed result and copied module paths |
| upstream_environment | blocked/not_run: Gym absent and upstream Python/NumPy differ; no environment change authorized |
| gpu_smoke | not_run: P1 does not change/run generation |
| full_benchmark | not_run: synthetic fixtures only, no verified real manifest/prediction join |
| paper_protocol_verified | false |

The unresolved independent-review concern remains: real source→manifest→prediction identity must be established before interpreting benchmark scores. P1's synthetic contract fixtures do not resolve it. In particular, do not copy a raw table and call it the ILVR paper evaluation manifest.

### Self-review coverage ledger

| Approved requirement | Task / concrete witness |
|---|---|
| Typed separate ModelInput/Target, no gold extraction | 1/2/3; signature check and unknown-key injection cases |
| Stable run identity, bytes manifest hash, no self-hash | 1/3; reordered generation keys, changed generation, bad run/manifest hash |
| Strict parser, nested/malformed last boxed, no deleted X | 2; explicit extraction/grammar examples |
| Goal/hole/boundary/step-limit semantics | 2; explicit position/step/termination assertions |
| Valid map without gold path | 2/3; target map-only fixture |
| ID/metadata/asset preflight, zero prediction file | 3; boundary table and explicit partial/empty cases |
| Fixed denominator, categories, infra versus model failures | 4; exact expected counts/nulls/coverage |
| Existing source/scoring preserved | 5/final; 11-test compatibility file plus source hashes |
| Explicit registry, errors and file storage | 4/5; unknown-profile, CLI exit/output cases |
| No heavy/Reference runtime dependency | 5; denied-import copied-workspace subprocess |
| Distinct validation levels and pending real-data concern | 5/docs and final matrix |

Execution status: P1 Tasks 1–5 implemented and reviewed in `.worktrees/evaluation-p1`; final focused suite 105 passed, documented CLI quick-start exit 0. No commit/merge/push. See the worktree docs/evaluation/validation.md for evidence and unverified boundaries.

Root self-review checked approved requirement coverage, cross-task types/names, concrete RED tests, file boundaries and verification claims. A context-free shower read covered the entire plan and reported minor handoff gaps: the P0 evidence locator/procedure and legacy comparison details were added above. Its remaining conditional branch-finishing question is explicitly assigned to the implementation controller because no implementation workspace or branch handoff is selected in this planning turn. Documentation syntax checks do not execute the proposed implementation or establish test results.

Implementation review correction (2026-09-10): JSONL records split on LF, preserving valid U+2028/U+0085 inside strings. Asset paths reject absolute paths and resolved escapes; a contained normalized relative path is allowed even when it contains `..`. Both corrections preserve the approved input contract and require causal IO regressions before completion.
