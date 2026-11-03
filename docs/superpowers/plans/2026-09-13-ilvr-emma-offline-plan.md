# ILVR EMMA Offline Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 고정 EMMA-mini 400개를 준비하고 저장 prediction을 `emma_safe_rules_v1`로 CPU 채점한다.

**Architecture:** 기존 Sample/Prediction, manifest/run hash, prepare/score CLI와 집계를 재사용한다. EMMA 전용 target과 추출·비교 모듈을 명시적 benchmark 분기로 연결하고 수식 parser는 제한된 입력만 처리한다. 모델 추론은 연결하지 않는다.

**Tech Stack:** `./venv/ilvr-venv/bin/python`의 Python 3.11, SymPy 1.14.0, 기존 PyYAML·pytest; 데이터 준비에만 기존 pyarrow 21.0.0·Pillow.

**Spec:** [승인된 design commitment v3](../specs/2026-09-13-ilvr-emma-offline-design.md). 실행자는 이 계획과 spec을 모두 읽는다. 사용자가 2026-09-13 v2 설계와 한계를 승인했다. 계획 독해에서 발견한 별도 4,096비트 한도를 v3에 명시했다. **최신 사용자 `진행해`로 v3 수정과 이 계획의 실행을 함께 승인받았다.**

## Global Constraints

- 활성 작업 루트는 `/root/workspace/ILVR/.worktrees/evaluation-p1`, 기존 branch는 `codex/evaluation-p1`, HEAD는 `2497ee7bafad1b51ef6c2aba6f40f084830916c4`다. 새 worktree를 만들지 않는다. 실행 시 using-git-worktrees로 기존 격리를 확인한다.
- Python/pip/tests는 반드시 `./venv/ilvr-venv/bin/`를 사용한다. 시스템 Python과 legacy `./ilvr-venv`를 사용하지 않는다. 패키지 설치·업그레이드가 없다.
- 데이터·이미지·원본·manifest·fixture·cache는 `data/`, 보고서는 `results/evaluation/<run-id>/`, 로그는 `logs/evaluation/<run-id>/` 아래에 둔다. data/checkpoints symlink를 보존한다. 모델 파일은 만들지 않는다.
- `benchmark=emma`, `profile=emma_safe_rules_v1`, dataset `luckychao/EMMA-mini`, revision `6b9ae9a74733bb57f0b741213f8d0ec9ebae067a`, test 400개, Chemistry/Coding/Math/Physics 각 100개, 객관식 341개, 자유응답 59개, 이미지 614개다. full이나 임의 subset 준비를 추가하지 않는다.
- 추출은 response와 ModelInput만 읽는다. target/정답/solution을 추출·model input으로 넘기지 않는다. 비교와 평가용 분류 집계만 target을 읽는다.
- 기존 VSP/CoMT/VisuLogic target·profile·결과 필드는 바꾸지 않는다. P2 추론 bridge와 `run_eval.py`의 VSP 범위도 유지한다. 기존 runtime, patched Transformers, prompts, 생성 설정은 변경하지 않는다.
- Reference는 read-only 근거이며 런타임 의존성이 아니다. source 비교 전에 independent root/origin/commit/clean/hash를 lock과 확인한다. EMMA 코드를 복사하거나 설치하지 않는다.
- 신규 scorer CPU 테스트에는 모델 로딩·torch·CUDA·Reference·실제 dataset이 필요하지 않다. PyArrow/Pillow는 source preparation 테스트에서만 사용한다.
- 문항을 자동으로 필터링하지 않는다. 누락/infrastructure error는 미완료/null accuracy이고, empty/invalid/지원하지 않는 응답은 관측된 오답이다. unknown benchmark/profile은 실패한다.
- `protocol_kind=modified`, `paper_protocol_verified=false`. 원본 함수 관찰·CPU 계약·원본 대응·GPU·실제 전체 모델 평가·논문 프로토콜을 분리한다. 실제 모델별 first-nail은 이번 실행에 포함되지 않는다.
- 자동 commit/push/merge/branch 정리하지 않는다. 각 task의 commit 단계는 owned files만 stage하고 task report를 남기는 것으로 대체한다. 기존 staged P1–P4 변경을 보존한다.
- 계획 승인 후 사용 가능한 SDD 경로로 task를 순차 실행하고 task별 계속 여부를 다시 묻지 않는다. Worker/reviewer는 paperthin을 호출하거나 agent를 생성하지 않는다. 공유 파일에 동시 구현 worker를 배치하지 않는다.

## Files, interfaces and ownership

| Task | 생성 | 수정 | 독립 결과 |
|---|---|---|---|
| 1 | `src/evaluation/emma_math.py`, `tests/evaluation/test_emma_math.py` | 없음 | 제한 수식 비교와 진단 |
| 2 | `src/evaluation/benchmarks/emma.py`, `tests/evaluation/emma_fixtures.py`, `test_emma_offline.py`, `test_emma_contract.py`, `configs/eval/emma_safe_rules_v1.yaml` | `schema.py`, `io.py`, `registry.py` | EMMA target·추출·저장 입력 채점 |
| 3 | `tests/evaluation/test_emma_report.py` | `src/evaluation/report.py` | EMMA 세부 집계·누락 계약 |
| 4 | `src/evaluation/emma_prepare.py`, `tests/evaluation/emma_source_fixtures.py`, `test_emma_prepare.py`, `test_emma_pipeline.py` | `scripts/prepare_eval.py` | 고정 원본 준비 CLI |
| 5 | 검증 스크립트는 해당 run의 `logs/`에만 작성 | `docs/evaluation/protocols.md`, `validation.md` | 원본 전수 대응·통합/회귀 증거·사용법 |

표의 test 파일은 모두 `tests/evaluation/` 아래다. 공통 모듈은 모두 `src/evaluation/` 아래다. `scripts/score_eval.py`는 수정하지 않는다. 현재 `read_manifest`가 모든 ID에 `benchmark/subtask/source_id`를 요구하므로 Task 2가 EMMA만 원본 pid를 허용하는 분기를 추가해야 한다.

공통 데이터 클래스의 기존 필드 순서는 유지한다. 추가 `EmmaTarget`의 필드는 순서대로 `question_type: str`, `answer: str`, `index: int | None`, `content: str`, `category: str`, `task: str`다. Sample.target의 union만 넓힌다. Score 필드를 추가하지 않는다.

Task 1 public API:

- `ComparisonResult(correctness: bool, error: str | None)` — frozen dataclass.
- `require_emma_math_runtime() -> None` — SymPy의 실제 import와 버전 1.14.0 확인, 실패는 ContractError. 버전 검사 실패를 오답으로 바꾸지 않는다.
- `unwrap_math(value: str) -> str` — strip 후 전체 `$...$` 또는 `\(...\)` 한 겹만 제거, 내부 공백은 보존한다.
- `is_nonfinite_literal(value: str) -> bool` — 대소문자 무관한 signed nan/inf/infinity, `∞`, `\infty`의 독립 표기만 검사한다. 문장 부분 문자열이나 변수 `i`를 거부하지 않는다.
- `compare_values(left: str, right: str) -> ComparisonResult` — 정답을 모르는 대칭 값 비교. wrapper 정규화, 비유한 표기 거부, 비어 있지 않은 정확 일치, 제한 수식 비교 순서다.

Task 2 public API (`evaluation.benchmarks.emma`):

- `CATEGORIES = ('Chemistry', 'Coding', 'Math', 'Physics')`, `PROFILE = 'emma_safe_rules_v1'`.
- `validate_target(target: EmmaTarget, model_input: ModelInput) -> None`.
- `extract_answer(response: str, model_input: ModelInput) -> ParsedAnswer`.
- `compare_answer(parsed: ParsedAnswer, target: EmmaTarget, model_input: ModelInput) -> ComparisonResult`.
- `score_emma(sample: Sample, prediction: Prediction) -> Score`.

Task 3 API: `emma_summary(samples: tuple[Sample, ...], scores: list[Score]) -> dict` in report.py. Task 4 API: `prepare_emma(source_root: Path, output: Path, *, repo_root: Path) -> dict`.

## Run discipline and preflight

Root reads existing worktree instructions and records the initial owned-file contents/hashes and unchanged code boundary under `logs/evaluation/p5-implementation-<fresh-id>/`. This is a small fixed baseline, not a new recovery framework. Existing P4 verification commands are in `logs/evaluation/p4-final-20260913-01/verify.py`; read them, do not execute that old run ID.

All test commands below run at the active worktree root with:

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src:tests/evaluation:.
export CUDA_VISIBLE_DEVICES=''
```

Each RED/GREEN command gets a fresh basetemp and logs directory. Before using a literal run ID shown below, check that its data/results/logs destinations do not exist; choose a fresh suffix if needed and record the chosen ID. Save stdout/stderr in its logs directory and read the complete output. An expected RED is a missing behavior/import from inside a test, not collection failure or environment damage.

## Task 1: 제한 수식 비교

**Files:** create `src/evaluation/emma_math.py`, `tests/evaluation/test_emma_math.py`.

**Interfaces:** consumes ContractError only; produces the five public APIs listed above. Private parser units are `MathNode(kind: str, value: str | int | None, children: tuple[MathNode, ...])`, `MathInputError(code: str)`, `MathValue(expression, had_symbol: bool, variables: frozenset[str], degree: int, terms: int)`, `_lex(text)`, `_parse(tokens)`, `_analyze(node)`, `_evaluate(node)`. They are internal to this file; other tasks never call them. `_lex` returns tokens with kind/text; `_parse` returns an immutable MathNode tree; `_analyze` checks whole-tree structural limits before arithmetic; `_evaluate` returns a SymPy expression plus original-symbol/degree/term-bound metadata needed to enforce restrictions before cancellation.

- [x] Write failing mathematical decision tests with literal expected answers.

```python
import pytest

@pytest.mark.parametrize('left,right,correct', [
    ('0.5', '1/2', True),
    (r'\frac{1}{2}', '0.5', True),
    ('sqrt(4)', '2', True),
    (r'\sqrt{4}', '2', True),
    ('pi', '3.14', True),
    ('1.5e-3', '0', True),
    ('(1e-100)^2', '0', True),
    ('1.004', '1', True),
    ('1.006', '1', False),
    ('2.675', '2.67', True),
    ('x+x', '2*x', True),
    ('x+x', '3*x', False),
    ('(x+1)^2', 'x^2+2*x+1', True),
    ('X', 'x', False),
    ('-2^2', '-4', True),
    ('(2^3)^2', '64', True),
    ('sin(x)', 'sin(x)', True),
    ('a b', 'ab', False),
])
def test_math_contract(left, right, correct):
    from evaluation.emma_math import compare_values
    assert compare_values(left, right).correctness is correct

@pytest.mark.parametrize('value,code', [
    ('2x', 'unsupported_expression'),
    ('x/x', 'unsupported_expression'),
    ('sqrt(x)', 'unsupported_expression'),
    ('2^3^2', 'unsupported_expression'),
    ('2^(3^2)', 'unsupported_expression'),
    ('x^-1', 'unsupported_expression'),
    ('__import__("os")', 'unsupported_expression'),
    ('nan', 'non_finite_value'),
    ('1/0', 'non_finite_value'),
    ('sqrt(-1)', 'non_finite_value'),
    ('2^9', 'math_limit_exceeded'),
    ('1e101', 'math_limit_exceeded'),
    ('((1e-100)^8)^8', 'math_limit_exceeded'),
])
def test_declared_rejections(value, code):
    from evaluation.emma_math import compare_values
    result = compare_values(value, '7')
    assert result.correctness is False
    assert result.error == code

def test_exact_long_content_bypasses_parser_limit():
    from evaluation.emma_math import compare_values
    code = 'long code choice\n' * 300
    assert compare_values(code, code).correctness is True
    assert compare_values(code, code).error is None
```

- [x] Run RED: `./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_emma_math.py -q -p no:cacheprovider --basetemp data/evaluation/p5-t1-red-01`. Confirm tests collect and fail for the missing module/API before implementation.
- [x] Implement independent lexer/parser. Use anchored number tokens `[0-9]+(?:\.[0-9]*)?|\.[0-9]+` followed by optional `[eE][+-]?[0-9]+`; tokenize `sqrt`/`pi` as complete names and one-letter ASCII variables otherwise. Reject unknown multi-letter identifiers, attributes, brackets, quotes, unexpected backslashes and unconsumed tokens. No deletion-based LaTeX cleanup.

The recursive-descent grammar is fixed here; `integer_literal` may be signed and optionally enclosed in one matching pair of parentheses or braces. `power` consumes at most one exponent at its current level; parentheses permit a new level.

```text
sum      := product (('+'|'-') product)*
product  := unary (('*'|'/'|'\\cdot'|'\\times') unary)*
unary    := ('+'|'-') unary | power
power    := atom (('^'|'**') integer_literal)?
atom     := number | variable | 'pi' | '\\pi'
          | '(' sum ')' | 'sqrt' '(' sum ')'
          | '\\sqrt' '{' sum '}' | '\\frac' '{' sum '}' '{' sum '}'
```

- [x] Add whole-input limits before evaluation: text ≤512 chars; arithmetic/unary/function nodes ≤64; grouped/recursive nesting ≤16; numeric mantissa digits ≤64; scientific exponent magnitude ≤100; literal power exponent magnitude ≤8. Count limits from original tokens/tree, before zero terms or cancellations disappear. Flat sums must not be rejected solely because a binary storage tree is left-deep; nesting means syntactic grouping/unary recursion.
- [x] Evaluate only whitelisted constructors. Numbers are parsed through Decimal/Fraction or explicit integer numerator/denominator, then passed as integers to SymPy Rational. Variable constructors receive a validated single ASCII letter. Only known sqrt/pi/arithmetic operations are used. Never pass model strings to SymPy string parsers.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ComparisonResult:
    correctness: bool
    error: str | None = None

# Numeric text has already passed the full token grammar and digit limits.
# Construct exact rational literals without a SymPy string parser.
from decimal import Decimal
from fractions import Fraction
number = Fraction(Decimal(token_text))
expression = sympy.Rational(number.numerator, number.denominator)
```

Before construction/expansion, carry original variable set, total-degree bound and expanded-term upper bound. For addition/subtraction use term-bound sum and degree max; multiplication uses term-bound product and degree sum; nonnegative integer power uses `terms**exponent` and `degree*exponent`. Enforce variables≤8, degree≤8, terms≤128. Constants count as one term. Keep `had_symbol` even when an expression cancels to a constant. A symbolic division requires its original denominator to contain no symbol and evaluate to a nonzero Rational. A symbolic power may not be negative; sqrt may not receive a symbolic operand; numeric irrational coefficients may not mix with symbolic expressions. Only after these checks form bounded Poly values over QQ for exact coefficient comparison.

Numeric intermediate real/finite/magnitude checks happen at every operation, not only at the final float. Apply v3's separate 4,096-bit limit to rational numerator/denominator integers, including symbolic coefficients: use cheap projected-size checks before creating a large power or expanded coefficient. Keep the original real-value magnitude limit `1e100` independently. `(1e-100)^2` must remain supported; `((1e-100)^8)^8` must return `math_limit_exceeded`. Record a constructor-call witness that the oversized power is rejected before the expensive constructor runs. Do not construct a huge denominator and then discover float underflow. The previous plan's numerator/denominator `1e100` restriction is withdrawn; the bit limit is an explicit v3 design change, not an interpretation of v2. No worker process, timeout service or retry layer is added.

- [x] Implement the public comparator around those internal units. `_evaluate` must return normalized expressions with the pre-cancellation restrictions already checked.

```python
def compare_values(left, right):
    require_emma_math_runtime()
    left, right = unwrap_math(left), unwrap_math(right)
    if is_nonfinite_literal(left) or is_nonfinite_literal(right):
        return ComparisonResult(False, 'non_finite_value')
    if not left or not right:
        return ComparisonResult(False, 'unsupported_expression')
    if left == right:
        return ComparisonResult(True)
    try:
        a_tree, b_tree = _parse(_lex(left)), _parse(_lex(right))
        _analyze(a_tree)
        _analyze(b_tree)
        a, b = _evaluate(a_tree), _evaluate(b_tree)
        if not a.expression.free_symbols and not b.expression.free_symbols:
            return ComparisonResult(round(float(a.expression.evalf()), 2) ==
                                    round(float(b.expression.evalf()), 2))
        # Both expressions have passed rational-polynomial admission.
        symbols = sorted(a.expression.free_symbols | b.expression.free_symbols,
                         key=str)
        difference = sympy.Poly(a.expression - b.expression, *symbols, domain='QQ')
        return ComparisonResult(difference.is_zero)
    except MathInputError as exc:
        return ComparisonResult(False, exc.code)
```

In the implementation, use a local `import sympy` immediately after `require_emma_math_runtime()` inside compare_values and other arithmetic consumers; the code above shows the comparison dispatch, not a module-level heavy import. `_evaluate` returns MathValue. The runtime checker uses a local import and exact `sympy.__version__` check; no module-level SymPy import or cross-run success flag may hide a version mismatch. If one non-rational numeric expression is paired with a symbolic polynomial, return unsupported_expression before constructing a QQ Poly. Explicitly map expected parser/domain failures to MathInputError; propagate unexpected exceptions.

- [x] Add limit boundary tests (at/beyond 512 chars, 64 nodes, depth16, exponent8, degree8, terms128), original-symbol cancellation cases `x/x`, `sqrt(x-x)`, irrational coefficient cancellation, finite and nonfinite wrapper forms, internal spaces, long exact content, missing/mismatched SymPy version, and a subprocess import guard blocking torch/transformers/Reference. A call-shaped response must produce a diagnostic without executing any callable. These are CPU contract tests, not general symbolic equivalence evidence.
- [x] GREEN: run the Task 1 file with fresh basetemp `data/evaluation/p5-t1-green-01`, inspect full output, stage its two owned files, and write RED/GREEN task report. Do not commit.

## Task 2: EMMA target, extraction, IO and scorer

**Files:** create `src/evaluation/benchmarks/emma.py`, `configs/eval/emma_safe_rules_v1.yaml`, `tests/evaluation/emma_fixtures.py`, `test_emma_offline.py`, `test_emma_contract.py`; modify `schema.py`, `io.py`, `registry.py`.

**Interfaces:** consumes Task 1 public API plus existing ModelInput/ParsedAnswer/Prediction/Score. Produces EmmaTarget and Task 2 public API. Fixture `make_case(count=4)` returns root/data/results/manifest/predictions/config/rows/prediction_rows/run_id with the same meanings as `visulogic_fixtures.make_case`; `write_rows(path, rows)` writes JSONL. Generate independent synthetic fixtures under active data/results, never import another benchmark's fixture as the EMMA oracle.

- [x] Write RED tests for target validation, extraction and counterfactual comparison.

```python
from dataclasses import replace
import pytest

def test_extraction_is_gold_independent():
    from evaluation.benchmarks.emma import extract_answer, compare_answer
    from evaluation.schema import EmmaTarget, ModelInput
    model_input = ModelInput('Q', (), ('first', 'second', 'third', 'fourth', 'fifth'))
    gold = EmmaTarget('multiple_choice', 'b', 1, 'second', '', '')
    other = replace(gold, answer='A', index=0, content='first')
    parsed = extract_answer(r'\boxed{B}', model_input)
    assert parsed.value == 'B'
    assert compare_answer(parsed, gold, model_input).correctness is True
    assert compare_answer(parsed, other, model_input).correctness is False
    assert extract_answer(r'\boxed{B}', model_input) == parsed

@pytest.mark.parametrize('response,value', [
    ('The final answer is 0.5', '0.5'),
    ('The final answer is -2.75.', '-2.75'),
    ('answer: 1.5e-3', '1.5e-3'),
    (r'\boxed{\frac{1}{2}}', r'\frac{1}{2}'),
    (r'\boxed{3} then \boxed{4', '3'),
    ('twelve', 'twelve'),
])
def test_decimal_and_boxed_extraction(response, value):
    from evaluation.benchmarks.emma import extract_answer
    from evaluation.schema import ModelInput
    assert extract_answer(response, ModelInput('Q', (), ())).value == value

def test_wrong_valid_label_does_not_fall_back_to_content():
    from evaluation.benchmarks.emma import extract_answer, compare_answer
    from evaluation.schema import EmmaTarget, ModelInput
    model_input = ModelInput('Q', (), ('B', 'other'))
    target = EmmaTarget('multiple_choice', 'A', 0, 'B', '', '')
    assert compare_answer(extract_answer('B', model_input), target, model_input).correctness is False
```

- [x] Add IO RED tests: raw pid equals provenance.source_id for EMMA, old benchmark prefixes remain required; type/index/content/choices mismatch and bool index reject; question type unknown rejects; category/task nonstrings reject; exact raw lowercase answer survives parse/asdict. Config allows canonical subject subsets as an allowlist and max_steps null/omitted only; a Math-only config with a Chemistry manifest row rejects without filtering. Unknown profile routes to the existing registry error.
- [x] Run RED new offline/contract files using fresh basetemp `data/evaluation/p5-t2-red-01`.
- [x] Implement the exact target and EMMA-only dispatch.

```python
@dataclass(frozen=True)
class EmmaTarget:
    question_type: str
    answer: str
    index: int | None
    content: str
    category: str
    task: str

# In read_manifest, preserve the existing non-EMMA identity rule.
expected_id = (sample.provenance.source_id if sample.benchmark == 'emma' else
               f'{sample.benchmark}/{sample.subtask}/{sample.provenance.source_id}')
```

Add parse_sample's benchmark-specific exact-key target constructor and validate_target call. Add config's canonical subject/null-max_steps branch. Add registry's exact tuple selection that lazily returns score_emma; **registry lookup alone must not import SymPy** because prepare CLI calls it too. Put the EMMA dependency preflight at read_inputs entry, before scoring, and also at direct score_emma entry. read_manifest/preparation alone do not require SymPy.

- [x] Implement extraction in the approved order: valid MCQ label/uppercase prefix, finite raw number, last complete balanced box, ordered marker kind with last occurrence, whole raw fallback. Strip outer response whitespace first. Preserve case/content; do not lower the selected substring. Return methods `upper_label`/`lower_label` for pure MCQ letters, `upper_label` for uppercase prefix, `boxed`, `marker`, `raw` otherwise. Use existing method literals only. Empty original response returns empty/raw; empty selected content returns invalid_format/empty_extraction; known nonfinite candidate returns invalid_format/non_finite_value. Empty original response has error=None. Apply nonfinite checks after selected candidate and math wrapper normalization. A full `\text{...}` wrapper is removed once only in the box path.

Comparison first handles parsed status, then normalizes a possible MCQ label using current choices. A valid wrong label ends as wrong. Other candidates compare against content (MCQ) or answer (open-ended) via compare_values. Scorer combines parsed.error or comparison.error into existing Score.error; simulation is None and scorer_version is `'1'`.

```python
def score_emma(sample, prediction):
    require_emma_math_runtime()
    if prediction.status != 'ok' or not isinstance(sample.target, EmmaTarget):
        raise ContractError('EMMA requires an ok prediction and EMMA target')
    validate_target(sample.target, sample.model_input)
    parsed = extract_answer(prediction.response, sample.model_input)
    comparison = compare_answer(parsed, sample.target, sample.model_input)
    return Score(sample.id, sample.subtask, 'emma', PROFILE, '1',
                 parsed.status, parsed.value, parsed.method,
                 comparison.correctness, None, parsed.error or comparison.error)
```

- [x] Create YAML config with schema_version1, benchmark emma, profile emma_safe_rules_v1, all four subjects, dataset_root `data/evaluation/emma-mini400-v1`, omitted max_steps, run_metadata model_identity `external:model`, prompt_template_identity `emma-mini400-v1`, generation_config `{}`. Explain in a comment that actual scoring requires actual generation identity/settings; this config does not define an EMMA generation prompt.
- [x] GREEN new files plus existing `test_offline_schema.py`, `test_offline_io.py`, `test_manifest_preflight.py`, `test_vsp_offline.py`, `test_comt_offline.py`, `test_visulogic_offline.py`. Use fresh basetemp `data/evaluation/p5-t2-green-01`; stage only owned files and report. Include serialization checks that old Sample/Score dictionaries have exactly their prior keys.

## Task 3: 과목·유형·category·task 집계

**Files:** modify `src/evaluation/report.py`; create `tests/evaluation/test_emma_report.py`.

**Interfaces:** consumes Task 2 EmmaTarget/scorer, existing PreparedEvaluation/Score/category_stats. Produces emma_summary and evaluate's EMMA-only `summary['emma']`.

- [x] Write a RED test with independent literal score outcomes, a missing prediction, duplicate Coding tag and empty task/category. Use a four-sample synthetic case; do not derive expected correctness from the implementation.

```python
def test_secondary_denominators_and_missing():
    from dataclasses import replace
    from evaluation.report import emma_summary
    from evaluation.schema import EmmaTarget, ModelInput, Provenance, Sample, Score
    provenance = Provenance('synthetic:emma', 'fixture', 'Coding', 'fixture', 'a', {})
    target = EmmaTarget('multiple_choice', 'A', 0, 'first', 'Grid;Grid;Line', 'Choose')
    sample = Sample(1, 'a', 'emma', 'Coding',
                    ModelInput('Q', (), ('first', 'second')), target, provenance)
    missing_sample = replace(sample, id='b', provenance=replace(provenance, source_id='b'))
    correct = Score('a', 'Coding', 'emma', 'emma_safe_rules_v1', '1',
                    'parsed', 'A', 'upper_label', True, None, None)
    missing = replace(correct, id='b', status='missing', correctness=None,
                      parsed_answer=None, extraction_method=None,
                      error='prediction not received')
    result = emma_summary((sample, missing_sample), [correct, missing])
    assert result['categories']['Coding']['Grid']['count'] == 2
    assert result['categories']['Coding']['Grid']['accuracy'] is None
    assert result['tasks']['Coding_Choose']['observed'] == 1
    assert result['question_types']['open_ended']['count'] == 0
    assert result['question_types']['open_ended']['accuracy'] is None
```

- [x] Run RED `test_emma_report.py` with basetemp `data/evaluation/p5-t3-red-01`.
- [x] Implement grouping from samples joined to scores by ID; reject duplicate IDs or mismatched sets in direct helper calls. Reuse category_stats for every group. Initialize both canonical question_type buckets. Subject/category and subject_task groups are made from manifest metadata, not only observed responses. Coding tags use a per-row set of stripped nonempty parts; non-Coding category retains original nonempty string. Empty/whitespace-only metadata makes no detail group. comparison_diagnostics counts exactly unsupported_expression, non_finite_value and math_limit_exceeded. Do not count empty_extraction, missing or infrastructure error text there; empty_extraction remains in per-row error and invalid_format status_counts.

```python
# evaluate(), after base summarize and metadata have been assembled:
if config.benchmark == 'emma':
    summary['emma'] = emma_summary(prepared.samples, scores)

# Coding tag membership, applied once for each manifest row:
tags = sorted({part.strip() for part in sample.target.category.split(';')
               if part.strip()})
```

- [x] Add assertions for overall/category denominators remaining unchanged, unsupported counted as observed false, all predictions missing, empty configured subject, no automatic subject filtering, case-preserved non-Coding category, empty task skipped, diagnostic counts excluding transport errors, and no new `emma` field in other benchmark reports.
- [x] GREEN `test_emma_report.py` and `test_offline_report.py` with fresh basetemp `data/evaluation/p5-t3-green-01`; stage owned files and report.

## Task 4: 고정 mini 준비와 CLI

**Files:** create `src/evaluation/emma_prepare.py`, `tests/evaluation/emma_source_fixtures.py`, `test_emma_prepare.py`, `test_emma_pipeline.py`; modify `scripts/prepare_eval.py`.

**Interfaces:** consumes EmmaTarget/validate_target, canonical_json, existing data path/IO conventions. Produces prepare_emma. Source fixture `make_source()` returns root/source/output/rows/assets/run_id; `pin_source(monkeypatch, case)` replaces only expected parquet hashes/sizes with synthetic file facts in tests. There is no production test mode or hash override CLI.

Source root: `data/evaluation/emma-source-discovery-20260913-01`. Read `EMMA-mini-dataset-info.json` (id/sha) and `EMMA-mini/<subject>/test-00000-of-00001.parquet`. Do not require current HF access, the discovery inventory or Reference at runtime. Pin these file facts directly in `PARQUET_FILES` keyed by subject:

| Subject | Bytes | SHA-256 |
|---|---:|---|
| Chemistry | 8503247 | b5849a452f8114d114f9d355e64a4c89c4c87f6a759d30a7be6d16650996872b |
| Coding | 25725653 | f161007210600bf6f254d0cff8f5007bdb386529cea350ed7880f1c1305ddec0 |
| Math | 7317246 | 929bee1405aaf98ab25ba082ef5dd9a67537977286ac108a757bdc7bb3110490 |
| Physics | 9067825 | 8663a7e70267110f16be9609923d97e8be903a5315b6a16397f78d212ee676c5 |

prepare_emma's source/output paths must resolve within repo_root/data; source and each accessed parquet must remain within the selected source root. Check fresh output and all input data/images before making output. Use pyarrow.parquet.read_table().to_pylist() without the datasets loader so no implicit cache/download/image transforms occur. Preserve source row order within canonical subject order.

- [x] Build independent synthetic parquet fixtures under data. Generate 100 records per subject with Chemistry 78/22 and Math63/37 MCQ/open-ended, the other subjects 100 MCQ each. Generate exactly 614 tiny image blobs: 25 rows with 5 slots, 26 with 4, 36 with 2, 313 with 1; swap one two-image allocation with `phy_6` and use slots `[2,3]` there. Put 21 lowercase MCQ labels in Physics and include examples with 2,3,4,5,7,8 choices, null/[] open options, PNG/JPEG/WEBP bytes, context, and a >512-character correct content. Metadata/hash fixtures are computed from synthetic parquets only. Generated images and parquets remain under data.
- [x] Write RED preservation and invalid-source tests.

```python
def test_preparation_preserves_raw_ids_and_slots(monkeypatch):
    from evaluation.emma_prepare import prepare_emma
    from emma_source_fixtures import make_source, pin_source
    import json
    case = make_source()
    pin_source(monkeypatch, case)
    summary = prepare_emma(case['source'], case['output'], repo_root=case['root'])
    assert summary['N_expected'] == 400
    assert summary['image_count'] == 614
    rows = [json.loads(line) for line in (case['output']/'manifest.jsonl').read_text().splitlines()]
    by_id = {row['id']:row for row in rows}
    assert by_id['phy_6']['provenance']['source_id'] == 'phy_6'
    assert by_id['phy_6']['model_input']['images'] == [
        'images/phy_6/image_2.png', 'images/phy_6/image_3.png']
    assert '<image_2><image_3>' in by_id['phy_6']['model_input']['text']
    assert summary['excluded_rows'] == 0
    assert summary['paper_protocol_verified'] is False
```

Invalid cases: metadata id/revision, any byte/hash mismatch, duplicate/unsafe pid, unknown subject/type, wrong subject/type population, missing field/extra field, empty question, non-string category/task, invalid/non-string answer, empty/one/nine choices, nonempty open options, missing referenced slot, missing/corrupt image bytes, zero images, input/output path escape, existing output. Invalid-source fixtures recompute only test hashes when testing the content contract rather than hash rejection. Assert output absent and source bytes unchanged.
- [x] RED `test_emma_prepare.py` and EMMA dispatch tests with basetemp `data/evaluation/p5-t4-red-01`.
- [x] Implement source preflight and output. Source columns are exactly pid/question/options/answer/image_1–image_5/solution/subject/task/category/source/type/context. solution/source are retained only in source-selection audit data, never model input. Each image cell is either null or contains nonempty bytes; do not dereference its path field. Validate referenced `<image_n>` tokens in question/context/options against existing slots without renumbering. Decode all image bytes once with Pillow and retain original bytes. Name assets `images/<pid>/image_<slot>.<extension>` with PNG→png, JPEG→jpg, WEBP→webp for the pinned source formats.

```python
# Given a validated raw record and its slots/asset hashes:
choices = tuple(row['options'] or ())
kind = row['type'].lower()
index = ord(row['answer'].upper()) - ord('A') if kind == 'multiple choice' else None
target = EmmaTarget('multiple_choice' if index is not None else 'open_ended',
                    row['answer'], index,
                    choices[index] if index is not None else row['answer'],
                    row['category'], row['task'])
text = row['context'] + '\n\n' + row['question'] if row['context'] else row['question']
model_input = ModelInput(text, tuple(asset_paths), choices)
validate_target(target, model_input)
sample = Sample(1, row['pid'], 'emma', row['subject'], model_input, target,
                Provenance(DATASET_ID, SOURCE_REVISION, row['subject'], 'test',
                           row['pid'], asset_hashes))
```

Write `manifest.jsonl` as canonical_json(asdict(sample))+newline. Write `source-selection.json` with schema_version1 and rows containing source_id, subject, source_row (zero-based in its parquet), raw question/context/options/answer/type/category/task/source/solution, and image_slots. Write `SOURCE.json` with schema_version1, returned summary and source_root relative to data. No new dataset-specific cache or orchestration framework.

Returned summary keys: N_expected, categories (subject counts), question_types (canonical counts), image_count, excluded_rows=0, dataset_id, revision, dataset_profile=`emma-mini400-v1`, split=`test`, dataset_root (repo data-relative), source_files (subject→path/bytes/sha256), manifest_sha256, protocol_kind=`modified`, paper_protocol_verified=false, ood_verified=`unknown`. This summary is passed through existing prepare CLI to results/evaluation/<output-name>/prepare.json.

- [x] Add prepare CLI's `elif args.benchmark == 'emma'` calling prepare_emma and an EMMA label for rejecting CoMT-only arguments; keep old VSP/VisuLogic error strings and dispatch intact. No new flags. get_scorer still validates the explicit profile without importing SymPy; preparation imports PyArrow/Pillow only when dispatched.
- [x] Pipeline tests run the existing prepare main with a synthetic source, then score a small independent synthetic manifest with one correct, one wrong and one missing response. Assert prepare exit0, invalid input2, incomplete score1, complete score0; scorer/report identity fields and parser diagnostics persist. Do not fabricate a model accuracy result using gold-generated responses for the actual 400-row manifest. In an isolated copied source package, block imports of torch/transformers/Reference and run the score path. Separately guard that run_eval rejects EMMA before runtime/model loader.
- [x] GREEN new preparation/pipeline tests plus `test_visulogic_pipeline.py`, `test_comt_pipeline.py`, `test_vsp_pipeline.py`, using fresh basetemp `data/evaluation/p5-t4-green-01`. Stage owned files and report.

## Task 5: 실제 데이터 대응·통합 증거·문서

**Files:** modify `docs/evaluation/protocols.md`, `docs/evaluation/validation.md`. Root-owned witness scripts and logs stay under `logs/evaluation/<run-id>/`; generated summaries under matching results. Do not create a new permanent runner to wrap existing commands.

**Interfaces:** consumes all four tasks and existing CLI. This task produces the actual mini manifest and records exact test/source comparison scope. Root handles actual preparation after task reviews; all external data already exists locally.

- [x] Run one task-scoped integration check before actual preparation: all new EMMA tests as a set. Save complete output. If the focused task tests already ran on identical final bytes and the final CPU command below is imminent, use that final command instead of running both.
- [x] Verify the existing data/checkpoints symlinks, current worktree, pinned parquet hashes and fresh output directories. Record source inventory and protected old file hashes before source comparison. Reuse the lock validation code from prior verification records; never execute old scripts that create already-used run IDs.
- [x] Prepare with the existing entrypoint:

```bash
./venv/ilvr-venv/bin/python scripts/prepare_eval.py \
  --benchmark emma --profile emma_safe_rules_v1 \
  --source-root data/evaluation/emma-source-discovery-20260913-01 \
  --output-dir data/evaluation/emma-mini400-v1
```

If this ID already exists, choose a fresh explicit suffix and update the new EMMA config/documentation to that output. On a runtime failure preserve the failed log, identify the cause, make one targeted fix plus one causal test, and retry with a fresh run ID. Do not add recovery infrastructure.

- [x] Independently validate prepared rows against pre-implementation `mini-record-inventory.json` and `mini-image-inventory.json` under the discovery data root. Match every pid, source answer (including lowercase), choice, subject, context+question construction, normalized type, target index/content, all 614 image byte hashes and slot names. Confirm manifest hash, no exclusions, 400 unique rows, subject/type/image counts, source files unchanged and read_manifest accepts all rows with the new config. Store actual hashes/counts, not only a boolean.

```python
import hashlib, json
from pathlib import Path
root = Path.cwd()
dataset = root/'data/evaluation/emma-mini400-v1'
discovery = root/'data/evaluation/emma-source-discovery-20260913-01'
original = {r['pid']:r for r in json.loads((discovery/'mini-record-inventory.json').read_text())}
rows = [json.loads(line) for line in (dataset/'manifest.jsonl').read_text().splitlines()]
assert len(rows) == len({r['id'] for r in rows}) == 400
assert set(original) == {r['id'] for r in rows}
for row in rows:
    source = original[row['id']]
    assert row['target']['answer'] == source['answer']
    assert row['model_input']['choices'] == (source['options'] or [])
    expected_text = (source['context']+'\n\n'+source['question']
                     if source['context'] else source['question'])
    assert row['model_input']['text'] == expected_text
    assert row['subtask'] == source['subject']
images = json.loads((discovery/'mini-image-inventory.json').read_text())
formats = {'PNG':'png', 'JPEG':'jpg', 'WEBP':'webp'}
for image in images:
    relative = f"images/{image['pid']}/image_{image['slot']}.{formats[image['format']]}"
    assert hashlib.sha256((dataset/relative).read_bytes()).hexdigest() == image['sha256']
```

Extend that witness with the target index/content/type and each row's asset_sha256 checks stated above; all required input values are in the two inventories. Store the script itself and complete output as evidence.

- [x] Perform bounded original-source comparisons only after lock verification. Reuse AST isolation of `is_number`, `extract_full_boxed_content`, `fast_extract_answer` with re only. Compare the existing 16 discovery cases plus MCQ 2/8-choice, lowercase, empty, marker case, wrong-label/content ambiguity and long raw content; list matched cases and intentional changes individually. Do not import the upstream evaluator, stub parser equivalence or report full upstream environment pass. Preserve the earlier controlled comparator probe as historical evidence, not a new scorer equivalence test. No real model scoring is launched.
- [x] Request the required final code review with the full relevant change package. Resolve correctness/security/compatibility findings; do not claim runtime equivalence from declarations or one implementation's tests. Review evidence may reuse tests on unchanged bytes. Preserve SDD task reports and any rulings; do not let cleanup remove source evidence, data, the shared worktree or existing P1–P4 work.
- [x] Determine finishing-a-development-branch applicability before final test selection. This migration plan does not request merge, PR or branch handoff and P6 remains, so use claim-specific verification below. If the user has since explicitly requested branch handoff, apply that finish gate and its full-suite requirement; do not call the following scoped suite a full project suite.
- [x] Run the final CPU contracts once on reviewed final bytes. Expected: all pass; do not invent the final count.

```bash
./venv/ilvr-venv/bin/python -m pytest \
  tests/evaluation/test_offline_schema.py tests/evaluation/test_vsp_offline.py \
  tests/evaluation/test_offline_io.py tests/evaluation/test_offline_report.py \
  tests/evaluation/test_offline_cli.py tests/evaluation/test_scoring.py \
  tests/evaluation/test_manifest_preflight.py tests/evaluation/test_vsp_prepare.py \
  tests/evaluation/test_inference_bridge.py tests/evaluation/test_run_eval.py \
  tests/evaluation/test_vsp_pipeline.py tests/evaluation/test_comt_offline.py \
  tests/evaluation/test_comt_contract.py tests/evaluation/test_comt_prepare.py \
  tests/evaluation/test_comt_pipeline.py tests/evaluation/test_visulogic_offline.py \
  tests/evaluation/test_visulogic_prepare.py tests/evaluation/test_visulogic_pipeline.py \
  tests/evaluation/test_emma_math.py tests/evaluation/test_emma_offline.py \
  tests/evaluation/test_emma_contract.py tests/evaluation/test_emma_report.py \
  tests/evaluation/test_emma_prepare.py tests/evaluation/test_emma_pipeline.py \
  -q -p no:cacheprovider --basetemp data/evaluation/p5-final-cpu-01
```

- [x] Run the smallest existing runtime integration witness separately; these are stubs, not GPU/model inference:

```bash
./venv/ilvr-venv/bin/python -m pytest \
  tests/evaluation/test_inference_bridge_runtime.py \
  tests/stage2/test_eval_integration.py::test_stage1_eval_uses_continuous_latent_controller_without_model_generate \
  tests/stage2/test_eval_integration.py::test_prompt_preparation_excludes_gold_from_tensors_and_identity \
  tests/stage2/test_eval_integration.py::test_eval_uses_controller_with_original_visual_inputs_and_preserves_decoding \
  -q -p no:cacheprovider --basetemp data/evaluation/p5-final-stubs-01
```

- [x] Confirm protected files, Reference roots/commits/clean/hash and symlinks unchanged. Record the final actual manifest/hash/census and CPU/stub outcomes in validation.md. Add protocol documentation with the prepared paths, CLI example, modified rules, diagnostics, numeric rounding, unsupported examples, deferred first-nail, and no-paper claim. Existing historical evidence is not overwritten with new pass labels.
- [x] Stage only owned changes, report P5 offline implementation/data preparation scope and all remaining not_run/unclear evidence. Before the final response run the repository ntfy summary hook with the accurate scoped outcome. Do not ask task-by-task permission after execution approval; do not auto-commit or merge.

## Plan self-review and execution handoff

Coverage: spec 1–2 map to Tasks 2/4; spec3 to Task2; spec4 and parser refinements to Task1/2; spec5 to Tasks3/5; spec6–9 evidence/limitations to Task5. File identities, names, target fields, ComparisonResult fields, source output keys, IDs and report nesting are fixed above. Source discovery artifacts are required only by root's actual-data witness; synthetic tests and runtime scorer do not depend on them.

The approved spec remains the authority for routine execution rulings. Unexpected plan defects are recorded and resolved within that scope; a material design change returns to the design commitment gate. The v3 4,096-bit rational-component bound is a separately reviewed design change; it must be checked before large construction and cannot be replaced by the withdrawn rational-component 1e100 bound. The approved v2 real-value bound remains unchanged.

After plan review, use **SDD in this session** as prescribed by AGENTS.md when available: a fresh implementer per task, task review, then one final review. Inline executing-plans is the fallback if that path is unavailable or the user explicitly chooses it. The writing-plans execution-choice question is already routed by the user's AGENTS.md; approval of the v3 correction and this concrete plan is recorded below. The user can approve both documents together; after that approval no additional execution-choice or task-by-task question is needed.

## 실행 승인 기록 — 2026-09-13

최신 사용자 `진행해`로 제시된 design commitment v3와 상세 실행 계획을 함께 승인받았다. 위 승인 대기 문구는 당시 검토 이력이다. 현재 상태는 **승인 완료·P5 오프라인 구현 및 데이터 준비 완료**이며 task별 재승인을 요청하지 않는다.

## 실행 완료 기록 — 2026-09-13

P5 오프라인 구현과 고정 mini 준비를 완료했다. 실제 출력은 `data/evaluation/emma-mini400-v1-r02/`이며 400문항·614이미지·제외 0개를 구현 전 원본 목록과 전수 대조했다. 최종 P1–P5 CPU 계약 550개와 별도 런타임 스텁 7개가 통과했다. 실제 모델 응답 평가·모델별 first-nail·원본 전체 환경·GPU·논문 프로토콜은 미실행이며 P6는 남아 있다. 검증 기록: `docs/evaluation/validation.md`, `results/evaluation/p5-final-20260913-01/verification.json`. 앞선 승인/실행 대기 문장은 당시 이력으로 보존한다.
