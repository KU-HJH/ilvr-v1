# Task 2 report — EMMA target, extraction, IO, scorer

Status: DONE

## Changed files

- `src/evaluation/schema.py`: added frozen `EmmaTarget` and widened only `Sample.target`.
- `src/evaluation/io.py`: added explicit EMMA config/target parsing, raw EMMA source IDs, and scoring-input SymPy preflight.
- `src/evaluation/registry.py`: added lazy exact EMMA profile dispatch.
- `src/evaluation/benchmarks/emma.py`: added validation, gold-independent extraction, comparison, and scorer entry point.
- `configs/eval/emma_safe_rules_v1.yaml`: added the approved fixed mini-400 scoring configuration.
- `tests/evaluation/emma_fixtures.py`: added independent synthetic EMMA JSONL fixtures.
- `tests/evaluation/test_emma_offline.py`: added extraction, comparison, and scorer behavior tests.
- `tests/evaluation/test_emma_contract.py`: added target, config, identity, serialization, and lazy registry contract tests.

## RED

Command (repository root, repository venv):

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES=''
./venv/ilvr-venv/bin/pytest tests/evaluation/test_emma_offline.py tests/evaluation/test_emma_contract.py --basetemp=data/evaluation/p5-t2-red-01/pytest
```

Complete output: `logs/evaluation/p5-t2-red-01/pytest.log`.

Result: `20 failed, 9 passed in 0.09s`. Expected failures showed the absent EMMA module/schema/config/registry behavior.

## GREEN and regression check

The first focused green run found the tested empty-marker tail raised `IndexError`; the smallest fix treats the absent tail as an empty selected candidate. The test then observes `invalid_format/empty_extraction`.

Final command:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES=''
./venv/ilvr-venv/bin/pytest tests/evaluation/test_emma_offline.py tests/evaluation/test_emma_contract.py tests/evaluation/test_offline_schema.py tests/evaluation/test_offline_io.py tests/evaluation/test_manifest_preflight.py tests/evaluation/test_vsp_offline.py tests/evaluation/test_comt_offline.py tests/evaluation/test_visulogic_offline.py --basetemp=data/evaluation/p5-t2-green-02/pytest
```

Complete output: `logs/evaluation/p5-t2-green-02/pytest.log`.

Result:

```text
collected 159 items
============================= 159 passed in 0.40s ==============================
```

`git diff --check` over all owned implementation and test files also completed with no output.

## Self-review

- Extraction accepts only `response` and `ModelInput`; target data enters only comparison.
- Selected substrings and raw lowercase target answers retain case.
- A valid wrong multiple-choice label returns false without falling through to matching choice content.
- Only EMMA uses raw provenance IDs. Existing benchmark ID construction remains unchanged.
- Registry import loads the EMMA module and Task 1 helpers but does not import SymPy; actual runtime validation occurs in `read_inputs` and `score_emma`.
- Existing `Sample` and `Score` serialized top-level keys remain exact.
- No dependency, legacy CLI/runtime, report, inference, or preparation changes were made.

## Concerns

None within Task 2 scope. Actual-data comparison and the final broad suite remain assigned to the root task.

## Round 1 review fix

The review found that the uppercase-prefix matcher classified general prose such as `A reasonable explanation follows` as choice `A`. A causal regression test first required the entire sentence to remain a parsed raw answer.

RED command:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES=''
./venv/ilvr-venv/bin/pytest tests/evaluation/test_emma_offline.py -q --basetemp=data/evaluation/p5-t2-review1-red-01/pytest
```

Complete output: `logs/evaluation/p5-t2-review1-red-01/pytest.log`.

Result: `1 failed, 16 passed in 0.19s`; the regression observed `A/upper_label` instead of the literal raw sentence.

The matcher now accepts an uppercase prefix only when the label is immediately followed by `:` or `.`, while whole one-letter uppercase and lowercase labels retain their existing methods. Literal `C:` and `D.` cases cover both valid prefix delimiters.

GREEN command:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES=''
./venv/ilvr-venv/bin/pytest tests/evaluation/test_emma_offline.py tests/evaluation/test_emma_contract.py --basetemp=data/evaluation/p5-t2-review1-green-01/pytest
```

Complete output: `logs/evaluation/p5-t2-review1-green-01/pytest.log`.

Result:

```text
collected 31 items
============================== 31 passed in 0.21s ==============================
```

No remaining concern from the round 1 finding.
