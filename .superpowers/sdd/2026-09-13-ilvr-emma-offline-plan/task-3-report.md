# Task 3 report — EMMA secondary aggregation

## Result

Implemented `emma_summary(samples, scores)` and the EMMA-only `summary["emma"]`
integration in `evaluate`. The helper joins samples to scores by ID, rejects
duplicate or mismatched populations, uses manifest metadata for question type,
subject/category, and subject/task denominators, deduplicates Coding tags within
each row, and limits comparison diagnostics to the three approved error names.
Existing benchmark summaries retain their previous shape.

## TDD evidence

The initial RED command placed the missing import at module scope and therefore
failed collection. It is retained as process evidence but is not accepted as the
required RED:

```text
mkdir -p logs/evaluation/p5-t3-red-01 && export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' && set -o pipefail && ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_emma_report.py -q -p no:cacheprovider --basetemp=data/evaluation/p5-t3-red-01 2>&1 | tee logs/evaluation/p5-t3-red-01/pytest.log

ERROR tests/evaluation/test_emma_report.py
1 error in 0.08s
exit 2
```

After moving the feature import inside the tests, the accepted RED command was:

```text
for p in data/evaluation/p5-t3-red-02 logs/evaluation/p5-t3-red-02; do test ! -e "$p" || { echo "occupied: $p"; exit 2; }; done && mkdir -p logs/evaluation/p5-t3-red-02 && export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' && set -o pipefail && ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_emma_report.py -q -p no:cacheprovider --basetemp=data/evaluation/p5-t3-red-02 2>&1 | tee logs/evaluation/p5-t3-red-02/pytest.log

6 failed in 0.05s
exit 1
```

The failures were the missing `emma_summary` API from inside five collected test
cases and the absent EMMA field in `evaluate`; these are the intended missing
behaviors.

After implementation, the first GREEN command was:

```text
for p in data/evaluation/p5-t3-green-01 logs/evaluation/p5-t3-green-01; do test ! -e "$p" || { echo "occupied: $p"; exit 2; }; done && mkdir -p logs/evaluation/p5-t3-green-01 && export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' && set -o pipefail && ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_emma_report.py tests/evaluation/test_offline_report.py -q -p no:cacheprovider --basetemp=data/evaluation/p5-t3-green-01 2>&1 | tee logs/evaluation/p5-t3-green-01/pytest.log

14 passed in 0.05s
exit 0
```

After strengthening the assertions for `empty_extraction`, empty configured
subjects, base denominators, and accuracy, the final GREEN command was:

```text
for p in data/evaluation/p5-t3-green-02 logs/evaluation/p5-t3-green-02; do test ! -e "$p" || { echo "occupied: $p"; exit 2; }; done && mkdir -p logs/evaluation/p5-t3-green-02 && export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' && set -o pipefail && ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_emma_report.py tests/evaluation/test_offline_report.py -q -p no:cacheprovider --basetemp=data/evaluation/p5-t3-green-02 2>&1 | tee logs/evaluation/p5-t3-green-02/pytest.log

14 passed in 0.05s
exit 0
```

Final whitespace validation:

```text
git diff --check -- src/evaluation/report.py tests/evaluation/test_emma_report.py

(no output)
exit 0
```

## Self-review

The expectations use literal `Score.correctness` outcomes rather than scorer
results. The four-row primary case covers missing denominators, duplicate Coding
tags, blank category/task metadata, case-preserved non-Coding metadata, false
observed outcomes, and diagnostic whitelisting. Separate cases cover all missing,
duplicate/mismatched direct calls, an empty configured subject, an unfiltered
subject, and unchanged non-EMMA reports.

No production behavior outside `src/evaluation/report.py` changed. No dependency,
model, dataset, result, reference, or inference files were changed. Concern: the
tests prove the approved synthetic aggregation contract; they do not constitute a
full EMMA evaluation or paper-protocol verification.

## Fix round 1

Review found two missing assertions rather than a production defect. The primary
literal-score case now asserts that `unsupported_expression` contributes
`observed=1` and `correct=0` to its detail group. A separate test runs the actual
EMMA scorer through `evaluate` with `answer:`, then asserts the resulting
`empty_extraction` remains in the row error and `status_counts["invalid_format"]`
while remaining absent from `comparison_diagnostics`.

These additions characterize behavior that the implementation already provided,
so they passed on their first run; no new RED was claimed and production code was
unchanged.

```text
for p in data/evaluation/p5-t3-fix1-green-01 logs/evaluation/p5-t3-fix1-green-01; do test ! -e "$p" || { echo "occupied: $p"; exit 2; }; done && mkdir -p logs/evaluation/p5-t3-fix1-green-01 && export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' && set -o pipefail && ./venv/ilvr-venv/bin/python -m pytest tests/evaluation/test_emma_report.py tests/evaluation/test_offline_report.py -q -p no:cacheprovider --basetemp=data/evaluation/p5-t3-fix1-green-01 2>&1 | tee logs/evaluation/p5-t3-fix1-green-01/pytest.log

...............                                                          [100%]
15 passed in 0.20s
exit 0
```

Fix-round concern remains unchanged: this is synthetic CPU contract evidence, not
a full benchmark or paper-protocol result.
