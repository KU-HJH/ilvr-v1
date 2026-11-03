# Final P5 consolidated fix wave

Status: DONE for the assigned I1–I4/M1 wave; root final validation remains pending.

## Scope and causal trace

- I1: `score_emma -> extract_answer -> _finish` previously overwrote the selected candidate with `unwrap_math`, then `compare_answer -> compare_values` unwrapped it again. `_finish` now inspects a separate normalized representation for empty/nonfinite checks and retains the candidate in ParsedAnswer/Score. `compare_answer` inspects one normalized label but passes the original selected value to `compare_values`. Both comparison operands therefore undergo exactly one layer of normalization; wrapped wrong labels remain decisive. Regression witnesses retain `$0.5$`, `\(0.5\)`, wrapped nonfinite/whitespace representations and wrong wrapped B labels, and reject numeric equivalence after double wrappers. The opposite operand's double-wrapper case was already passing and remains a boundary control.
- I2: `parse_sample` constructs an EmmaTarget and calls the same `validate_target` used by direct scoring. That validator now requires 2–8 choices before range/label/content checks. JSON-shaped one-choice A and nine-choice I/index 8 cases failed causally in RED; two-choice b and eight-choice h controls already passed.
- I3: `evaluate -> score_rows` preserves infrastructure errors as free text without scoring. Aggregation previously matched those strings alone. Diagnostic aggregation now restricts matches to parsed/invalid_format response statuses. A real evaluate-level fixture with all three code-like infrastructure messages preserves the three unobserved rows and null accuracy while producing zero comparison diagnostics.
- I4: the old startswith/endswith test did not prove enclosure. `_finish` now counts balanced braces from the initial text wrapper and removes it only when its matching close is the candidate's final character. Exact-content scoring preserves `\text{a}+\text{b}` from the box. Nested brace and nested text positive controls retain the one-layer behavior.
- M1: restored VisuLogic's historical `-m scripts.prepare_eval` spelling and changed the EMMA historical recipe to `scripts/prepare_eval.py`. Protocol prose now records comparison-only math normalization and retained extraction evidence.

No math helper API, math parser grammar, dependencies, actual preparation inputs/artifacts, runtime/model code, or validation.md changed. No full suite, controlled runtime stub, model, GPU, upstream environment or paper reproduction was run. No commits, branches, source-data rebuilds, or subagents were created.

## Changed files

- src/evaluation/benchmarks/emma.py
- src/evaluation/report.py
- tests/evaluation/test_emma_offline.py
- tests/evaluation/test_emma_contract.py
- tests/evaluation/test_emma_report.py
- docs/evaluation/protocols.md
- .superpowers/sdd/2026-09-13-ilvr-emma-offline-plan/final-fix-report.md (this report)

## RED

The selected new causal tests ran before production edits. Both run IDs were checked absent under data/results/logs before execution. Synthetic make_case fixtures separately use fresh UUID directories under data/results.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_emma_offline.py tests/evaluation/test_emma_contract.py tests/evaluation/test_emma_report.py -k 'preserves_math_wrappers or wrapped_wrong_label or box_text_wrapper or choice_population or choice_boundaries or infrastructure_error_codes' --basetemp=data/evaluation/p5-final-fix-red-20260913-01 > logs/evaluation/p5-final-fix-red-20260913-01/pytest.log 2>&1
cat logs/evaluation/p5-final-fix-red-20260913-01/pytest.log
```

Result: 12 expected assertion failures, 5 passing boundary controls, 38 deselected. No collection/environment error. The shell wrapper's exit was 0 because cat followed pytest; the recorded pytest failure summary is the RED evidence.

Full output:

```text
FFF.FFFFFF..FF..F                                                        [100%]
=================================== FAILURES ===================================
_ test_score_preserves_math_wrappers_and_normalizes_one_layer[$0.5$-0.5-True-parsed-None] _

response = '$0.5$', answer = '0.5', correctness = True, status = 'parsed'
error = None

    @pytest.mark.parametrize('response,answer,correctness,status,error', [
        ('$0.5$', '0.5', True, 'parsed', None),
        (r'\(0.5\)', '0.5', True, 'parsed', None),
        ('$$0.5$$', '0.5', False, 'parsed', 'unsupported_expression'),
        ('0.5', '$$0.5$$', False, 'parsed', 'unsupported_expression'),
        ('$NaN$', '0', False, 'invalid_format', 'non_finite_value'),
        (r'\(-inf\)', '0', False, 'invalid_format', 'non_finite_value'),
        ('$ $', '0', False, 'invalid_format', 'empty_extraction'),
    ])
    def test_score_preserves_math_wrappers_and_normalizes_one_layer(response, answer, correctness, status, error):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                        EmmaTarget('open_ended', answer, None, answer, '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
        assert (score.correctness, score.status, score.error) == (correctness, status, error)
>       assert score.parsed_answer == response
E       AssertionError: assert '0.5' == '$0.5$'
E         
E         - $0.5$
E         ? -   -
E         + 0.5

tests/evaluation/test_emma_offline.py:89: AssertionError
_ test_score_preserves_math_wrappers_and_normalizes_one_layer[\\(0.5\\)-0.5-True-parsed-None] _

response = '\\(0.5\\)', answer = '0.5', correctness = True, status = 'parsed'
error = None

    @pytest.mark.parametrize('response,answer,correctness,status,error', [
        ('$0.5$', '0.5', True, 'parsed', None),
        (r'\(0.5\)', '0.5', True, 'parsed', None),
        ('$$0.5$$', '0.5', False, 'parsed', 'unsupported_expression'),
        ('0.5', '$$0.5$$', False, 'parsed', 'unsupported_expression'),
        ('$NaN$', '0', False, 'invalid_format', 'non_finite_value'),
        (r'\(-inf\)', '0', False, 'invalid_format', 'non_finite_value'),
        ('$ $', '0', False, 'invalid_format', 'empty_extraction'),
    ])
    def test_score_preserves_math_wrappers_and_normalizes_one_layer(response, answer, correctness, status, error):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                        EmmaTarget('open_ended', answer, None, answer, '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
        assert (score.correctness, score.status, score.error) == (correctness, status, error)
>       assert score.parsed_answer == response
E       AssertionError: assert '0.5' == '\\(0.5\\)'
E         
E         - \(0.5\)
E         + 0.5

tests/evaluation/test_emma_offline.py:89: AssertionError
_ test_score_preserves_math_wrappers_and_normalizes_one_layer[$$0.5$$-0.5-False-parsed-unsupported_expression] _

response = '$$0.5$$', answer = '0.5', correctness = False, status = 'parsed'
error = 'unsupported_expression'

    @pytest.mark.parametrize('response,answer,correctness,status,error', [
        ('$0.5$', '0.5', True, 'parsed', None),
        (r'\(0.5\)', '0.5', True, 'parsed', None),
        ('$$0.5$$', '0.5', False, 'parsed', 'unsupported_expression'),
        ('0.5', '$$0.5$$', False, 'parsed', 'unsupported_expression'),
        ('$NaN$', '0', False, 'invalid_format', 'non_finite_value'),
        (r'\(-inf\)', '0', False, 'invalid_format', 'non_finite_value'),
        ('$ $', '0', False, 'invalid_format', 'empty_extraction'),
    ])
    def test_score_preserves_math_wrappers_and_normalizes_one_layer(response, answer, correctness, status, error):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                        EmmaTarget('open_ended', answer, None, answer, '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
>       assert (score.correctness, score.status, score.error) == (correctness, status, error)
E       AssertionError: assert (True, 'parsed', None) == (False, 'pars...d_expression')
E         
E         At index 0 diff: True != False
E         Use -v to get more diff

tests/evaluation/test_emma_offline.py:88: AssertionError
_ test_score_preserves_math_wrappers_and_normalizes_one_layer[$NaN$-0-False-invalid_format-non_finite_value] _

response = '$NaN$', answer = '0', correctness = False, status = 'invalid_format'
error = 'non_finite_value'

    @pytest.mark.parametrize('response,answer,correctness,status,error', [
        ('$0.5$', '0.5', True, 'parsed', None),
        (r'\(0.5\)', '0.5', True, 'parsed', None),
        ('$$0.5$$', '0.5', False, 'parsed', 'unsupported_expression'),
        ('0.5', '$$0.5$$', False, 'parsed', 'unsupported_expression'),
        ('$NaN$', '0', False, 'invalid_format', 'non_finite_value'),
        (r'\(-inf\)', '0', False, 'invalid_format', 'non_finite_value'),
        ('$ $', '0', False, 'invalid_format', 'empty_extraction'),
    ])
    def test_score_preserves_math_wrappers_and_normalizes_one_layer(response, answer, correctness, status, error):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                        EmmaTarget('open_ended', answer, None, answer, '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
        assert (score.correctness, score.status, score.error) == (correctness, status, error)
>       assert score.parsed_answer == response
E       AssertionError: assert 'NaN' == '$NaN$'
E         
E         - $NaN$
E         ? -   -
E         + NaN

tests/evaluation/test_emma_offline.py:89: AssertionError
_ test_score_preserves_math_wrappers_and_normalizes_one_layer[\\(-inf\\)-0-False-invalid_format-non_finite_value] _

response = '\\(-inf\\)', answer = '0', correctness = False
status = 'invalid_format', error = 'non_finite_value'

    @pytest.mark.parametrize('response,answer,correctness,status,error', [
        ('$0.5$', '0.5', True, 'parsed', None),
        (r'\(0.5\)', '0.5', True, 'parsed', None),
        ('$$0.5$$', '0.5', False, 'parsed', 'unsupported_expression'),
        ('0.5', '$$0.5$$', False, 'parsed', 'unsupported_expression'),
        ('$NaN$', '0', False, 'invalid_format', 'non_finite_value'),
        (r'\(-inf\)', '0', False, 'invalid_format', 'non_finite_value'),
        ('$ $', '0', False, 'invalid_format', 'empty_extraction'),
    ])
    def test_score_preserves_math_wrappers_and_normalizes_one_layer(response, answer, correctness, status, error):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                        EmmaTarget('open_ended', answer, None, answer, '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
        assert (score.correctness, score.status, score.error) == (correctness, status, error)
>       assert score.parsed_answer == response
E       AssertionError: assert '-inf' == '\\(-inf\\)'
E         
E         - \(-inf\)
E         + -inf

tests/evaluation/test_emma_offline.py:89: AssertionError
_ test_score_preserves_math_wrappers_and_normalizes_one_layer[$ $-0-False-invalid_format-empty_extraction] _

response = '$ $', answer = '0', correctness = False, status = 'invalid_format'
error = 'empty_extraction'

    @pytest.mark.parametrize('response,answer,correctness,status,error', [
        ('$0.5$', '0.5', True, 'parsed', None),
        (r'\(0.5\)', '0.5', True, 'parsed', None),
        ('$$0.5$$', '0.5', False, 'parsed', 'unsupported_expression'),
        ('0.5', '$$0.5$$', False, 'parsed', 'unsupported_expression'),
        ('$NaN$', '0', False, 'invalid_format', 'non_finite_value'),
        (r'\(-inf\)', '0', False, 'invalid_format', 'non_finite_value'),
        ('$ $', '0', False, 'invalid_format', 'empty_extraction'),
    ])
    def test_score_preserves_math_wrappers_and_normalizes_one_layer(response, answer, correctness, status, error):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                        EmmaTarget('open_ended', answer, None, answer, '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
>       assert (score.correctness, score.status, score.error) == (correctness, status, error)
E       AssertionError: assert (False, 'pars...d_expression') == (False, 'inva...y_extraction')
E         
E         At index 1 diff: 'parsed' != 'invalid_format'
E         Use -v to get more diff

tests/evaluation/test_emma_offline.py:88: AssertionError
_______ test_wrapped_wrong_label_remains_decisive_at_score_boundary[$B$] _______

response = '$B$'

    @pytest.mark.parametrize('response', ['$B$', r'\(B\)'])
    def test_wrapped_wrong_label_remains_decisive_at_score_boundary(response):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ('B', 'other')),
                        EmmaTarget('multiple_choice', 'A', 0, 'B', '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
        assert score.correctness is False
        assert score.error is None
>       assert score.parsed_answer == response
E       AssertionError: assert 'B' == '$B$'
E         
E         - $B$
E         + B

tests/evaluation/test_emma_offline.py:102: AssertionError
_____ test_wrapped_wrong_label_remains_decisive_at_score_boundary[\\(B\\)] _____

response = '\\(B\\)'

    @pytest.mark.parametrize('response', ['$B$', r'\(B\)'])
    def test_wrapped_wrong_label_remains_decisive_at_score_boundary(response):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ('B', 'other')),
                        EmmaTarget('multiple_choice', 'A', 0, 'B', '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
        assert score.correctness is False
        assert score.error is None
>       assert score.parsed_answer == response
E       AssertionError: assert 'B' == '\\(B\\)'
E         
E         - \(B\)
E         + B

tests/evaluation/test_emma_offline.py:102: AssertionError
_ test_box_text_wrapper_must_enclose_whole_candidate[\\boxed{\\text{a}+\\text{b}}-\\text{a}+\\text{b}] _

response = '\\boxed{\\text{a}+\\text{b}}', answer = '\\text{a}+\\text{b}'

    @pytest.mark.parametrize('response,answer', [
        (r'\boxed{\text{a}+\text{b}}', r'\text{a}+\text{b}'),
        (r'\boxed{\text{a{b}c}}', 'a{b}c'),
        (r'\boxed{\text{\text{a}}}', r'\text{a}'),
    ])
    def test_box_text_wrapper_must_enclose_whole_candidate(response, answer):
        from evaluation.benchmarks.emma import score_emma
        from evaluation.schema import EmmaTarget, ModelInput, Prediction, Provenance, Sample
        sample = Sample(1, 'math_0', 'emma', 'Math', ModelInput('Q', (), ()),
                        EmmaTarget('open_ended', answer, None, answer, '', ''),
                        Provenance('synthetic:emma', 'fixture', 'Math', 'test', 'math_0', {}))
        score = score_emma(sample, Prediction('math_0', 'manifest', 'run', response, 'ok'))
>       assert score.parsed_answer == answer
E       AssertionError: assert 'a}+\\text{b' == '\\text{a}+\\text{b}'
E         
E         - \text{a}+\text{b}
E         + a}+\text{b

tests/evaluation/test_emma_offline.py:117: AssertionError
_ test_manifest_rejects_mcq_choice_population_outside_two_to_eight[choices0-A-0] _

choices = ['only'], answer = 'A', index = 0

    @pytest.mark.parametrize('choices,answer,index', [
        (['only'], 'A', 0),
        (['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'], 'I', 8),
    ])
    def test_manifest_rejects_mcq_choice_population_outside_two_to_eight(choices, answer, index):
        from evaluation.io import parse_sample
        row = make_case(1)['rows'][0]
        row['model_input']['choices'] = choices
        row['target'].update(answer=answer, index=index, content=choices[index])
>       with pytest.raises(ContractError):
E       Failed: DID NOT RAISE ContractError

tests/evaluation/test_emma_contract.py:91: Failed
_ test_manifest_rejects_mcq_choice_population_outside_two_to_eight[choices1-I-8] _

choices = ['one', 'two', 'three', 'four', 'five', 'six', ...], answer = 'I'
index = 8

    @pytest.mark.parametrize('choices,answer,index', [
        (['only'], 'A', 0),
        (['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'], 'I', 8),
    ])
    def test_manifest_rejects_mcq_choice_population_outside_two_to_eight(choices, answer, index):
        from evaluation.io import parse_sample
        row = make_case(1)['rows'][0]
        row['model_input']['choices'] = choices
        row['target'].update(answer=answer, index=index, content=choices[index])
>       with pytest.raises(ContractError):
E       Failed: DID NOT RAISE ContractError

tests/evaluation/test_emma_contract.py:91: Failed
_ test_evaluate_excludes_infrastructure_error_codes_from_comparison_diagnostics _

    def test_evaluate_excludes_infrastructure_error_codes_from_comparison_diagnostics():
        samples = tuple(_sample(sample_id) for sample_id in ('a', 'b', 'c'))
        diagnostics = ('unsupported_expression', 'non_finite_value', 'math_limit_exceeded')
        predictions = {
            sample.id: Prediction(sample.id, 'manifest', 'run', None, 'infrastructure_error', error)
            for sample, error in zip(samples, diagnostics)
        }
        config = EvalConfig(1, 'emma', 'emma_safe_rules_v1', ('Coding',),
                            Path('data/evaluation/fixture'), None,
                            RunMetadata('fixture', 'fixture', {}))
        scores, summary = evaluate(PreparedEvaluation(samples, predictions, 'manifest', 'run'), config)
        assert [score.error for score in scores] == list(diagnostics)
        assert summary['status_counts'] == {'infrastructure_error': 3}
        assert (summary['N_expected'], summary['observed'], summary['accuracy']) == (3, 0, None)
        assert summary['emma']['question_types']['multiple_choice'] == dict(
            count=3, observed=0, correct=0, complete=False, accuracy=None)
>       assert summary['emma']['comparison_diagnostics'] == {
            'unsupported_expression': 0, 'non_finite_value': 0, 'math_limit_exceeded': 0,
        }
E       AssertionError: assert {'unsupported..._exceeded': 1} == {'unsupported..._exceeded': 0}
E         
E         Differing items:
E         {'unsupported_expression': 1} != {'unsupported_expression': 0}
E         {'non_finite_value': 1} != {'non_finite_value': 0}
E         {'math_limit_exceeded': 1} != {'math_limit_exceeded': 0}
E         Use -v to get more diff

tests/evaluation/test_emma_report.py:222: AssertionError
=========================== short test summary info ============================
FAILED tests/evaluation/test_emma_offline.py::test_score_preserves_math_wrappers_and_normalizes_one_layer[$0.5$-0.5-True-parsed-None]
FAILED tests/evaluation/test_emma_offline.py::test_score_preserves_math_wrappers_and_normalizes_one_layer[\\(0.5\\)-0.5-True-parsed-None]
FAILED tests/evaluation/test_emma_offline.py::test_score_preserves_math_wrappers_and_normalizes_one_layer[$$0.5$$-0.5-False-parsed-unsupported_expression]
FAILED tests/evaluation/test_emma_offline.py::test_score_preserves_math_wrappers_and_normalizes_one_layer[$NaN$-0-False-invalid_format-non_finite_value]
FAILED tests/evaluation/test_emma_offline.py::test_score_preserves_math_wrappers_and_normalizes_one_layer[\\(-inf\\)-0-False-invalid_format-non_finite_value]
FAILED tests/evaluation/test_emma_offline.py::test_score_preserves_math_wrappers_and_normalizes_one_layer[$ $-0-False-invalid_format-empty_extraction]
FAILED tests/evaluation/test_emma_offline.py::test_wrapped_wrong_label_remains_decisive_at_score_boundary[$B$]
FAILED tests/evaluation/test_emma_offline.py::test_wrapped_wrong_label_remains_decisive_at_score_boundary[\\(B\\)]
FAILED tests/evaluation/test_emma_offline.py::test_box_text_wrapper_must_enclose_whole_candidate[\\boxed{\\text{a}+\\text{b}}-\\text{a}+\\text{b}]
FAILED tests/evaluation/test_emma_contract.py::test_manifest_rejects_mcq_choice_population_outside_two_to_eight[choices0-A-0]
FAILED tests/evaluation/test_emma_contract.py::test_manifest_rejects_mcq_choice_population_outside_two_to_eight[choices1-I-8]
FAILED tests/evaluation/test_emma_report.py::test_evaluate_excludes_infrastructure_error_codes_from_comparison_diagnostics
12 failed, 5 passed, 38 deselected in 0.27s
```

## GREEN

The branch finishing gate belongs to the root controller; this worker used the assigned focused verification command and reserved the final full suite for root.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/evaluation:. CUDA_VISIBLE_DEVICES='' ./venv/ilvr-venv/bin/python -m pytest -q tests/evaluation/test_emma_math.py tests/evaluation/test_emma_offline.py tests/evaluation/test_emma_contract.py tests/evaluation/test_emma_report.py --basetemp=data/evaluation/p5-final-fix-green-20260913-01 > logs/evaluation/p5-final-fix-green-20260913-01/pytest.log 2>&1
emma_test_exit=$?
cat logs/evaluation/p5-final-fix-green-20260913-01/pytest.log
exit "$emma_test_exit"
```

Exit 0. Full output:

```text
........................................................................ [ 47%]
........................................................................ [ 94%]
........                                                                 [100%]
152 passed in 0.55s
```

## Remaining concerns

No unresolved contract ambiguity identified for this bounded fix wave. This focused result establishes the changed contracts and included math regressions only; it does not replace root's final P1–P5 CPU run, controlled runtime witness, preservation checks, or review. The original actual-data r02 remains the fixed prepared run and was not rebuilt or revalidated here.
