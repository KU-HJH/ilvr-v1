# Stage1-v5 External Content Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development, the user's retained implementation choice, to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not switch to root implementation.

**Goal:** Give adaptive target selection the ILVR/legacy placeholder-conditioned Teacher observation and compute relative history relevance outside the Teacher.

**Architecture:** A parameter-free, FP32 cosine metric reads current final text hidden states and earlier raw selected visual rows. Adaptive selection shares the literal legacy language observation, keeps discounted text and existing candidate selection, and replaces injected-prefix native attention. Activate the path and its strict saved identity atomically; historical native-adaptive contracts remain inspectable but cannot resume training under new semantics.

**Tech Stack:** Repository Python 3.11 venv, PyTorch, vendored Transformers Qwen2.5-VL, PEFT DoRA, pytest, existing Stage1-v5 contracts and verification script.

---

## Bound scope, authorization and starting evidence

Work in `/root/.codex/worktrees/6b9d/ILVR`, existing isolated `Stage1-v5`.
Starting control commit `b39e0f2`; product baseline `1c86472`.
Canonical checkout and existing untracked files are outside this task.
Read canonical AGENTS and worktree AGENTS. Preserve data/checkpoints/venv links.

Bound design: `docs/superpowers/specs/2026-10-17-stage1-v5-content-metric-design.md`,
SHA256 `113658fdd4c8ddf8357d3cace7dff21753e8148969a4c566961357a05aaa1d3d`.
The user's subsequent `$superpowers:writing-plans` advances this presented
commitment; the exact response and prior retained SDD selection are recorded in
`docs/superpowers/2026-10-17-stage1-v5-content-metric-workflow.md`.

Teacher means target selection; reference means DIL scoring. Keep frozen DoRA
weights/EMA1, reference rules, student target forcing, losses and global-token
DIL denominators, pooling/top-p/first-K, workers, sampler and training lifecycle.
Legacy is the default and must preserve outputs, forward arguments and RNG.
No real training, evaluation, GPU campaign, dependency install or costly premise
probe. Cosine calibration/usefulness remains uncertain; arithmetic tests do not
prove accuracy or speed. Matching ILVR inputs does not prove original training
equivalence. Do not label relative metric weights native attention mass.

Fresh focused pre-change baseline: 176 passed in 11.87s, exit 0, complete output
`logs/stage1-v5-content-metric-20261017/baseline.log`. The previously reviewed
merged product full suite passed 4935 tests/11 skips. Reuse this baseline; run
one final full suite after independent reviews, not another baseline full suite.

## File responsibility map

| Files | Responsibility |
| --- | --- |
| Create `src/stage1/history_metric.py` | Validated chunked cosine-relative history and equal-coefficient query composition; no model access |
| Create `tests/stage1/test_history_metric.py` | Independent literal math, invalid/empty/chunk/no-grad and actual selector-boundary witnesses |
| Modify `src/stage1/targets.py` | Shared legacy observation; adaptive consumes final text rows plus raw earlier targets, preserving ownership/selector order |
| Modify `src/stage1/query.py` | New strict schema2 identity, historical schema1 reconstruction, current training validation; retain old utility math |
| Modify `src/stage1/v5_resume.py` | Separate read-only contract validation from early incompatible training-resume rejection |
| Modify `src/utils_deepseed.py`, `src/stage1/composition.py` | CLI and direct composition fail before loaders for invalid current layers/old native resume |
| Modify `src/stage1/v5_trainer.py`, `scripts/verify_stage1_v5.py` | Actual metric source binding and exact compatible verifier source map |
| Modify `tests/stage1/test_history_query.py`, `tests/stage1/test_history_targets.py` | Identity migration and real Teacher/selection/ownership witnesses |
| Modify `tests/stage1/test_cli.py`, `tests/stage1/test_composition.py` | Pre-model parser/direct composition guard witnesses |
| Modify `tests/stage1/v5/test_resume.py`, `tests/stage1/v5/test_protocol.py`, `tests/stage1/v5/test_workflow.py`, `tests/stage1/v5/test_history_integration.py` | Historical inspection, current contracts, actual parent/shared target/reference/gradient and verifier consumers |
| Modify `README.md` | Current adaptive formula, role/input distinction, final-layer restriction and fresh-run requirement |
| Inspect `run_stage1_training.sh`, `scripts/stage1_v5_train_eval.sh`, `src/multinode_runtime.py` | Existing flags/default last must stay compatible; change only affected wording or assumptions |
| Preserve `src/stage1/history_teacher.py`, `src/stage1/history_attention.py` and their utility tests | Historical code remains callable in isolation; never used by new adaptive execution |

The two tasks are sequential. Task 1 adds an unused pure module and makes a
safe independent commit. Task 2 is one atomic activation because observation,
formula identity, resume guards and source consumers must agree. Do not split
activation into commits that serialize new semantics while executing old code.
Within each task, follow the small RED/GREEN actions below and record commands,
exit codes and complete log locations. Keep requested fixes in implementer agents.

All Python test commands below use this exact prefix from the worktree:

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.:src:transformers/src
```

Logs live under `logs/stage1-v5-content-metric-20261017/`; results are metadata
only under the corresponding results directory. Existing CPU scalar checkpoint
fixtures must keep using the repository checkpoint namespace. Do not serialize
real model artifacts in logs/results or arbitrary temporary directories.

### Task 1: Add the pure external metric and discriminating selector test

**Files:** Create `src/stage1/history_metric.py` and
`tests/stage1/test_history_metric.py`. No runtime activation in this task.

- [ ] **Step 1: Write the independent cosine/aggregation RED test.**

```python
import pytest
import torch
from stage1.history_metric import content_history, compose_content_query


def test_content_history_uses_global_softmax_and_raw_values_without_grad():
    text = torch.tensor([[1., 0.], [1., 0.], [0., 1.]], requires_grad=True)
    values = torch.tensor([[4., 0.], [0., 1.]], requires_grad=True)
    # Literal cosine scores; independent of the production normalizer/chunks.
    expected = torch.softmax(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]), 1).mean(0)
    result = content_history(text, values, query_chunk_size=1, key_chunk_size=1)
    assert result.present
    torch.testing.assert_close(result.weights, expected)
    torch.testing.assert_close(result.summary, (expected[:, None] * values.detach()).sum(0, keepdim=True))
    assert result.weights.dtype == result.summary.dtype == torch.float32
    assert not result.weights.requires_grad and not result.summary.requires_grad
    assert text.grad is None and values.grad is None
```

- [ ] **Step 2: Run RED and read the complete failure.**

```bash
./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_metric.py
```

Expected: collection fails because `stage1.history_metric` does not exist.
Save the output as `task1-red.log`; a dependency/import environment failure
unrelated to that module is not the expected RED.

- [ ] **Step 3: Implement the validated metric interface.**

Use this complete module. The two key passes are a global softmax at fixed
tau1: cosine logits lie in [-1,1], so exponentiation is bounded; no per-chunk
normalization or full query-by-history matrix is needed.

```python
"""External relative content relevance for Stage1 target selection."""
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class ContentHistory:
    weights: Tensor
    summary: Tensor
    present: bool


def _matrix(name: str, value: Tensor) -> Tensor:
    if (not isinstance(value, Tensor) or value.ndim != 2
            or value.shape[1] == 0 or not value.is_floating_point()):
        raise ValueError(f'{name} must be a floating [N,H] tensor with H positive')
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f'{name} must be finite in FP32')
    return value


@torch.no_grad()
def content_history(text: Tensor, values: Tensor, *,
                    query_chunk_size: int = 64,
                    key_chunk_size: int = 2048) -> ContentHistory:
    for name, size in (('query_chunk_size', query_chunk_size),
                       ('key_chunk_size', key_chunk_size)):
        if type(size) is not int or size <= 0:
            raise ValueError(f'{name} must be a positive integer')
    text, values = _matrix('current text', text), _matrix('history values', values)
    if text.shape[1] != values.shape[1] or text.device != values.device:
        raise ValueError('text and history must share hidden width and device')
    weights = values.new_zeros(values.shape[0])
    empty = values.new_zeros((1, values.shape[1]))
    if not text.shape[0] or not values.shape[0]:
        return ContentHistory(weights, empty, False)
    text_norm = torch.linalg.vector_norm(text, dim=1, keepdim=True)
    value_norm = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    for norm in (text_norm, value_norm):
        if not bool(torch.isfinite(norm).all()) or bool((norm <= 1e-8).any()):
            raise FloatingPointError('scoring vector norms must be finite and greater than 1e-8')
    queries, keys = text / text_norm, values / value_norm
    for q0 in range(0, queries.shape[0], query_chunk_size):
        query = queries[q0:q0 + query_chunk_size]
        denominator = query.new_zeros((query.shape[0], 1))
        for k0 in range(0, keys.shape[0], key_chunk_size):
            logits = query @ keys[k0:k0 + key_chunk_size].T
            denominator += logits.exp().sum(1, keepdim=True)
        for k0 in range(0, keys.shape[0], key_chunk_size):
            logits = query @ keys[k0:k0 + key_chunk_size].T
            weights[k0:k0 + key_chunk_size] += (logits.exp() / denominator).sum(0)
    weights /= text.shape[0]
    summary = (weights[:, None] * values).sum(0, keepdim=True)
    if not bool(torch.isfinite(summary).all()):
        raise FloatingPointError('history summary must be finite in FP32')
    return ContentHistory(weights, summary, True)


@torch.no_grad()
def compose_content_query(global_intent: Tensor, text_summary: Tensor | None,
                          history: ContentHistory) -> Tensor:
    intent = _matrix('global intent', global_intent)
    if intent.shape[0] != 1:
        raise ValueError('global intent must have shape [1,H]')
    parts = [intent]
    for name, value in (('text summary', text_summary),
                        ('history summary', history.summary if history.present else None)):
        if value is None:
            continue
        row = _matrix(name, value)
        if row.shape != intent.shape or row.device != intent.device:
            raise ValueError(f'{name} must share query shape and device')
        parts.append(row)
    query = torch.stack(parts).mean(0)
    norm = torch.linalg.vector_norm(query)
    if not bool(torch.isfinite(norm)) or bool(norm <= 1e-8):
        raise FloatingPointError('history query norm must be finite and greater than 1e-8')
    return query
```

- [ ] **Step 4: Add chunk/empty/invalid-input tests before changing any failing behavior.**

```python
@pytest.mark.parametrize('q,k', [(1, 1), (2, 2), (64, 2048)])
def test_chunk_boundaries_preserve_one_softmax(q, k):
    text = torch.tensor([[1., 2.], [-3., 1.], [2., -1.]])
    values = torch.tensor([[3., 1.], [-1., 2.], [4., -1.], [2., 5.], [-2., -3.]])
    logits = torch.nn.functional.normalize(text, dim=1) @ torch.nn.functional.normalize(values, dim=1).T
    expected = logits.softmax(1).mean(0)
    result = content_history(text, values, query_chunk_size=q, key_chunk_size=k)
    torch.testing.assert_close(result.weights, expected)
    torch.testing.assert_close(result.summary, (expected[:, None] * values).sum(0, keepdim=True))


@pytest.mark.parametrize('n,m', [(0, 2), (2, 0), (0, 0)])
def test_empty_metric_omits_history_instead_of_uniform_fallback(n, m):
    history = content_history(torch.ones(n, 2), torch.ones(m, 2))
    assert not history.present and history.weights.shape == (m,)
    assert not history.weights.any() and not history.summary.any()
    intent, text = torch.tensor([[1., 0.]]), torch.tensor([[0., 2.]])
    torch.testing.assert_close(compose_content_query(intent, text, history), (intent + text) / 2)


@pytest.mark.parametrize('bad', [torch.tensor([[0., 0.]]), torch.tensor([[1e-12, 0.]]),
                                torch.tensor([[float('nan'), 1.]]),
                                torch.tensor([[float('inf'), 1.]])])
@pytest.mark.parametrize('side', ['text', 'values'])
def test_invalid_scoring_rows_fail_explicitly(bad, side):
    args = [torch.ones(1, 2), torch.ones(1, 2)]
    args[0 if side == 'text' else 1] = bad
    with pytest.raises(FloatingPointError):
        content_history(*args)


@pytest.mark.parametrize('options', [{'query_chunk_size': 0}, {'key_chunk_size': True},
                                   {'key_chunk_size': 1.5}])
def test_bad_chunks_fail(options):
    with pytest.raises(ValueError):
        content_history(torch.ones(1, 2), torch.ones(1, 2), **options)


@pytest.mark.parametrize('text,values', [(torch.ones(2), torch.ones(1, 2)),
                                        (torch.ones(1, 2), torch.ones(1, 3)),
                                        (torch.ones(1, 0), torch.ones(1, 0)),
                                        (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2))])
def test_malformed_metric_tensors_fail(text, values):
    with pytest.raises(ValueError):
        content_history(text, values)


def test_zero_composed_query_fails():
    history = content_history(torch.empty(0, 2), torch.empty(0, 2))
    with pytest.raises(FloatingPointError):
        compose_content_query(torch.ones(1, 2), -torch.ones(1, 2), history)
```

Existing `update_text_history` gamma tests remain the recurrence oracle; do not
replace that function. Add a finite-large norm case if FP32 conversion/norm
overflow is not covered by the above invalid rows. Device mismatch is guarded
without requiring a CUDA test.

- [ ] **Step 5: Add the actual selector-boundary witness.**

```python
from stage1.history_metric import ContentHistory
from stage1.targets import Stage1TeacherTargetBuilder
from tests.stage1.history_fixtures import make_history_case


def test_relative_history_changes_actual_selected_candidate_vs_uniform():
    case = make_history_case()
    selector = Stage1TeacherTargetBuilder(case.tokenizer, coverage_p=.1,
                                          helper_group_L=0, collect_diagnostics=True)
    # Fixed observation and raw row ledger; only relative weighting changes.
    text = torch.tensor([[1., 0.]])
    values = torch.tensor([[4., 0.], [0., 4.]])
    intent = torch.tensor([[0., 1.5]])
    metric = content_history(text, values, query_chunk_size=1, key_chunk_size=1)
    uniform = ContentHistory(torch.full((2,), .5), values.mean(0, keepdim=True), True)
    candidates = torch.eye(2)
    metric_query = compose_content_query(intent, text, metric)
    uniform_query = compose_content_query(intent, text, uniform)
    metric_selected = selector._select(candidates, metric_query, max_count=1)
    uniform_selected = selector._select(candidates, uniform_query, max_count=1)
    assert metric_selected[1] == (0,)
    assert uniform_selected[1] == (1,)
    assert metric_selected[1] != uniform_selected[1]
```

This is a known directional, tie-free selector test, not an expectation that
every real sample changes or improves. Final Teacher isolation has a separate
real-model witness in Task 2.

- [ ] **Step 6: Run the focused GREEN set, self-review and commit only Task 1.**

```bash
./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_metric.py tests/stage1/test_history_query.py
git add src/stage1/history_metric.py tests/stage1/test_history_metric.py
git commit -m "feat: add external Stage1 cosine history metric"
```

Expected: all focused tests pass, old runtime unmodified. Record complete output
as `task1-green.log`, commit identity and diff. Independent spec reviewer must
approve before an independent quality reviewer; fix and re-review findings
through this implementer before advancing to Task 2.

### Task 2: Activate placeholder-conditioned adaptive selection and bind its identity

**Files:** All Modify/Test rows in the responsibility map. Task 1 is required.
Keep this task atomic across runtime, contract and current verification consumers.

- [ ] **Step 1: Write a real-model RED witness for literal legacy observation.**

Replace adaptive injection expectations, keeping the legacy frozen-blob oracle.
Add this test beside `observe_build` in `tests/stage1/test_history_targets.py`:

```python
def test_adaptive_observes_literal_legacy_full_input_once_without_target_injection():
    case = make_history_case()
    qwen = resolve_qwen2_5_vl_model(case.teacher)
    calls, hidden_calls = [], []
    handle = qwen.model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append({
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()}), with_kwargs=True)
    def capture_hidden(module, args, output):
        states = output.hidden_states
        last = states[-1] if isinstance(states, (tuple, list)) else states
        hidden_calls.append(last.detach().clone())
    output_handle = case.teacher.register_forward_hook(capture_hidden)
    try:
        # Literal pre-change wrapper call; does not use a new observation helper.
        with torch.no_grad():
            expected = case.teacher(
                input_ids=case.inputs['input_ids'],
                attention_mask=case.inputs['attention_mask'],
                pixel_values=case.inputs['pixel_values'],
                image_grid_thw=case.inputs['image_grid_thw'],
                output_hidden_states=True, return_dict=True)
        literal = calls.pop()
        literal_hidden = hidden_calls.pop()
        target_builder = builder(case)
        result, actual, visuals = observe_build(case, target_builder)
    finally:
        handle.remove()
        output_handle.remove()
    assert len(actual) == 1
    assert len(hidden_calls) == 1
    torch.testing.assert_close(hidden_calls[0], literal_hidden, rtol=0, atol=0)
    assert actual[0]['inputs_embeds'].shape[1] == case.inputs['input_ids'].shape[1]
    for key in ('inputs_embeds', 'position_ids', 'attention_mask'):
        torch.testing.assert_close(actual[0][key], literal[key], rtol=0, atol=0)
    positions = [p for span in case.metadata.samples[0].latent_spans for p in span.latent_positions]
    with torch.no_grad():
        pads = qwen.model.embed_tokens(case.inputs['input_ids'])[0, positions]
    torch.testing.assert_close(actual[0]['inputs_embeds'][0, positions], pads, rtol=0, atol=0)
    assert len(visuals) == 2  # Original and helper visual operations are separate.
    assert not result.latent_values.requires_grad
    assert all(not parameter.requires_grad for parameter in case.teacher.parameters())
```

The output hook above captures the actual wrapper hidden states. The independent
literal call must remain independent of the new shared helper. Compare hidden
states at fixed weights/inputs, not changed targets.

- [ ] **Step 2: Run the new witness RED.**

```bash
./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_targets.py::test_adaptive_observes_literal_legacy_full_input_once_without_target_injection
```

Expected: old adaptive makes multiple prefix calls and fails the one-call/full
input witness. Record `task2-observation-red.log`; do not weaken the assertion.

- [ ] **Step 3: Write schema/current-layer RED tests before activation.**

The new serialized mapping is exactly this (no unused head chunk or layer list):

```python
NEW_QUERY = {
    'schema_version': 2, 'mode': 'adaptive',
    'formula_id': 'stage1_placeholder_discounted_segments_cosine_relative_v2',
    'teacher_context': 'ilvr_placeholder_full_sequence_v1',
    'mask_mode': 'native_legacy_v1',
    'text_mode': 'discounted_segments', 'text_gamma': .9,
    'latent_mode': 'external_cosine_relative',
    'hidden_state': 'final_normalized', 'temperature': 1.0,
    'query_chunk_size': 64, 'key_chunk_size': 2048,
}


def test_new_adaptive_execution_identity_roundtrips():
    from stage1.query import (Stage1QueryConfig, query_execution_config,
                             resolve_query_config, validate_query_execution_config)
    config = Stage1QueryConfig(mode='adaptive')
    saved = query_execution_config(config, ())
    assert saved == NEW_QUERY
    assert validate_query_execution_config(saved) == config
    assert resolve_query_config({}, saved=saved) == config


@pytest.mark.parametrize('selector', ['last4', 'all'])
def test_new_training_rejects_nonfinal_layers(selector):
    from stage1.query import resolve_query_config
    with pytest.raises(ValueError, match='final'):
        resolve_query_config({'query_mode': 'adaptive', 'history_attention_layers': selector})
```

Run these tests and save `task2-contract-red.log`. They must fail on missing
validator/new schema or acceptance of unsupported current layers.

- [ ] **Step 4: Implement strict historical read/current training separation.**

In `query.py`, preserve the old serializer body as private
`_historical_query_execution_config(config, resolved_layers)` and its old formula
string. `Stage1QueryConfig` continues to represent historical last4/all and
head chunks so old observer utilities/read-only reconstruction still work.
Add this current training gate and current serializer:

```python
_CONTENT_FORMULA = 'stage1_placeholder_discounted_segments_cosine_relative_v2'


def validate_query_training_config(config: Stage1QueryConfig) -> None:
    if not isinstance(config, Stage1QueryConfig):
        raise ValueError('query training requires Stage1QueryConfig')
    if config.mode == 'adaptive' and config.attention_layers != 'last':
        raise ValueError('adaptive uses final-normalized hidden states; only last is supported')
    if config.mode == 'adaptive' and config.head_chunk_size != 4:
        raise ValueError('adaptive content metric has no head chunk setting')


def query_execution_config(config: Stage1QueryConfig,
                           resolved_layers: tuple[int, ...]) -> dict[str, object]:
    validate_query_training_config(config)
    if config.mode == 'legacy':
        return {'schema_version': 1, 'mode': 'legacy', 'formula_id': _LEGACY_FORMULA}
    return {
        'schema_version': 2, 'mode': 'adaptive', 'formula_id': _CONTENT_FORMULA,
        'teacher_context': 'ilvr_placeholder_full_sequence_v1',
        'mask_mode': 'native_legacy_v1', 'text_mode': 'discounted_segments',
        'text_gamma': config.text_gamma, 'latent_mode': 'external_cosine_relative',
        'hidden_state': 'final_normalized', 'temperature': 1.0,
        'query_chunk_size': config.query_chunk_size,
        'key_chunk_size': config.key_chunk_size,
    }
```

`resolved_layers` is retained only for call compatibility, not serialized as
an unused current knob. Current runtime passes `()`. Historical reconstruction
uses its own serializer and validates the exact old resolved layer list.

Implement the read-only validator separately from the training resolver:

```python
def validate_query_execution_config(saved: Mapping[str, object]) -> Stage1QueryConfig:
    if not isinstance(saved, Mapping):
        raise ValueError('saved query config must be an execution mapping')
    mode, version = saved.get('mode'), saved.get('schema_version')
    if type(version) is not int:
        raise ValueError('saved query schema must be an integer')
    if mode == 'legacy' and version == 1:
        inherited = Stage1QueryConfig()
        expected = _historical_query_execution_config(inherited, ())
    elif mode == 'adaptive' and version == 1:
        names = ('text_gamma', 'attention_layers', *_CHUNKS)
        if any(name not in saved for name in (*names, 'resolved_layer_indices')):
            raise ValueError('saved adaptive query config has missing required fields')
        indices = saved['resolved_layer_indices']
        if type(indices) is not list:
            raise ValueError('saved resolved layer indices must be a JSON list')
        _validate_layers(indices)
        inherited = Stage1QueryConfig(mode='adaptive', **{name: saved[name] for name in names})
        expected = _historical_query_execution_config(inherited, tuple(indices))
    elif mode == 'adaptive' and version == 2:
        names = ('text_gamma', 'query_chunk_size', 'key_chunk_size')
        if any(name not in saved for name in names):
            raise ValueError('saved content query config has missing required fields')
        if type(saved.get('temperature')) is not float:
            raise ValueError('saved content temperature must be a JSON float')
        inherited = Stage1QueryConfig(mode='adaptive', **{name: saved[name] for name in names})
        expected = query_execution_config(inherited, ())
    else:
        raise ValueError('saved query config has invalid mode or schema')
    if dict(saved) != expected:
        raise ValueError('saved query config has invalid schema or semantic fields')
    return inherited
```

Replace the public training resolver with explicit inheritance plus the old
CLI conflict check, rejecting historical native semantics before returning:

```python
def resolve_query_config(explicit, *, saved=None):
    unknown = set(explicit) - _CLI_FIELDS.keys()
    if unknown:
        raise ValueError(f'unknown query option: {sorted(unknown)}')
    values = {_CLI_FIELDS[name]: value for name, value in explicit.items()}
    explicit_config = Stage1QueryConfig(**values)
    if saved is None:
        validate_query_training_config(explicit_config)
        return explicit_config
    inherited = validate_query_execution_config(saved)
    if saved['mode'] == 'adaptive' and saved['schema_version'] == 1:
        raise ValueError('historical native-adaptive training requires its original reviewed code; '
                         'external content adaptive requires a fresh run')
    for name, value in values.items():
        if value != getattr(inherited, name):
            raise ValueError(f'explicit {name} conflicts with saved query resume config')
    validate_query_training_config(inherited)
    return inherited
```

In `v5_resume._execution`, call `validate_query_execution_config`, not the
training resolver. `load_v5_contract` remains strict and read-only for both
histories. In `validate_v5_resume`, after the existing outer schema1/block-mean
rejection and before objective inheritance, call this training query guard:

```python
execution = saved['execution']
if 'query_config' in execution:
    resolve_query_config({}, saved=execution['query_config'])
```

Keep all outer DIL schema/formula/coefficients, source/reference/schedule checks.
Add `'history_metric'` to the missing-query source-binding guard in
`resolve_v5_query_resume`; retain all historical keys. `composition` must call
`validate_query_training_config(query_config)` immediately after extracting it
and before model/processor loaders. Parser already calls the resolver before
loaders: preserve that dispatch and add runtime witnesses for both entrypoints.

- [ ] **Step 5: Share the literal observation and replace adaptive history loop.**

Factor only the unchanged wrapper call and extraction out of legacy:

```python
def _observe_teacher(self, teacher, sample, sample_ids, sample_attention,
                     original_pixels, original_grid):
    outputs = teacher(input_ids=sample_ids, attention_mask=sample_attention,
                      pixel_values=original_pixels, image_grid_thw=original_grid,
                      output_hidden_states=True, return_dict=True)
    hidden_pre, hidden_last = self._hidden_states(
        outputs, (1, sample.unpadded_length), validate=self.validate)
    embeddings = self._language_input_embeddings(teacher, outputs, sample_ids)
    return hidden_pre, hidden_last, embeddings
```

Legacy uses this helper in the same position before helper visual extraction;
leave its remaining selector code and textual_states behavior unchanged.
New adaptive keeps existing Qwen/Stage1/DoRA/SDPA, single assistant, markers,
original/helper ownership and finite candidate checks. Add the current training
config gate before model work. Remove the old observer-specific full-prefix
sliding-window rejection: native mask behavior is the legacy wrapper's behavior.
No-latent samples retain validated empty output without encoding helper images.

Replace all adapter/prompt/prefix/observer operations with this code after the
existing validated `counts`, `text_spans`, and nonempty latent-span check:

```python
hidden_pre, hidden_last, embeddings = self._observe_teacher(
    teacher, sample, sample_ids, sample_attention, original_pixels, original_grid)
row_hidden_last = hidden_last[0]
user_query = self._user_query(
    teacher=teacher, sample=sample, sample_ids=row_ids,
    hidden_pre=hidden_pre[0], hidden_last=row_hidden_last,
    input_embeddings=embeddings[0], assistant_start=assistant_start,
    image_token_id=image_token_id, original_grid=original_grid,
    original_counts=counts)
helper_features = teacher.visual(helper_pixels, grid_thw=helper_grid)
if not bool(torch.isfinite(helper_features).all()):
    raise FloatingPointError('teacher helper visual features must be finite')
helper_images = self._split_visual_features(helper_features, helper_grid, teacher)
candidates = self._sample_helper_candidates(helper_images, sample)
if any(values.shape[0] == 0 for values in candidates):
    raise ValueError('adaptive history requires nonempty candidates for every block')
packed, block_counts, block_indices, block_margins = [], [], [], []
state = DiscountedTextState()
for index, span in enumerate(sample.latent_spans):
    text_values = row_hidden_last[list(text_spans[index])].float()
    current_mean = text_values.mean(0, keepdim=True) if text_values.shape[0] else None
    state, text_summary = update_text_history(state, current_mean,
                                               gamma=self.query_config.text_gamma)
    previous = torch.cat(packed, 0) if packed else row_hidden_last[:0]
    history = content_history(text_values, previous,
                              query_chunk_size=self.query_config.query_chunk_size,
                              key_chunk_size=self.query_config.key_chunk_size)
    query = compose_content_query(user_query, text_summary, history)
    selected, indices, margin = self._select(
        candidates[index], query, max_count=len(span.latent_positions))
    if (selected.ndim != 2 or not 0 < selected.shape[0] <= len(span.latent_positions)
            or selected.shape[1] != row_hidden_last.shape[-1]
            or not bool(torch.isfinite(selected).all())):
        raise ValueError('adaptive selection must contain finite raw first-Kstar targets')
    packed.append(selected)
    block_counts.append(int(selected.shape[0]))
    block_indices.append(indices)
    block_margins.append(margin)
return (packed, tuple(block_counts), tuple(block_indices), tuple(block_margins),
        row_hidden_last[:0])
```

Import `content_history`, `compose_content_query` from the new module; remove
unused old observer/adapter imports from targets only. Old utility files stay.
Do not pass helper pixels, target tensors, latent_hidden_states or image_out_mask
to the Teacher observation. Existing builder eval/no-grad scope stays unchanged.

- [ ] **Step 6: Add historical inspection/early rejection RED-to-GREEN witnesses.**

Use a literal historical native query, not the new serializer, in resume tests:

```python
OLD_NATIVE_QUERY = {
    'schema_version': 1, 'mode': 'adaptive',
    'formula_id': 'stage1_injected_prefix_discounted_segments_attention_mass_v1',
    'text_mode': 'discounted_segments', 'latent_mode': 'attention_mass',
    'text_gamma': .55, 'attention_layers': 'last4',
    'resolved_layer_indices': [0, 1, 2, 3],
    'query_chunk_size': 64, 'key_chunk_size': 2048, 'head_chunk_size': 4,
    'mask_mode': 'full_causal_2d_v1',
}


def test_historical_native_inspection_does_not_authorize_training(tmp_path):
    import json
    from stage1.query import validate_query_execution_config
    saved = contract()  # Existing outer token-mean contract fixture.
    saved['execution']['query_config'] = OLD_NATIVE_QUERY
    path = tmp_path / 'stage1_v5.json'
    path.write_text(json.dumps(saved))
    assert api().load_v5_contract(tmp_path) == saved
    assert validate_query_execution_config(OLD_NATIVE_QUERY).attention_layers == 'last4'
    with pytest.raises(ValueError, match='original reviewed code'):
        api().validate_v5_resume(saved, {})
    with pytest.raises(ValueError, match='fresh run'):
        api().resolve_v5_query_resume(saved, {})
```

Retain inspection of an existing historical sidecar found read-only under the
checkpoint/results namespace; record its path/hash, schema and validation result
without modifying it. A synthetic contract alone does not establish real artifact
compatibility. If an artifact predates outer token-mean DIL, report both identities.

Parametrize current query tampering over every semantic field in NEW_QUERY plus
unknown/missing keys and wrong scalar types. Preserve existing strict JSON-list
tests for old resolved indices, explicit null rejection, outer loss identity,
reference/source/schedule binding and resume coefficients. Current fixtures
use `last`; historical last4/all stay separate and inspectable. In parser and
composition tests, install model-loader spies that fail if called, request
last4/all or old native resume, assert the intended ValueError and zero calls.
Do not change a historical artifact to make a new resume test pass.

- [ ] **Step 7: Bind actual current source files and repair the exact verifier map.**

In `_build_v5_runtime_contract`, replace native attention layer resolution with
`query_execution_config(query_config, ())` and add the metric module to the
existing explicit source map. The resulting keys must be exactly:

```python
SOURCE_PATHS = {
    'v5_trainer': 'src/stage1/v5_trainer.py',
    'stage1_trainer': 'src/stage1/trainer.py',
    'transformers_trainer': 'transformers/src/transformers/trainer.py',
    'query': 'src/stage1/query.py',
    'targets': 'src/stage1/targets.py',
    'history_attention': 'src/stage1/history_attention.py',
    'history_teacher': 'src/stage1/history_teacher.py',
    'history_metric': 'src/stage1/history_metric.py',
}
```

In `scripts/verify_stage1_v5.py`, include each of those files in the audit
`code_identity.files` inventory, preserving its existing data/script entries.
Replace the stale three-key validator map with this explicit mapping and keep
exact equality, not a subset/ignore-extra comparison:

```python
expected_sources = {key: audited['code_identity']['files'][path]
                    for key, path in SOURCE_PATHS.items()}
if execution['source_sha256'] != expected_sources:
    raise ValueError('contract source binding differs from audited execution code')
```

Use the verifier's existing failure convention if its callers expect a different
exception/message; do not weaken the condition. Update protocol fixtures to
represent every actual source key and add a witness that removes or changes
`history_metric` and is rejected. Keep the legacy protocol profile selection and
all checkpoint/model/data/package equality checks. This is a local explicit-map
correction, not a new provenance framework or a real audit run.

- [ ] **Step 8: Replace old adaptive integration expectations with current witnesses.**

Keep actual tiny DoRA and parent trainer fixtures; use the following concrete
assertions when replacing native-prefix expectations:

```python
assert len(prefixes) == 1
assert len(visuals) == 2
assert prefixes[0]['inputs_embeds'].shape[1] == case.inputs['input_ids'].shape[1]
assert result.selected_counts == ((1, 1),)
assert result.latent_mask.nonzero().tolist() == [[0, 14], [0, 20]]
assert not result.latent_values.requires_grad
```

Add the following independent intervention to the real-builder test using a
subclass that overrides only the first selected raw row; observe its language
calls/hidden outputs with scoped hooks in the same test:

```python
class FixedFirstTarget(RecordingBuilder):
    def __init__(self, tokenizer, value):
        super().__init__(tokenizer, query_config=Stage1QueryConfig(mode='adaptive'),
                         coverage_p=.1, helper_group_L=0)
        self.value = value

    def _select(self, candidates, query, max_count):
        selected, indices, margin = super()._select(candidates, query, max_count)
        if len(self.queries) == 1:
            selected = self.value[None].to(selected)
        return selected, indices, margin
```

Use two nonzero directional rows at the fixture's width (e0/e1, magnitude4),
same case/weights/input. Assert identical full Teacher embeddings, mask/MRoPE,
hidden states and first query; second external metric/query changes. The pure
Task 1 fixture already proves actual selected indices can change under fixed
observation/candidates; do not demand random real tiny weights select differently.

Keep future-text/helper tests but compare only causally relevant hidden rows
and pre-current queries. Full full-sequence embeddings may legitimately differ
at a changed future text token; asserting identical whole inputs there is wrong.
Current helper features may change current candidates, but never Teacher states
or current query. Earlier selected targets may affect future queries externally.
Keep assistant/special exclusions, source ordering and first-K mask packing,
sample isolation, all input tensors unchanged, RNG unchanged, no persistent
hooks, native mask/window preservation and next-build state reset witnesses.

In `tests/stage1/v5/test_history_integration.py`, retain real parent dispatch,
shared detached target object identity, fixed-identical-target reference logits,
student gradients and DIL token/aux-block scales. The helper build now uses one
language wrapper pass instead of prompt-plus-prefix passes. Reference logits
with different selected targets are not an invariance claim. Current saved
adaptive config `.55,last` replaces old `.55,all`; the latter belongs only to
historical inspection/rejection tests. No loss/gradient normalization changes.

- [ ] **Step 9: Record current semantics in existing README and CLI wording.**

Replace the current adaptive paragraph with this content, preserving unrelated
training examples and historical design/results documents:

```text
Stage1-v5 defaults to --query_mode legacy. --query_mode adaptive observes the
target-selection Teacher once with the ILVR/legacy full input IDs, original
images and latent-pad token embeddings. It computes relative history weights
outside the Teacher: per-current-text-token cosine softmax over earlier raw
selected visual rows, then token-mean weights and a raw-value weighted sum.
The metric uses FP32 and fixed temperature 1.0. Available global intent,
gamma-discounted segment text summary (--history_text_gamma, default 0.9), and
history summary enter the query with equal coefficients. Empty current text
or history omits the history term. This is relative content relevance, not
native attention mass. --history_attention_layers accepts last for new adaptive
training; last4/all are historical only. New adaptive query schema2 cannot
resume a native injected-prefix adaptive run; use its original code or a fresh
run. Reference means DIL scoring and retains its existing input rules/weights.
Metric usefulness, training accuracy and production cost have not been measured.
```

Existing launcher pass-through of mode/gamma/last is sufficient. If CLI help
still advertises last4/all as current supported training, replace that string
with `Final hidden-state selector for adaptive training (last only)`; retain
historical read support in the model-independent validator. Add no temperature,
head or third-mode public option.

- [ ] **Step 10: Run one focused set and smallest actual-parent integration, then commit.**

```bash
./venv/ilvr-venv/bin/python -m pytest -q tests/stage1/test_history_metric.py tests/stage1/test_history_query.py tests/stage1/test_history_targets.py tests/stage1/test_cli.py tests/stage1/test_composition.py tests/stage1/v5/test_resume.py tests/stage1/v5/test_history_integration.py tests/stage1/v5/test_protocol.py tests/stage1/v5/test_workflow.py
git diff --check
```

Expected: all selected tests pass and diff check is clean. Record complete
`task2-focused.log` and exact counts/exit codes. Keep historical utility tests
and frozen throughput parity for the final suite; do not run repeated broad
cycles. If a failure requires a code fix, first add the smallest causal RED
test and then rerun the affected test plus the required focused set once the
task is ready. Do not relabel a changed expected output without explaining
why the new semantics require it.

Self-review exact Teacher calls, query/history ownership, contract/type equality,
early guards and unchanged loss paths. Stage only the mapped product/test/docs
files, then commit:

```bash
git commit -m "feat: use ILVR Teacher inputs with external adaptive history"
```

Report DONE/DONE_WITH_CONCERNS/NEEDS_CONTEXT/BLOCKED, actual commit, changed
files, RED/GREEN evidence, original-symptom witnesses and uncertainties. Obtain
independent spec PASS followed by quality PASS. Implementer fixes findings and
the relevant reviewers re-review before task completion. Root does not fix code.

## Final review, fresh verification and branch handoff

- [ ] Root dispatches a final independent whole-change reviewer after both task
reviews pass. Provide baseline `b39e0f2`, final product commit and the exact bound
spec. Review actual entrypoints/parent hooks and strict old/new contracts, not
comments/test names. No code edits or advisory hook spawning by reviewers.
- [ ] Route any finding through the selected SDD implementer/fixer, then obtain
the required re-review. Preserve unresolved issues honestly; tests alone are not
approval. Record IDs, verdicts, commits and exact evidence in the workflow ledger.
- [ ] Read/apply verification-before-completion and finishing-a-development-branch
before choosing the final command. The final branch gate applies when handing
off the completed feature. Its project full suite is selected as the shared
fresh test-pass witness, run once at the independently reviewed final revision:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.:src:transformers/src ./venv/ilvr-venv/bin/python -m pytest -q > logs/stage1-v5-content-metric-20261017/final-full-suite.log 2>&1
```

Expected: exit0 with all applicable tests passing; established skips may remain.
Read complete output and preserve counts/duration/warnings. A configured process
or test filename does not prove a runtime claim. Capture separate claim-specific
query/Teacher/resume witnesses once if the suite did not already run them.
- [ ] Run `git diff --check`, verify reviewed product commit identity, audit
changed-path scope and control hashes. Persist minimal approval/agent/review/
verification evidence; leave pre-existing caches, worktrees and checkpoints.
- [ ] Present the finishing skill's applicable human branch disposition choices.
Do not merge, publish, delete worktrees/branches or start training without the
applicable existing or subsequent authorization. Report implementation test
coverage separately from absent real-data relevance/GPU/cost/accuracy evidence.
- [ ] Before final response, run the required ntfy summary writer with the
actual success/partial outcome. Completion of this plan artifact alone is not
completion of implementation.

## Root plan self-review record

Spec coverage: Teacher call/input and one observation -> Task2.1/2/5/8;
math/no-grad/chunks/empty/invalid/selection boundary -> Task1.1-6;
gamma recurrence and ownership -> unchanged tested utility plus Task2.5/8;
new identity/historical inspection/early resume/CLI guard -> Task2.3/4/6;
actual source binding/current consumers -> Task2.7; reference/student/DIL
invariance -> Task2.8/10; current docs/launcher behavior -> Task2.9;
independent reviews/fresh verification/handoff -> per-task and final gates.
No real training or claimed semantic evidence is included.

Type/interface consistency: `ContentHistory(weights, summary, present)`,
`content_history(text, values, *, query_chunk_size, key_chunk_size)`,
`compose_content_query(intent, text_summary, history)` and the read-only versus
training query validators are named consistently. Legacy serializer signature
is retained; native layer resolution is historical only. Planned test snippets
reuse existing `RecordingBuilder`, `builder`, `observe_build`, `contract`, `api`
and tiny DoRA fixture definitions in their named files. Current and historical
fixtures are deliberately separate. Every changed behavior has an identified
RED and runtime/contract witness. No unassigned requirement remains.

Execution clarification within the approved FP32 requirement: an enclosing
CPU/CUDA autocast context can downcast matrix multiplication despite explicit
float inputs. Disable autocast locally around metric scoring/aggregation using
the validated input device type, and retain an enclosing CPU BF16 autocast
regression against independent FP32 math. The Task 1 implementer observed this
causal RED before the targeted fix. This is implementation detail, not a formula
or design-scope change; no accuracy claim follows.
