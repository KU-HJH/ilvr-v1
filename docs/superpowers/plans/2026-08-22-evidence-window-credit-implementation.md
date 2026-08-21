# Evidence–Window Credit Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add measurable, configurable evidence–window credit learning to ILVR while preserving the exact existing loss path when the feature is disabled.

**Architecture:** Keep `CustomTrainerStage1` and the modified Qwen2.5-VL forward path. Add a focused `src/evidence_window/` package whose pure tensor modules extract native layouts, create zero-latent counterfactuals, aggregate causal token scores, compute preference/relation/residual losses, and summarize metrics. A small engine orchestrates policy/reference/EMA conditions, while the trainer supplies ILVR-selected latents and reuses its existing full forward.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu128, CUDA 12.8, FlashAttention 2.8.3, vendored Transformers 4.50.0.dev0, TRL 0.15.2, PEFT 0.17.1, Accelerate/DeepSpeed, pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-evidence-window-credit-design.md`

## Global Constraints

- Use `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python` for Python, pytest, training, and evaluation.
- Preserve `CustomTrainerStage1`, the EMA teacher, modified Qwen2.5-VL latent injection, multimodal positions, gradient checkpointing, and BF16 behavior.
- Never use the EMA teacher as the fixed DPO-style reference.
- Resolve latent and special-token IDs from the tokenizer/model configuration; do not add hard-coded Qwen token IDs.
- The first intervention is `zero_latent`. It must not delete tokens or alter IDs, sequence length, attention masks, positions, unrelated visual spans, or text.
- One selected evidence span requires at most one counterfactual full-sequence forward per model role, never one forward per evidence–window pair.
- EW-DPO training uses adjacent evidence–window pairs. Diagnostics and relation KD retain all temporally valid predecessor pairs.
- All new coefficients default to zero. With coefficients zero and diagnostics disabled, construct no reference/layout, run no auxiliary forward, and return the original ILVR path within numerical tolerance.
- Include both adapter-disabled LoRA reference and explicit frozen-copy reference for ZeRO-2 in the first implementation.
- Include logit-residual KD. Hidden-state residual KD remains outside this plan pending its profiling and approval gate.
- Do not claim sparse dependence is always bad, every window needs every evidence, or fixed-history LOO is a total causal effect.
- Before a validation expected to read more than 10 GB or exceed five minutes, reuse equivalent evidence or obtain explicit approval. Never repeat an already successful expensive immutable-input validation.

## File Structure

- Create `src/evidence_window/__init__.py`: stable public exports.
- Create `src/evidence_window/config.py`: strict versioned JSON schema and required-condition derivation.
- Create `src/evidence_window/layout.py`: native evidence/window extraction and padded batch masks.
- Create `src/evidence_window/interventions.py`: deterministic LOO selection and compact zero-latent replacement.
- Create `src/evidence_window/scores.py`: causal log-probabilities and masked aggregation.
- Create `src/evidence_window/losses.py`: preference, relation, and chunked logit-residual losses.
- Create `src/evidence_window/metrics.py`: sufficient statistics, distributed reduction, and rank-local matrix dumps.
- Create `src/evidence_window/reference.py`: adapter-disabled and frozen-copy providers.
- Create `src/evidence_window/engine.py`: condition cache, forward orchestration, and objective composition.
- Modify `src/utils_deepseed.py`: `--ew_config` and `--deepspeed_config`.
- Modify `src/main.py`: configuration/token-ID wiring and selectable DeepSpeed JSON.
- Modify `src/trainer.py`: preserve original loss and add the auxiliary engine.
- Create `configs/evidence_window/*.json`: source-edit-free research variants.
- Create `tests/evidence_window/`: CPU/synthetic unit and integration suite.
- Create `tests/evidence_window/conftest.py`: shared tiny latent model, deterministic ILVR inputs, trainer factory, and parameter snapshots used by integration/smoke tests.

The spec's components form one dependency chain—layout → scoring → intervention/reference → engine → trainer—rather than independent products, so this is one sequential plan. Every task ends in an independently reviewable commit.

---

### Task 1: Strict Configuration, CLI, and Disabled Contract

**Files:**
- Create: `src/evidence_window/__init__.py`
- Create: `src/evidence_window/config.py`
- Create: `tests/evidence_window/test_config.py`
- Modify: `src/utils_deepseed.py:15-63`
- Modify: `src/main.py:98-226`

**Interfaces:**
- Consumes: optional JSON path from `--ew_config`.
- Produces: `EvidenceWindowConfig.from_json(path)`, `EvidenceWindowConfig.disabled()`, `is_disabled`, `required_conditions(role)`, `args.ew_config`, and `args.deepspeed_config`.

- [ ] **Step 1: Write failing strict-schema tests**

```python
# tests/evidence_window/test_config.py
import json
import pytest
from evidence_window.config import EvidenceWindowConfig

def test_disabled_requests_nothing():
    cfg = EvidenceWindowConfig.disabled()
    assert cfg.is_disabled
    assert cfg.required_conditions("policy") == frozenset()
    assert cfg.required_conditions("reference") == frozenset()

def test_unknown_key_fails(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "unknown": True}))
    with pytest.raises(ValueError, match="unknown"):
        EvidenceWindowConfig.from_json(path)

def test_preference_requires_reference(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({
        "version": 1,
        "reference": {"mode": "none"},
        "preference": {"lambda_ew": 1.0},
    }))
    with pytest.raises(ValueError, match="reference"):
        EvidenceWindowConfig.from_json(path)

def test_relation_only_needs_policy_and_ema_loo(tmp_path):
    path = tmp_path / "relation.json"
    path.write_text(json.dumps({"version": 1, "relation": {"lambda_jsd": 1.0}}))
    cfg = EvidenceWindowConfig.from_json(path)
    assert cfg.required_conditions("policy") == frozenset({"full", "loo"})
    assert cfg.required_conditions("ema") == frozenset({"full", "loo"})
    assert cfg.required_conditions("reference") == frozenset()
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_config.py -q
```

Expected: collection fails because `evidence_window.config` does not exist.

- [ ] **Step 3: Implement exact nested dataclasses**

```python
# src/evidence_window/config.py
from dataclasses import dataclass, field, fields
from pathlib import Path
import json
from typing import Literal

@dataclass(frozen=True)
class ReferenceConfig:
    mode: Literal["none", "adapter_disabled", "frozen_copy"] = "none"

@dataclass(frozen=True)
class PreferenceConfig:
    beta: float = 0.1
    lambda_global: float = 0.0
    lambda_window: float = 0.0
    lambda_ew: float = 0.0

@dataclass(frozen=True)
class RelationConfig:
    lambda_jsd: float = 0.0
    lambda_magnitude: float = 0.0
    temperature: float = 1.0
    huber_delta: float = 1.0

@dataclass(frozen=True)
class ResidualConfig:
    lambda_logit: float = 0.0
    metric: Literal["huber", "cosine"] = "huber"
    huber_delta: float = 1.0
    vocab_chunk_size: int = 4096
    max_tokens_per_window: int = 0

@dataclass(frozen=True)
class LOOConfig:
    intervention: Literal["zero_latent"] = "zero_latent"
    max_spans_per_example: int = 0

@dataclass(frozen=True)
class DiagnosticsConfig:
    global_saturation: bool = False
    contribution_threshold: float = 0.0
    matrix_dump_interval: int = 0
    matrix_dump_dir: str = "evidence_window_matrices"

@dataclass(frozen=True)
class EvidenceWindowConfig:
    version: int = 1
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    relation: RelationConfig = field(default_factory=RelationConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    loo: LOOConfig = field(default_factory=LOOConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
```

Implement `from_json` by rejecting unknown top-level and nested keys before constructing these dataclasses. `validate` rejects version other than 1, negative coefficients/caps, nonpositive beta/temperature/chunk size, unsupported interventions, and preference/reference-adjusted diagnostics without a fixed reference. `is_disabled` is true only when all new coefficients, diagnostic flags, and dump intervals are zero/off. `required_conditions` returns the minimal set:
- global/window: policy+reference `full/all_null`;
- EW: policy+reference `full/loo`;
- relation/residual: policy+EMA `full/loo`;
- disabled: empty.

- [ ] **Step 4: Add CLI and main wiring**

```python
# src/utils_deepseed.py, in get_args()
parser.add_argument("--ew_config", type=str, default=None)
parser.add_argument("--deepspeed_config", type=str, default="configs/config_stage3.json")
```

```python
# src/main.py
ew_config = (
    EvidenceWindowConfig.disabled()
    if args.ew_config is None
    else EvidenceWindowConfig.from_json(args.ew_config)
)

# SFTConfig
deepspeed=args.deepspeed_config,

# trainer constructor
evidence_window_config=ew_config,
latent_token_ids={
    "start": int(latent_start_idx),
    "end": int(latent_end_idx),
    "pad": int(latent_token_idx),
    "padding": int(pad_token_idx),
    "special": tuple(int(x) for x in processor.tokenizer.all_special_ids),
},
```

Do not initialize the engine or reference in `main.py`; runtime model ownership belongs to the trainer.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_config.py -q
git add src/evidence_window src/utils_deepseed.py src/main.py tests/evidence_window/test_config.py
git commit -m "feat: add evidence-window configuration contract"
```

Expected: all configuration tests pass.

---

### Task 2: ILVR-Native Evidence and Reasoning Windows

**Files:**
- Create: `src/evidence_window/layout.py`
- Create: `tests/evidence_window/test_layout.py`
- Modify: `src/evidence_window/__init__.py`

**Interfaces:**
- Consumes: `input_ids [B,T]`, `attention_mask [B,T]`, `labels [B,T]`, post-selection `active_latent_mask [B,T]`, and `TokenIds`.
- Produces: `extract_interleaved_layout(input_ids, attention_mask, labels, active_latent_mask, token_ids) -> InterleavedLayout` with evidence/window masks, validity, `causal_pair_mask`, `training_pair_mask`, and `native_predecessor`.

- [ ] **Step 1: Write exact synthetic layout tests**

```python
# tests/evidence_window/test_layout.py
import torch
import pytest
from evidence_window.layout import TokenIds, extract_interleaved_layout

TOKENS = TokenIds(90, 91, 92, 0, frozenset({0, 90, 91, 99}))

def test_extracts_r0_v1_r1_v2_r2_v3_r3():
    ids = torch.tensor([[10,90,92,92,91,20,21,90,92,91,30,90,92,91,40,0]])
    labels = ids.clone()
    labels[:, -1] = -100
    active = torch.tensor([[0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0]], dtype=torch.bool)
    layout = extract_interleaved_layout(ids, ids.ne(0), labels, active, TOKENS)
    assert layout.evidence_token_mask[0].nonzero().tolist() == [[0,2],[0,3],[1,8],[2,12]]
    assert layout.window_token_mask[0].nonzero().tolist() == [[0,0],[1,5],[1,6],[2,10],[3,14]]
    assert layout.native_predecessor[0].tolist() == [-1,0,1,2]
    assert layout.training_pair_mask[0].nonzero().tolist() == [[0,1],[1,2],[2,3]]

def test_padding_ignored_and_special_tokens_are_excluded():
    ids = torch.tensor([[90,92,91,20,99,21,0]])
    labels = torch.tensor([[-100,-100,-100,20,99,-100,-100]])
    layout = extract_interleaved_layout(ids, ids.ne(0), labels, ids.eq(92), TOKENS)
    assert layout.window_token_mask[0,0].nonzero().flatten().tolist() == [3]

def test_adjacent_spans_create_no_empty_window():
    ids = torch.tensor([[90,92,91,90,92,91,7]])
    layout = extract_interleaved_layout(ids, torch.ones_like(ids), ids, ids.eq(92), TOKENS)
    assert layout.window_valid.sum().item() == 1
    assert layout.native_predecessor[0,0].item() == 1

def test_unmatched_delimiter_fails():
    ids = torch.tensor([[90,92,7]])
    with pytest.raises(ValueError, match="unmatched"):
        extract_interleaved_layout(ids, torch.ones_like(ids), ids, ids.eq(92), TOKENS)
```

Add named tests for first/last span, inactive span, `R_0` without predecessor, padding, and a two-example batch with different evidence/window counts.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_layout.py -q
```

Expected: import failure for `evidence_window.layout`.

- [ ] **Step 3: Implement the immutable layout records**

```python
# src/evidence_window/layout.py
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class TokenIds:
    latent_start: int
    latent_end: int
    latent_pad: int
    padding: int
    special: frozenset[int]

@dataclass(frozen=True)
class InterleavedLayout:
    evidence_token_mask: torch.Tensor
    window_token_mask: torch.Tensor
    evidence_valid: torch.Tensor
    window_valid: torch.Tensor
    causal_pair_mask: torch.Tensor
    training_pair_mask: torch.Tensor
    native_predecessor: torch.Tensor
```

`extract_interleaved_layout` scans every row once. It rejects nested/unmatched delimiters, records delimiter interiors, intersects interiors with `active_latent_mask`, and retains inactive spans in structural parsing while marking them invalid for pairs. A reasoning token requires attention=1, label!=-100, exclusion from every latent range, and exclusion from `TokenIds.special`. Omit empty inter-latent windows. Pack row lists to batch maxima with false masks and predecessor -1. Compute temporal validity from evidence-end < window-start, and adjacent training pairs from each window's stored native predecessor.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_layout.py -q
git add src/evidence_window/layout.py src/evidence_window/__init__.py tests/evidence_window/test_layout.py
git commit -m "feat: extract ILVR evidence-window layouts"
```

Expected: all window-extraction and heterogeneous-batch cases pass.

---

### Task 3: Causal Scores, Preference Losses, and Saturation Diagnostics

**Files:**
- Create: `src/evidence_window/scores.py`
- Create: `src/evidence_window/losses.py`
- Create: `tests/evidence_window/test_scores.py`
- Create: `tests/evidence_window/test_preference_losses.py`
- Modify: `src/evidence_window/__init__.py`

**Interfaces:**
- Consumes: logits, labels, token/window masks, full/counterfactual policy/reference scores, beta, and pair masks.
- Produces: `causal_token_logps`, `masked_token_sums`, `reference_adjusted_effect`, `evidence_window_margins`, and three preference losses.

- [ ] **Step 1: Write causal alignment and exact masking tests**

```python
# tests/evidence_window/test_scores.py
import torch
from evidence_window.scores import causal_token_logps, masked_token_sums

def test_t_minus_one_predicts_t():
    logits = torch.tensor([[[0.,4.],[4.,0.],[0.,4.]]])
    labels = torch.tensor([[-100,1,0]])
    scores = causal_token_logps(logits, labels)
    logp = logits.log_softmax(-1)
    assert torch.allclose(scores[0,1], logp[0,0,1])
    assert torch.allclose(scores[0,2], logp[0,1,0])
    assert scores[0,0] == 0

def test_window_sum_excludes_every_outside_token():
    scores = torch.tensor([[1.,2.,4.,8.,16.]])
    masks = torch.tensor([[[0,1,1,0,0],[0,0,0,1,0]]], dtype=torch.bool)
    assert masked_token_sums(scores, masks).tolist() == [[6.,8.]]
```

- [ ] **Step 2: Write prefix identity and research-motivation gradient test**

```python
# tests/evidence_window/test_preference_losses.py
import torch
from evidence_window.losses import global_preference_loss, window_preference_loss
from evidence_window.scores import evidence_window_margins

def test_prefix_cancellation_equals_direct_window_sum():
    delta = torch.tensor([[[0.,3.,5.,7.,11.]]])
    masks = torch.tensor([[[[0,1,1,0,0],[0,0,0,1,1]]]], dtype=torch.bool)
    direct = evidence_window_margins(delta, masks)
    cumulative = delta.cumsum(-1)
    assert direct[0,0,0] == cumulative[0,0,2] - cumulative[0,0,0]
    assert direct[0,0,1] == cumulative[0,0,4] - cumulative[0,0,2]

def test_global_saturation_suppresses_m2_but_window_loss_retains_signal():
    margins = torch.tensor([[20.,0.]], requires_grad=True)
    valid = torch.ones_like(margins, dtype=torch.bool)
    global_preference_loss(margins, valid, beta=1.).backward(retain_graph=True)
    global_grad = margins.grad[0,1].abs().item()
    margins.grad.zero_()
    window_preference_loss(margins, valid, beta=1.).backward()
    local_grad = margins.grad[0,1].abs().item()
    assert global_grad < 1e-7
    assert local_grad > 0.2
```

- [ ] **Step 3: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_scores.py tests/evidence_window/test_preference_losses.py -q
```

Expected: score/loss imports fail.

- [ ] **Step 4: Implement stable tensor arithmetic**

```python
# src/evidence_window/scores.py
def causal_token_logps(logits, labels):
    safe = labels[:,1:].clamp_min(0)
    selected = logits[:,:-1].log_softmax(-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    selected = selected * labels[:,1:].ne(-100)
    return torch.cat([logits.new_zeros((logits.shape[0],1)), selected], dim=1)

def masked_token_sums(scores, masks):
    return torch.einsum("bt,bnt->bn", scores, masks.to(scores.dtype))

def reference_adjusted_effect(policy_full, policy_cf, ref_full, ref_cf):
    return (policy_full-policy_cf) - (ref_full-ref_cf)

def evidence_window_margins(delta_by_evidence, masks):
    return torch.einsum("bjt,bjkt->bjk", delta_by_evidence, masks.to(delta_by_evidence.dtype))
```

```python
# src/evidence_window/losses.py
def global_preference_loss(margins, valid, beta):
    return -F.logsigmoid(beta * (margins * valid).sum(-1)).mean()

def window_preference_loss(margins, valid, beta):
    values = -F.logsigmoid(beta * margins)
    return (values * valid).sum() / valid.sum().clamp_min(1)

def evidence_window_preference_loss(margins, pair_mask, beta, weights=None):
    weights = pair_mask.to(margins.dtype) if weights is None else weights * pair_mask
    return (-F.logsigmoid(beta*margins) * weights).sum() / weights.sum().clamp_min(1)
```

Add shape validation and differentiable zero for empty valid sets.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_scores.py tests/evidence_window/test_preference_losses.py -q
git add src/evidence_window/scores.py src/evidence_window/losses.py src/evidence_window/__init__.py tests/evidence_window/test_scores.py tests/evidence_window/test_preference_losses.py
git commit -m "feat: add evidence-window preference primitives"
```

Expected: exact masking, prefix cancellation, and saturation tests pass.

---

### Task 4: Contribution Metrics, Distributed Reduction, and Matrix Dumps

**Files:**
- Create: `src/evidence_window/metrics.py`
- Create: `tests/evidence_window/test_metrics.py`
- Create: `tests/evidence_window/test_distributed_metrics.py`
- Modify: `src/evidence_window/__init__.py`

**Interfaces:**
- Consumes: relation matrix `[B,J,K]`, pair mask, threshold, beta/margins.
- Produces: `MetricBatch`, `contribution_summaries`, `reduce_sufficient_statistics`, and `MatrixJsonlWriter`.

- [ ] **Step 1: Write metric and privacy tests**

```python
# tests/evidence_window/test_metrics.py
def test_positive_mass_summaries():
    c = torch.tensor([[[3.,0.],[1.,-2.]]])
    pairs = torch.ones_like(c, dtype=torch.bool)
    out = contribution_summaries(c, pairs, threshold=.5)
    assert torch.allclose(out.top1_ratio[0,0], torch.tensor(.75))
    assert out.above_threshold_count[0,0] == 2
    assert out.entropy_valid[0,0]
    assert not out.entropy_valid[0,1]

def test_rank_local_jsonl_contains_no_sample_payload(tmp_path):
    writer = MatrixJsonlWriter(tmp_path, rank=2)
    writer.write(7, 3, torch.tensor([[1.]]), torch.tensor([[True]]))
    record = json.loads((tmp_path/"rank-00002.jsonl").read_text())
    assert record["observation_id"] == "r2-s7-o3"
    assert set(record) == {"observation_id","rank","step","relation","pair_mask"}
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_metrics.py -q
```

Expected: metrics import fails.

- [ ] **Step 3: Implement positive-mass descriptive summaries**

```python
positive = relations.clamp_min(0) * pair_mask
mass = positive.sum(dim=1)
probability = positive / mass.unsqueeze(1).clamp_min(torch.finfo(relations.dtype).eps)
candidate_count = pair_mask.sum(dim=1)
top1 = probability.max(dim=1).values
entropy = -(probability.clamp_min(torch.finfo(relations.dtype).eps).log()*probability).sum(dim=1)
entropy_valid = mass.gt(0) & candidate_count.gt(1)
normalized_entropy = entropy / candidate_count.clamp_min(2).log()
above_threshold_count = ((relations > threshold) & pair_mask).sum(dim=1)
```

Every logged mean is represented as numerator+count. Reduce both with `torch.distributed.all_reduce(SUM)` only when distributed is initialized, then divide. Compute the exact outer coefficient `beta * sigmoid(-beta*M_global)`. The writer accepts only numeric tensors plus rank/step/counter and writes one JSONL file per rank.

- [ ] **Step 4: Test two-process gloo reduction**

Use `torch.multiprocessing.spawn` with a file init method. Rank 0 contributes numerator 2/count 1; rank 1 contributes 4/1; both assert reduced mean 3.

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_metrics.py tests/evidence_window/test_distributed_metrics.py -q
```

Expected: scalar, entropy-edge, JSONL-privacy, and distributed tests pass.

- [ ] **Step 5: Commit Milestone 1 diagnostics**

```bash
git add src/evidence_window/metrics.py src/evidence_window/__init__.py tests/evidence_window/test_metrics.py tests/evidence_window/test_distributed_metrics.py
git commit -m "feat: add evidence-window diagnostics"
```

---

### Task 5: Zero-Latent LOO and Deterministic Span Cap

**Files:**
- Create: `src/evidence_window/interventions.py`
- Create: `tests/evidence_window/test_interventions.py`
- Modify: `src/evidence_window/__init__.py`

**Interfaces:**
- Consumes: compact latents, active mask, evidence masks, evidence ordinal, and cap.
- Produces: `apply_zero_latent_loo`, `apply_all_null`, and `select_loo_spans`.

- [ ] **Step 1: Write locality and cap tests**

```python
# tests/evidence_window/test_interventions.py
def test_loo_zeros_only_selected_compact_rows():
    values = torch.tensor([[[1.,1.],[2.,2.],[3.,3.]]])
    active = torch.tensor([[0,1,1,0,1]], dtype=torch.bool)
    evidence = torch.tensor([[[0,1,1,0,0],[0,0,0,0,1]]], dtype=torch.bool)
    result = apply_zero_latent_loo(values, active, evidence, 0)
    assert result.shape == values.shape
    assert result.tolist() == [[[0.,0.],[0.,0.],[3.,3.]]]
    assert values.tolist() == [[[1.,1.],[2.,2.],[3.,3.]]]

def test_all_null_zeros_all_compact_rows():
    assert torch.count_nonzero(apply_all_null(torch.ones(1,3,2))) == 0

def test_cap_is_deterministic_and_includes_endpoints():
    valid = torch.tensor([[True,True,True,True,True]])
    assert select_loo_spans(valid, 3) == [[0,2,4]]
    assert select_loo_spans(valid, 0) == [[0,1,2,3,4]]
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_interventions.py -q
```

Expected: intervention import fails.

- [ ] **Step 3: Implement row-major compact mapping**

```python
def apply_zero_latent_loo(values, active_mask, evidence_masks, evidence_index):
    source = values.reshape(-1, values.shape[-1])
    active_positions = active_mask.reshape(-1).nonzero().flatten()
    if source.shape[0] != active_positions.numel():
        raise ValueError("compact latent order does not match image_out_mask")
    selected_token_rows = evidence_masks[:, evidence_index].reshape(-1)
    zero_rows = selected_token_rows.index_select(0, active_positions)
    result = source.clone()
    result[zero_rows] = 0
    return result.reshape_as(values)

def apply_all_null(values):
    return torch.zeros_like(values)
```

Cap selection uses evenly spaced integer positions in sorted valid indices, deduplicates rounding collisions, and fills remaining temporal positions until exactly `min(cap, valid_count)` are selected. Add a fixture asserting unchanged `input_ids`, `attention_mask`, `position_ids`, length, and unrelated latent rows.

- [ ] **Step 4: Run GREEN and commit**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_interventions.py -q
git add src/evidence_window/interventions.py src/evidence_window/__init__.py tests/evidence_window/test_interventions.py
git commit -m "feat: add zero-latent leave-one-out"
```

Expected: all locality and invariance assertions pass.

---

### Task 6: Fixed References, Including ZeRO-2 Frozen Copy

**Files:**
- Create: `src/evidence_window/reference.py`
- Create: `tests/evidence_window/test_reference.py`
- Create: `tests/evidence_window/test_zero2_reference.py`
- Create: `tests/evidence_window/zero2_reference_worker.py`
- Modify: `src/evidence_window/__init__.py`

**Interfaces:**
- Consumes: SFTTrainer-initialized policy after PEFT application but before optimizer steps, accelerator, and reference config.
- Produces: `AdapterDisabledReference.forward(wrapped_policy, inputs)`, `FrozenCopyReference.create(policy, accelerator)`, `ReferenceProvider.parameters()`, invariance witnesses, and `build_reference_provider`.

- [ ] **Step 1: Write provider invariance tests**

```python
# tests/evidence_window/test_reference.py
def test_adapter_disabled_reference_ignores_adapter_update(adapter_toy):
    provider = AdapterDisabledReference(adapter_toy)
    before = provider.forward(adapter_toy, {"x": torch.ones(1,2)})
    with torch.no_grad():
        adapter_toy.adapter.weight.add_(1)
    after = provider.forward(adapter_toy, {"x": torch.ones(1,2)})
    assert torch.equal(before, after)
    assert adapter_toy.adapter_enabled

def test_frozen_copy_is_no_grad_and_unchanged_after_policy_step():
    policy = torch.nn.Linear(2,2,bias=False)
    provider = FrozenCopyReference.create(policy, Accelerator(cpu=True))
    snapshot = [p.clone() for p in provider.model.parameters()]
    loss = policy(torch.ones(1,2)).sum()
    loss.backward()
    torch.optim.SGD(policy.parameters(), lr=1).step()
    assert all(not p.requires_grad for p in provider.model.parameters())
    assert all(torch.equal(a,b) for a,b in zip(snapshot, provider.model.parameters()))
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_reference.py -q
```

Expected: reference import fails.

- [ ] **Step 3: Implement wrapper-safe adapter-disabled reference**

```python
class AdapterDisabledReference:
    def __init__(self, unwrapped_policy):
        self.peft_model = locate_disable_adapter_owner(unwrapped_policy)
        validate_only_adapter_parameters_trainable(self.peft_model)
        self.base_witness = parameter_witness(self.peft_model, adapters=False)

    def forward(self, wrapped_policy, inputs):
        with torch.no_grad(), self.peft_model.disable_adapter():
            return wrapped_policy(**inputs)

    def parameters(self):
        return self.peft_model.parameters()

    def named_parameters(self):
        return self.peft_model.named_parameters()
```

The locator walks `.module` and `.model` with cycle detection. The forward uses the active wrapped policy, not a direct unwrapped ZeRO-3 call. Reject full-parameter tuning and any trainable non-adapter parameter. Restore adapter state through the context manager even on exceptions.

- [ ] **Step 4: Implement replicated frozen-copy reference**

```python
class FrozenCopyReference:
    @classmethod
    def create(cls, initialized_policy, accelerator):
        model = copy.deepcopy(initialized_policy)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model = accelerator.prepare_model(model, evaluation_mode=True)
        return cls(model, parameter_witness(model, adapters=True))

    def forward(self, wrapped_policy, inputs):
        del wrapped_policy
        with torch.no_grad():
            return self.model(**inputs)

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()
```

Create this immediately after `SFTTrainer.__init__` returns: PEFT has been applied and `self.accelerator` exists, but the optimizer has not stepped. Do not attach it beneath `self.model` or place its parameters in optimizer groups/checkpoints. Under ZeRO-2 it is intentionally replicated per rank.

- [ ] **Step 5: Add two-rank tiny ZeRO-2 smoke**

`test_zero2_reference.py` launches `torch.distributed.run --standalone --nproc_per_node=2 tests/evidence_window/zero2_reference_worker.py`. The worker uses a tiny model and temporary ZeRO-2 config, then asserts on both ranks: reference parameters are local, require no gradients, are absent from optimizer groups, and equal their pre-step snapshot.

- [ ] **Step 6: Run GREEN and commit**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_reference.py tests/evidence_window/test_zero2_reference.py -q
git add src/evidence_window/reference.py src/evidence_window/__init__.py tests/evidence_window/test_reference.py tests/evidence_window/test_zero2_reference.py tests/evidence_window/zero2_reference_worker.py
git commit -m "feat: add fixed references for LoRA and ZeRO-2"
```

Expected: adapter/frozen invariance and two-process ZeRO-2 tests pass.

---

### Task 7: Condition Engine and Prefix-Cancelled EW-DPO

**Files:**
- Create: `src/evidence_window/engine.py`
- Create: `tests/evidence_window/test_engine_preference.py`
- Modify: `src/evidence_window/__init__.py`

**Interfaces:**
- Consumes: config, layout, compact latents, reused full-policy output, and a role/condition forward callback.
- Produces: `Condition`, `ModelConditionOutput`, `EvidenceWindowResult`, and `EvidenceWindowEngine.compute`.

- [ ] **Step 1: Write call-count, pair-policy, and exact-margin tests**

```python
def test_one_loo_forward_scores_all_downstream_windows(layout_fixture, ew_config):
    calls = []
    def forward(role, condition, values, need_hidden):
        calls.append((role, condition.kind, condition.evidence_index))
        return synthetic_output(role, condition, layout_fixture)
    result = EvidenceWindowEngine(ew_config).compute(
        layout_fixture, torch.ones(1,3,4), synthetic_full_output(), forward
    )
    assert calls.count(("policy","loo",0)) == 1
    assert calls.count(("policy","loo",1)) == 1
    assert result.ew_margins.shape[-2:] == layout_fixture.training_pair_mask.shape[-2:]
    assert torch.equal(result.preference_pair_mask, layout_fixture.training_pair_mask)
    assert torch.equal(result.relation_pair_mask, layout_fixture.causal_pair_mask)

def test_margin_matches_boxed_reference_adjusted_formula():
    result = run_scalar_engine(
        policy_full=10, policy_loo=7, reference_full=8, reference_loo=6
    )
    assert result.ew_margins.item() == 1
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_engine_preference.py -q
```

Expected: engine import fails.

- [ ] **Step 3: Implement records and exact condition cache**

```python
@dataclass(frozen=True)
class Condition:
    kind: Literal["full","all_null","loo"]
    evidence_index: int | None = None

@dataclass
class ModelConditionOutput:
    token_logps: torch.Tensor
    hidden_states: torch.Tensor | None = None

@dataclass
class EvidenceWindowResult:
    total_loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    metric_sums: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ew_margins: torch.Tensor | None
    relations: dict[str, torch.Tensor]
    preference_pair_mask: torch.Tensor
    relation_pair_mask: torch.Tensor
```

Cache keys are exactly `(role, condition.kind, condition.evidence_index)`. Seed policy/full from the trainer's existing output. Derive minimal conditions from config. Reduce logits to token scores immediately and retain hidden states only for logit residual. Compute all-null global/window objectives and LOO reference-adjusted `m_jk` with direct window masks. Use adjacent pairs for EW loss and all causal pairs for diagnostics. Raise on non-finite enabled results with numeric layout counts only.

- [ ] **Step 4: Add policy/reference gradient assertions and run GREEN**

Add tests proving EW policy scores receive gradients, reference outputs are detached, earlier-window token changes do not alter a later window's direct masked margin, and empty pair sets yield finite differentiable zero.

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_engine_preference.py -q
git add src/evidence_window/engine.py src/evidence_window/__init__.py tests/evidence_window/test_engine_preference.py
git commit -m "feat: orchestrate evidence-window preference"
```

Expected: O(1+J), exact margin, prefix isolation, and gradient tests pass.

---

### Task 8: Teacher–Student Evidence–Window Relation KD

**Files:**
- Modify: `src/evidence_window/losses.py`
- Modify: `src/evidence_window/engine.py`
- Create: `tests/evidence_window/test_relation_kd.py`

**Interfaces:**
- Consumes: policy/EMA full and LOO window sums, lengths, all-causal pair mask.
- Produces: `relation_matrix`, `masked_relation_jsd`, `masked_relation_huber`, and student/teacher `C` matrices.

- [ ] **Step 1: Write signed relation and mask tests**

```python
def test_relation_is_length_normalized():
    full = torch.tensor([[6.,9.]])
    loo = torch.tensor([[[2.,5.],[4.,3.]]])
    lengths = torch.tensor([[2.,3.]])
    pairs = torch.tensor([[[1,1],[0,1]]], dtype=torch.bool)
    actual = relation_matrix(full, loo, lengths, pairs)
    assert torch.allclose(actual, torch.tensor([[[2.,4/3],[0.,2.]]]))

def test_jsd_excludes_single_candidate_window():
    teacher = torch.tensor([[[2.,1.],[0.,3.]]])
    student = torch.tensor([[[1.,1.],[9.,2.]]], requires_grad=True)
    pairs = torch.tensor([[[1,1],[0,1]]], dtype=torch.bool)
    loss, count = masked_relation_jsd(student, teacher, pairs, 1.)
    assert count == 1
    loss.backward()
    assert student.grad[0,1,0] == 0

def test_huber_retains_negative_relations():
    student = torch.tensor([[[-1.,2.]]], requires_grad=True)
    teacher = torch.tensor([[[1.,-2.]]])
    loss, count = masked_relation_huber(student, teacher, torch.ones_like(student,dtype=torch.bool), 1.)
    assert count == 2
    loss.backward()
    assert student.grad.abs().sum() > 0
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_relation_kd.py -q
```

Expected: relation functions are missing.

- [ ] **Step 3: Implement relation and separate structure/magnitude losses**

```python
def relation_matrix(full_window_sums, loo_window_sums, lengths, pair_mask):
    values = (full_window_sums.unsqueeze(1)-loo_window_sums) / lengths.unsqueeze(1).clamp_min(1)
    return values.masked_fill(~pair_mask, 0)

def masked_relation_huber(student, teacher, mask, delta):
    values = F.huber_loss(student, teacher.detach(), delta=delta, reduction="none")
    return masked_mean_and_count(values, mask)
```

For JSD, iterate windows with at least two valid evidence candidates, mask both student and teacher to identical indices, compute softmax(C/temperature), then `0.5*(KL(q_s||m)+KL(q_t||m))`. Detach EMA targets. Do not clamp signs and do not add entropy maximization.

- [ ] **Step 4: Wire all-causal cached policy/EMA conditions**

When relation or magnitude is active, request policy and EMA full plus identical selected LOO indices. One LOO forward scores every downstream window. Store separate student/teacher matrices and metrics. Assert EMA has no gradients.

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_relation_kd.py tests/evidence_window/test_engine_preference.py -q
git add src/evidence_window/losses.py src/evidence_window/engine.py tests/evidence_window/test_relation_kd.py
git commit -m "feat: distill evidence-window relations"
```

Expected: signed magnitude, masked JSD, all-causal coverage, EMA detachment, and condition reuse pass.

---

### Task 9: Chunked Logit-Residual KD

**Files:**
- Modify: `src/evidence_window/losses.py`
- Modify: `src/evidence_window/engine.py`
- Create: `tests/evidence_window/test_logit_residual.py`

**Interfaces:**
- Consumes: final normalized full/LOO hidden states, student/EMA LM-head weights, adjacent pair mask, token cap, chunk size.
- Produces: `select_residual_tokens` and `chunked_logit_residual_loss`.

- [ ] **Step 1: Write exact small-vocabulary equivalence tests**

```python
def test_chunked_huber_matches_full_projection():
    student = torch.tensor([[1.,2.],[0.,1.]], requires_grad=True)
    teacher = torch.tensor([[0.,1.],[1.,1.]])
    weight = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    expected = F.huber_loss(student@weight.T, teacher@weight.T)
    actual = chunked_logit_residual_loss(student, teacher, weight, weight, "huber", 2, 1.)
    assert torch.allclose(actual, expected)

def test_chunked_cosine_matches_full_projection():
    student = torch.tensor([[1.,2.]])
    teacher = torch.tensor([[2.,1.]])
    weight = torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    expected = 1-F.cosine_similarity(student@weight.T, teacher@weight.T).mean()
    actual = chunked_logit_residual_loss(student, teacher, weight, weight, "cosine", 2, 1.)
    assert torch.allclose(actual, expected)

def test_token_cap_includes_endpoints():
    mask = torch.tensor([False,True,True,True,True,True])
    assert select_residual_tokens(mask, 3).tolist() == [1,3,5]
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_logit_residual.py -q
```

Expected: residual functions are missing.

- [ ] **Step 3: Implement vocabulary-chunk accumulation**

```python
def chunked_logit_residual_loss(student_dh, teacher_dh, student_weight, teacher_weight,
                                metric, chunk_size, delta):
    if metric == "huber":
        total = student_dh.sum()*0
        count = 0
        for sw, tw in zip(student_weight.split(chunk_size), teacher_weight.split(chunk_size)):
            sz = student_dh @ sw.T
            tz = teacher_dh @ tw.T
            total += F.huber_loss(sz, tz.detach(), delta=delta, reduction="sum")
            count += sz.numel()
        return total/max(count,1)
    dots = student_dh.new_zeros(student_dh.shape[0])
    sn = dots.clone()
    tn = dots.clone()
    for sw, tw in zip(student_weight.split(chunk_size), teacher_weight.split(chunk_size)):
        sz = student_dh @ sw.T
        tz = teacher_dh @ tw.T
        dots += (sz*tz.detach()).sum(-1)
        sn += sz.square().sum(-1)
        tn += tz.detach().square().sum(-1)
    return (1-dots/(sn.sqrt()*tn.sqrt()).clamp_min(1e-8)).mean()
```

For target token t, use final normalized predictor hidden state t-1 from `outputs.hidden_states`. Compute evidence-specific `h(full)-h(loo(j))` on selected adjacent-window targets. Use the policy and EMA teacher's own bias-free LM-head weights. Never materialize `[selected_tokens,vocabulary]` for all chunks at once.

- [ ] **Step 4: Retain hidden states only when active and run GREEN**

Add tests that student parameters receive gradients, EMA parameters do not, deterministic token count is logged, and monkeypatched allocation tracing never sees the full residual shape.

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_logit_residual.py -q
git add src/evidence_window/losses.py src/evidence_window/engine.py tests/evidence_window/test_logit_residual.py
git commit -m "feat: add chunked logit-residual distillation"
```

Expected: Huber/cosine equivalence, cap, gradient, and allocation tests pass.

---

### Task 10: Thin Trainer Integration and Exact Backward Compatibility

**Files:**
- Modify: `src/trainer.py:22-486`
- Modify: `src/main.py:213-226`
- Create: `tests/evidence_window/conftest.py`
- Create: `tests/evidence_window/test_backward_compatibility.py`
- Create: `tests/evidence_window/test_trainer_integration.py`

**Interfaces:**
- Consumes: config, token IDs, engine/reference, current EMA-selected latents/mask, original full output/loss.
- Produces: new optional trainer arguments and enabled auxiliary composition without changing the disabled path; shared `tiny_trainer_factory`, `snapshot_parameters`, and deterministic input fixtures.

- [ ] **Step 1: Characterize both existing loss branches before refactoring**

```python
def test_disabled_matches_recorded_legacy_teacher_latent_branch(trainer, model, inputs):
    trainer.evidence_window_config = EvidenceWindowConfig.disabled()
    loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
    assert torch.allclose(loss, inputs["recorded_legacy_loss"], atol=1e-6, rtol=1e-6)
    assert model.forward_count == inputs["recorded_legacy_forward_count"]

def test_disabled_matches_recorded_legacy_ce_fallback(trainer, model, no_latent_inputs):
    loss = trainer.compute_loss(model, no_latent_inputs)
    assert torch.allclose(loss, no_latent_inputs["recorded_ce_loss"], atol=1e-6, rtol=1e-6)
    assert trainer._ew_engine is None
    assert trainer._reference is None
```

Run this against the committed trainer first and record the fixture's deterministic values.

`tests/evidence_window/conftest.py` defines `TinyLatentLM`, whose forward returns deterministic `loss`, `logits`, final normalized `hidden_states`, and `inputs_embeds`, while counting full/all-null/LOO invocations. Its factory constructs `CustomTrainerStage1` without a 7B checkpoint, patches `_teacher_build_latents` to return fixed compact latents plus `firstK_mask`, and exposes inputs containing explicit legacy CE/sim expectations. `snapshot_parameters(module)` returns `{name: tensor.detach().clone()}` so all integration assertions use the same comparison rule.

- [ ] **Step 2: Run the characterization**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_backward_compatibility.py -q
```

Expected: both legacy branches pass before integration.

- [ ] **Step 3: Extract the existing body without arithmetic changes**

```python
@dataclass
class _ILVRLossContext:
    loss: torch.Tensor
    outputs: object
    latent_hidden_states: torch.Tensor | None
    image_out_mask: torch.Tensor | None
    model_inputs: dict

def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    context = self._compute_ilvr_loss_context(model, inputs, num_items_in_batch)
    if self._ew_engine is None or context.latent_hidden_states is None:
        return (context.loss, context.outputs) if return_outputs else context.loss
    auxiliary = self._compute_evidence_window(model, context)
    loss = context.loss + auxiliary.total_loss
    return (loss, context.outputs) if return_outputs else loss
```

Move the present lines 412–468 into `_compute_ilvr_loss_context`. Replace every early return with the corresponding context, preserving current CE/sim weighting and CE fallback exactly.

- [ ] **Step 4: Implement condition input/callback boundary**

```python
def _condition_inputs(self, context, latent_values):
    values = dict(context.model_inputs)
    values.pop("labels", None)
    values.pop("pixel_values_latent", None)
    values.pop("image_grid_thw_latent", None)
    values["latent_hidden_states"] = latent_values
    values["image_out_mask"] = context.image_out_mask
    values["return_dict"] = True
    return values

def _forward_policy_condition(self, model, inputs, labels, need_hidden):
    outputs = model(**inputs)
    return ModelConditionOutput(
        causal_token_logps(outputs.logits, labels),
        outputs.hidden_states if need_hidden else None,
    )
```

Reference forwards go through the provider under no-grad. EMA forwards use `self._ema.teacher` under no-grad and move copied inputs to its device. LOO policy forwards retain gradients. Reuse the original policy/full output rather than repeating it.

- [ ] **Step 5: Initialize runtime objects conditionally**

After `super().__init__`:
- disabled config sets engine/reference to None;
- enabled config constructs `TokenIds` and engine;
- preference modes construct the requested provider;
- frozen copy is snapshotted immediately, before optimizer updates;
- startup logs reference parameter count, dtype bytes, ZeRO stage, replication under ZeRO-2, EMA coexistence, selected intervention/cap, and nonzero dropout warnings.

- [ ] **Step 6: Add enabled gradient and forward-count tests**

```python
def test_enabled_ew_has_policy_grad_only(enabled_trainer, inputs):
    loss = enabled_trainer.compute_loss(enabled_trainer.model, inputs)
    loss.backward()
    assert any(p.grad is not None for p in enabled_trainer.model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in enabled_trainer._reference.parameters())
    assert all(p.grad is None for p in enabled_trainer._ema.teacher.parameters())

def test_three_spans_add_three_policy_loo_forwards(enabled_trainer, three_span_inputs):
    enabled_trainer.model.reset_counts()
    enabled_trainer.compute_loss(enabled_trainer.model, three_span_inputs)
    assert enabled_trainer.model.loo_forward_count == 3
```

- [ ] **Step 7: Run GREEN and commit**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_backward_compatibility.py tests/evidence_window/test_trainer_integration.py -q
git add src/trainer.py src/main.py tests/evidence_window/test_backward_compatibility.py tests/evidence_window/test_trainer_integration.py
git commit -m "feat: integrate evidence-window losses into ILVR trainer"
```

Expected: disabled outputs/loss/forward count match legacy; enabled policy gradients exist; reference and EMA gradients do not.

---

### Task 11: Experiment Configs and Bounded Research Logging

**Files:**
- Create: `configs/evidence_window/baseline.json`
- Create: `configs/evidence_window/diagnostic_global.json`
- Create: `configs/evidence_window/window_preference.json`
- Create: `configs/evidence_window/loo_window_preference.json`
- Create: `configs/evidence_window/relation_kd.json`
- Create: `configs/evidence_window/logit_residual.json`
- Create: `configs/evidence_window/loo_relation_combined.json`
- Create: `configs/evidence_window/frozen_reference_zero2.json`
- Create: `tests/evidence_window/test_experiment_configs.py`
- Modify: `src/evidence_window/engine.py`
- Modify: `src/trainer.py`

**Interfaces:**
- Consumes: strict schema and engine result.
- Produces: required source-edit-free variants, distributed scalar logs, optional numeric matrix dumps.

- [ ] **Step 1: Write config matrix test**

```python
EXPECTED = {
    "baseline.json": set(),
    "diagnostic_global.json": {"full","all_null"},
    "window_preference.json": {"full","all_null"},
    "loo_window_preference.json": {"full","loo"},
    "relation_kd.json": {"full","loo"},
    "logit_residual.json": {"full","loo"},
    "loo_relation_combined.json": {"full","loo"},
    "frozen_reference_zero2.json": {"full","loo"},
}

def test_shipped_configs_load_and_request_expected_policy_conditions():
    for name, expected in EXPECTED.items():
        cfg = EvidenceWindowConfig.from_json(Path("configs/evidence_window")/name)
        assert set(cfg.required_conditions("policy")) == expected
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_experiment_configs.py -q
```

Expected: JSON files are absent.

- [ ] **Step 3: Add explicit JSON variants**

Every file uses version 1 and spells out non-default choices. The ZeRO-2 variant is:

```json
{
  "version": 1,
  "reference": {"mode": "frozen_copy"},
  "preference": {"beta": 0.1, "lambda_global": 0.0, "lambda_window": 0.0, "lambda_ew": 1.0},
  "relation": {"lambda_jsd": 0.0, "lambda_magnitude": 0.0, "temperature": 1.0, "huber_delta": 1.0},
  "residual": {"lambda_logit": 0.0, "metric": "huber", "huber_delta": 1.0, "vocab_chunk_size": 4096, "max_tokens_per_window": 0},
  "loo": {"intervention": "zero_latent", "max_spans_per_example": 0},
  "diagnostics": {"global_saturation": true, "contribution_threshold": 0.0, "matrix_dump_interval": 0, "matrix_dump_dir": "evidence_window_matrices"}
}
```

Preference configs use adapter-disabled reference. Relation/residual-only configs use no reference. Baseline is `is_disabled`.

- [ ] **Step 4: Add bounded scalar logging**

Reduce numerator/count across ranks and pass these stable keys to `self.log`:

```text
loss/ilvr
loss/global_preference
loss/window_preference
loss/ew_dpo
loss/relation_jsd
loss/relation_magnitude
loss/logit_residual
preference/global_margin
preference/global_outer_weight
preference/window_margin_mean
preference/ew_margin_mean
relation/student/top1_ratio
relation/student/normalized_entropy
relation/student/above_threshold_count
relation/teacher/top1_ratio
relation/teacher/normalized_entropy
relation/teacher/above_threshold_count
loo/selected_span_count
```

Accumulate `preference/window_margin/{k}` and `preference/ew_margin/{k}` by logical ordinal so each observed native window/adjacent pair remains inspectable. Reduce their numerator/count pairs across ranks before logging. At configured intervals, write full numeric matrices through the rank-local writer using rank, global step, and opaque counter, never raw text or image paths.

- [ ] **Step 5: Run GREEN and commit**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window/test_experiment_configs.py tests/evidence_window/test_metrics.py -q
git add configs/evidence_window src/evidence_window/engine.py src/trainer.py tests/evidence_window/test_experiment_configs.py
git commit -m "feat: add evidence-window experiment variants"
```

Expected: every config validates and activates only its intended conditions.

---

### Task 12: Smoke Matrix, Documentation, and Completion Evidence

**Files:**
- Create: `tests/evidence_window/test_smoke_variants.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-21-evidence-window-credit-design.md` (status only after verification)

**Interfaces:**
- Consumes: all prior tasks and the repository venv.
- Produces: tiny one-step smoke evidence for all implemented losses, launch documentation, and final verification record.

- [ ] **Step 1: Add parameterized tiny optimizer smokes**

```python
@pytest.mark.parametrize("name", [
    "baseline.json",
    "diagnostic_global.json",
    "window_preference.json",
    "loo_window_preference.json",
    "relation_kd.json",
    "logit_residual.json",
    "loo_relation_combined.json",
])
def test_one_tiny_optimizer_step(name, tiny_trainer_factory):
    trainer = tiny_trainer_factory("configs/evidence_window/"+name)
    policy_before = snapshot_parameters(trainer.model)
    reference_before = None if trainer._reference is None else snapshot_parameters(trainer._reference)
    ema_before = snapshot_parameters(trainer._ema.teacher)
    loss = trainer.one_optimizer_step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(value, dict(trainer.model.named_parameters())[key])
               for key, value in policy_before.items())
    if reference_before is not None:
        assert all(torch.equal(value, dict(trainer._reference.named_parameters())[key])
                   for key, value in reference_before.items())
    assert all(parameter.grad is None for parameter in trainer._ema.teacher.parameters())
    assert trainer.ema_update_count == 1
    assert any(not torch.equal(value, dict(trainer._ema.teacher.named_parameters())[key])
               for key, value in ema_before.items())
```

- [ ] **Step 2: Run the complete CPU/synthetic suite once**

```bash
PYTHONPATH=src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest tests/evidence_window -q
```

Expected: all tests pass, including two-process tiny ZeRO-2.

- [ ] **Step 3: Run static and relevant existing checks**

```bash
/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m compileall -q src tests
```

If the vendored Qwen model file changed, also run its focused CPU tests:

```bash
PYTHONPATH=transformers/src /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/python -m pytest transformers/tests/models/qwen2_5_vl -q
```

If it did not change, record “not run—model path unchanged” and do not spend duplicate compute.

- [ ] **Step 4: Document exact launch examples**

```bash
source /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/venv/ilvr-venv/bin/activate
python src/main.py --use_lora --lora_dropout 0 --ew_config configs/evidence_window/loo_window_preference.json
python src/main.py --ew_config configs/evidence_window/frozen_reference_zero2.json --deepspeed_config configs/config_stage2.json
```

Document exact `V_j`/`R_k` definitions, zero-latent fixed-history semantics, `O(1+J_selected)` per-role forwards, replicated ZeRO-2 reference memory, and zero-default compatibility.

- [ ] **Step 5: Run each approved large-model smoke at most once**

First inspect recorded evidence. If none exists, obtain explicit approval because this loads more than 10 GB and may exceed five minutes. Run one minimal optimizer step for baseline-disabled, global diagnostic, window preference, EW-DPO adapter reference, EW-DPO frozen reference under ZeRO-2, relation JSD, relation magnitude, logit residual, and explicit combined objective. Record exact command/status, losses, forwards, gradient ownership, peak memory, and runtime. Reuse the result; do not rerun an unchanged model/checkpoint.

- [ ] **Step 6: Invoke verification-before-completion**

Run the required skill, then:

```bash
git status --short
git diff --check
```

The final report must state exact files changed, implemented mathematical objective, `V_j`/`R_k` definitions, LOO semantics, complexity, configs, tests/results, limitations, deviations, and reference/EMA invariance.

- [ ] **Step 7: Update verified status and commit documentation**

Only after required checks pass, update the spec status to “implemented and verified” and commit:

```bash
git add README.md docs/superpowers/specs/2026-08-21-evidence-window-credit-design.md tests/evidence_window/test_smoke_variants.py
git commit -m "docs: document evidence-window training variants"
```

Do not claim hidden-state residual KD or attention blocking; both remain deferred.

---

## Self-Review Coverage Map

- Spec §§1–3 (hypothesis, repository findings, goals/non-goals): global constraints plus Tasks 3, 7, and 12.
- Spec §4 (native layout): Task 2.
- Spec §5 (component boundaries): file structure and Tasks 1–9.
- Spec §6 (conditions, reuse, cap, dropout): Tasks 5, 7, and 10. The optional EMA selection-forward optimization is deliberately not taken, so no disabled-path work or equivalence burden is introduced.
- Spec §7 (fixed reference roles and memory): Task 6, including first-implementation ZeRO-2 frozen copy.
- Spec §8 (causal scores and preference losses): Tasks 3 and 7.
- Spec §9 (relation matrices and research metrics): Tasks 4, 8, and 11.
- Spec §10 (logit residual and deferred hidden residual): Task 9 and the explicit global exclusion of hidden residual.
- Spec §11 (strict config and objective): Tasks 1 and 11.
- Spec §12 (backward compatibility/errors): Tasks 1, 2, 7, and 10.
- Spec §13 (memory/compute): Tasks 5–7, 9, 10, and final smoke reporting in Task 12.
- Spec §14 (TDD milestones): Tasks 2–9 in the same milestone order.
- Spec §15 (integration/final verification): Tasks 6, 10, and 12.
- Spec §§16–17 (limitations and approved decisions): global constraints and Task 12 documentation.

Self-review found no uncovered mandatory spec requirement. Hidden-state residual remains conditional by design, and the optional EMA repeated-forward optimization is not selected for this implementation.

---

## Dependency and Review Order

```text
Task 1 config/disabled contract
  -> Task 2 native layout
  -> Task 3 score/preference math
  -> Task 4 diagnostics
  -> Task 5 zero-latent LOO
  -> Task 6 adapter and ZeRO-2 frozen references
  -> Task 7 EW condition engine
  -> Task 8 relation KD
  -> Task 9 logit residual KD
  -> Task 10 trainer/backward compatibility
  -> Task 11 configs/logging
  -> Task 12 verification/documentation
```

Each task receives a fresh review gate. Use `superpowers:systematic-debugging` for every unexplained failure. Request code review after Tasks 4, 7, 10, and before completion because those boundaries close a scientific milestone or trainer integration boundary.
