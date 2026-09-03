# Step Preference Tree-Packed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete TODO steps 9–11 by replacing the naive repeated final-decoder branches with one FlexAttention tree-packed decoder call, proving naive/packed loss and gradient equivalence, and measuring a real time or memory improvement.

**Architecture:** Keep the clean Stage 2 latent rollout as the single root computation. Append short sidecar query-row sequences for each V-off, T-off, and valid rejected completion; a block-sparse tree mask lets each sidecar see only clean ancestors and its own earlier rows, while copied logical positions preserve multimodal RoPE. The existing naive path stays unchanged and remains the correctness oracle behind an explicit opt-in flag.

**Tech Stack:** Python 3.11, PyTorch 2.7.1 CUDA 12.8 `torch.nn.attention.flex_attention`, vendored Transformers Qwen2.5-VL, Hugging Face Trainer, pytest, NVIDIA H100/B200.

**Spec:** `docs/superpowers/specs/2026-09-03-stepwise-modality-dpo-design.md`

## Global Constraints

- Work only in `/root/workspace/ILVR/.worktrees/ours1` and use `./venv/ilvr-venv` with `PYTHONPATH="$PWD/transformers/src:$PWD/src"`.
- The naive Stage 2B implementation and its Task 10 real-data witness are the oracle; do not change its call graph or numerical behavior.
- Packing is opt-in, Stage 2B-only, batch-size one, and disabled by default.
- Each sidecar attends only clean ancestors and its own previous/current rows; sidecar-to-sidecar attention is forbidden.
- V/T-off sidecars retain the exact shifted blocked edge. Rejected sidecars have clean context and no modality block.
- Logical 3-axis multimodal position IDs are copied from the clean sequence; never derive them from packed physical offsets.
- FlexAttention must skip masked blocks. Reducing Python call count with a dense attention mask is not completion.
- Deterministic FP32 tiny equivalence thresholds are `epsilon_loss=1e-5` and `epsilon_grad=1e-4`, over every trainable parameter.
- BF16 production thresholds are `epsilon_loss=1e-3` and `epsilon_grad=1e-2`, over the first trainable decoder `q_proj`, `o_proj`, and `down_proj` parameters.
- Benchmark both paths in one process on the same GPU/model/precision/fixed sample, with three warmups, ten measured iterations, CUDA synchronization around every timed iteration, and peak-memory reset after warmup.
- Completion requires at least 5% lower median iteration time or 1% lower peak allocated memory.
- Model artifacts stay under `checkpoints/`; generated metrics under `results/`; execution logs under `logs/`.
- A runtime failure gets one causal regression and the smallest targeted fix, followed by a fresh run ID.
- Do not claim that weak-label negatives are semantically worse or that model quality improved.

## Normative notation and existing contracts

All intervals are half-open. `ReasoningStepSpan(step_id, prev_text_start,
prev_text_end, latent_start, latent_end, target_text_start, target_text_end,
helper_image_idx)` uses clean full-sequence token coordinates. For target tokens
`[a, b)`, causal prediction/query rows are `[a - 1, b - 1)`. The packed root is
the original clean sequence in physical positions `[0, S)`, so its physical and
logical token indices are identical.

For each step, branches are appended in this exact order:

```text
visual_off, textual_off, visual_rejected if valid, textual_rejected if valid
```

An ablation branch duplicates clean input rows `[a - 1, b - 1)`. Its clean
ancestor interval is `[0, a - 1)`, its own local causal rows include itself, and
its blocked clean-root keys are `[latent_start, latent_end)` for visual-off or
`[prev_text_start, prev_text_end)` for textual-off.

For a rejected encoding, `completion_token_positions=(p0, ..., pn)` names target
tokens in that complete-string encoding, and prediction rows are
`(p0 - 1, ..., pn - 1)`. Let `lcp` be the length of the longest equal token-ID
prefix of the clean and rejected complete strings. The rejected branch starts
at `r0 = min(lcp, p0 - 1)` and ends at `r1 = pn`; it duplicates rejected input
rows `[r0, r1)`, sees clean ancestors `[0, r0)`, and selects packed positions
`packed_start + (pi - 1 - r0)`. It must have `r0 >= latent_end` for its joined
step, all completion positions must be strictly increasing, and every copied
logical position must exist in the clean full trajectory; otherwise packing
fails closed instead of changing tokenization or truncating.

`StepNegative` and the joined-step mapping retain the Phase A schema already
enforced by `src/step_preference/negatives.py` and `src/stage2/data.py`:

```python
{
    "span": ReasoningStepSpan,
    "negative": StepNegative,
    "visual_rejected": None | {
        "model_inputs": {"input_ids": LongTensor[1, L],
                         "attention_mask": LongTensor[1, L]},
        "completion_token_positions": tuple[int, ...],
        "completion_token_ids": tuple[int, ...],
        "rendered_text": str,
    },
    "textual_rejected": None | dict,
}
```

Packing does not support non-contiguous clean attention masks, batch sizes other
than one, empty target spans, duplicate/non-contiguous step IDs, or a rejected
branch that diverges before its completed latent prefix. These cases raise
`ValueError`. The only padding is right padding of the final packed sequence to
a multiple of 128: `token_ids=0`, `clean_embedding_sources=-1`,
`logical_position_sources=0`, `branch_id=-1`, `branch_local_position=0`, and
`valid_slots=False`. A padding query attends only itself so FlexAttention never
receives an all-masked row; padding output is never selected or scored.

The existing `_compute_stage2b_loss` in `src/stage2/trainer.py` is normative for
chosen/rejected mean log-probability, DPO weighting, entropy-margin reduction,
valid-pair denominators, and all existing metrics. Packed integration extracts
that reduction unchanged; the plan does not define a second objective.

---

### Task 1: Pure tree layout and block-sparse visibility contract

**Files:**
- Create: `src/step_preference/packing.py`
- Create: `tests/step_preference/test_packing.py`

**Interfaces:**
- Consumes: `ReasoningStepSpan`; joined `step_preference["steps"]` mappings containing `StepNegative` and rejected `model_inputs`.
- Produces: `PackedBranch`, `PackedTreePlan`, `build_packed_tree_plan`, `dense_visibility`, and `build_flex_block_mask`.

- [ ] **Step 1: Write failing layout tests**

Cover a two-step input with four ablation branches and three valid rejected branches. Assert exact packed branch bounds, shifted selected query rows, target IDs, longest-common-prefix branch starts, root/sidecar visibility, V/T blocked edges, sidecar isolation, end padding exclusion, and logical-position source indices.

```python
def test_tree_visibility_preserves_ancestors_blocks_one_edge_and_isolates_sidecars():
    clean_ids, spans, joined = _two_step_fixture()
    plan = build_packed_tree_plan(
        clean_input_ids=clean_ids,
        attention_mask=torch.ones_like(clean_ids),
        reasoning_spans=spans,
        joined_steps=joined,
        pad_to_multiple=128,
    )
    visible = dense_visibility(plan)
    visual = plan.branch("step-0-visual-off")
    assert visual.logical_query_positions == (3, 4)
    assert not visible[visual.selected_positions[0], 2]
    assert visible[visual.selected_positions[0], 1]
    assert visible[visual.selected_positions[1], visual.selected_positions[0]]
    rejected = plan.branch("step-0-visual-rejected")
    assert not visible[visual.selected_positions[0], rejected.packed_start]
```

- [ ] **Step 2: Run the tests and observe the missing module failure**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_packing.py
```

Expected: FAIL because `step_preference.packing` does not exist.

- [ ] **Step 3: Implement immutable layout records and validation**

Use these public records; tensors are one-dimensional CPU tensors until `to_model_inputs` moves them.

```python
@dataclass(frozen=True)
class PackedBranch:
    name: str
    kind: Literal["visual_off", "textual_off", "visual_rejected", "textual_rejected"]
    step_id: int
    packed_start: int
    packed_end: int
    clean_ancestor_end: int
    blocked_start: int
    blocked_end: int
    logical_query_positions: tuple[int, ...]
    selected_positions: tuple[int, ...]
    target_token_ids: tuple[int, ...]

@dataclass(frozen=True)
class PackedTreePlan:
    root_length: int
    unpadded_length: int
    padded_length: int
    token_ids: torch.Tensor
    clean_embedding_sources: torch.Tensor
    logical_position_sources: torch.Tensor
    branch_ids: torch.Tensor
    branch_local_positions: torch.Tensor
    clean_ancestor_ends: torch.Tensor
blocked_starts: torch.Tensor
    blocked_ends: torch.Tensor
    valid_slots: torch.Tensor
    root_selected_positions: tuple[int, ...]
    branches: tuple[PackedBranch, ...]

    def branch(self, name: str) -> PackedBranch:
        matches = [branch for branch in self.branches if branch.name == name]
        if len(matches) != 1:
            raise KeyError(name)
        return matches[0]

    def selected_positions(self) -> torch.Tensor:
        positions = list(self.root_selected_positions)
        for branch in self.branches:
            positions.extend(branch.selected_positions)
        return torch.tensor(positions, dtype=torch.long)

    def to_model_inputs(self, device: torch.device) -> dict[str, object]:
        return {
            "block_mask": build_flex_block_mask(self, device),
            "token_ids": self.token_ids.to(device),
            "clean_embedding_sources": self.clean_embedding_sources.to(device),
            "logical_position_sources": self.logical_position_sources.to(device),
            "valid_slots": self.valid_slots.to(device),
            "root_length": self.root_length,
        }
```

Root slots use `branch_id=0` and `clean_embedding_sources=0..S-1`. Ablation slots gather the corresponding clean final embedding. Rejected slots use `clean_embedding_sources=-1` and their complete-string token IDs. The rejected branch starts at `min(longest_common_prefix, first_completion_position - 1)` and must start at or after that step's `latent_end`; otherwise fail closed. Every logical source index must be in the clean root because the clean full trajectory is longer than the current-step completion.

- [ ] **Step 4: Implement one visibility predicate for both dense and Flex masks**

The predicate is:

```python
root_visible = (q_branch == 0) & (kv_branch == 0) & (kv_logical <= q_logical)
blocked_source = (kv_logical >= blocked_start) & (kv_logical < blocked_end)
sidecar_visible = (q_branch > 0) & (
    ((kv_branch == 0) & (kv_logical < ancestor_end) & ~blocked_source)
    | ((kv_branch == q_branch) & (kv_local <= q_local))
)
padding_visible = (~q_valid) & (q_idx == kv_idx)
return (q_valid & kv_valid & (root_visible | sidecar_visible)) | padding_visible
```

`dense_visibility` returns a CPU `torch.bool[P, P]` matrix. `build_flex_block_mask` moves the plan metadata to the requested CUDA device and calls local PyTorch's `create_block_mask(mask_mod, B=1, H=None, Q_LEN=P, KV_LEN=P, device=device, BLOCK_SIZE=128)`. It raises `ValueError` for a non-CUDA device. The returned `BlockMask.sparsity()` must be positive for the multi-branch test, proving a compressed block-sparse representation rather than a dense additive mask; the Task 7 runtime benchmark, not this structural assertion, proves actual benefit.

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_packing.py tests/step_preference/test_ablation.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/step_preference/packing.py tests/step_preference/test_packing.py
git commit -m "feat: define step preference tree packing"
```

### Task 2: Qwen FlexAttention decoder boundary

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Create: `tests/step_preference/test_qwen_flex_attention.py`

**Interfaces:**
- Consumes: a PyTorch `BlockMask` and explicit copied `[3, 1, packed_length]` logical position IDs.
- Produces: a `Qwen2_5_VLModel` decoder path that accepts `BlockMask` without changing the configured eager/SDPA default.

- [ ] **Step 1: Write a failing dense-versus-Flex decoder test**

On CUDA, initialize two identical tiny Qwen models with dropout zero. Disable TF32 for the test and enable deterministic algorithms. Feed the same embeddings and non-monotonic copied logical position IDs through (a) eager attention plus `torch.where(visible, 0.0, torch.finfo(torch.float32).min)[None, None]`, and (b) the BlockMask path. Backpropagate `last_hidden_state.square().mean()` in both models, compare last hidden states and `q_proj`, `o_proj`, and `down_proj` gradients in FP32, then restore the global deterministic/TF32 settings. Also record every decoder layer's received mask object with a `forward_pre_hook(with_kwargs=True)` and assert it is the same `BlockMask`.

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_qwen_flex_decoder_matches_dense_tree_mask_with_copied_positions():
    torch.manual_seed(101)
    dense_model = tiny_stage2_qwen(attn_implementation="eager").cuda().float().train()
    flex_model = copy.deepcopy(dense_model)
    dense_out = dense_model.model(inputs_embeds=embeds, attention_mask=dense_mask,
                                  position_ids=logical_positions, return_dict=True)
    flex_out = flex_model.model(inputs_embeds=embeds.clone(), attention_mask=block_mask,
                                position_ids=logical_positions, return_dict=True)
    torch.testing.assert_close(flex_out.last_hidden_state, dense_out.last_hidden_state,
                               rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run the exact test and verify the BlockMask type failure**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_qwen_flex_attention.py
```

Expected: FAIL because `_update_causal_mask` and the Qwen attention classes only accept tensor masks.

- [ ] **Step 3: Add an opt-in Flex dispatch to the vendored Qwen decoder**

Import `BlockMask` only when FlexAttention is available; when unavailable, passing a BlockMask-shaped packed request raises `RuntimeError("packed Stage 2 requires torch FlexAttention")`. Add this method contract:

```python
def forward_flex(
    self,
    hidden_states: torch.Tensor,
    *,
    block_mask: BlockMask,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    output_attentions: bool = False,
    use_cache: bool = False,
) -> tuple[torch.Tensor, None, None]:
```

It rejects `output_attentions`, `use_cache`, and `self.training and self.attention_dropout != 0.0`. It performs the existing Q/K/V projections into `[B, Hq|Hkv, P, D]`, applies the existing `apply_multimodal_rotary_pos_emb`, and calls vendored `transformers.integrations.flex_attention.flex_attention_forward` with `scaling=self.scaling`; that helper supplies GQA and returns `[B, P, H]`. Reuse the same `q_proj/k_proj/v_proj/o_proj`. Dispatch depends only on the mask being a `BlockMask`; the configured eager/SDPA class continues to own every tensor-mask call.

In `Qwen2_5_VLDecoderLayer.forward`, dispatch to `forward_flex` only when `attention_mask` is a `BlockMask`. In `Qwen2_5_VLModel.forward`, pass a BlockMask directly instead of calling `_update_causal_mask`. Tensor masks retain their exact existing path.

- [ ] **Step 4: Run Flex, eager, and SDPA regressions**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_qwen_flex_attention.py \
  tests/step_preference/test_decoder_mask_probe.py \
  tests/stage2/test_stage2_modeling.py
```

Expected: PASS with the CUDA Flex test executed, not skipped, on this host.

- [ ] **Step 5: Commit**

```bash
git add transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  tests/step_preference/test_qwen_flex_attention.py
git commit -m "feat: add qwen tree flex attention"
```

### Task 3: One-rollout packed Stage 2 model forward

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Create: `tests/step_preference/test_packed_stage2_model.py`

**Interfaces:**
- Consumes: `stage2_packed_inputs=PackedTreePlan.to_model_inputs(device)` and packed `stage2_logit_positions`.
- Produces: one clean latent rollout, one packed final decoder call, root-only clean CE, and ordered selected packed logits.

- [ ] **Step 1: Write a failing real tiny-Qwen packed-forward test**

Use one clean root, one V-off sidecar, one T-off sidecar, and one rejected sidecar. Compare each packed selected-logit slice against the corresponding existing independent eager forward. Wrap `model.stage2_rollout_engine.rollout` to count one rollout and register a hook on `model.model.norm` to count one full-length packed final decoder completion; rollout prefix decoder calls have shorter lengths and are not counted as final calls. Backpropagate `stage2_selected_logits.float().square().mean()` and assert finite gradients reach `model.model.embed_tokens.weight` at a clean-prefix token row and the first decoder `q_proj.weight`.

- [ ] **Step 2: Run the exact test and verify the unexpected keyword failure**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_packed_stage2_model.py
```

Expected: FAIL because `stage2_packed_inputs` is not yet a Stage 2 model argument.

- [ ] **Step 3: Extend only the Stage 2 final decoder boundary**

Add `stage2_packed_inputs: Optional[Mapping[str, object]] = None` to `forward` and `_forward_stage2`; the already-existing `stage2_logit_positions: Optional[LongTensor]` remains the ordered list of root-selected positions followed by every branch's selected positions. Validate the exact packed keys and one-dimensional tensor lengths:

```python
{
    "block_mask": BlockMask,
    "token_ids": LongTensor[P],
    "clean_embedding_sources": LongTensor[P],
    "logical_position_sources": LongTensor[P],
    "valid_slots": BoolTensor[P],
    "root_length": int,
}
```

After the existing clean latent rollout, construct packed embeddings differentiably:

```python
direct = self.model.embed_tokens(token_ids.unsqueeze(0))
safe_sources = clean_embedding_sources.clamp_min(0)
shared = rollout.final_inputs_embeds.index_select(1, safe_sources)
packed_inputs = torch.where(
    (clean_embedding_sources >= 0).view(1, -1, 1), shared, direct
)
packed_positions = full_position_ids.index_select(2, logical_position_sources)
```

Call the final decoder once with the BlockMask and `packed_positions`. Compute CE only from root hidden rows and original labels. Index `stage2_selected_logits` from packed hidden rows. Preserve the current output and latent-state behavior when `stage2_packed_inputs is None`.

Packed output retains the existing `Qwen2_5_VLCausalLMOutputWithPast` type: `loss` is root-only clean CE; `logits` is `None` for `stage2_loss_only=True` or `stage2_selective_logits_only=True`; `stage2_selected_logits` has shape `[1, N_selected, vocab]` in the caller-provided position order; `inputs_embeds` is the packed tensor; and `stage2_latent_states`/diagnostics still describe the single clean rollout. Packed mode requires `return_dict=True`, `use_cache=False`, `output_attentions=False`, and no gradient checkpointing. `output_hidden_states=True` remains supported and returns packed decoder states.

- [ ] **Step 4: Add fail-closed boundary cases**

Reject CPU BlockMasks, mismatched vector lengths/devices/dtypes, negative or out-of-root logical position sources, clean embedding sources other than `-1` or `[0, S)`, a root length unequal to the original input length, packed mode with cache/output attentions/gradient checkpointing, duplicate or unsorted selected positions, empty selections, and any selected position outside the packed length. Non-root valid direct-token slots reject image/video/vision-start and latent-start/pad control IDs. A direct latent-end token is allowed only when the clean root has the identical latent-end token at that slot's copied logical source; a mismatched latent-end token is rejected. This narrow case is required when the first rejected completion token immediately follows the latent block and its causal prediction row is `<|latent_end|>`. Padding slots are exempt from the direct-token control-ID check and use the valid vocabulary ID zero.

- [ ] **Step 5: Run focused model regressions**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_stage2_model.py \
  tests/step_preference/test_qwen_flex_attention.py \
  tests/stage2/test_stage2_modeling.py tests/stage2/test_rollout.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  tests/step_preference/test_packed_stage2_model.py
git commit -m "feat: add packed stage 2 final forward"
```

### Task 4: Packed objective integration behind an exact opt-in

**Files:**
- Modify: `src/step_preference/config.py`
- Modify: `src/main.py`
- Modify: `src/stage2/trainer.py`
- Modify: `run_stage2_training.sh`
- Modify: `tests/step_preference/test_config.py`
- Modify: `tests/step_preference/test_stage2b_trainer.py`
- Modify: `tests/stage2/test_routing.py`
- Modify: `tests/stage2/test_launcher.py`

**Interfaces:**
- Consumes: Tasks 1–3 layout/model interfaces.
- Produces: CLI/env flag `--step_preference_packing`, `StepPreferenceConfig.packing_enabled`, and `_compute_stage2b_packed_loss` with the same scalar objective/log fields as naive Stage 2B.

- [ ] **Step 1: Write failing config, routing, and fake-policy tests**

Assert the default is false, launcher passthrough is exact, packing outside Stage 2B fails, and disabled/naive model calls are byte-for-byte unchanged. A packed fake policy must receive one model call with `stage2_packed_inputs`, return deterministic logits for every root/branch slice, and yield the same hand-computed CE/DPO/entropy scalar and component metrics as the current naive fake policy.

```python
def test_packed_stage2b_uses_one_policy_call_and_same_objective(monkeypatch):
    naive_loss, naive_metrics = run_fake_stage2b(packing=False)
    packed_loss, packed_metrics, calls = run_fake_stage2b(packing=True)
    torch.testing.assert_close(packed_loss, naive_loss, rtol=0, atol=0)
    assert packed_metrics["loss/dpo_visual"] == naive_metrics["loss/dpo_visual"]
    assert len(calls) == 1
    assert calls[0]["stage2_packed_inputs"] is not None
```

- [ ] **Step 2: Run the exact tests and observe missing flag/path failures**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_routing.py tests/stage2/test_launcher.py
```

Expected: FAIL on the absent flag and packed trainer call.

- [ ] **Step 3: Add the immutable flag and exact CLI/env plumbing**

Add `packing_enabled: bool = False` to `StepPreferenceConfig`, sourced only from the `argparse` `store_true` option `--step_preference_packing`. Add `STEP_PREFERENCE_PACKING` to the launcher: unset, empty, `0`, `false`, or `False` omits the flag; `1`, `true`, or `True` appends it; any other value exits before launch. `StepPreferenceConfig.__post_init__` raises when packing is true but `is_stage2b` is false. Do not change the existing accepted eager/SDPA setting because Flex replaces only the final packed decoder.

- [ ] **Step 4: Refactor objective reduction without changing the naive graph**

Extract only the post-logit scalar reduction into a private helper taking clean CE, chosen/rejected scores, reference scores, validity masks, and clean/ablated entropies. Keep the current naive forward loop and call counts intact. Route to `_compute_stage2b_packed_loss` only when `packing_enabled` is true.

The packed path builds one `PackedTreePlan`, creates its CUDA BlockMask, requests root chosen rows in step order followed by every branch's selected rows in the normative branch order, slices returned logits by each `PackedBranch`, and calls the shared reduction. `stats/branch_forwards` counts final policy decoder graphs only, not the internal clean latent-rollout prefix calls or any frozen-reference work already materialized offline. It emits every existing metric name plus:

```text
stats/branch_forwards = 1
stats/packed_tokens
stats/packed_mask_sparsity
```

- [ ] **Step 5: Run integration and baseline-preservation tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_config.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/stage2/test_routing.py tests/stage2/test_launcher.py \
  tests/stage2/test_data.py tests/stage2/test_stage2_modeling.py
```

Expected: PASS, including exact default/naive preservation.

- [ ] **Step 6: Commit**

```bash
git add src/step_preference/config.py src/main.py src/stage2/trainer.py \
  run_stage2_training.sh tests/step_preference/test_config.py \
  tests/step_preference/test_stage2b_trainer.py tests/stage2/test_routing.py \
  tests/stage2/test_launcher.py
git commit -m "feat: integrate packed step preference training"
```

### Task 5: Deterministic loss and gradient equivalence verifier

**Files:**
- Create: `src/step_preference/equivalence.py`
- Create: `scripts/verify_packed_step_preference.py`
- Create: `tests/step_preference/test_packed_equivalence.py`

**Interfaces:**
- Consumes: one model, fixed collated Stage 2B batch, and naive/packed `Stage2Trainer` configurations.
- Produces: `relative_l2_error`, `compare_scalar_metrics`, `run_tiny_fp32_equivalence`, `run_production_bf16_equivalence`, and schema-1 JSON evidence.

- [ ] **Step 1: Write failing math and end-to-end tiny-model tests**

Unit-test zero-reference gradients, a known relative error, missing/nonfinite gradients, and per-component tolerance failures. Then run two identical CUDA FP32 tiny Qwen models on the same one-step batch, backward both losses, and require all trainable parameter gradients to satisfy the spec.

```python
def test_tiny_fp32_naive_and_packed_match_all_trainable_gradients():
    result = run_tiny_fp32_equivalence(device="cuda:0")
    assert result.status == "pass"
    assert result.max_loss_error <= 1e-5
    assert result.max_gradient_relative_l2 <= 1e-4
    assert set(result.gradients) == set(result.trainable_parameters)
```

- [ ] **Step 2: Run the verifier tests and confirm missing implementation**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_packed_equivalence.py
```

Expected: FAIL because the equivalence module/script is absent.

- [ ] **Step 3: Implement exact comparison functions**

Use float64 accumulation for relative L2 norms:

```python
def relative_l2_error(reference: torch.Tensor, candidate: torch.Tensor,
                      epsilon: float = 1e-12) -> float:
    ref = reference.detach().double().cpu()
    cand = candidate.detach().double().cpu()
    return float(torch.linalg.vector_norm(cand - ref) /
                 (torch.linalg.vector_norm(ref) + epsilon))
```

Both gradients must exist, have equal shapes, and be finite. If the reference
norm is below `1e-12`, pass only when the candidate difference norm is also at
most the active absolute gradient tolerance; otherwise use the relative error
above. Scalar/component loss error is absolute difference. Missing or nonfinite
values are recorded as named errors and fail the result rather than being
coerced to zero.

Reset RNG and gradients before each path. Tiny equivalence uses deep-copied identical models and compares every trainable parameter. Production equivalence uses one frozen model state sequentially: naive forward/backward, copy named gradients to CPU, clear gradients and graph, then packed forward/backward. Select the first trainable decoder parameters whose names end in `q_proj.weight`, `o_proj.weight`, and `down_proj.weight` after wrappers.

- [ ] **Step 4: Implement a strict evidence CLI**

The CLI accepts `--checkpoint_path`, `--data_path`, `--image_root`, `--negative_path`, `--negative_manifest`, `--device`, and `--output_path`. It reuses `build_stage2_training_components` and the real collator, selects the lowest manifest `selected_row_indices` entry that has at least one valid pair, and verifies that its collated `sample_id` joins the authenticated negative store. It runs both equivalence levels with dropout zero, TF32 disabled, identical RNG state, and no autocast for tiny FP32 versus BF16 autocast for production. It refuses to overwrite `output_path`.

Write this schema-1 shape and exit nonzero on any failed predicate:

```json
{
  "schema_version": 1,
  "status": "pass",
  "identity": {"checkpoint_path": "/raid/checkpoints/stage2a",
               "sample_id": "sample-a",
               "negative_sha256": "85da2e3b9fd048b7b6d38e24f35c0916ad2e131effa84103118f754640543cef",
               "device": "cuda:0",
               "git_commit": "0123456789abcdef0123456789abcdef01234567"},
  "tiny_fp32": {"loss_errors": {"loss/total": 0.0},
                  "gradient_relative_l2": {"model.layers.0.self_attn.q_proj.weight": 0.0},
                  "epsilon_loss": 1e-5, "epsilon_grad": 1e-4},
  "production_bf16": {"loss_errors": {"loss/total": 0.0},
                       "gradient_relative_l2": {"q_proj": 0.0,
                                                  "o_proj": 0.0,
                                                  "down_proj": 0.0},
                       "epsilon_loss": 1e-3, "epsilon_grad": 1e-2},
  "errors": []
}
```

`identity.checkpoint_path` is resolved; `negative_sha256` comes from the validated manifest; `git_commit` is the full `git rev-parse HEAD` captured before model loading. Production parameter selection sorts `model.named_parameters()` lexically and records the first trainable name containing `.layers.` and ending in each of `q_proj.weight`, `o_proj.weight`, and `down_proj.weight`; absence is a hard failure. Here “same frozen model state” means weights are unchanged because no optimizer step occurs, not that `requires_grad` is disabled.

- [ ] **Step 5: Run focused equivalence and trainer tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_equivalence.py \
  tests/step_preference/test_stage2b_trainer.py \
  tests/step_preference/test_packed_stage2_model.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/step_preference/equivalence.py scripts/verify_packed_step_preference.py \
  tests/step_preference/test_packed_equivalence.py
git commit -m "test: verify packed step preference equivalence"
```

### Task 6: Synchronized CUDA benchmark harness

**Files:**
- Create: `src/step_preference/benchmark.py`
- Create: `scripts/benchmark_packed_step_preference.py`
- Create: `tests/step_preference/test_packed_benchmark.py`

**Interfaces:**
- Consumes: the same loaded policy, collated fixed batch, and naive/packed loss callables.
- Produces: `benchmark_training_path`, `evaluate_benchmark_gate`, and schema-1 benchmark JSON.

- [ ] **Step 1: Write failing protocol and threshold tests**

Inject a fake CUDA runtime and deterministic clock. Assert exactly three unrecorded warmups and ten recorded iterations per path, zero-grad before each iteration, synchronize immediately before and after timing, peak reset after warmup, median rather than mean, and the disjunctive threshold.

```python
def test_gate_accepts_time_or_memory_improvement():
    assert evaluate_benchmark_gate(naive_ms=100, packed_ms=94.9,
                                   naive_bytes=1000, packed_bytes=1000).passed
    assert evaluate_benchmark_gate(naive_ms=100, packed_ms=100,
                                   naive_bytes=1000, packed_bytes=990).passed
    assert not evaluate_benchmark_gate(naive_ms=100, packed_ms=96,
                                       naive_bytes=1000, packed_bytes=995).passed
```

- [ ] **Step 2: Run the tests and confirm missing harness failure**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q tests/step_preference/test_packed_benchmark.py
```

Expected: FAIL because the benchmark module/script is absent.

- [ ] **Step 3: Implement the fixed training-iteration benchmark**

Prebuild/collate the fixed batch before timing. Each iteration includes trainer routing, packed-plan and BlockMask construction, the complete Stage 2B forward/loss, and backward; it excludes model loading, data/image decoding, JSON logging, and optimizer mutation. Each iteration executes `model.zero_grad(set_to_none=True)` and restores the same RNG state before computing the loss, so dropout or routing cannot drift.

Run naive first and packed second because packed Flex compilation is path-specific; each path receives exactly three warmups that absorb its own setup/compilation. After each path's warmups, call `torch.cuda.synchronize(device)`, record `baseline_allocated_bytes=torch.cuda.memory_allocated(device)`, and call `torch.cuda.reset_peak_memory_stats(device)`. For each of ten measurements: synchronize; record the start CUDA event; execute forward/backward; record the end event; synchronize; and read `start.elapsed_time(end)`. Report the median of the ten event times, `1000 / median_ms`, absolute peak allocated bytes, and `peak_increment_bytes=max(0, peak-baseline)`. The required 1% gate uses absolute peak allocated bytes as specified; the incremental value is diagnostic only.

The unit-test seam is `benchmark_training_path(step_fn, runtime, warmups=3, measurements=10)`, where `runtime` exposes `synchronize`, `memory_allocated`, `reset_peak_memory_stats`, `max_memory_allocated`, `new_event`, and `elapsed_ms`. This makes call order observable without CUDA.

- [ ] **Step 4: Implement the CLI and fail-closed gate**

Accept the same artifact arguments as the equivalence CLI plus `--output_path`. Refuse to overwrite the output. Load the model and fixed sample once, run naive then packed against unchanged weights, require both paths to report the same `sample_id`, and compute:

```python
time_reduction = 1.0 - packed_median_ms / naive_median_ms
memory_reduction = 1.0 - packed_peak_bytes / naive_peak_bytes
passed = time_reduction >= 0.05 or memory_reduction >= 0.01
```

Write JSON even on a threshold miss and exit nonzero. Do not save model state.

The schema records `schema_version`, `status`, the same identity object as the equivalence file, per-path ten raw times/median/iterations-per-second/baseline/absolute peak/incremental peak, both reduction fractions, threshold constants, and named errors. Runtime exceptions are preserved in the tee'd log; the CLI writes a failure JSON only when the output directory is writable and a trustworthy identity was already established.

- [ ] **Step 5: Run benchmark unit tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference/test_packed_benchmark.py \
  tests/step_preference/test_packed_equivalence.py
```

Expected: PASS without launching a 7B benchmark from pytest.

- [ ] **Step 6: Commit**

```bash
git add src/step_preference/benchmark.py scripts/benchmark_packed_step_preference.py \
  tests/step_preference/test_packed_benchmark.py
git commit -m "perf: benchmark packed step preference"
```

### Task 7: Real BF16 equivalence and performance gate

**Artifacts:**
- Read model: `checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827`
- Read data: `data/TRAIN.jsonl`
- Read negatives: `results/step-preference-phase-a-20260903T133617Z/negatives.jsonl`
- Read manifest: `results/step-preference-phase-a-20260903T133617Z/negative-manifest.json`
- Write results: `results/step-preference-packed-<UTC-run-id>/`
- Write logs: `logs/step-preference-packed-<UTC-run-id>/`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: green tiny FP32, production BF16, and performance JSON witnesses for TODO 9–11.

- [ ] **Step 1: Allocate one collision-free run ID**

```bash
export PACKED_RUN_ID=step-preference-packed-$(date -u +%Y%m%dT%H%M%SZ)
test ! -e "results/$PACKED_RUN_ID"
test ! -e "logs/$PACKED_RUN_ID"
mkdir "results/$PACKED_RUN_ID" "logs/$PACKED_RUN_ID"
```

Keep Steps 1–4 in the same shell session so `PACKED_RUN_ID` cannot drift.

- [ ] **Step 2: Run one focused combined suite**

```bash
set -o pipefail
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest -q \
  tests/step_preference tests/stage2 \
  2>&1 | tee "logs/$PACKED_RUN_ID/focused-suite.log"
```

Expected: every test passes; every CUDA-required packed test executes rather than skips. Record passed and skipped counts from pytest's final summary. Do not repeat this broad suite if green.

- [ ] **Step 3: Run equivalence**

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/verify_packed_step_preference.py \
  --checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --data_path data/TRAIN.jsonl --image_root data/images_comt \
  --negative_path results/step-preference-phase-a-20260903T133617Z/negatives.jsonl \
  --negative_manifest results/step-preference-phase-a-20260903T133617Z/negative-manifest.json \
  --device cuda:0 --output_path "results/$PACKED_RUN_ID/equivalence.json" \
  2>&1 | tee "logs/$PACKED_RUN_ID/equivalence.log"
```

Require tiny FP32 loss `<=1e-5`, all-gradient relative L2 `<=1e-4`, BF16 loss `<=1e-3`, and named production-gradient relative L2 `<=1e-2`.

- [ ] **Step 4: Run the synchronized benchmark in the same fresh run**

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python scripts/benchmark_packed_step_preference.py \
  --checkpoint_path checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827 \
  --data_path data/TRAIN.jsonl --image_root data/images_comt \
  --negative_path results/step-preference-phase-a-20260903T133617Z/negatives.jsonl \
  --negative_manifest results/step-preference-phase-a-20260903T133617Z/negative-manifest.json \
  --device cuda:0 --output_path "results/$PACKED_RUN_ID/benchmark.json" \
  2>&1 | tee "logs/$PACKED_RUN_ID/benchmark.log"
```

Require at least 5% lower packed median time or 1% lower packed peak allocated memory. If this threshold misses, Task 7 is not complete; use the recorded block sparsity, raw iteration times, and absolute/incremental peak allocations to identify the first causal packed-kernel/layout defect, add one regression, and rerun under a fresh run ID.

- [ ] **Step 5: Cross-check and record the runtime evidence**

Assert `equivalence.json` and `benchmark.json` both have `status="pass"`, the same resolved checkpoint path, `sample_id`, negative SHA-256, device, dtype, and current pre-runtime git commit. Record the exact commands, pytest pass/skip counts, both evidence paths, and this identity cross-check in the SDD Task 7 report. The claim is limited to numerical equivalence and measured execution efficiency.

- [ ] **Step 6: Leave runtime payloads uncommitted**

Task 7 requires no commit when no causal code fix was needed. Do not commit `results/$PACKED_RUN_ID`, runtime logs, checkpoints, compiler caches, or generated model state. If a threshold miss required a causal code/test fix, commit only those source/test files before allocating the fresh retry ID; the new evidence JSON records that fix commit.

---

## Plan self-review record

- Spec coverage: Task 1 covers the tree topology, edge masking, sidecar isolation, logical-position map, and block sparsity; Tasks 2–4 cover Qwen execution, clean-prefix gradients, opt-in integration, and unchanged disabled/naive behavior; Task 5 covers both declared equivalence levels; Tasks 6–7 cover the exact benchmark protocol and improvement gate.
- Placeholder scan: no deferred implementation steps or unspecified error-handling instructions remain.
- Type consistency: `PackedTreePlan.to_model_inputs` is the sole cross-boundary model mapping; `stage2_packed_inputs` is used consistently by Tasks 3–5; both runtime CLIs consume the same authenticated checkpoint/data/negative inputs.
