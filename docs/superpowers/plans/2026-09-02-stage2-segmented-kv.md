# Stage 2 Segmented KV-Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and evidence-gate the opt-in `segmented_kv` Stage 2 training recurrence while preserving `no_cache` as the default, semantic oracle, evaluation/inference backend, and legacy-resume meaning.

**Architecture:** Add an oracle-independent Qwen-local rollout engine that feeds monotonic embedding deltas into one fresh autograd-connected `DynamicCache` per sample, then leaves the existing complete-sequence `use_cache=False` loss pass untouched. Install that engine only after strict active-model preflight, bind each training run and checkpoint to an exact recurrence identity, and let a separate evidence validator derive adoption from immutable per-rank measurements and independent verification commands.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu128, vendored Transformers 4.50.0.dev0, PEFT 0.17.1, Accelerate 1.10.1, FlashAttention 2.8.3, pytest, Bash, 2-node/4-rank NVIDIA H100 NVL DDP.

**Spec:** `documents/stage-2-kv-cache-20260902.md`

## Understood Scope

Understood as: implement the approved design commitment v2 in the existing `feat/vcsd-ilvr-stage2` worktree, stop at any fail-closed prerequisite or adoption gate, preserve all pre-existing dirty changes, and do not launch a production training run.

## Observed Planning Baseline and Required Inputs

- Planning branch/commit: `feat/vcsd-ilvr-stage2` at `ae83093aa9c6331f9ca028d4affe4d807257204c`, a descendant of the reviewed `13564855b42d3ca78dad462d4e2930073bcd4c60` baseline.
- Local host: `172.16.11.40` (`korea-20`) with `2× NVIDIA H100 NVL`, 95,830 MiB each, driver `580.173.02`; `bond0.11` is up.
- Dataset: `/root/workspace/ILVR/data/TRAIN.jsonl`; image root: `/root/workspace/ILVR/data`.
- The current checkpoint tree contains tiny Stage 1 test artifacts but no discovered production-sized Stage 1 initialization artifact, and the current result tree contains no discovered sealed Stage 2 length manifest. Task 1 must not guess either path. Before its profile step, the operator must bind `STAGE1_CHECKPOINT` and either bind a matching `STAGE2_LENGTH_MANIFEST` or create it with `stage2.length_manifest`; both paths must pass the repository's existing artifact/manifest validators.
- Existing user changes overlap recurrence checkpointing, PEFT preflight, and Trainer save files. Before implementation, invoke `superpowers:using-git-worktrees`; prefer a clean implementation worktree based on `ae83093` so qualification can satisfy the clean-runtime digest gate without absorbing the current dirty changes.

## Global Constraints

- Run Python, pytest, training, and evaluation only through `./venv/ilvr-venv` with `PYTHONPATH=transformers/src:src` where repository imports are needed.
- Store model-bearing state only below `./checkpoints/`, result JSON below `./results/`, and logs below `./logs/`; keep the existing `./checkpoints/` symlink intact.
- Start from commit `13564855b42d3ca78dad462d4e2930073bcd4c60` or a reviewed descendant and record the implementation commit as 40 lowercase hexadecimal characters.
- Preserve `LatentRolloutEngine` unchanged as the independent oracle; candidate code lives in `transformers/src/transformers/models/qwen2_5_vl/stage2_kv_rollout.py`.
- Preserve rank-local batch size one, latent/scaffold semantics, three-axis RoPE, masks, label rows, trainable-parameter boundaries, effective batch, sampler membership, and the mandatory final decoder/loss pass with `use_cache=False`.
- Permit `segmented_kv` only for training with all decoder/adapter dropout equal to `0.0`, gradient checkpointing disabled, all-one attention masks, a fresh `DynamicCache` per sample, and no cache reuse or persistence.
- Keep `no_cache` as the omitted/default backend, evaluation/inference backend, legacy-checkpoint meaning, and same-run failure fallback prohibition.
- Single-device structural oracle is eager FP32. BF16 FlashAttention 2 and four-rank DDP remain unauthorized until their respective H100 gates pass.
- Adoption requires every correctness/distributed/capacity/save-resume gate, at least `0.15` median rank-max step-time improvement, and at most `0.05` peak-allocated-memory regression.
- Never reinterpret schema-v1 efficiency evidence as KV authorization. The final comparison validator alone derives `accepted`.
- Before every edit, compare the target file's `git diff` with the baseline captured at plan start. `src/stage2/checkpointing.py`, `src/stage2/peft.py`, `src/stage2/trainer.py`, `tests/stage2/test_peft.py`, and `tests/stage2/test_peft_checkpointing.py` already contain user changes; never stage or commit those files wholesale.
- For a clean file or a newly created file, commit only the task's explicit paths after inspecting `git diff --cached`. For a pre-dirty overlapping file, leave the task uncommitted unless the task's hunks can be isolated without staging the user's hunks.

## Embedded Runtime and Evidence Contract

The recurrence runtime digest has exactly these keys:

```text
transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py
transformers/src/transformers/models/qwen2_5_vl/stage2_kv_rollout.py
transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
transformers/src/transformers/cache_utils.py
src/main.py
src/task_deepseed.py
src/artifact_paths.py
src/peft_common.py
src/stage2/efficiency_probe.py
src/stage2/kv_evidence.py
src/stage2/checkpointing.py
src/stage2/trainer.py
src/stage2/peft.py
src/stage2/sampler.py
run_stage2_training.sh
transformers/src/transformers/trainer.py
```

`stage2-recurrence.json` has exactly `schema_version`, `backend`, `algorithm_version`, `git_commit`, `runtime_sha256`, `model_identity`, `execution`, `data_identity`, `batch_geometry`, and `global_step`. `model_identity` has exactly `base_reference`, `base_tree_sha256`, `adapter_reference`, `adapter_tree_sha256`, `tuning_method`, and `peft_version`. `execution` has exactly `attention_implementation`, `precision`, `decoder_dropout`, `attention_dropout`, `adapter_dropout`, `gradient_checkpointing`, `latent_size`, and `supervise_latent_boundaries`. `data_identity` has exactly `manifest_sha256`, `dataset_sha256`, and `sampler_schedule_sha256`. `batch_geometry` has exactly `world_size`, `per_device_batch`, `gradient_accumulation`, and `effective_batch`.

Each rank-evidence file has exactly `schema_version`, `experiment_id`, `backend`, `global_rank`, `world_size`, `runtime_identity_sha256`, `schedule_sha256`, `benchmark_cursor`, three `warmup_frame_sha256` entries, ten `measured_frame_sha256` entries, ten `steps`, `p95_capacity`, and `maximum_capacity`. Each step has exactly `index`, `frame_sha256`, `step_ms`, `data_ms`, `forward_loss_ms`, `backward_ms`, `sync_ms`, `optimizer_ms`, `overhead_ms`, `peak_allocated_bytes`, `peak_reserved_bytes`, `query_token_count`, `key_token_count`, `latent_block_count`, `loss`, `gradient_norm`, and `occurrence_sha256`. Each capacity object has exactly `raw_index`, `recurrence_work`, `completed`, `peak_allocated_bytes`, and `finite`.

The comparison has exactly six metrics: `oracle_rank_max_median_step_ms`, `candidate_rank_max_median_step_ms`, `speed_improvement_fraction`, `oracle_peak_allocated_bytes`, `candidate_peak_allocated_bytes`, and `memory_regression_fraction`. Its nine gates are exactly `runtime_identity`, `workload_identity`, `fp32_correctness`, `bf16_correctness`, `distributed_correctness`, `capacity`, `save_resume`, `speed`, and `memory`; `accepted` is their conjunction.

The qualification DDP topology is fixed: rank-0/master `172.16.11.41`, worker `172.16.11.40`, interface `bond0.11`, two H100s per node, four processes total, per-device batch `1`, accumulation `2`, effective batch `8`, DoRA dropout `0.0`, and checkpointing disabled. These short controlled qualification runs are not the prohibited production training run.

---

### Task 1: Seal the compatible no-cache prerequisite

**Files:**
- Modify: `src/stage2/efficiency_probe.py`
- Modify: `tests/stage2/test_efficiency_probe.py`
- Read: `src/stage2/length_manifest.py`
- Generate: `results/zero-stage-compatibility/stage2-kv-prereq-no-cache-20260902/stage2-efficiency.json`
- Generate: `logs/zero-stage-compatibility/stage2-kv-prereq-no-cache-20260902/`

**Interfaces:**
- Consumes: a validated `Stage2LengthManifest`, prepared retained examples, and their raw indices.
- Produces: `Stage2RecurrenceWorkExample(raw_index, sequence_length, latent_positions, scaffold_sha256)` selected by linear p50/p95 recurrence-work and maximum recurrence-work; a compatible H100 no-cache evidence result with `segmented_kv_authorized == false`.

- [ ] **Step 1: Add failing recurrence-work selection tests**

Add independent expected-value calculations to `tests/stage2/test_efficiency_probe.py`:

```python
def test_representatives_use_sequence_plus_every_latent_position():
    def recurrence_row(*, raw_index, sequence_length, latent_positions):
        return Stage2RecurrenceWorkExample(
            raw_index=raw_index,
            sequence_length=sequence_length,
            latent_positions=latent_positions,
            scaffold_sha256="0" * 64,
        )

    rows = (
        recurrence_row(raw_index=9, sequence_length=10, latent_positions=(2, 3)),
        recurrence_row(raw_index=4, sequence_length=20, latent_positions=(5, 8)),
        recurrence_row(raw_index=7, sequence_length=30, latent_positions=(10, 20)),
    )
    selected = select_recurrence_work_examples(rows)
    assert [row.recurrence_work for row in selected] == [33, 60, 60]
    assert selected[-1].raw_index == 7

def test_representative_tie_breaks_on_raw_index():
    rows = (
        Stage2RecurrenceWorkExample(8, 10, (5,), "0" * 64),
        Stage2RecurrenceWorkExample(2, 12, (3,), "1" * 64),
    )
    assert select_recurrence_work_examples(rows)[0].raw_index == 2
```

Also assert that parsing occurs before model loading and that the emitted metadata records the three raw indices, manifest/scaffold digests, sequence lengths, exact ordered latent positions, and recurrence-work values.

- [ ] **Step 2: Run the focused tests and observe the intended failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_efficiency_probe.py -k 'recurrence_work or representative'
```

Expected: FAIL because recurrence-work records and selection are not implemented.

- [ ] **Step 3: Implement independent recurrence-work records and selection**

In `src/stage2/efficiency_probe.py`, add immutable records and pure selection helpers:

```python
@dataclass(frozen=True)
class Stage2RecurrenceWorkExample:
    raw_index: int
    sequence_length: int
    latent_positions: tuple[int, ...]
    scaffold_sha256: str

    @property
    def recurrence_work(self) -> int:
        return self.sequence_length + sum(self.latent_positions)

def select_recurrence_work_examples(
    examples: tuple[Stage2RecurrenceWorkExample, ...],
) -> tuple[Stage2RecurrenceWorkExample, ...]:
    values = tuple(row.recurrence_work for row in examples)
    targets = (_linear_quantile(values, 0.50), _linear_quantile(values, 0.95))
    chosen = [min(examples, key=lambda row: (abs(row.recurrence_work - target), row.raw_index)) for target in targets]
    chosen.append(min(examples, key=lambda row: (-row.recurrence_work, row.raw_index)))
    return tuple(chosen)
```

Build these records from prepared retained examples by parsing their actual scaffold IDs with `parse_latent_spans`; do not infer latent positions from block count and do not load weights until the records and digests have been sealed.

- [ ] **Step 4: Run focused tests**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_efficiency_probe.py tests/stage2/test_length_manifest.py
```

Expected: PASS, while every schema-v1 assertion still requires `segmented_kv_authorized is False`.

- [ ] **Step 5: Run the compatible H100 no-cache profile once**

Set `STAGE1_CHECKPOINT` to the exact validated Stage 1 initialization artifact and `STAGE2_LENGTH_MANIFEST` to its matching sealed manifest. Validate both with `test -e`, compute `STAGE2_LENGTH_MANIFEST_SHA256` with `sha256sum`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=transformers/src:src \
  ./venv/ilvr-venv/bin/python -m stage2.efficiency_probe \
  --experiment-id stage2-kv-prereq-no-cache-20260902 \
  --model-source "$STAGE1_CHECKPOINT" \
  --data-path /root/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /root/workspace/ILVR/data \
  --stage2-length-manifest "$STAGE2_LENGTH_MANIFEST" \
  --stage2-length-manifest-sha256 "$STAGE2_LENGTH_MANIFEST_SHA256" \
  --latent-size 8 --tuning-method dora \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.0 \
  --attention-implementation flash_attention_2 \
  --no-gradient-checkpointing --dataloader-num-workers 2
```

Expected: immutable evidence and log under the repository `results/` and `logs/` trees, active device `NVIDIA H100 NVL`, three sealed representatives, finite measurements, and `prefix_decoder_share >= 0.30`. If the share is below `0.30`, stop the entire plan and retain `no_cache` as the only backend.

- [ ] **Step 6: Commit only clean task hunks**

```bash
git diff -- src/stage2/efficiency_probe.py tests/stage2/test_efficiency_probe.py
git add src/stage2/efficiency_probe.py tests/stage2/test_efficiency_probe.py
git diff --cached --check
git commit -m "perf(stage2): seal segmented KV prerequisite"
```

---

### Task 2: Implement the standalone segmented cache engine

**Files:**
- Create: `transformers/src/transformers/models/qwen2_5_vl/stage2_kv_rollout.py`
- Create: `tests/stage2/test_kv_rollout.py`
- Read without modifying: `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`

**Interfaces:**
- Consumes: the same `language_model`, base embeddings, attention mask, three-axis positions, `LatentSpan` tuple, and diagnostics flags as `LatentRolloutEngine.rollout`.
- Produces: `LatentRolloutResult` with final embeddings, graph-attached latent states, prefix lengths, and optional diagnostics; candidate-only `SegmentTrace` observations remain internal/test-only and never expose a cache in the result.
- Private helpers are local to the candidate module: `_ordered_latent_positions(latent_spans)`, `_assemble_segment(base, replacements, start, stop)`, `_assemble_complete_inputs(base, replacements)`, `_validate_dynamic_cache(cache, *, expected_layers, expected_length)`, and `_build_segmented_diagnostics(states, block_count, *, batch_size, device)`.

- [ ] **Step 1: Write failing cursor, delta, mask, cache, and graph tests**

Use a fake cache-aware decoder that records independent call arguments. Cover zero spans, one block, consecutive slots, two blocks separated by ordinary tokens, three-axis positions, non-all-one masks, batch size two, empty initial prefix, cache replacement, wrong cache length, detached cache tensors, and `retain_latent_grads`.

```python
def test_segmented_kv_advances_cursor_to_slot_not_after_slot():
    result = DifferentiableSegmentedKVRolloutEngine(cache_factory=RecordingCache).rollout(
        language_model=decoder,
        base_inputs_embeds=base,
        attention_mask=torch.ones(1, 11, dtype=torch.long),
        full_position_ids=positions,
        latent_spans=(LatentSpan(1, (2, 3), 4), LatentSpan(6, (7, 8), 9)),
    )
    assert decoder.delta_lengths == [2, 1, 4, 1]
    assert decoder.cache_positions == [(0, 1), (2,), (3, 4, 5, 6), (7,)]
    assert decoder.cache_lengths == [2, 3, 7, 8]
    assert result.prefix_lengths == (2, 3, 7, 8)
```

Expected deltas must be constructed directly in the test, never by calling a candidate helper.

- [ ] **Step 2: Run the new engine tests and observe import failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q tests/stage2/test_kv_rollout.py
```

Expected: FAIL because `stage2_kv_rollout.py` does not exist.

- [ ] **Step 3: Implement the minimal engine**

Implement this public shape:

```python
class DifferentiableSegmentedKVRolloutEngine:
    def __init__(self, *, cache_factory=DynamicCache):
        self._cache_factory = cache_factory

    def rollout(
        self,
        *,
        language_model,
        base_inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        full_position_ids: torch.Tensor,
        latent_spans: tuple[LatentSpan, ...],
        retain_latent_grads: bool = False,
        collect_diagnostics: bool = False,
    ) -> LatentRolloutResult:
        replacements: list[tuple[int, torch.Tensor]] = []
        states: list[torch.Tensor] = []
        prefix_lengths: list[int] = []
        positions = _ordered_latent_positions(latent_spans)
        if not positions:
            diagnostics = (
                _build_segmented_diagnostics([], 0, batch_size=1, device=base_inputs_embeds.device)
                if collect_diagnostics
                else None
            )
            return LatentRolloutResult(base_inputs_embeds, (), (), diagnostics)
        cache = self._cache_factory()
        cursor = 0
        for position in positions:
            delta = _assemble_segment(base_inputs_embeds, replacements, cursor, position)
            outputs = language_model(
                inputs_embeds=delta,
                past_key_values=cache,
                use_cache=True,
                attention_mask=attention_mask[:, :position],
                position_ids=full_position_ids[..., cursor:position],
                cache_position=torch.arange(cursor, position, device=delta.device),
                return_dict=True,
            )
            if outputs.past_key_values is not cache:
                raise RuntimeError("segmented KV decoder replaced its DynamicCache")
            _validate_dynamic_cache(
                cache,
                expected_layers=language_model.func.config.num_hidden_layers,
                expected_length=position,
            )
            latent = outputs.last_hidden_state[:, -1, :]
            if retain_latent_grads:
                latent.retain_grad()
            replacements.append((position, latent))
            states.append(latent)
            prefix_lengths.append(position)
            cursor = position
        final = _assemble_complete_inputs(base_inputs_embeds, replacements)
        diagnostics = (
            _build_segmented_diagnostics(
                states,
                len(latent_spans),
                batch_size=base_inputs_embeds.shape[0],
                device=base_inputs_embeds.device,
            )
            if collect_diagnostics
            else None
        )
        return LatentRolloutResult(
            final, tuple(states), tuple(prefix_lengths), diagnostics
        )
```

For each ordered position `p`, build `[cursor, p)` out of immutable base slices plus replacements, pass `attention_mask[:, :p]`, `full_position_ids[..., cursor:p]`, and `torch.arange(cursor, p, device=delta.device)`, require returned cache identity and exact per-layer sequence length, then set `cursor = p`. Inspect `DynamicCache.key_cache`, `value_cache`, and `get_seq_length()`: after each call their layer count equals `num_hidden_layers`, every key/value sequence dimension equals `p`, and graph-enabled cache tensors have `requires_grad` plus a non-null `grad_fn`. Reject empty deltas, padding masks, detached latent/cache tensors, layer-count changes, and shape mismatches. A zero-span sample returns the original assembled value without allocating a cache.

- [ ] **Step 4: Run engine and oracle regressions**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_kv_rollout.py tests/stage2/test_rollout.py
```

Expected: PASS and `git diff -- transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py` remains empty relative to the task baseline.

- [ ] **Step 5: Commit the isolated new engine**

```bash
git add transformers/src/transformers/models/qwen2_5_vl/stage2_kv_rollout.py \
  tests/stage2/test_kv_rollout.py
git diff --cached --check
git commit -m "feat(stage2): add differentiable segmented KV rollout"
```

---

### Task 3: Add deterministic active-model preflight and training-only routing

**Files:**
- Modify: `src/stage2/peft.py`
- Modify: `src/main.py`
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Modify: `tests/stage2/test_peft.py`
- Modify: `tests/stage2/test_qwen_forward.py`
- Modify: `tests/stage2/test_inference.py`
- Modify: `tests/stage2/test_eval_integration.py`

**Interfaces:**
- Produces: `validate_segmented_kv_training_model(model, *, gradient_checkpointing: bool) -> None` and `install_stage2_training_rollout_engine(model, backend: str) -> None`.
- Runtime routing: `stage2_training_rollout_engine` is `None` by default; `_forward_stage2` uses it only when `self.training`, otherwise `stage2_rollout_engine`.

- [ ] **Step 1: Write failing preflight and routing tests**

Parameterize rejection over active `nn.Dropout(p>0)`, adapter dropout, attention dropout, stochastic router marker, enabled gradient checkpointing, batch size other than one, and nontraining use. Assert that evaluation on a model with an installed candidate calls only the oracle.

```python
def test_eval_uses_oracle_when_training_candidate_is_installed():
    model = tiny_stage2_qwen()
    model.stage2_rollout_engine = oracle = RecordingRolloutEngine()
    model.stage2_training_rollout_engine = candidate = RecordingRolloutEngine()
    model.eval()(input_ids=ids, attention_mask=torch.ones_like(ids))
    assert oracle.calls == 1
    assert candidate.calls == 0

def test_segmented_kv_preflight_inspects_active_wrapped_dropouts():
    tuning = Stage2TuningConfig(
        method="dora",
        r=2,
        alpha=4,
        dropout=0.05,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), tuning)
    with pytest.raises(ValueError, match="dropout"):
        validate_segmented_kv_training_model(wrapped, gradient_checkpointing=False)
```

- [ ] **Step 2: Run the focused tests and observe failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_peft.py tests/stage2/test_qwen_forward.py \
  tests/stage2/test_inference.py tests/stage2/test_eval_integration.py \
  -k 'segmented or training_candidate or oracle'
```

Expected: FAIL because preflight, candidate slot, and routing are absent.

- [ ] **Step 3: Implement active-module inspection and candidate installation**

Resolve through DDP/PEFT wrappers with `resolve_stage2_qwen_model`. Inspect every reachable `torch.nn.modules.dropout._DropoutNd`, each active PEFT configuration's `lora_dropout`, decoder/config `attention_dropout`, gradient-checkpoint flags, and any configured expert/router module. The current supported Qwen decoder must have no positive `num_experts`, router configuration, or active module whose qualified name ends in `.router`. Never mutate dropout or checkpointing. After successful composition/preflight, install a fresh `DifferentiableSegmentedKVRolloutEngine`; for `no_cache`, leave the candidate slot `None`.

In Qwen initialization and routing, make only these semantic changes:

```python
self.stage2_rollout_engine = LatentRolloutEngine()
self.stage2_training_rollout_engine = None

rollout_engine = (
    self.stage2_training_rollout_engine
    if self.training and self.stage2_training_rollout_engine is not None
    else self.stage2_rollout_engine
)
rollout = rollout_engine.rollout(
    language_model=language_model,
    base_inputs_embeds=base_inputs_embeds,
    attention_mask=attention_mask,
    full_position_ids=full_position_ids,
    latent_spans=latent_spans,
    retain_latent_grads=retain_stage2_latent_grads,
    collect_diagnostics=collect_stage2_diagnostics,
)
```

Keep the existing final `language_model(inputs_embeds=rollout.final_inputs_embeds, attention_mask=attention_mask, position_ids=full_position_ids, use_cache=False, return_dict=True)` behavior unchanged, including its output flags.

- [ ] **Step 4: Run routing, forward, inference, and generation regressions**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_peft.py tests/stage2/test_qwen_forward.py \
  tests/stage2/test_inference.py tests/stage2/test_eval_integration.py \
  tests/stage2/test_generation_regression.py
```

Expected: PASS; ordinary generation and Stage 1 cache behavior remain unchanged.

- [ ] **Step 5: Preserve pre-existing dirty hunks**

Inspect the five overlapping dirty files named in Global Constraints. Commit only clean `src/main.py`, model, inference, and evaluation hunks if they are isolated; leave `peft.py` and `test_peft.py` unstaged when isolation would include prior work.

---

### Task 4: Add the explicit selector and fail-closed launcher contract

**Files:**
- Modify: `src/main.py`
- Modify: `run_stage2_training.sh`
- Modify: `tests/stage2/test_launcher.py`
- Modify: `tests/stage2/test_routing.py`
- Modify: `tests/stage2/test_comt_readiness.py`
- Modify: `tests/stage2/integration/test_distributed_contract.py`

**Interfaces:**
- Consumes: CLI `--stage2-recurrence-backend {no_cache,segmented_kv}` and environment `STAGE2_RECURRENCE_BACKEND`.
- Produces: normalized backend plus algorithm version `no-cache-v1` or `segmented-kv-v1`; dry run prints selector, algorithm, dropout values, checkpointing, and attention backend without loading a model or creating artifacts.

- [ ] **Step 1: Add failing CLI and launcher tests**

Assert omitted/default is `no_cache`, unknown values fail before model loading, dry run includes both `--stage2-recurrence-backend no_cache` and `no-cache-v1`, `segmented_kv` rejects nonzero adapter dropout and checkpointing, and resume never silently changes backend.

```python
def test_launcher_defaults_to_no_cache():
    tokens = dry_run_tokens(run_launcher())
    assert option_value(tokens, "--stage2-recurrence-backend") == "no_cache"

def test_launcher_rejects_kv_with_dropout():
    result = run_launcher(env_overrides={
        "STAGE2_RECURRENCE_BACKEND": "segmented_kv",
        "LORA_DROPOUT": "0.05",
    })
    assert result.returncode == 2
    assert "requires LORA_DROPOUT=0.0" in result.stderr
```

- [ ] **Step 2: Run launcher tests and observe failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_launcher.py tests/stage2/test_routing.py \
  tests/stage2/test_comt_readiness.py \
  tests/stage2/integration/test_distributed_contract.py -k 'launcher or recurrence'
```

- [ ] **Step 3: Implement selector parsing inside hashed runtime code**

Parse and normalize the selector in `src/main.py` before model loading, so its behavior is covered by the mandated `src/main.py` runtime digest. Add shell validation and the managed option to `run_stage2_training.sh`. For dry run, print a stable summary line to stderr and the launch argv to stdout; do not call path-creation helpers.

Reject candidate selection unless all statically knowable conditions pass: Stage 2, rank-local batch one, `LORA_DROPOUT=0.0`, `GRADIENT_CHECKPOINTING=false`, attention backend `eager` or `flash_attention_2` (never `sdpa` in version 1), no DeepSpeed/ZeRO/FSDP selector, and fresh or candidate-compatible resume intent. Active-model conditions remain Task 3's responsibility.

- [ ] **Step 4: Run launcher suite and Bash syntax check**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_launcher.py tests/stage2/test_routing.py \
  tests/stage2/test_comt_readiness.py tests/stage2/integration/test_distributed_contract.py
bash -n run_stage2_training.sh
```

- [ ] **Step 5: Commit clean selector files only**

```bash
git add src/main.py run_stage2_training.sh tests/stage2/test_launcher.py \
  tests/stage2/test_routing.py tests/stage2/test_comt_readiness.py \
  tests/stage2/integration/test_distributed_contract.py
git diff --cached --check
git commit -m "feat(stage2): add recurrence backend selector"
```

---

### Task 5: Implement strict recurrence identity and artifact-tree hashing

**Files:**
- Modify: `src/stage2/checkpointing.py`
- Create: `tests/stage2/test_recurrence_checkpointing.py`

**Interfaces:**
- Produces: `Stage2RecurrenceIdentity`, `build_stage2_recurrence_identity(repository_root, backend, model_identity, execution, data_identity, batch_geometry)`, `read_stage2_recurrence_sidecar(directory)`, `write_stage2_recurrence_sidecar(directory, identity, *, global_step)`, and `validate_stage2_recurrence_resume(checkpoint, expected, *, trainer_global_step)`.
- Sidecar: exact filename `stage2-recurrence.json`, schema version `1`, exact field sets and canonical encoding from specification Section 7.

- [ ] **Step 1: Write strict parsing and identity tests**

Cover exact top-level and nested key sets, duplicate JSON keys, Boolean-as-integer, NaN/Infinity, noncanonical commit/digest strings, symlinks, artifact paths escaping their roots, tracked runtime file mismatch, untracked import shadowing, batch multiplication mismatch, global-step mismatch, legacy no-cache allowance, and legacy-to-KV rejection.

```python
def test_recurrence_sidecar_is_compact_canonical_and_immutable(tmp_path):
    path = write_stage2_recurrence_sidecar(tmp_path, identity(), global_step=3)
    encoded = path.read_bytes()
    assert encoded.endswith(b"\n")
    assert b'": ' not in encoded
    assert read_stage2_recurrence_sidecar(tmp_path).global_step == 3
    with pytest.raises(FileExistsError):
        write_stage2_recurrence_sidecar(tmp_path, identity(backend="segmented_kv"), global_step=3)
```

- [ ] **Step 2: Run the new tests and observe import failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_recurrence_checkpointing.py
```

- [ ] **Step 3: Implement exact identity types and canonical I/O**

Use `json.loads(raw_text, object_pairs_hook=_reject_duplicate_keys)`, exact `type(value) is int/bool`, `math.isfinite`, compact sorted UTF-8 JSON with one newline, and exclusive publication. Compute an artifact-tree digest from canonical newline-delimited records with exact keys `path`, `size`, and `sha256`; reject symlinks and resolved paths outside the root.

Hash exactly the 16 runtime paths embedded near the top of this plan. Require each path to be tracked at the recorded commit and clean on both nodes. For each imported runtime module, resolve `importlib.util.find_spec(module_name).origin`, require it to equal the canonical expected path, and require `git ls-files --error-unmatch` to succeed; this is the fail-closed untracked-shadow check.

- [ ] **Step 4: Run recurrence and existing checkpoint tests**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_recurrence_checkpointing.py \
  tests/stage2/test_checkpointing.py tests/stage2/test_peft_checkpointing.py
```

- [ ] **Step 5: Do not stage the pre-dirty checkpoint module wholesale**

Record `git diff -- src/stage2/checkpointing.py` and commit only the new clean test file. Leave the implementation unstaged unless its hunks can be isolated without including the pre-existing PEFT compatibility fix.

---

### Task 6: Bind recurrence identity to composition, saves, and resume

**Files:**
- Modify: `src/main.py`
- Modify: `src/stage2/trainer.py`
- Modify: `src/stage2/checkpointing.py`
- Modify: `tests/stage2/test_routing.py`
- Modify: `tests/stage2/test_peft_checkpointing.py`
- Modify: `tests/stage2/integration/tiny_stage2_train.py`
- Modify: `tests/stage2/integration/test_distributed_contract.py`

**Interfaces:**
- `Stage2TrainingComponents` carries one immutable recurrence identity into `Stage2Trainer`.
- `Stage2Trainer` accepts the keyword-only `stage2_recurrence_identity: Stage2RecurrenceIdentity` in addition to its existing constructor arguments, validates resume before training collectives, and publishes the sidecar at every numbered checkpoint and final artifact.

- [ ] **Step 1: Add failing save/resume lifecycle tests**

Assert numbered and final artifacts contain `stage2-recurrence.json`; sidecar `global_step` matches directory and `TrainerState`; all ranks agree before the designated writer publishes; a present malformed/mismatched sidecar fails before collectives; a missing sidecar resumes only under explicit `no_cache`; no cache tensor appears in any saved file.

```python
def test_numbered_checkpoint_writes_recurrence_identity(tmp_path, monkeypatch):
    trainer = recurrence_trainer(tmp_path, backend="segmented_kv")
    trainer.state.global_step = 3
    trainer._save_checkpoint(trainer.model, trial=None)
    sidecar = read_stage2_recurrence_sidecar(tmp_path / "checkpoint-3")
    assert sidecar.backend == "segmented_kv"
    assert sidecar.global_step == 3
```

- [ ] **Step 2: Run lifecycle tests and observe failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_routing.py tests/stage2/test_peft_checkpointing.py \
  tests/stage2/integration/test_distributed_contract.py -k 'recurrence or legacy'
```

- [ ] **Step 3: Integrate identity without replacing existing save machinery**

Build identity after model/PEFT composition and all input/runtime identities exist, but before distributed preparation. On `_save_checkpoint` and `save_model`, first make all ranks validate the same canonical digest, then let the designated writer publish the common sidecar in addition to the existing model/adapter, sparse rows, sampler, runtime, processor, optimizer, scheduler, RNG, and Trainer state.

Legacy resume semantics are exact: missing sidecar plus explicit `no_cache` may proceed through every existing compatibility check and writes a full no-cache sidecar at the next successful save; missing sidecar plus `segmented_kv` fails.

- [ ] **Step 4: Run checkpoint and tiny distributed suites**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_checkpointing.py tests/stage2/test_peft_checkpointing.py \
  tests/stage2/test_routing.py tests/stage2/integration/test_distributed_contract.py
```

- [ ] **Step 5: Preserve all prior dirty trainer/checkpoint/test hunks**

Do not commit overlapping files wholesale. If the task is executed by a subagent, return the exact changed-line summary and test output to the controller for controlled integration.

---

### Task 7: Add sampler schedule-v2 logical frames and benchmark sealing

**Files:**
- Modify: `src/stage2/sampler.py`
- Create: `tests/stage2/test_kv_schedule.py`
- Modify: `src/stage2/trainer.py`

**Interfaces:**
- Produces: immutable schedule-v2 logical frame records, `sampler_schedule_sha256`, ordered occurrence digests, and `select_kv_benchmark_frames(schedule) -> KVBenchmarkSelection`.
- Selection: at least 13 completed frames; cursor from `SHA256(b"stage2-kv-benchmark-v1\0" + schedule_sha256.encode("ascii"))[:8]`, three warm-ups, then ten measured frames.

- [ ] **Step 1: Write failing deterministic schedule tests**

Use a small known schedule to assert byte-for-byte canonical digest, big-endian cursor computation, exact frame slice, occurrence ordering, and failure for fewer than 13 completed frames.

- [ ] **Step 2: Run and observe failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q tests/stage2/test_kv_schedule.py
```

- [ ] **Step 3: Implement versioned frame records without changing membership**

Derive frame records from the existing balanced sampler stream and effective-batch-eight accumulation geometry. Do not reshuffle or pad differently. Hash canonical occurrence records and expose them to neutral Trainer timing hooks; candidate code must not create or alter benchmark frames.

- [ ] **Step 4: Run sampler and schedule tests**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_sampler.py tests/stage2/test_kv_schedule.py
```

- [ ] **Step 5: Commit clean sampler/schedule hunks**

```bash
git add src/stage2/sampler.py tests/stage2/test_kv_schedule.py
git diff --cached --check
git commit -m "feat(stage2): seal KV benchmark schedule"
```

---

### Task 8: Implement neutral rank evidence and the independent validator

**Files:**
- Create: `src/stage2/kv_evidence.py`
- Create: `tests/stage2/test_kv_evidence.py`
- Modify: `src/stage2/trainer.py`
- Modify: `src/artifact_paths.py`

**Interfaces:**
- Trainer writes one immutable `stage2-kv-rank-evidence.json` per backend/global rank.
- `validate_kv_comparison(*, oracle_rank_paths: Sequence[Path], candidate_rank_paths: Sequence[Path], verification_commands: Mapping[str, Sequence[str]], output_root: Path, comparison_id: str, oracle_run_id: str, candidate_run_id: str) -> Path` reads exactly eight rank files, verifies command evidence, calculates metrics/gates, and writes immutable `stage2-kv-comparison.json`.

- [ ] **Step 1: Write failing strict-schema and gate tests**

Cover duplicate keys, extra/missing/wrong-type/non-finite fields, wrong rank coverage, identity or frame mismatch, invalid wall partition, `sync_ms` double counting, incomplete capacity objects, failed verification commands, exact rank-max median math, `0.15` speed boundary, `0.05` memory boundary, and non-overridable conjunction.

```python
def test_validator_derives_acceptance_from_all_gates(tmp_path):
    inputs = write_eight_matching_rank_files(tmp_path, oracle_ms=100.0, candidate_ms=85.0)
    result = read_json(validate_kv_comparison(inputs, verification_commands=passing_commands()))
    assert result["metrics"]["speed_improvement_fraction"] == pytest.approx(0.15)
    assert result["gates"] == {name: True for name in EXPECTED_GATE_NAMES}
    assert result["accepted"] is True
```

- [ ] **Step 2: Run validator tests and observe import failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q tests/stage2/test_kv_evidence.py
```

- [ ] **Step 3: Implement strict readers, neutral timing records, and comparison derivation**

Match every exact field embedded near the top of this plan and its type/range from the specification. Store each verification command argv array, exit code, and complete stdout/stderr SHA256 in command-output records under `results/`; the validator must not import the candidate module and must reject caller-supplied gate or acceptance bits.

Use `T[b, i] = max_rank(step_ms[b, rank, i])`, median over ten `T` values, maximum allocated bytes over ranks/steps, and the exact threshold inequalities. Publish with exclusive-create semantics so an experiment ID cannot be rewritten.

- [ ] **Step 4: Run evidence and artifact-path tests**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_kv_evidence.py tests/test_artifact_paths.py
```

- [ ] **Step 5: Commit the independent validator**

```bash
git add src/stage2/kv_evidence.py tests/stage2/test_kv_evidence.py \
  src/artifact_paths.py
git diff --cached --check
git commit -m "feat(stage2): validate segmented KV evidence"
```

---

### Task 9: Pass the exact FP32 first nail and CPU regression matrix

**Files:**
- Create: `tests/stage2/test_kv_correctness.py`
- Modify only if a causal failure is found: `transformers/src/transformers/models/qwen2_5_vl/stage2_kv_rollout.py`

**Interfaces:**
- Consumes: cloned tiny-Qwen state bytes and a separate hand-unrolled toy causal decoder.
- Produces: direct forward/backward/optimizer parity evidence for full, LoRA, and DoRA boundaries; proof of final `use_cache=False`.

- [ ] **Step 1: Build the exact fixture and independent toy oracle**

Use seed `1234`, vocab/hidden/intermediate `128/32/64`, one decoder layer, `4` attention heads, `2` KV heads, eager FP32, dropout zero, untied embeddings, latent pad/start/end IDs `90/91/92`, image/vision-start/video IDs `80/81/82`, input IDs `[1,91,90,90,92,2,91,90,90,92,3]`, latent positions `[2,3,7,8]`, eleven-one attention mask, labels `[-100,-100,-100,-100,92,-100,-100,-100,-100,92,3]`, mRoPE `{"type":"mrope","mrope_section":[2,1,1]}`, and AdamW `lr=1e-3`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`. The unused vision config is depth `1`, hidden/intermediate `32/64`, heads `4`, patch/spatial-merge/temporal-patch sizes `2/1/1`, window `2`, output hidden `32`, and full-attention block indices `[0]`; pass no pixels.

The toy hidden size is `4`; its causal cumulative-prefix equations may use only ordinary tensor operations and must not import Qwen attention, `DynamicCache`, or candidate cursor helpers.

- [ ] **Step 2: Compare the complete required boundary**

For full, LoRA rank 2/alpha 4/dropout 0, and DoRA rank 2/alpha 4/dropout 0, assert `atol=1e-6`, `rtol=1e-5` for every latent, final embedding, supervised logit, loss, retained latent gradient, input-embedding gradient, every trainable parameter gradient, and one optimizer update. Assert call counts, deltas `[2, 1, 4, 1]`, cache positions, lengths, and final no-cache call directly.

- [ ] **Step 3: Run the first nail alone**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_kv_correctness.py
```

Expected: PASS. Any mismatch, missing gradient, detachment, cursor error, or final-pass cache use stops all CUDA work.

- [ ] **Step 4: Run the focused CPU Stage 2 matrix once**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_rollout.py tests/stage2/test_kv_rollout.py \
  tests/stage2/test_kv_correctness.py tests/stage2/test_qwen_forward.py \
  tests/stage2/test_routing.py tests/stage2/test_launcher.py \
  tests/stage2/test_recurrence_checkpointing.py tests/stage2/test_kv_schedule.py \
  tests/stage2/test_kv_evidence.py
```

- [ ] **Step 5: Commit the durable first-nail tests**

```bash
git add tests/stage2/test_kv_correctness.py
git diff --cached --check
git commit -m "test(stage2): prove segmented KV FP32 parity"
```

---

### Task 10: Pass the single-H100 BF16 and capacity gates

**Files:**
- Modify: `tests/stage2/integration/test_cuda_backends.py`
- Generate: `results/zero-stage-compatibility/stage2-kv-single-h100-bf16-20260902-r1/`
- Generate: `logs/zero-stage-compatibility/stage2-kv-single-h100-bf16-20260902-r1/`

**Interfaces:**
- Produces: observed BF16 FlashAttention 2 parity for single-token and multi-token deltas plus p95 and maximum-work capacity objects.

- [ ] **Step 1: Add the H100-only candidate/oracle parity test**

Clone exact state bytes, train both backends, and compare loss/latents/supervised logits with `atol=rtol=2e-2`; compare every nonzero trainable gradient with cosine similarity `>=0.999` and relative L2 `<=0.02`; compare zero-reference gradients with `atol=2e-2`, `rtol=0`; require finite values and identical non-`None` gradient/frozen sets.

- [ ] **Step 2: Run the CUDA integration test once on GPU 0**

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=transformers/src:src \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/integration/test_cuda_backends.py -k segmented_kv
```

Expected: PASS on `NVIDIA H100 NVL`, including observed bottom-right causal alignment for query length smaller than key length.

- [ ] **Step 3: Run p95 then maximum-work admission probes**

Use the representatives sealed in Task 1, `CUDA_VISIBLE_DEVICES=0`, BF16, FlashAttention 2, dropout zero, checkpointing disabled, and fresh run IDs. Each evidence object must exist even on OOM/failure and records `completed`, `finite`, peak allocated bytes, raw index, and recurrence work.

- [ ] **Step 4: Stop on any capacity failure**

Proceed only if both probes complete with finite loss/gradients. Do not reduce sequence length, latent size, rank-local batch, or effective batch.

- [ ] **Step 5: Commit the CUDA test**

```bash
git add tests/stage2/integration/test_cuda_backends.py
git diff --cached --check
git commit -m "test(stage2): gate segmented KV on H100"
```

---

### Task 11: Run the four-rank DDP save/resume and matched adoption benchmark

**Files:**
- Modify only for a demonstrated causal defect: files covered by Tasks 3-8
- Generate: `checkpoints/zero-stage-compatibility/stage2-kv-oracle-ddp-20260902-r1/`
- Generate: `checkpoints/zero-stage-compatibility/stage2-kv-candidate-ddp-20260902-r1/`
- Generate: matching `results/zero-stage-compatibility/` and `logs/zero-stage-compatibility/` directories

**Interfaces:**
- Consumes: two H100 NVL GPUs on each of `172.16.11.41` rank 0 and `172.16.11.40` rank 1, `bond0.11`, four global processes, per-device batch one, accumulation two, and the same sealed schedule frames/weights.
- Produces: eight immutable rank files plus DDP/save/resume command evidence for Task 8's validator.

- [ ] **Step 1: Dry-run both node commands before rendezvous**

Export `STAGE1_CHECKPOINT`, `/root/workspace/ILVR/data/TRAIN.jsonl` as `DATA_PATH`, `/root/workspace/ILVR/data` as `IMAGE_ROOT`, backend-specific `OUTPUT_DIR`, `TUNING_METHOD=dora`, `LORA_R=32`, `LORA_ALPHA=64`, `LORA_DROPOUT=0.0`, the seven canonical projection suffixes, `LATENT_SIZE=8`, `SUPERVISE_LATENT_BOUNDARIES=true`, `ATTENTION_IMPLEMENTATION=flash_attention_2`, `GRADIENT_CHECKPOINTING=false`, `GRADIENT_ACCUMULATION_STEPS=2`, `STAGE2_DIAGNOSTICS_INTERVAL=0`, `MASTER_ADDR=172.16.11.41`, `MASTER_PORT=29520`, `NCCL_SOCKET_IFNAME='=bond0.11'`, `GLOO_SOCKET_IFNAME=bond0.11`, `NCCL_SOCKET_FAMILY=AF_INET`, and `CUDA_VISIBLE_DEVICES=0,1` on both nodes. Run `STAGE2_DRY_RUN=true ./run_stage2_training.sh` and compare argv except backend/run-ID outputs.

- [ ] **Step 2: Run the candidate DDP correctness/save-resume gate**

Launch `.41` with `STAGE2_DISTRIBUTED_MODE=ddp STAGE2_NUM_MACHINES=2 STAGE2_NUM_PROCESSES=4 STAGE2_MACHINE_RANK=0 STAGE2_MAIN_PROCESS_IP=172.16.11.41 STAGE2_MAIN_PROCESS_PORT=29520`; launch `.40` with only `STAGE2_MACHINE_RANK=1` changed. Require identical recurrence identity, optimizer step, occurrence records, finite loss/gradients, synchronization only at accumulation boundaries, rank-zero checkpoint/adapter/sidecar publication, and a same-topology one-step resume.

- [ ] **Step 3: Run negative resume probes**

Before training collectives, verify backend mismatch and malformed present sidecar fail; verify missing-sidecar legacy resume succeeds only with explicit `no_cache` and legacy-to-KV fails. Do not attempt cross-backend optimizer resume.

- [ ] **Step 4: Run matched oracle and candidate timings**

For each backend, start from identical initial weights and the same presealed cursor/frames. Execute three warm-up optimizer steps and ten measured steps, synchronizing device work at boundaries. Neutral Trainer hooks record step and component times, call/token counts, peak allocated/reserved memory, throughput, loss, gradient norm, and cache lengths/bytes.

- [ ] **Step 5: Derive the immutable comparison**

Run `src/stage2/kv_evidence.py` over the exact eight rank files and fixed correctness/save-resume commands. Accept only if all nine named gates are true, median rank-max speed improves by at least `0.15`, and peak allocated memory regression is at most `0.05`. A false gate is the final result for those run IDs; do not alter thresholds.

---

### Task 12: Verify the implementation and record the adoption decision

**Files:**
- Modify: `README.md`
- Read: `documents/stage-2-kv-cache-20260902.md`
- Read: immutable result from Task 11

**Interfaces:**
- Produces: one operator-facing description of the selector, compatibility boundary, artifact paths, and observed accepted/rejected decision without broadening production authorization.

- [ ] **Step 1: Document only observed behavior**

Add launcher examples for explicit `no_cache` and eligible `segmented_kv`, the dropout/checkpointing constraints, sidecar name, evaluation/inference oracle routing, failure behavior, and the exact immutable comparison path. State `segmented_kv` is production-authorized only when that comparison has `accepted: true`.

- [ ] **Step 2: Run the full Stage 2 suite once**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q tests/stage2
```

Expected: PASS with complete output read once.

- [ ] **Step 3: Run claim-specific syntax/import checks once**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/python -m py_compile \
  src/main.py src/stage2/checkpointing.py src/stage2/efficiency_probe.py \
  src/stage2/kv_evidence.py src/stage2/peft.py src/stage2/sampler.py \
  src/stage2/trainer.py \
  transformers/src/transformers/models/qwen2_5_vl/stage2_kv_rollout.py \
  transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
bash -n run_stage2_training.sh
```

Expected: both commands exit zero.

- [ ] **Step 4: Verify storage and cache non-persistence claims**

```bash
test -L checkpoints
! rg --files results | rg '\.(safetensors|bin|pt|pth|ckpt)$'
! rg -n 'past_key_values|key_cache|value_cache' \
  checkpoints/zero-stage-compatibility/*/stage2-recurrence.json \
  results/zero-stage-compatibility/*/stage2-kv-*.json
```

Expected: checkpoint root remains a symlink; no model-bearing result files or serialized KV fields are found.

- [ ] **Step 5: Review requirements against the spec**

Map the non-negotiable semantics, deterministic compatibility boundary, training-only selector, recurrence identity, prerequisite, correctness, distributed/save-resume, capacity, speed, memory, and fail-closed requirements to passing tests and immutable evidence. Report verified scope separately for FP32 CPU, single-H100 BF16, four-rank DDP, capacity, save/resume, and performance; do not generalize a partial gate into production authorization.

- [ ] **Step 6: Commit only isolated documentation changes**

`README.md` is pre-dirty. Leave the new documentation unstaged unless its hunks can be isolated from the existing user edits. The human owns merge, PR, and keep-as-is decisions.
