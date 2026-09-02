# Stage 2 No-Cache Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove exact, measurable waste from the existing Stage 2 no-cache training path and measure on the active target training server whether a separately reviewed segmented KV experiment may be justified.

**Architecture:** Keep the current no-cache recurrence as the semantic oracle. Add three isolated model/trainer fast paths, replace per-rank image-aware length filtering with one launcher-produced manifest, then profile the combined path. This plan records whether repeated-prefix decoder work remains at least 30% of median end-to-end step time, while schema v1 keeps segmented KV authorization fail-closed pending separate review.

**Tech Stack:** Python 3.11, PyTorch 2.7.1+cu128, local Transformers 4.50.0.dev0, TRL Trainer, PEFT, DeepSpeed, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-09-02-stage2-efficiency-design.md`

## Global Constraints

- Run every Python command through `./venv/ilvr-venv/bin/` with `PYTHONPATH=transformers/src:src`.
- Preserve the exact Stage 2 latent scaffold, recurrence definition, graph attachment, label mask, final full-sequence decoder pass, and shifted cross-entropy objective.
- `no_cache` remains the only production backend in this plan.
- Checkpoint-bearing output stays below `./checkpoints/`; metrics and manifests stay below `./results/`; logs stay below `./logs/`.
- The worktree already contains user changes, including changes in `run_stage2_training.sh` and `src/stage2/trainer.py`. Inspect each diff and stage only task-owned hunks; never stage unrelated changes or `__pycache__` files.
- Each behavior change follows red-green-refactor TDD. Do not weaken the FP32 oracle limits (`atol=1e-6`, `rtol=1e-5`) after seeing results.
- Do not patch PEFT's DoRA implementation, change sampler policy, change batch size, or add retry/recovery infrastructure.

---

### Task 1: Supervised-row LM head and loss-only model contract

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` (`_forward_stage2`, `forward`)
- Modify: `src/stage2/trainer.py` (`compute_loss`)
- Test: `tests/stage2/test_qwen_forward.py`
- Test: `tests/stage2/test_routing.py`

**Interfaces:**
- Consumes: existing Stage 2 `labels` with `-100` masking and `Qwen2_5_VLCausalLMOutputWithPast`.
- Produces: model keyword `stage2_loss_only: bool = False`; when true, `loss` is populated and `logits is None`; all ordinary calls retain full logits.

- [ ] **Step 1: Add failing model tests for loss, head shape, and all trainable gradients**

Parameterize the helper over `full`, `lora`, and `dora` with adapter dropout zero. Clone one tiny model, run the ordinary full-logit path and the loss-only path from identical weights, and compare every named trainable gradient. Record the tensor entering `lm_head` to prove the optimized call contains only supervised shifted rows:

```python
def test_stage2_loss_only_matches_full_loss_and_all_trainable_gradients():
    torch.manual_seed(17)
    full = tiny_stage2_qwen().train()
    selective = deepcopy(full).train()
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2, 3]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels[input_ids == 90] = -100
    seen = []
    handle = selective.lm_head.register_forward_pre_hook(
        lambda _module, args: seen.append(tuple(args[0].shape))
    )
    full_output = full(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)
    selective_output = selective(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        stage2_loss_only=True,
    )
    handle.remove()
    full_output.loss.backward()
    selective_output.loss.backward()
    active = int((labels[:, 1:] != -100).sum())
    assert seen == [(active, selective.config.hidden_size)]
    assert selective_output.logits is None
    torch.testing.assert_close(selective_output.loss, full_output.loss, atol=1e-6, rtol=1e-5)
    for (full_name, full_parameter), (fast_name, fast_parameter) in zip(
        full.named_parameters(), selective.named_parameters(), strict=True
    ):
        assert full_name == fast_name
        if full_parameter.requires_grad:
            torch.testing.assert_close(
                fast_parameter.grad, full_parameter.grad, atol=1e-6, rtol=1e-5
            )
```

Also test that `stage2_loss_only=True` without labels raises `ValueError`, and that the default call still returns `[1, sequence, vocab]` logits.

- [ ] **Step 2: Run the new model tests and observe the missing keyword failure**

Run:

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_qwen_forward.py -k 'loss_only'
```

Expected: FAIL because `forward()` does not accept `stage2_loss_only`.

- [ ] **Step 3: Implement the minimal supervised-row loss helper**

Thread `stage2_loss_only` through `forward()` and `_forward_stage2()`. In `_forward_stage2`, reject missing labels and compute the selective loss without creating full logits:

```python
if stage2_loss_only and labels is None:
    raise ValueError("Stage 2 loss-only forward requires labels")

if stage2_loss_only:
    shift_labels = labels[..., 1:].contiguous().view(-1).to(
        final_outputs.last_hidden_state.device
    )
    active = shift_labels != -100
    shifted_hidden = final_outputs.last_hidden_state[..., :-1, :].contiguous().view(
        -1, final_outputs.last_hidden_state.shape[-1]
    )
    active_logits = self.lm_head(shifted_hidden[active]).float()
    loss = CrossEntropyLoss()(active_logits, shift_labels[active])
    logits = None
else:
    logits = self.lm_head(final_outputs.last_hidden_state)
    loss = None
    if labels is not None:
        shift_logits = logits.float()[..., :-1, :].contiguous().view(
            -1, self.config.vocab_size
        )
        shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
        loss = CrossEntropyLoss()(shift_logits, shift_labels)
```

Reject an all-masked label batch with a clear `ValueError` instead of passing an empty tensor to cross-entropy. Leave the tuple-return path unchanged by rejecting `stage2_loss_only and not return_dict`.

- [ ] **Step 4: Make the trainer request loss-only output only when outputs are not requested**

In `Stage2Trainer.compute_loss`, call:

```python
outputs = model(
    **inputs,
    stage2_loss_only=not return_outputs,
    collect_stage2_diagnostics=True,
)
```

Add a routing test with a recording fake model proving `return_outputs=False` passes true and `return_outputs=True` passes false.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_qwen_forward.py tests/stage2/test_routing.py
```

Expected: PASS.

- [ ] **Step 6: Commit only Task 1 hunks**

```bash
git diff -- transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  src/stage2/trainer.py tests/stage2/test_qwen_forward.py tests/stage2/test_routing.py
git add -p transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  src/stage2/trainer.py tests/stage2/test_qwen_forward.py tests/stage2/test_routing.py
git commit -m "perf(stage2): project only supervised loss rows"
```

---

### Task 2: Opt-in diagnostics scheduling

**Files:**
- Modify: `src/utils_deepseed.py` (`parse_args`)
- Modify: `src/main.py` (`run_stage2`)
- Modify: `src/stage2/trainer.py` (`__init__`, `compute_loss`)
- Modify: `run_stage2_training.sh`
- Test: `tests/stage2/test_routing.py`
- Test: `tests/stage2/test_comt_readiness.py`

**Interfaces:**
- Consumes: `stage2_diagnostics_interval: int` from CLI/launcher.
- Produces: `Stage2Trainer(..., stage2_diagnostics_interval: int = 0)` and deterministic microbatch sampling; interval zero performs no diagnostic tensor work.

- [ ] **Step 1: Add failing CLI and trainer scheduling tests**

Test `parse_args(["--stage", "stage2"])` gives zero and negative values fail. Use a recording model to call `compute_loss` five times with interval two and assert `collect_stage2_diagnostics` is `[True, False, True, False, True]`. Assert interval zero yields only false.

```python
@pytest.mark.parametrize("interval", [-1, -2])
def test_stage2_rejects_negative_diagnostics_interval(interval):
    with pytest.raises(ValueError, match="diagnostics interval"):
        parse_args(["--stage", "stage2", "--stage2_diagnostics_interval", str(interval)])
```

- [ ] **Step 2: Run the scheduling tests and observe failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_routing.py tests/stage2/test_comt_readiness.py \
  -k 'diagnostics_interval'
```

Expected: FAIL because the argument and trainer field do not exist.

- [ ] **Step 3: Implement the counter without checkpoint semantics**

Add the parser argument and validation:

```python
parser.add_argument("--stage2_diagnostics_interval", type=int, default=0)
if args.stage2_diagnostics_interval < 0:
    raise ValueError("Stage 2 diagnostics interval must be non-negative")
```

Initialize in `Stage2Trainer.__init__`:

```python
self.stage2_diagnostics_interval = stage2_diagnostics_interval
self._stage2_microbatch_index = 0
```

At the beginning of `compute_loss`:

```python
interval = self.stage2_diagnostics_interval
collect_diagnostics = interval > 0 and self._stage2_microbatch_index % interval == 0
self._stage2_microbatch_index += 1
```

The counter is observability-only and is intentionally absent from checkpoints.

- [ ] **Step 4: Wire the launcher**

Add `STAGE2_DIAGNOSTICS_INTERVAL=${STAGE2_DIAGNOSTICS_INTERVAL:-0}`, validate it as a non-negative integer, own `--stage2_diagnostics_interval`, and pass it to `src/main.py`. Pass the parsed value from `run_stage2()` into `Stage2Trainer`.

- [ ] **Step 5: Run focused tests and launcher dry run**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_routing.py tests/stage2/test_comt_readiness.py
STAGE2_DRY_RUN=true STAGE1_CHECKPOINT=/tmp/stage1 DATA_PATH=/tmp/data.jsonl \
  IMAGE_ROOT=/tmp/images OUTPUT_DIR=/root/workspace/ILVR/checkpoints/zero-stage-compatibility/stage2-diagnostics-dry \
  ./run_stage2_training.sh
```

Expected: tests PASS and dry-run command contains `--stage2_diagnostics_interval 0`.

- [ ] **Step 6: Commit Task 2 hunks**

```bash
git add -p src/utils_deepseed.py src/main.py src/stage2/trainer.py \
  run_stage2_training.sh tests/stage2/test_routing.py tests/stage2/test_comt_readiness.py
git commit -m "perf(stage2): make rollout diagnostics opt in"
```

---

### Task 3: Prefix-sized embedding assembly

**Files:**
- Modify: `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`
- Test: `tests/stage2/test_qwen_forward.py`

**Interfaces:**
- Consumes: immutable `base_inputs_embeds` and ordered `(position, latent_state)` replacements.
- Produces: `_assemble_latent_inputs(base_inputs_embeds, replacements, stop)` and one final complete tensor; no full suffix is copied per latent slot.

- [ ] **Step 1: Add a test-only oracle for the current replacement algorithm**

Copy the current no-cache recurrence into a test helper named `_reference_full_copy_rollout`; keep it in the test file so production code cannot share candidate assembly logic. On a two-block scaffold, compare final embeddings, every latent state, prefix lengths, scalar loss, base embedding gradient, and decoder parameter gradient at FP32 tolerances.

Also record decoder input lengths and assert they remain `[2, 3, 7, 8, 11]` for the existing two-block test.

- [ ] **Step 2: Run the new test against a deliberately missing helper**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_qwen_forward.py -k 'prefix_sized or full_copy_oracle'
```

Expected: FAIL because `_assemble_latent_inputs` is absent.

- [ ] **Step 3: Implement functional prefix assembly**

Replace `_replace_embedding` with:

```python
def _assemble_latent_inputs(
    base: torch.Tensor,
    replacements: list[tuple[int, torch.Tensor]],
    *,
    stop: int,
) -> torch.Tensor:
    pieces = []
    cursor = 0
    for position, state in replacements:
        if position >= stop:
            break
        pieces.append(base[:, cursor:position])
        pieces.append(state.unsqueeze(1))
        cursor = position + 1
    pieces.append(base[:, cursor:stop])
    return torch.cat(pieces, dim=1)
```

In `rollout`, build only `stop=position` before each decoder call, append the newly generated state to `replacements`, and call the helper once with `stop=base_inputs_embeds.shape[1]` for `final_inputs_embeds`. Do not mutate `base_inputs_embeds` in place.

- [ ] **Step 4: Run oracle, PEFT, and checkpointing tests**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_qwen_forward.py \
  -k 'prefix_sized or full_copy_oracle or multiblock or gradient_checkpointing'
```

Expected: PASS with unchanged decoder call count and input lengths.

- [ ] **Step 5: Commit Task 3**

```bash
git add transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py \
  tests/stage2/test_qwen_forward.py
git commit -m "perf(stage2): assemble only required rollout prefixes"
```

---

### Task 4: Immutable Stage 2 length manifest

**Files:**
- Create: `src/stage2/length_manifest.py`
- Modify: `src/stage2/data.py` (reuse existing preparation and length measurement)
- Create: `tests/stage2/test_length_manifest.py`

**Interfaces:**
- Consumes: ordered JSONL, image root, loaded processor, latent size, boundary flag, and maximum length.
- Produces: `Stage2LengthManifest`, `build_stage2_length_manifest(...)`, `write_stage2_length_manifest(path, manifest)`, and `read_stage2_length_manifest(path)`.

- [ ] **Step 1: Add failing schema, identity, and stale-input tests**

Create two tiny PNGs and a two-row JSONL. Tests must assert retained raw indices and lengths, deterministic JSON round-trip, data-order hash changes after row reordering, and image identity changes after dimensions or `mtime_ns` change.

```python
def test_manifest_selects_exact_raw_indices_and_round_trips(tmp_path, processor):
    manifest = build_stage2_length_manifest(
        data_path=tmp_path / "train.jsonl",
        image_root=tmp_path,
        processor=processor,
        processor_source="/models/stage1",
        latent_size=2,
        supervise_latent_boundaries=True,
        max_sequence_length=8,
    )
    assert manifest.retained_indices == (0,)
    path = write_stage2_length_manifest(tmp_path / "manifest.json", manifest)
    assert read_stage2_length_manifest(path) == manifest
```

- [ ] **Step 2: Run tests and observe the missing module failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_length_manifest.py
```

Expected: FAIL because `stage2.length_manifest` does not exist.

- [ ] **Step 3: Implement strict dataclasses and canonical hashing**

Use immutable dataclasses with exact schema fields:

```python
@dataclass(frozen=True)
class Stage2ImageIdentity:
    path: str
    size: int
    mtime_ns: int
    width: int
    height: int

@dataclass(frozen=True)
class Stage2LengthManifest:
    schema_version: int
    data_sha256: str
    processor_sha256: str
    latent_size: int
    supervise_latent_boundaries: bool
    max_sequence_length: int
    images: tuple[Stage2ImageIdentity, ...]
    lengths: tuple[int, ...]
    latent_block_counts: tuple[int, ...]
    retained_indices: tuple[int, ...]
    stats: Stage2LengthStats
```

Hash dataset bytes directly. Build `processor_sha256` from canonical sorted JSON containing processor/image-processor class names and `to_dict()` values, tokenizer class, sorted vocabulary, special-token mappings converted to strings, tokenizer initialization arguments, and chat template. Reject symlink/non-regular manifest files, unknown fields, booleans in integer fields, length/count cardinality mismatch, and retained indices that do not exactly equal `length <= max_sequence_length`.

- [ ] **Step 4: Implement one-pass build and exclusive publication**

Prepare each raw record once, measure its exact untruncated processor length once, collect image `stat()` plus PIL dimensions, and compute `Stage2LengthStats` from the collected lengths. Publish with `path.open("x", encoding="utf-8")`; if the path exists, read it and require exact identity equality instead of overwriting it.

- [ ] **Step 5: Run manifest and existing data tests**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_length_manifest.py tests/stage2/test_data.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/stage2/length_manifest.py tests/stage2/test_length_manifest.py
git add -p src/stage2/data.py
git commit -m "feat(stage2): publish exact length manifests"
```

---

### Task 5: Launcher manifest consumption and explicit data workers

**Files:**
- Modify: `src/stage2/length_manifest.py` (CLI)
- Modify: `src/utils_deepseed.py`
- Modify: `src/main.py`
- Modify: `src/artifact_paths.py`
- Modify: `run_stage2_training.sh`
- Modify: `tests/stage2/test_routing.py`
- Modify: `tests/stage2/test_comt_readiness.py`
- Modify: `tests/test_artifact_paths.py`

**Interfaces:**
- Consumes: `--stage2_length_manifest`, `--stage2_length_manifest_sha256`, `--dataloader_num_workers`.
- Produces: one pre-distributed manifest under the run's canonical result directory and identical retained dataset membership on every rank.

- [ ] **Step 1: Add failing CLI, artifact-path, and dataset-selection tests**

Assert Stage 2 rejects a missing manifest, wrong manifest SHA-256, negative worker count, and a manifest whose data/config fields disagree. Assert `build_stage2_training_components` uses `retained_indices` and never calls `build_prepared_example_length_measure` when a valid manifest is supplied. Assert `SFTConfig.dataloader_num_workers` receives the requested value.

- [ ] **Step 2: Run focused tests and observe failures**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_routing.py tests/stage2/test_comt_readiness.py tests/test_artifact_paths.py \
  -k 'length_manifest or dataloader_num_workers'
```

Expected: FAIL because the arguments and result ownership field are missing.

- [ ] **Step 3: Add strict CLI fields and training consumption**

Add:

```python
parser.add_argument("--stage2_length_manifest")
parser.add_argument("--stage2_length_manifest_sha256")
parser.add_argument("--dataloader_num_workers", type=int, default=0)
```

For Stage 2, require both manifest fields together and a non-negative worker count. In `build_stage2_training_components`, hash the manifest bytes, require the passed digest, strictly read it, validate dataset/config/processor identity, and select:

```python
kept = [prepared[index] for index in manifest.retained_indices]
```

Remove the per-rank image-aware `filter_stage2_examples` call from the manifest path. Keep the old direct-call path only for unit tests or non-launcher callers that omit both manifest arguments.

Set `dataloader_num_workers=args.dataloader_num_workers` in `SFTConfig`; set `dataloader_prefetch_factor=2` only when the worker count is positive.

- [ ] **Step 4: Add the manifest CLI and launcher preflight**

The module CLI accepts the exact fields needed to call `AutoProcessor.from_pretrained`, `build_stage2_length_manifest`, and `write_stage2_length_manifest`. In the launcher resolve:

```bash
result_dir="/root/workspace/ILVR/results/zero-stage-compatibility/$experiment_id"
length_manifest="$result_dir/stage2-length-manifest.json"
processor_source=${RESUME_CHECKPOINT:-$STAGE1_CHECKPOINT}
```

Before `accelerate launch`, validate the result path, create `result_dir` for a fresh run, run the manifest CLI once, compute `sha256sum`, and append both manifest arguments. On resume, require and validate the existing manifest. Add `DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}` and pass it explicitly.

- [ ] **Step 5: Run tests and launcher dry run**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_length_manifest.py tests/stage2/test_data.py \
  tests/stage2/test_routing.py tests/stage2/test_comt_readiness.py tests/test_artifact_paths.py
STAGE2_DRY_RUN=true STAGE1_CHECKPOINT=/tmp/stage1 DATA_PATH=/tmp/data.jsonl \
  IMAGE_ROOT=/tmp/images OUTPUT_DIR=/root/workspace/ILVR/checkpoints/zero-stage-compatibility/stage2-manifest-dry \
  DATALOADER_NUM_WORKERS=2 ./run_stage2_training.sh
```

Expected: tests PASS; dry run owns the manifest options and emits worker count two without creating files.

- [ ] **Step 6: Commit Task 5 hunks**

```bash
git add -p src/stage2/length_manifest.py src/utils_deepseed.py src/main.py \
  src/artifact_paths.py run_stage2_training.sh tests/stage2/test_routing.py \
  tests/stage2/test_comt_readiness.py tests/test_artifact_paths.py
git commit -m "perf(stage2): share one length manifest across ranks"
```

---

### Task 6: Focused active-target-server efficiency probe and hard KV gate

**Files:**
- Create: `src/stage2/efficiency_probe.py`
- Create: `tests/stage2/test_efficiency_probe.py`
- Modify: `README.md` (probe command and gate)

**Interfaces:**
- Consumes: an already validated Stage 2 manifest, model/run arguments, representative quantiles, warmup count three, measured count ten.
- Produces: `results/<experiment-id>/stage2-efficiency.json` and a matching log; `prefix_threshold_met` records whether prefix decoder time share is at least `0.30`, while schema-v1 `segmented_kv_authorized` remains false.

- [ ] **Step 1: Add failing selection, aggregation, and gate tests**

Test representative selection is deterministic for median and p95 length/block count. Test aggregation uses medians, computes prefix share as `prefix_decoder_ms / step_ms`, rejects non-finite/negative samples, and fixes the gate at 0.30.

```python
def test_prefix_gate_is_fixed_at_thirty_percent():
    evidence = summarize_samples([
        Stage2EfficiencySample(step_ms=100.0, prefix_decoder_ms=29.9, peak_bytes=10)
    ])
    assert evidence.segmented_kv_authorized is False
```

- [ ] **Step 2: Run tests and observe missing module failure**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_efficiency_probe.py
```

Expected: FAIL because `stage2.efficiency_probe` is absent.

- [ ] **Step 3: Implement deterministic sample selection and evidence schema**

Define frozen `Stage2EfficiencySample` and `Stage2EfficiencyEvidence` dataclasses. Select the closest retained examples to `(p50 length, p50 blocks)` and `(p95 length, p95 blocks)`, breaking ties by raw index. Write strict schema version 1 JSON exclusively beneath the selected result directory.

- [ ] **Step 4: Implement direct runtime measurement**

Load the same Stage 2 components used by training. Time collation with `time.perf_counter_ns`. Resolve the underlying Qwen model and attach CUDA-event pre/post hooks to its decoder, visual module, and `lm_head`. For each training step:

1. reset peak memory;
2. run the existing trainer loss path;
3. backward;
4. perform the configured optimizer step;
5. synchronize once;
6. classify the last decoder call as the mandatory final full forward and all preceding calls as recurrence prefix work;
7. record `lm_head` time separately from decoder time;
8. record decoder input lengths and the sum of prefix lengths.

Remove every hook in `finally`. Warm up three times and retain exactly ten measured samples per representative example. Log the component breakdown while writing the same scalar fields to the immutable result JSON. Fail on any non-finite timing, loss, or gradient.

- [ ] **Step 5: Run unit tests and one tiny CUDA smoke probe**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_efficiency_probe.py tests/stage2/test_qwen_forward.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=transformers/src:src \
  ./venv/ilvr-venv/bin/python -m stage2.efficiency_probe \
  --self-test --experiment-id stage2-efficiency-self-test
```

Expected: tests PASS and the smoke result contains finite timing and memory values under `results/zero-stage-compatibility/stage2-efficiency-self-test/`.

- [ ] **Step 6: Benchmark worker counts and the optimized no-cache path**

Construct the actual Stage 2 `DataLoader` with worker counts `0`, `2`, and `4`; after two warm-up batches, time twenty collated batches or one full retained epoch when it has fewer than twenty entries. Select the fastest count that stays within host-memory capacity. Then run the GPU probe with that fixed count on the actual full/LoRA/DoRA configuration selected by the user. Use FlashAttention 2 and the best feasible no-cache checkpointing choice as separate labeled runs; do not compare different examples or effective batch sizes.

Expected: one immutable result JSON and one log per run ID. Record `prefix_threshold_met`; verify that schema-v1 `segmented_kv_authorized` remains false.

- [ ] **Step 7: Commit the probe and documentation**

```bash
git add src/stage2/efficiency_probe.py tests/stage2/test_efficiency_probe.py
git add -p README.md
git commit -m "perf(stage2): add representative efficiency gate"
```

---

### Task 7: Combined regression verification and phase decision

**Files:**
- Modify: `docs/superpowers/plans/2026-09-02-stage2-no-cache-efficiency.md` (checkboxes and observed gate result only)
- Read: `results/zero-stage-compatibility/<run-id>/stage2-efficiency.json`

**Interfaces:**
- Consumes: Tasks 1-6 and fresh test/probe output.
- Produces: a verified no-cache optimization result and either an explicit stop or authorization to invoke `superpowers:writing-plans` for a separate segmented-KV experiment plan.

- [ ] **Step 1: Run the focused Stage 2 suite once**

```bash
PYTHONPATH=transformers/src:src ./venv/ilvr-venv/bin/pytest -q tests/stage2
```

Expected: all Stage 2 tests PASS.

- [ ] **Step 2: Run syntax and launcher checks once**

```bash
./venv/ilvr-venv/bin/python -m py_compile \
  src/main.py src/utils_deepseed.py src/stage2/data.py \
  src/stage2/length_manifest.py src/stage2/efficiency_probe.py \
  transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py
bash -n run_stage2_training.sh
```

Expected: exit code zero.

- [ ] **Step 3: Decide from the predeclared gate**

Read the newest optimized no-cache evidence. If `prefix_decoder_share < 0.30`, stop and report that segmented KV is not justified. If it is at least `0.30`, invoke `superpowers:writing-plans` to create the separate conditional segmented-KV plan without changing the threshold.

- [ ] **Step 4: Commit only the recorded plan result when evidence exists**

```bash
git add docs/superpowers/plans/2026-09-02-stage2-no-cache-efficiency.md
git commit -m "docs(stage2): record no-cache efficiency gate"
```
