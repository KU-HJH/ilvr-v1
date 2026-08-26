# ILVR Stage 1 Batched Full/DoRA Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a batch-safe ILVR Stage 1 full/DoRA training path, a complete adapter-and-latent-token checkpoint lifecycle, and a reproducible two-B200 three-epoch CoMT timing comparison that can hand Stage 1 DoRA state to Stage 2 DoRA without merging.

**Architecture:** Stage 1-specific data ownership, teacher-target construction, EMA updates, PEFT composition, sparse latent-token rows, checkpointing, and benchmark evidence live behind focused components in `src/stage1/`. The trainer remains the integration boundary but delegates the batch algorithm and never forwards helper images to the student; shared Qwen architecture and decoder-target validation are stage-neutral, while Stage 2 gains only the schema-v3 and adapter-continuation plumbing needed to consume Stage 1 DoRA artifacts.

**Tech Stack:** Python 3.11, PyTorch 2.7.1 + CUDA 12.8, Transformers local Qwen2.5-VL fork, PEFT 0.17.1, TRL/SFTTrainer, DeepSpeed ZeRO-3, FlashAttention 2.8.3, safetensors, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-08-26-ilvr-stage1-dora-batched-training-design.md`

## Global Constraints

- Use only `./venv/ilvr-venv` for Python, pip, tests, training, and evaluation.
- Execute implementation in an isolated worktree created or verified through `superpowers:using-git-worktrees`; do not implement directly on `main` or `master`.
- Do not push, merge, publish, or create a PR without explicit user permission.
- The official model is exactly `Qwen/Qwen2.5-VL-3B-Instruct`.
- The official dataset is the complete local CoMT `TRAIN.jsonl` containing 3,450 samples.
- The official experiment uses both local NVIDIA B200 GPUs with world size 2.
- Run exactly 3 epochs for each official full and DoRA timed training.
- Use learning rate `1e-5` for both full and DoRA.
- Use gradient accumulation `2` for both methods.
- Probe per-GPU micro-batch independently for full and DoRA, starting at 8; try 16, 32, and higher powers of two until the first capacity failure, or descend through 4, 2, and 1 if 8 fails.
- Use seed `42`, FlashAttention 2, and the same `config_stage3.json` DeepSpeed ZeRO-3 configuration for both methods.
- DoRA uses `r=32`, `alpha=64`, `dropout=0.05`, and decoder suffixes `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- LoRA is not an experiment variant; preserve the legacy Stage 1 `--use_lora` CLI without interpreting it as DoRA.
- Freeze the complete visual module for full and DoRA; teacher visual computation is under `torch.no_grad()` and language/adapter computation remains differentiable.
- Stage 1 supports true `B>1` student forwards; it must not simulate batching with one complete student forward per sample.
- Stage 2 remains explicitly `B=1` until its separately approved batch-safe recurrence work.
- Stage 1 DoRA to Stage 2 DoRA continues the same DoRA and latent-token row deltas without merging, stacking, or resuming Stage 1 optimizer/scheduler/global-step state.
- Stage 1 DoRA trains exactly three latent input-token rows and exactly three sparse output-token row deltas in addition to DoRA parameters; ordinary vocabulary rows, the base decoder, LM head, and visual module stay frozen.
- Intermediate checkpoints are disabled in each timed three-epoch run; save one final checkpoint and time final serialization separately.
- Every production behavior change follows `superpowers:test-driven-development`.
- Any unexpected test, numerical, gradient, OOM, DeepSpeed, FlashAttention, wrapper, or integration failure invokes `superpowers:systematic-debugging` before a fix.
- Do not repeat an expensive successful validator or reread/hash an unchanged large model/checkpoint tree solely to create evidence; reuse the recorded witness.
- Do not modify generic `GenerationMixin` or the authoritative Stage 2 continuous-latent recurrence.
- Do not claim paper-level reproduction from this three-epoch timing and readiness experiment.

---

## File Structure

### New production files

- `src/peft_common.py` — stage-neutral decoder-target resolution and strict Qwen2.5-VL architecture snapshot comparison shared by Stage 1 and Stage 2.
- `src/stage1/__init__.py` — stable public exports for Stage 1 components.
- `src/stage1/types.py` — immutable batch metadata, latent-span, teacher-target, tuning, and benchmark result dataclasses.
- `src/stage1/data.py` — CoMT preparation with stable source-row identity, batch-safe collation, latent-span parsing, image-grid ownership, and structural validation.
- `src/stage1/targets.py` — one-batch EMA-teacher text/vision observation and sample-local helper selection packed in row-major latent order.
- `src/stage1/ema.py` — name/shape-checked EMA topology and ZeRO-aware updates.
- `src/stage1/token_rows.py` — sparse three-row output adapter plus non-merged save/load and token-ID validation.
- `src/stage1/peft.py` — Stage 1 full/legacy-LoRA/DoRA composition, decoder target validation, selective input-token rows, visual freezing, and trainable-boundary reporting.
- `src/stage1/checkpointing.py` — Stage 1 full/PEFT artifact classification, metadata schema, save/load, resume, and handoff validation.
- `src/stage1/trainer.py` — thin Stage 1 Trainer integration delegating targets/EMA/checkpointing and enforcing helper-image removal.
- `src/stage1/composition.py` — model/processor/token setup, fresh-vs-resume resolution, training arguments, and final Stage 1 runtime bundle.
- `src/stage1/benchmark.py` — deterministic capacity candidates, high-cost sample audit, timing/memory/parameter metrics, evidence read/write, and command manifests.
- `run_stage1_training.sh` — explicit two-GPU full/DoRA launcher with no repository-local constants hidden in the script.
- `run_stage1_batch_probe.sh` — fresh-process, one-candidate-per-run capacity probe controller.

### Modified production files

- `src/main.py` — explicit Stage 1/Stage 2 routing; Stage 1 delegates to the new composition, collator, and trainer.
- `src/trainer.py` — compatibility re-export of `CustomTrainerStage1` after its implementation moves into `src/stage1/trainer.py`.
- `src/utils_deepseed.py` — new validated Stage 1 tuning/data/benchmark arguments, explicit `--stage1_adapter_path`, source-row preservation, and use of `args.learning_rate`.
- `src/task_deepseed.py` — explicit CoMT image-root injection and source provenance instead of relying on the hard-coded root in the official launcher.
- `src/stage2/peft.py` — consume shared decoder/Qwen contracts and continue an existing Stage 1 DoRA adapter plus sparse rows.
- `src/stage2/checkpointing.py` — schema v3, strict complete-v2 migration, conditional sparse-row artifact requirements, and Stage 1 adapter provenance.
- `src/stage2/trainer.py` — include sparse token rows in Stage 2 adapter-only saves.
- `src/evaluate_deepseed.py` and `eval.py` — resolve and load sparse token rows with the active Stage 2 adapter before inference.
- `README.md` — exact launcher, artifact, resume/handoff, batch-probe, and evidence instructions.

### New and modified tests

- `tests/stage1/helpers.py` — deterministic tiny language/vision/processor fixtures and metadata builders.
- `tests/stage1/test_data.py` — per-sample ownership, span parsing, structural failures, variable lengths/counts, and source identity.
- `tests/stage1/test_targets.py` — B=1 legacy parity, B>1 row-major equivalence/isolation, multi-block semantics, and visual call count.
- `tests/stage1/test_ema.py` — name/shape topology, full/DoRA update sets, and mocked ZeRO partition behavior.
- `tests/stage1/test_token_rows.py` — exact gradient rows, logits, fp32-safe round trip, token mismatch, and wrapper resolution.
- `tests/stage1/test_peft.py` — target ownership, parameter boundary, full/DoRA composition, latent input rows, and legacy LoRA CLI compatibility.
- `tests/stage1/test_trainer.py` — true batched student call, loss parity, helper removal, gradients, and missing-target failure.
- `tests/stage1/test_checkpointing.py` — full/DoRA final/numbered artifacts, save/resume, mixed/incomplete rejection, and metadata round trip.
- `tests/stage1/test_composition.py` — fresh/resume routing, argument propagation, visual freeze, and checkpoint-state distinctions.
- `tests/stage1/test_benchmark.py` — search order, collision guards, command invariants, metrics schema, and evidence reuse.
- `tests/stage1/test_distributed_smoke.py` — opt-in two-rank ZeRO-3 B>1 full/DoRA save-resume witness.
- `tests/stage2/test_peft_checkpointing.py`, `tests/stage2/test_peft.py`, `tests/stage2/test_stage1_checkpoint_handoff.py`, and focused inference tests — schema-v3 migration/relocation, adapter continuation, token-row logits, fresh Stage 2 state, and current full paths.

## Execution Dependency Map

`Task 1` establishes shared contracts. `Tasks 2–4` establish Stage 1 arguments, types, data ownership, and teacher targets. `Tasks 5–7` establish EMA and DoRA/token-row mechanics. `Task 8` integrates the batched loss; `Tasks 9–10` establish artifacts and runtime routing. `Tasks 11–12` add Stage 2 continuation and inference loading. `Task 13` builds experiment evidence, `Task 14` closes CPU regressions/docs, and `Tasks 15–17` are the only GPU-heavy execution gates. `Task 18` performs final review and verification without repeating already-authenticated expensive runs.

### Task 1: Extract Stage-Neutral Qwen PEFT Contracts

**Files:**
- Create: `src/peft_common.py`
- Modify: `src/stage2/peft.py`
- Modify: `src/stage2/checkpointing.py`
- Test: `tests/stage2/test_peft.py`
- Test: `tests/stage2/test_peft_checkpointing.py`

**Interfaces:**
- Consumes: existing `normalize_stage2_target_modules`, `resolve_stage2_target_module_names`, `_capture_base_model_architecture`, `_normalize_base_model_architecture`, and `_first_architecture_difference` behavior.
- Produces: `resolve_qwen2_5_vl_model(model: nn.Module) -> nn.Module`, `normalize_decoder_target_suffixes(value: str | Sequence[str], *, stage_label: str) -> tuple[str, ...]`, `resolve_qwen_decoder_target_names(model: nn.Module, requested_suffixes: Sequence[str]) -> tuple[str, ...]`, `capture_qwen2_5_vl_architecture(config: PretrainedConfig) -> dict[str, object]`, `normalize_qwen2_5_vl_architecture(value: Mapping[str, object]) -> dict[str, object]`, and `first_architecture_difference(expected: Mapping[str, object], actual: Mapping[str, object]) -> tuple[str, object, object] | None`.

- [ ] **Step 1: Write failing compatibility and ownership tests**

Add tests that lock existing Stage 2 outputs and reject a visual match:

```python
def test_shared_target_resolution_preserves_stage2_contract(tiny_qwen):
    requested = ("q_proj", "k_proj", "v_proj", "o_proj")
    assert resolve_qwen_decoder_target_names(tiny_qwen, requested) == (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
    )


def test_shared_target_resolution_rejects_visual_projection(model_with_visual_q_proj):
    with pytest.raises(ValueError, match="decoder-owned"):
        resolve_qwen_decoder_target_names(model_with_visual_q_proj, ("q_proj",))


def test_stage2_architecture_snapshot_is_unchanged(qwen_config):
    shared = capture_qwen2_5_vl_architecture(qwen_config)
    legacy = capture_stage2_base_model_architecture(qwen_config)
    assert shared == legacy
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_peft.py \
  tests/stage2/test_peft_checkpointing.py -q
```

Expected: collection fails because `src.peft_common` and its functions do not exist.

- [ ] **Step 3: Implement the shared module and compatibility wrappers**

Implement strict suffix normalization and full-name validation:

```python
ALLOWED_QWEN_DECODER_SUFFIXES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def resolve_qwen_decoder_target_names(
    model: nn.Module,
    requested_suffixes: Sequence[str],
) -> tuple[str, ...]:
    suffixes = tuple(requested_suffixes)
    unknown = sorted(set(suffixes) - ALLOWED_QWEN_DECODER_SUFFIXES)
    if unknown:
        raise ValueError(f"Unsupported decoder target suffixes: {unknown}")
    decoder = resolve_qwen2_5_vl_model(model).model
    decoder_module_ids = {id(module) for module in decoder.modules()}
    matches = tuple(
        name
        for name, module in model.named_modules()
        if id(module) in decoder_module_ids
        and any(name.endswith(f".{suffix}") for suffix in suffixes)
    )
    missing = [suffix for suffix in suffixes if not any(name.endswith(f".{suffix}") for name in matches)]
    if missing:
        raise ValueError(f"Requested decoder targets did not match: {missing}")
    return matches
```

Move the current Qwen wrapper-resolution logic and architecture field set unchanged into `src/peft_common.py`. The shared module must not import `src.stage1` or `src.stage2`. Keep the old Stage 2 public/helper names as direct wrappers so no current caller or artifact test changes semantics.

- [ ] **Step 4: Run the focused Stage 2 tests and confirm GREEN**

Run the command from Step 2.

Expected: all selected tests pass; serialized Stage 2 architecture snapshots remain byte-for-byte equivalent after JSON normalization.

- [ ] **Step 5: Commit**

```bash
git add src/peft_common.py src/stage2/peft.py src/stage2/checkpointing.py \
  tests/stage2/test_peft.py tests/stage2/test_peft_checkpointing.py
git commit -m "refactor: share qwen peft contracts"
```

### Task 2: Define Stage 1 Tuning and CLI Contracts

**Files:**
- Create: `src/stage1/__init__.py`
- Create: `src/stage1/types.py`
- Modify: `src/utils_deepseed.py`
- Test: `tests/stage1/test_cli.py`

**Interfaces:**
- Consumes: `normalize_decoder_target_suffixes` from Task 1.
- Produces: `Stage1TuningConfig`, `Stage1RuntimeConfig`, `resolve_stage1_tuning_config(args: argparse.Namespace) -> Stage1TuningConfig`, and parser fields `--tuning_method`, `--stage1_adapter_path`, `--data_path`, `--image_root`, `--per_device_train_batch_size`, `--gradient_accumulation_steps`, `--save_strategy`, `--final_save`/`--no_final_save`, and `--resume_from_checkpoint_path`.

- [ ] **Step 1: Write failing parser and validation tests**

```python
def test_stage1_dora_defaults_match_approved_experiment():
    args = parse_args([
        "--stage", "stage1",
        "--tuning_method", "dora",
        "--model", "Qwen/Qwen2.5-VL-3B-Instruct",
        "--data_path", "/data/comt/TRAIN.jsonl",
        "--image_root", "/data/comt",
    ])
    tuning = resolve_stage1_tuning_config(args)
    assert tuning.method == "dora"
    assert (tuning.rank, tuning.alpha, tuning.dropout) == (32, 64, 0.05)
    assert tuning.target_suffixes == (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--stage", "stage1", "--use_lora", "--tuning_method", "dora"],
         "cannot be combined"),
        (["--stage", "stage2", "--stage1_adapter_path", "/a",
          "--resume_from_checkpoint_path", "/b"], "mutually exclusive"),
        (["--stage", "stage1", "--learning_rate", "nan"], "finite"),
        (["--stage", "stage1", "--per_device_train_batch_size", "0"], "at least 1"),
    ],
)
def test_cli_rejects_contradictory_or_nonfinite_values(argv, message):
    with pytest.raises(ValueError, match=message):
        parse_args(argv)
```

Also lock legacy `--use_lora` as method `lora` and assert it never becomes `dora`.

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_cli.py -q
```

Expected: tests fail because the Stage 1 dataclasses and new validated flags are absent.

- [ ] **Step 3: Implement immutable configurations and parser validation**

Define:

```python
@dataclass(frozen=True)
class Stage1TuningConfig:
    method: Literal["full", "lora", "dora"]
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_suffixes: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@dataclass(frozen=True)
class Stage1RuntimeConfig:
    data_path: Path
    image_root: Path
    output_dir: Path
    model_name_or_path: str
    epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    seed: int
    attention_implementation: str
    deepspeed_config: Path | None
```

Make `--tuning_method` accept `full` or `dora` for the official path while mapping a sole legacy `--use_lora` to `lora`. Reject both flags together. Validate finite positive learning rate, finite dropout in `[0,1)`, positive rank/alpha/batch/accumulation, and the Stage 2 initializer/resume mutual exclusion. Preserve existing Stage 2 `B=1` validation.

- [ ] **Step 4: Run focused and parser regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_cli.py \
  tests/stage2/test_routing.py -q
```

Expected: all tests pass, including legacy Stage 2 argument behavior.

- [ ] **Step 5: Commit**

```bash
git add src/stage1/__init__.py src/stage1/types.py src/utils_deepseed.py \
  tests/stage1/test_cli.py
git commit -m "feat: define stage1 tuning contracts"
```

### Task 3: Build Batch-Safe CoMT Preparation and Collation

**Files:**
- Create: `src/stage1/data.py`
- Modify: `src/stage1/types.py`
- Modify: `src/task_deepseed.py`
- Modify: `src/utils_deepseed.py`
- Test: `tests/stage1/helpers.py`
- Test: `tests/stage1/test_data.py`

**Interfaces:**
- Consumes: processor output keys `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw` and current latent token replacement utilities.
- Produces: `Stage1LatentSpan(start: int, end: int, latent_positions: tuple[int, ...])`, `Stage1SampleMetadata`, `Stage1BatchMetadata`, `PreparedStage1Example`, `prepare_stage1_comt_examples(data_path: Path, image_root: Path) -> list[PreparedStage1Example]`, and `Stage1DataCollator(processor, token_ids, latent_size)` whose output includes trainer-only key `stage1_batch_metadata`.

- [ ] **Step 1: Write structural RED tests**

Use a deterministic processor fixture and cover unequal samples:

```python
def test_collator_records_per_sample_ownership_and_row_identity(stage1_collator):
    batch = stage1_collator([
        prepared_example(source_index=7, original_images=1, helper_images=2, blocks=2),
        prepared_example(source_index=11, original_images=2, helper_images=1, blocks=1),
    ])
    metadata = batch["stage1_batch_metadata"]
    assert metadata.batch_size == 2
    assert [sample.source_index for sample in metadata.samples] == [7, 11]
    assert metadata.samples[0].original_grid_range == (0, 1)
    assert metadata.samples[1].original_grid_range == (1, 3)
    assert metadata.samples[0].helper_grid_range == (0, 2)
    assert metadata.samples[1].helper_grid_range == (2, 3)
    assert [len(sample.latent_spans) for sample in metadata.samples] == [2, 1]
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    assert batch["labels"].shape == batch["input_ids"].shape


@pytest.mark.parametrize(
    "tokens, helper_count, message",
    [
        (["start", "pad", "end"], 2, "latent blocks.*helper images"),
        (["start", "pad", "pad", "end"], 1, "exactly 3 latent slots"),
        (["start", "start", "pad", "end", "end"], 1, "nonnested"),
        (["start", "pad"], 1, "incomplete"),
    ],
)
def test_collator_rejects_malformed_sample(tokens, helper_count, message, collator_for_tokens):
    with pytest.raises(ValueError, match=message):
        collator_for_tokens(tokens, helper_count, source_index=19)
```

Also test variable padding lengths, multiple blocks, missing/duplicated special tokens, and raw-ID provenance where `source_index` remains the primary unique key.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_data.py -q
```

Expected: import/collection failure for `Stage1DataCollator` and metadata types.

- [ ] **Step 3: Implement explicit data and metadata types**

Add these immutable contracts:

```python
@dataclass(frozen=True)
class Stage1LatentSpan:
    start: int
    end: int
    latent_positions: tuple[int, ...]


@dataclass(frozen=True)
class Stage1SampleMetadata:
    source_index: int
    source_id: str | None
    original_grid_range: tuple[int, int]
    helper_grid_range: tuple[int, int]
    latent_spans: tuple[Stage1LatentSpan, ...]
    helper_to_block: tuple[int, ...]
    unpadded_length: int


@dataclass(frozen=True)
class Stage1BatchMetadata:
    samples: tuple[Stage1SampleMetadata, ...]
    batch_size: int
    sequence_length: int
```

`prepare_stage1_comt_examples` enumerates JSONL rows, calls the existing CoMT trajectory conversion with an explicit `image_root`, records optional raw IDs as `source_id`, and never uses a raw ID in place of `source_index`.

- [ ] **Step 4: Implement collation and fail-fast validation**

The collator must:

```python
class Stage1DataCollator:
    def __call__(self, examples: Sequence[PreparedStage1Example]) -> dict[str, object]:
        conversations = [example.conversation for example in examples]
        original = self._process_original_images(conversations)
        helper = self._process_helper_images(conversations)
        input_ids, latent_spans = self._replace_helper_tokens_with_latent_blocks(original)
        metadata = self._build_metadata(examples, original, helper, input_ids, latent_spans)
        self._validate(metadata, input_ids, original["attention_mask"], original["labels"])
        return {
            **original,
            "input_ids": input_ids,
            "pixel_values_latent": helper["pixel_values"],
            "image_grid_thw_latent": helper["image_grid_thw"],
            "stage1_batch_metadata": metadata,
        }
```

The helper keys exist only between collator and trainer target builder. Validate each block has exactly `latent_size` pad positions, boundaries are ordered and nonnested, helper count equals block count, grid ranges form exact non-overlapping partitions, and `input_ids`/`attention_mask`/`labels`/latent mask share `B,T`. Every error includes `source_index` and observed versus expected values.

- [ ] **Step 5: Run the data tests and one Stage 2 data regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_data.py \
  tests/stage2/test_data.py -q
```

Expected: all selected tests pass; Stage 2 collation remains unchanged.

- [ ] **Step 6: Commit**

```bash
git add src/stage1/data.py src/stage1/types.py src/task_deepseed.py \
  src/utils_deepseed.py tests/stage1/helpers.py tests/stage1/test_data.py
git commit -m "feat: add batch-safe stage1 collation"
```

### Task 4: Implement Batched EMA-Teacher Targets

**Files:**
- Create: `src/stage1/targets.py`
- Modify: `src/stage1/types.py`
- Test: `tests/stage1/test_targets.py`

**Interfaces:**
- Consumes: `Stage1BatchMetadata` from Task 3 and a finalized eval-only teacher model.
- Produces: `Stage1TeacherTargets(latent_values: Tensor, latent_mask: Tensor, selected_counts: tuple[tuple[int, ...], ...], diagnostics: Mapping[str, Tensor])` and `Stage1TeacherTargetBuilder.build(teacher: nn.Module, batch: MutableMapping[str, object], metadata: Stage1BatchMetadata) -> Stage1TeacherTargets`. The method consumes and removes `pixel_values_latent` and `image_grid_thw_latent` from `batch`.

- [ ] **Step 1: Write deterministic recurrence, isolation, and call-count tests**

```python
def test_batched_targets_equal_row_major_independent_results(target_builder, two_sample_batch):
    batched = target_builder.build(two_sample_batch.teacher, two_sample_batch.inputs, two_sample_batch.metadata)
    independent = [
        target_builder.build(sample.teacher, sample.inputs, sample.metadata)
        for sample in two_sample_batch.split()
    ]
    expected = torch.cat([result.latent_values for result in independent], dim=0)
    torch.testing.assert_close(batched.latent_values, expected)
    assert batched.latent_mask.tolist() == two_sample_batch.expected_mask


def test_second_sample_cannot_observe_first_sample_state(target_builder, two_sample_batch):
    baseline = target_builder.build_fresh(two_sample_batch)
    perturbed = target_builder.build_fresh(two_sample_batch.with_first_sample_scaled(1000.0))
    first_count = sum(baseline.selected_counts[0])
    torch.testing.assert_close(
        baseline.latent_values[first_count:],
        perturbed.latent_values[first_count:],
    )


def test_teacher_visual_encoder_runs_once_for_multiple_blocks(target_builder, multiblock_batch):
    result = target_builder.build(
        multiblock_batch.teacher,
        multiblock_batch.inputs,
        multiblock_batch.metadata,
    )
    assert result.latent_values.shape[0] == multiblock_batch.total_selected
    assert multiblock_batch.teacher.language_forward_count == 1
    assert multiblock_batch.teacher.original_visual_call_count == 1
    assert multiblock_batch.teacher.helper_visual_call_count == 1
```

Add a `B=1` golden fixture captured from the existing target-selection equations and assert target, mask, CE inputs, and alignment inputs within `rtol=1e-5, atol=1e-6`. Add unequal helper/block counts and multiple-block ordering cases.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_targets.py -q
```

Expected: tests fail because `Stage1TeacherTargetBuilder` is absent.

- [ ] **Step 3: Implement one language forward, one helper visual call, and row-major packing**

Implement the algorithm as explicit batch and sample-local phases:

```python
def build(self, teacher, batch, metadata):
    helper_pixels = batch.pop("pixel_values_latent")
    helper_grid = batch.pop("image_grid_thw_latent")
    with torch.no_grad():
        text_outputs = self._teacher_language_forward(teacher, batch)
        helper_features = teacher.visual(helper_pixels, grid_thw=helper_grid)

    latent_mask = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
    packed: list[Tensor] = []
    selected_counts: list[tuple[int, ...]] = []
    for sample_index, sample in enumerate(metadata.samples):
        sample_text = self._slice_text_outputs(text_outputs, sample_index, sample)
        sample_helpers = self._slice_helper_features(helper_features, sample)
        previous_selected = None
        block_counts: list[int] = []
        for block_index, span in enumerate(sample.latent_spans):
            selected = self._select_for_block(
                sample_text=sample_text,
                helper_features=sample_helpers,
                block_index=block_index,
                previous_selected=previous_selected,
            )
            packed.append(selected)
            latent_mask[sample_index, list(span.latent_positions)] = True
            previous_selected = selected
            block_counts.append(selected.shape[0])
        selected_counts.append(tuple(block_counts))
    latent_values = torch.cat(packed, dim=0)
    if latent_values.shape[0] != int(latent_mask.sum()):
        raise RuntimeError("row-major latent target count does not match latent mask")
    return Stage1TeacherTargets(
        latent_values=latent_values,
        latent_mask=latent_mask,
        selected_counts=tuple(selected_counts),
        diagnostics=self._diagnostics(latent_values),
    )
```

Preserve the existing coverage probability, image pooling, query/key attention, helper selection, and previous-block conditioning equations. The only semantic change is batch-safe indexing. Keep all teacher work under `no_grad` and return finite-value diagnostics without logging in the hot path.

- [ ] **Step 4: Run target tests and confirm GREEN**

Run the command from Step 2.

Expected: all target tests pass; the fake teacher records exactly one language call and one helper visual call per batch.

- [ ] **Step 5: Commit**

```bash
git add src/stage1/targets.py src/stage1/types.py tests/stage1/test_targets.py
git commit -m "feat: build batched stage1 teacher targets"
```

### Task 5: Replace Positional EMA with Name/Shape-Safe Updates

**Files:**
- Create: `src/stage1/ema.py`
- Test: `tests/stage1/test_ema.py`

**Interfaces:**
- Consumes: `Stage1TuningConfig` from Task 2 and generic named-parameter model fixtures; Task 7 later supplies the production-composed topology through the same `nn.Module` contract.
- Produces: `Stage1EmaTeacher(student: nn.Module, decay: float, method: Literal["full", "lora", "dora"])` with `model`, `update(student: nn.Module) -> None`, `state_dict()`, and `load_state_dict(state)`; `build_ema_parameter_map(student, teacher, method) -> tuple[EmaParameterPair, ...]` where every pair stores an exact parameter name and shape.

- [ ] **Step 1: Write RED tests for topology and method-specific updates**

```python
def test_ema_rejects_name_or_shape_mismatch(full_student):
    ema = Stage1EmaTeacher(full_student, decay=0.9, method="full")
    delattr(ema.model.model.layers[0].self_attn, "q_proj")
    with pytest.raises(RuntimeError, match="EMA topology mismatch.*q_proj"):
        ema.update(full_student)


def test_full_ema_updates_trainable_language_but_not_visual(full_student):
    ema = Stage1EmaTeacher(full_student, decay=0.5, method="full")
    before_visual = clone_named_parameters(ema.model.visual)
    add_to_trainable_language_parameters(full_student, 2.0)
    ema.update(full_student)
    assert any_parameter_changed(ema.model.model)
    assert_named_parameters_equal(ema.model.visual, before_visual)


def test_dora_ema_updates_only_adapter_and_token_rows(dora_student):
    ema = Stage1EmaTeacher(dora_student, decay=0.5, method="dora")
    expected = trainable_parameter_names(dora_student)
    assert expected
    assert all(
        "lora_" in name
        or "trainable_tokens_delta" in name
        or "output_delta" in name
        for name in expected
    )
    assert ema.parameter_names == tuple(sorted(expected))
```

Mock `deepspeed.zero.GatheredParameters` and a parameter carrying `ds_id`/`ds_status`; assert only the source/target pair being updated is gathered and no zero-sized shard is treated as a complete tensor.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_ema.py -q
```

Expected: import failure for `Stage1EmaTeacher`.

- [ ] **Step 3: Implement exact topology mapping and selective ZeRO gathering**

```python
@dataclass(frozen=True)
class EmaParameterPair:
    name: str
    shape: torch.Size


class Stage1EmaTeacher:
    def __init__(self, student, decay, method):
        self.model = copy.deepcopy(student).eval()
        self.model.requires_grad_(False)
        self.decay = float(decay)
        self.method = method
        self._pairs = build_ema_parameter_map(student, self.model, method)

    @torch.no_grad()
    def update(self, student):
        student_named = dict(student.named_parameters())
        teacher_named = dict(self.model.named_parameters())
        validate_parameter_map(self._pairs, student_named, teacher_named)
        for pair in self._pairs:
            source = student_named[pair.name]
            target = teacher_named[pair.name]
            with gathered_parameter_pair(source, target):
                target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
```

For `full`, select trainable non-visual language parameters. For `lora`/`dora`, select only parameters already marked trainable by the composed PEFT/token-row model. Fail on missing name, duplicate name, shape mismatch, empty update set, nonfinite decay, or a trainable visual parameter. Do not use positional `zip` anywhere.

- [ ] **Step 4: Run EMA tests and confirm GREEN**

Run the command from Step 2.

Expected: all tests pass, including the mocked ZeRO gather witness.

- [ ] **Step 5: Commit**

```bash
git add src/stage1/ema.py tests/stage1/test_ema.py
git commit -m "feat: add topology-safe stage1 ema"
```

### Task 6: Add the Sparse Latent Output-Token Row Adapter

**Files:**
- Create: `src/stage1/token_rows.py`
- Test: `tests/stage1/test_token_rows.py`

**Interfaces:**
- Consumes: exactly three tokenizer/config latent token IDs.
- Produces: `SparseOutputTokenRows(base_head: nn.Linear, token_ids: Sequence[int])`, `attach_sparse_output_token_rows(model, token_ids) -> SparseOutputTokenRows`, `find_sparse_output_token_rows(model) -> SparseOutputTokenRows`, `save_sparse_token_rows(model, output_dir: Path) -> Path`, and `load_sparse_token_rows(model, artifact_dir: Path, expected_token_ids: Sequence[int]) -> None`.

- [ ] **Step 1: Write exact-logit, gradient, and round-trip RED tests**

```python
def test_output_delta_changes_only_three_latent_logits():
    base = nn.Linear(5, 13, bias=False)
    wrapper = SparseOutputTokenRows(base, token_ids=(2, 7, 11))
    hidden = torch.randn(2, 4, 5)
    with torch.no_grad():
        wrapper.output_delta.copy_(torch.arange(15).reshape(3, 5))
    base_logits = base(hidden)
    logits = wrapper(hidden)
    unchanged = [index for index in range(13) if index not in (2, 7, 11)]
    torch.testing.assert_close(logits[..., unchanged], base_logits[..., unchanged])
    torch.testing.assert_close(
        logits[..., (2, 7, 11)] - base_logits[..., (2, 7, 11)],
        F.linear(hidden, wrapper.output_delta),
    )


def test_only_output_delta_receives_gradient():
    wrapper = SparseOutputTokenRows(nn.Linear(4, 9, bias=False), token_ids=(1, 3, 8))
    wrapper.base_head.requires_grad_(False)
    wrapper(torch.randn(2, 4)).sum().backward()
    assert wrapper.output_delta.grad is not None
    assert torch.isfinite(wrapper.output_delta.grad).all()
    assert wrapper.base_head.weight.grad is None


def test_sparse_rows_round_trip_in_fp32_container(tmp_path):
    original = wrapped_tiny_head(dtype=torch.bfloat16, token_ids=(4, 5, 6))
    save_sparse_token_rows(original, tmp_path)
    restored = wrapped_tiny_head(dtype=torch.bfloat16, token_ids=(4, 5, 6))
    load_sparse_token_rows(restored, tmp_path, expected_token_ids=(4, 5, 6))
    torch.testing.assert_close(
        find_sparse_output_token_rows(original).output_delta,
        find_sparse_output_token_rows(restored).output_delta,
        rtol=0,
        atol=0,
    )
```

Add failures for duplicate IDs, fewer/more than three IDs, out-of-vocabulary IDs, tensor shape mismatch, missing file, and stored/expected token-ID mismatch.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_token_rows.py -q
```

Expected: import failure for `SparseOutputTokenRows`.

- [ ] **Step 3: Implement the non-merged output wrapper**

```python
class SparseOutputTokenRows(nn.Module):
    def __init__(self, base_head: nn.Linear, token_ids: Sequence[int]):
        super().__init__()
        ids = validate_exact_latent_token_ids(token_ids, base_head.out_features)
        self.base_head = base_head
        self.base_head.requires_grad_(False)
        self.register_buffer("token_ids", torch.tensor(ids, dtype=torch.long), persistent=True)
        self.output_delta = nn.Parameter(
            torch.zeros(len(ids), base_head.in_features, dtype=base_head.weight.dtype)
        )

    @property
    def weight(self):
        return self.base_head.weight

    @property
    def in_features(self):
        return self.base_head.in_features

    @property
    def out_features(self):
        return self.base_head.out_features

    def forward(self, hidden_states):
        base_logits = self.base_head(hidden_states)
        delta_logits = F.linear(hidden_states, self.output_delta)
        return torch.index_add(
            base_logits,
            dim=-1,
            index=self.token_ids.to(base_logits.device),
            source=delta_logits.to(base_logits.dtype),
        )
```

Save `token_ids` as int64 and `output_delta` as float32 in `ilvr_token_rows.safetensors`; include `format_version=1` and training dtype in safetensors metadata. On load, validate IDs and shape before copying back to the module dtype. Use the same focused-parameter gather helper as EMA when `output_delta` is ZeRO-partitioned. Never alter `base_head.weight` or materialize a full-vocabulary delta.

- [ ] **Step 4: Run token-row tests and confirm GREEN**

Run the command from Step 2.

Expected: all tests pass and only three vocabulary logits differ.

- [ ] **Step 5: Commit**

```bash
git add src/stage1/token_rows.py tests/stage1/test_token_rows.py
git commit -m "feat: add sparse latent output rows"
```

### Task 7: Compose Full, Legacy LoRA, and DoRA Stage 1 Models

**Files:**
- Create: `src/stage1/peft.py`
- Modify: `src/stage1/__init__.py`
- Test: `tests/stage1/test_peft.py`

**Interfaces:**
- Consumes: `Stage1TuningConfig` from Task 2, shared decoder resolution from Task 1, and token-row wrapper from Task 6.
- Produces: `build_stage1_tuned_model(base_model: nn.Module, tuning: Stage1TuningConfig, latent_token_ids: Sequence[int]) -> nn.Module`, `validate_stage1_trainable_boundary(model, tuning, latent_token_ids) -> Stage1ParameterReport`, and `Stage1ParameterReport(trainable_names, trainable_parameters, total_parameters, categories)`.

- [ ] **Step 1: Write real tiny-Qwen parameter-boundary RED tests**

```python
def test_dora_targets_decoder_and_exact_latent_rows(tiny_qwen, latent_token_ids):
    model = build_stage1_tuned_model(
        tiny_qwen,
        Stage1TuningConfig(method="dora"),
        latent_token_ids,
    )
    report = validate_stage1_trainable_boundary(
        model, Stage1TuningConfig(method="dora"), latent_token_ids
    )
    assert report.categories["visual"] == 0
    assert report.categories["base_decoder"] == 0
    assert report.categories["ordinary_embedding_rows"] == 0
    assert report.categories["ordinary_lm_head_rows"] == 0
    assert report.categories["latent_input_rows"] == 3
    assert report.categories["latent_output_rows"] == 3
    assert any("lora_magnitude_vector" in name for name in report.trainable_names)
    assert all("visual" not in name for name in report.trainable_names)


def test_full_freezes_visual_but_trains_language(tiny_qwen, latent_token_ids):
    model = build_stage1_tuned_model(
        tiny_qwen, Stage1TuningConfig(method="full"), latent_token_ids
    )
    assert not any(parameter.requires_grad for parameter in model.visual.parameters())
    assert any(parameter.requires_grad for parameter in model.model.parameters())


def test_legacy_lora_remains_lora_not_dora(tiny_qwen, latent_token_ids):
    model = build_stage1_tuned_model(
        tiny_qwen, Stage1TuningConfig(method="lora"), latent_token_ids
    )
    assert any("lora_A" in name for name, _ in model.named_parameters())
    assert not any("lora_magnitude_vector" in name for name, _ in model.named_parameters())
```

Perturb latent input rows, backpropagate a latent-boundary CE fixture, and assert PEFT's `trainable_tokens_delta` has finite nonzero gradients while ordinary base embedding rows remain frozen.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_peft.py -q
```

Expected: import failure for `build_stage1_tuned_model`.

- [ ] **Step 3: Implement full and adapter composition**

For full tuning, freeze `model.visual` and leave language parameters trainable. For LoRA/DoRA:

```python
def build_stage1_tuned_model(base_model, tuning, latent_token_ids):
    qwen = resolve_qwen2_5_vl_model(base_model)
    freeze_visual(qwen)
    if tuning.method == "full":
        return base_model

    resolved = resolve_qwen_decoder_target_names(base_model, tuning.target_suffixes)
    peft_config = LoraConfig(
        r=tuning.rank,
        lora_alpha=tuning.alpha,
        lora_dropout=tuning.dropout,
        target_modules=list(resolved),
        use_dora=tuning.method == "dora",
        trainable_token_indices={"model.embed_tokens": list(latent_token_ids)},
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, peft_config)
    attach_sparse_output_token_rows(
        resolve_qwen2_5_vl_model(model),
        latent_token_ids,
    )
    validate_stage1_trainable_boundary(model, tuning, latent_token_ids)
    return model
```

Resolve full decoder-owned module names before PEFT construction. Attach the output-row wrapper after `get_peft_model` so PEFT's base-freezing pass cannot freeze `output_delta`. Reject visual/embedding/LM-head targets, unmatched suffixes, trainable base/visual parameters, fewer or more than three selective token rows, and absent DoRA magnitude parameters in `dora` mode. The report must count parameter elements and logical sparse rows separately so later benchmark evidence is unambiguous.

- [ ] **Step 4: Run PEFT and shared-contract tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_peft.py \
  tests/stage2/test_peft.py -q
```

Expected: all selected tests pass with no visual target and exactly three trainable latent input/output rows.

- [ ] **Step 5: Commit**

```bash
git add src/stage1/peft.py src/stage1/__init__.py tests/stage1/test_peft.py
git commit -m "feat: compose stage1 full and dora models"
```

### Task 8: Integrate True Batched Student Loss in the Stage 1 Trainer

**Files:**
- Create: `src/stage1/trainer.py`
- Modify: `src/trainer.py`
- Modify: `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- Test: `tests/stage1/test_trainer.py`
- Test: `tests/stage1/test_qwen_latent_injection.py`

**Interfaces:**
- Consumes: `Stage1TeacherTargetBuilder`, `Stage1EmaTeacher`, `Stage1BatchMetadata`, and a finalized full/PEFT model.
- Produces: `CustomTrainerStage1` with delegated `compute_loss`/`optimizer_step` and a model-side row-major `latent_hidden_states` plus `image_out_mask: BoolTensor[B,T]` contract.

- [ ] **Step 1: Write RED tests for one batched student call and gradient/loss parity**

```python
def test_trainer_runs_one_student_forward_for_batched_examples(stage1_trainer, batch_b2):
    loss = stage1_trainer.compute_loss(stage1_trainer.model, clone_batch(batch_b2))
    assert torch.isfinite(loss)
    assert stage1_trainer.model.student_forward_count == 1
    assert stage1_trainer.target_builder.teacher_language_calls == 1
    assert stage1_trainer.target_builder.helper_visual_calls == 1


def test_helper_images_never_reach_student(stage1_trainer, batch_b2):
    stage1_trainer.compute_loss(stage1_trainer.model, clone_batch(batch_b2))
    forwarded = stage1_trainer.model.last_forward_kwargs
    assert "pixel_values_latent" not in forwarded
    assert "image_grid_thw_latent" not in forwarded
    assert forwarded["image_out_mask"].shape == batch_b2["input_ids"].shape


def test_missing_teacher_target_is_not_a_fallback(stage1_trainer, valid_latent_batch):
    stage1_trainer.target_builder.return_empty_targets = True
    with pytest.raises(RuntimeError, match="valid latent block.*teacher target"):
        stage1_trainer.compute_loss(stage1_trainer.model, valid_latent_batch)
```

Add `B=1` CE/alignment/total-loss golden parity, `B=2` variable-count row-major masked scatter, finite full gradients, finite nonzero DoRA direction/magnitude/input-row/output-row gradients, and no visual/base gradients. Patch the fake model to raise if called once per sample.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_trainer.py \
  tests/stage1/test_qwen_latent_injection.py -q
```

Expected: failures expose the current row-zero assumptions and the Qwen helper-image `B != 1` branch.

- [ ] **Step 3: Generalize only the row-major latent injection boundary**

In the local Qwen fork, validate and inject precomputed targets:

```python
if latent_hidden_states is not None:
    if image_out_mask is None or image_out_mask.shape != input_ids.shape:
        raise ValueError("image_out_mask must have shape [B, T]")
    expected = int(image_out_mask.sum().item())
    if latent_hidden_states.ndim != 2 or latent_hidden_states.shape[0] != expected:
        raise ValueError(
            f"latent_hidden_states rows ({latent_hidden_states.shape[0]}) "
            f"must equal true image_out_mask entries ({expected})"
        )
    inputs_embeds = inputs_embeds.masked_scatter(
        image_out_mask.unsqueeze(-1),
        latent_hidden_states.to(inputs_embeds.dtype),
    )
```

Keep the legacy direct-helper branch for compatibility but make the official trainer unreachable from it. Do not add Stage 1 logic to Stage 2 recurrence or generation.

- [ ] **Step 4: Implement the delegated trainer flow**

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    metadata = inputs.pop("stage1_batch_metadata")
    targets = self.target_builder.build(self.ema_teacher.model, inputs, metadata)
    if int(targets.latent_mask.sum()) != targets.latent_values.shape[0]:
        raise RuntimeError("valid latent block has no matching teacher target")
    model_inputs = {
        key: value for key, value in inputs.items()
        if key not in {"pixel_values_latent", "image_grid_thw_latent"}
    }
    model_inputs["latent_hidden_states"] = targets.latent_values
    model_inputs["image_out_mask"] = targets.latent_mask
    outputs = model(**model_inputs)
    loss = self.ce_weight * outputs.loss + self.sim_weight * self._alignment_loss(
        outputs, targets
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Stage 1 loss is nonfinite")
    return (loss, outputs) if return_outputs else loss
```

Construct EMA only after `self.model` has its final full/PEFT/token-row topology. Update EMA after successful optimizer steps using Task 5. Before each optimizer step, validate the representative trainable gradient norm is finite; raise `FloatingPointError` on nonfinite loss or gradient rather than recording a successful step. Move implementation into `src/stage1/trainer.py` and leave `src/trainer.py` as a compatibility import. Preserve expected Trainer callback/logging behavior and report CE, alignment, latent count/norm, and gradient diagnostics only when requested.

- [ ] **Step 5: Run all Stage 1 semantic unit tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_data.py \
  tests/stage1/test_targets.py \
  tests/stage1/test_ema.py \
  tests/stage1/test_token_rows.py \
  tests/stage1/test_peft.py \
  tests/stage1/test_trainer.py \
  tests/stage1/test_qwen_latent_injection.py -q
```

Expected: all selected tests pass; the B=2 fixture reports one student forward.

- [ ] **Step 6: Commit**

```bash
git add src/stage1/trainer.py src/trainer.py \
  transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py \
  tests/stage1/test_trainer.py tests/stage1/test_qwen_latent_injection.py
git commit -m "feat: train stage1 with true batches"
```

### Task 9: Implement Stage 1 Full and Adapter Artifact Lifecycles

**Files:**
- Create: `src/stage1/checkpointing.py`
- Modify: `src/stage1/trainer.py`
- Test: `tests/stage1/test_checkpointing.py`

**Interfaces:**
- Consumes: architecture snapshot helpers, `Stage1TuningConfig`, PEFT adapter save/load, and sparse token rows.
- Produces: `Stage1PeftMetadata` schema v1, `classify_stage1_artifact(path: Path) -> Stage1ArtifactKind`, `save_stage1_peft_artifact(model, processor, output_dir, metadata)`, `load_stage1_peft_artifact(base_model, processor, artifact_dir, *, is_trainable) -> nn.Module`, and `resolve_stage1_resume(output_dir, explicit_resume) -> Stage1ResumeDecision`.

- [ ] **Step 1: Write classifier, metadata, and round-trip RED tests**

```python
def test_dora_artifact_is_adapter_only_and_complete(tmp_path, stage1_dora_bundle):
    save_stage1_peft_artifact(
        stage1_dora_bundle.model,
        stage1_dora_bundle.processor,
        tmp_path,
        stage1_dora_bundle.metadata,
    )
    assert classify_stage1_artifact(tmp_path) is Stage1ArtifactKind.PEFT
    assert (tmp_path / "adapter_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "ilvr_token_rows.safetensors").is_file()
    assert (tmp_path / "stage1_peft_metadata.json").is_file()
    assert not (tmp_path / "model.safetensors").exists()


@pytest.mark.parametrize(
    "missing, message",
    [
        ("adapter_model.safetensors", "incomplete"),
        ("ilvr_token_rows.safetensors", "token-row"),
        ("stage1_peft_metadata.json", "metadata"),
    ],
)
def test_incomplete_adapter_fails_before_base_load(tmp_path, missing, message, load_spy):
    write_complete_fake_stage1_adapter(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError, match=message):
        classify_and_load_stage1(tmp_path, load_spy)
    assert load_spy.large_weight_loads == 0
```

Add tests for mixed full+PEFT markers, legacy binary adapter, unknown schema, architecture/token/shape/tuning mismatch, processor token round trip, final directory, numbered checkpoint, explicit resume, and auto-resume only within the same Stage 1 output directory.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest tests/stage1/test_checkpointing.py -q
```

Expected: import failure for `Stage1PeftMetadata` and artifact helpers.

- [ ] **Step 3: Implement strict schema and pre-load classification**

Define the complete JSON sidecar:

```python
@dataclass(frozen=True)
class Stage1PeftMetadata:
    schema_version: int
    stage: Literal["stage1"]
    base_model_reference: str
    base_model_architecture: dict[str, object]
    base_vocab_size: int
    resized_vocab_size: int
    latent_token_ids: tuple[int, int, int]
    latent_token_strings: tuple[str, str, str]
    latent_size: int
    ce_weight: float
    sim_weight: float
    coverage_p: float
    image_pool_k: int
    tuning_method: Literal["lora", "dora"]
    rank: int
    alpha: int
    dropout: float
    requested_target_suffixes: tuple[str, ...]
    resolved_target_names: tuple[str, ...]
    input_row_format: Literal["peft_trainable_tokens_v1"]
    output_row_format: Literal["ilvr_sparse_output_rows_v1"]
    output_row_shape: tuple[int, int]
    peft_version: str
    repository_format_version: int
```

Use exact marker sets. Classify before any `from_pretrained` call. Reject mixed layouts and unknown files that claim to be model weight formats. Validate exact schema fields, stage, original architecture, base/resized vocab, token strings/IDs, all tuning values, resolved module names, and sparse shape.

- [ ] **Step 4: Integrate Trainer save/resume semantics**

For PEFT `save_model(output_dir)`, call normal adapter save, then save sparse rows, metadata, tokenizer, and processor. For a numbered Stage 1 checkpoint, let Trainer additionally own `optimizer.pt`, scheduler, RNG, and `trainer_state.json`. For final adapter output, do not copy full base weights. Full tuning retains normal Trainer full-model save/resume.

On Stage 1 PEFT resume, resize the base, load `PeftModel.from_pretrained(..., is_trainable=True)`, attach the output-row wrapper to the now-active Qwen model, and then load `ilvr_token_rows.safetensors`:

```python
decision = resolve_stage1_resume(args.output_dir, args.resume_from_checkpoint_path)
model = load_stage1_peft_artifact(
    base_model,
    processor,
    decision.checkpoint_path,
    is_trainable=True,
)
trainer.train(resume_from_checkpoint=str(decision.checkpoint_path))
```

This restores adapter/token rows plus Trainer optimizer/scheduler/RNG/global step. Do not use this function for Stage 1-to-Stage 2 initialization.

- [ ] **Step 5: Run checkpoint tests and selected Trainer tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_checkpointing.py \
  tests/stage1/test_trainer.py -q
```

Expected: all selected tests pass, adapter-only saves contain the four required Stage 1 PEFT files, and numbered resume restores Trainer state.

- [ ] **Step 6: Commit**

```bash
git add src/stage1/checkpointing.py src/stage1/trainer.py \
  tests/stage1/test_checkpointing.py
git commit -m "feat: save and resume stage1 adapters"
```

### Task 10: Route Stage 1 Through an Explicit Runtime Composition and Launcher

**Files:**
- Create: `src/stage1/composition.py`
- Create: `run_stage1_training.sh`
- Modify: `src/main.py`
- Modify: `run_training.sh`
- Test: `tests/stage1/test_composition.py`
- Test: `tests/stage1/test_launcher.py`
- Test: `tests/stage2/test_routing.py`

**Interfaces:**
- Consumes: Tasks 2–9.
- Produces: `Stage1TrainingComponents(model, processor, dataset, collator, trainer, resume_decision, parameter_report)`, `build_stage1_training_components(args) -> Stage1TrainingComponents`, `run_stage1(args) -> None`, and an executable `run_stage1_training.sh`.

- [ ] **Step 1: Write RED tests for source selection and argument propagation**

```python
def test_stage1_runtime_consumes_learning_rate_and_batch(monkeypatch, parsed_args):
    parsed_args.learning_rate = 1e-5
    parsed_args.per_device_train_batch_size = 8
    parsed_args.gradient_accumulation_steps = 2
    bundle = build_stage1_training_components(parsed_args)
    assert bundle.trainer.args.learning_rate == 1e-5
    assert bundle.trainer.args.per_device_train_batch_size == 8
    assert bundle.trainer.args.gradient_accumulation_steps == 2


def test_stage1_fresh_dora_and_resume_are_distinct(tmp_path, dora_artifact):
    fresh = resolve_stage1_sources(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        tuning_method="dora",
        resume_path=None,
    )
    resumed = resolve_stage1_sources(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        tuning_method="dora",
        resume_path=dora_artifact,
    )
    assert fresh.restore_trainer_state is False
    assert resumed.restore_trainer_state is True


def test_stage2_routing_does_not_import_stage1_trainer(monkeypatch, stage2_args):
    spy = monkeypatch.spy(stage2_main, "run_stage2")
    main(stage2_args)
    spy.assert_called_once()
```

Parse `run_stage1_training.sh` in a shell-free test and assert it forwards every approved value, requires explicit data/image/output paths, preserves spaces safely, and refuses an existing nonempty output directory unless an explicit resume is supplied.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_composition.py \
  tests/stage1/test_launcher.py \
  tests/stage2/test_routing.py -q
```

Expected: tests fail because composition and launcher do not exist and `main.py` still hard-codes Stage 1 settings.

- [ ] **Step 3: Implement token-preserving model/processor composition**

`build_stage1_training_components` must:

1. validate/classify all input artifacts before loading large weights;
2. load processor/tokenizer from the fresh base or Stage 1 resume artifact as appropriate;
3. add missing latent special tokens only on a fresh base, record the old vocabulary size, resize once, and validate exact config/tokenizer IDs;
4. load full or adapter weights according to `Stage1ResumeDecision`;
5. build the final full/LoRA/DoRA topology, then construct trainer/EMA;
6. prepare the complete CoMT data with stable row indices and explicit image root;
7. create `SFTConfig` using `args.learning_rate`, batch, accumulation, epochs, seed, attention, DeepSpeed, save strategy, and output path;
8. freeze visual and validate the trainable boundary before `trainer.train`;
9. call the final save exactly once only when `args.final_save` is true, timing that save separately.

Return a single bundle; `src/main.py` becomes:

```python
if args.stage == "stage1":
    run_stage1(args)
elif args.stage == "stage2":
    run_stage2(args)
else:
    raise ValueError(f"Unsupported stage: {args.stage}")
```

- [ ] **Step 4: Implement the explicit launcher**

`run_stage1_training.sh` accepts values through flags/environment and emits the exact command without hidden path edits:

```bash
./run_stage1_training.sh \
  --method dora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /absolute/path/to/CoMT/TRAIN.jsonl \
  --image-root /absolute/path/to/CoMT \
  --output-dir /absolute/path/to/results/stage1-dora \
  --micro-batch 8 \
  --epochs 3 \
  --learning-rate 1e-5 \
  --gradient-accumulation 2 \
  --seed 42 \
  --deepspeed-config "$PWD/config_stage3.json" \
  --attention-implementation flash_attention_2 \
  --save-strategy no \
  --final-save
```

Use `accelerate launch --num_processes 2 --main_process_port 29501 src/main.py` and Bash arrays so user paths are not evaluated as shell code. For `--method full` omit all PEFT flags; for `dora` forward rank 32, alpha 64, dropout 0.05, and the exact seven suffixes. Keep `run_training.sh` functional and forward legacy `--use_lora` when used.

- [ ] **Step 5: Run routing/launcher tests and a dry-run command render**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_composition.py \
  tests/stage1/test_launcher.py \
  tests/stage2/test_routing.py -q

./run_stage1_training.sh --dry-run \
  --method dora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /tmp/comt/TRAIN.jsonl \
  --image-root /tmp/comt \
  --output-dir /tmp/ilvr-stage1-dora-plan-check \
  --micro-batch 8
```

Expected: tests pass and dry-run prints one two-process command with LR `1e-5`, accumulation 2, epochs 3, FlashAttention 2, ZeRO-3, and DoRA settings without starting training.

- [ ] **Step 6: Commit**

```bash
git add src/stage1/composition.py src/main.py run_stage1_training.sh \
  run_training.sh tests/stage1/test_composition.py tests/stage1/test_launcher.py \
  tests/stage2/test_routing.py
git commit -m "feat: route explicit stage1 training"
```

### Task 11: Extend Stage 2 Schema v3 and Continue Stage 1 DoRA

**Files:**
- Modify: `src/stage2/checkpointing.py`
- Modify: `src/stage2/peft.py`
- Modify: `src/stage2/trainer.py`
- Modify: `src/main.py`
- Test: `tests/stage2/test_peft_checkpointing.py`
- Test: `tests/stage2/test_peft.py`
- Test: `tests/stage2/test_stage1_checkpoint_handoff.py`

**Interfaces:**
- Consumes: `Stage1PeftMetadata` and Stage 1 PEFT/token-row loaders from Task 9.
- Produces: `Stage2PeftMetadata` schema v3 fields `token_row_source: Literal["base", "sparse_adapter"]` and `stage1_adapter_reference: str | None`; `migrate_stage2_metadata(raw: Mapping[str, object]) -> Stage2PeftMetadata`; `load_stage1_dora_for_fresh_stage2(base_model, processor, stage1_adapter_path, stage2_tuning) -> nn.Module`.

- [ ] **Step 1: Write RED tests for strict migration and handoff**

```python
def test_complete_v2_metadata_migrates_to_v3(complete_v2_metadata):
    metadata = migrate_stage2_metadata(complete_v2_metadata)
    assert metadata.schema_version == 3
    assert metadata.token_row_source == "base"
    assert metadata.stage1_adapter_reference is None


def test_partial_or_v1_metadata_still_fails(complete_v2_metadata):
    partial = dict(complete_v2_metadata)
    partial.pop("base_model_architecture")
    with pytest.raises(ValueError, match="complete v2"):
        migrate_stage2_metadata(partial)
    with pytest.raises(ValueError, match="schema v1"):
        migrate_stage2_metadata({"schema_version": 1})


def test_stage1_dora_handoff_preserves_adapter_and_token_logits(
    original_base, stage1_dora_artifact, stage2_tuning, fixed_hidden
):
    before = load_stage1_for_inference(original_base(), stage1_dora_artifact)
    after = load_stage1_dora_for_fresh_stage2(
        original_base(), stage1_dora_artifact.processor,
        stage1_dora_artifact.path, stage2_tuning
    )
    assert_adapter_tensors_equal(before, after)
    torch.testing.assert_close(before.lm_head(fixed_hidden), after.lm_head(fixed_hidden))
    assert all_stage1_adapter_and_token_rows_trainable(after)
```

Add tests that Stage 1 adapter plus Stage 2 resume is invalid; rank/alpha/dropout/target/token/architecture mismatch fails before Trainer construction; Stage 1 DoRA→Stage 2 full/new adapter/LoRA fails; full→full and full→DoRA remain supported.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_peft_checkpointing.py \
  tests/stage2/test_peft.py \
  tests/stage2/test_stage1_checkpoint_handoff.py -q
```

Expected: v3 fields and Stage 1 PEFT initializer are absent.

- [ ] **Step 3: Implement strict v3 metadata and artifact classification**

New writes always serialize:

```json
{
  "schema_version": 3,
  "stage": "stage2",
  "token_row_source": "sparse_adapter",
  "stage1_adapter_reference": "/provenance/path/only"
}
```

The real sidecar retains every existing v2 field. Exact complete v2 inputs migrate in memory with `token_row_source="base"` and `stage1_adapter_reference=null`. For sparse source, require `ilvr_token_rows.safetensors` and reject mixed/full layouts. The Stage 2 final artifact copies the current sparse row state so it does not depend on the Stage 1 directory. Treat `stage1_adapter_reference` as provenance, not a load dependency.

- [ ] **Step 4: Implement same-adapter continuation with fresh Stage 2 state**

Load original compatible Qwen, resize from the Stage 1 processor/token metadata, load the existing PEFT adapter as trainable, attach sparse output rows to the active Qwen model, load sparse rows, and validate all Stage 1/Stage 2 DoRA settings exactly. Attaching after PEFT restoration keeps `output_delta` trainable. Do not call `get_peft_model` a second time and do not merge.

Construct Stage 2 Trainer with:

```python
assert trainer.state.global_step == 0
assert trainer.optimizer is None
trainer.train(resume_from_checkpoint=None)
```

The assertion belongs in a test/setup validation, not the hot training loop. Stage 2 resume continues to use `resume_from_checkpoint_path` and restores optimizer/scheduler/RNG/global step normally.

- [ ] **Step 5: Save sparse rows in Stage 2 adapter checkpoints and rerun tests**

Extend `Stage2Trainer.save_model` to save `ilvr_token_rows.safetensors` whenever metadata says `sparse_adapter`. Run the Step 2 command.

Expected: all selected tests pass; deleting the Stage 1 directory after saving Stage 2 does not break Stage 2 reload.

- [ ] **Step 6: Commit**

```bash
git add src/stage2/checkpointing.py src/stage2/peft.py src/stage2/trainer.py \
  src/main.py tests/stage2/test_peft_checkpointing.py \
  tests/stage2/test_peft.py \
  tests/stage2/test_stage1_checkpoint_handoff.py
git commit -m "feat: continue stage1 dora in stage2"
```

### Task 12: Load Sparse Token Rows in Stage 2 Inference

**Files:**
- Modify: `src/evaluate_deepseed.py`
- Modify: `eval.py`
- Modify: `src/stage2/peft.py`
- Test: `tests/stage2/test_peft_inference.py`
- Test: `tests/stage2/test_inference.py`
- Test: `tests/stage2/test_rollout.py`

**Interfaces:**
- Consumes: Stage 2 schema-v3 artifact and `load_sparse_token_rows`.
- Produces: `load_stage2_model_for_inference(base_model_path, stage2_artifact_path, processor) -> nn.Module` that resolves direct or DeepSpeed-wrapped Qwen models with one active adapter and optional sparse token rows.

- [ ] **Step 1: Write inference RED tests**

```python
def test_v3_inference_loads_adapter_and_sparse_rows_without_stage1_dir(
    original_base_path, relocated_stage2_artifact, fixed_hidden
):
    shutil.rmtree(relocated_stage2_artifact.original_stage1_dir)
    model = load_stage2_model_for_inference(
        original_base_path, relocated_stage2_artifact.path,
        relocated_stage2_artifact.processor,
    )
    logits = model.lm_head(fixed_hidden)
    torch.testing.assert_close(
        logits[..., relocated_stage2_artifact.latent_token_ids],
        relocated_stage2_artifact.expected_latent_logits,
    )


def test_deepspeed_wrapped_inference_finds_sparse_rows(wrapped_stage2_model):
    rows = find_sparse_output_token_rows(wrapped_stage2_model)
    assert tuple(rows.token_ids.tolist()) == wrapped_stage2_model.expected_token_ids
```

Retain existing train/inference continuous-latent parity and generation-controller tests; add a v3 sparse-row fixture so the same latent states/logits are compared after reload.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_peft_inference.py \
  tests/stage2/test_inference.py \
  tests/stage2/test_rollout.py -q
```

Expected: v3 sparse artifact cannot yet restore token-row logits.

- [ ] **Step 3: Implement model-specific inference loading**

Classify and validate the Stage 2 artifact before loading the base, load processor/token IDs, resize the original base once, load the single active adapter, attach sparse output rows to the active Qwen model, then load sparse rows. Route `eval.py` and `evaluate_deepseed.py` through this helper. Keep all latent inference state-machine behavior inside the existing model-specific Stage 2 controller.

- [ ] **Step 4: Run inference and generic generation regressions**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage2/test_peft_inference.py \
  tests/stage2/test_inference.py \
  tests/stage2/test_rollout.py \
  tests/stage2/test_generation_regression.py -q
```

Expected: all selected tests pass and generic generation behavior remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/evaluate_deepseed.py eval.py src/stage2/peft.py \
  tests/stage2/test_peft_inference.py \
  tests/stage2/test_inference.py \
  tests/stage2/test_rollout.py
git commit -m "feat: load stage2 sparse token rows"
```

### Task 13: Build Capacity-Probe and Timing Evidence

**Files:**
- Create: `src/stage1/benchmark.py`
- Create: `run_stage1_batch_probe.sh`
- Modify: `run_stage1_training.sh`
- Test: `tests/stage1/test_benchmark.py`
- Test: `tests/stage1/test_launcher.py`

**Interfaces:**
- Consumes: validated Stage 1 launcher from Task 10.
- Produces: `capacity_candidates(start: int, max_per_gpu: int) -> tuple[int, ...]`, `select_high_cost_probe_rows(examples, count, audit_path) -> tuple[int, ...]`, `Stage1BenchmarkEvidence`, `write_benchmark_evidence(path, evidence)`, `load_verified_evidence(path, expected_config_digest) -> Stage1BenchmarkEvidence | None`, a `python -m src.stage1.benchmark` CLI with the exact subcommands used in Tasks 15–18, and a fresh-process probe script.

- [ ] **Step 1: Write search-order, safety, and evidence RED tests**

```python
def test_capacity_candidates_follow_approved_order():
    assert capacity_candidates(start=8, max_per_gpu=64) == (8, 16, 32, 64)
    assert fallback_candidates_after_initial_failure(start=8) == (4, 2, 1)


def test_probe_stops_at_first_failure_without_reusing_process(probe_runner):
    result = probe_runner.run(outcomes={8: "success", 16: "success", 32: "oom"})
    assert result.attempted == (8, 16, 32)
    assert result.maximum_successful == 16
    assert probe_runner.process_ids_are_unique()
    assert result.requires_oom_classification is True


def test_timed_manifest_has_approved_invariants(timed_manifest):
    assert timed_manifest.epochs == 3
    assert timed_manifest.learning_rate == 1e-5
    assert timed_manifest.gradient_accumulation_steps == 2
    assert timed_manifest.world_size == 2
    assert timed_manifest.save_strategy == "no"
    assert timed_manifest.seed == 42
```

Add tests for full/DoRA output separation, probe-attempt path uniqueness, nonfinite metrics rejection, high-cost row audit reuse, config-digest mismatch, no repeated validated evidence, and effective batch formula `2 * micro_batch * 2`.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_benchmark.py \
  tests/stage1/test_launcher.py -q
```

Expected: benchmark module and probe controller are absent.

- [ ] **Step 3: Implement lightweight audit and evidence schema**

`Stage1BenchmarkEvidence` records:

```python
@dataclass(frozen=True)
class Stage1BenchmarkEvidence:
    status: Literal["success", "oom", "failure"]
    method: Literal["full", "dora"]
    micro_batch: int
    world_size: int
    gradient_accumulation_steps: int
    effective_global_batch: int
    optimizer_steps: int
    samples_seen: int
    model_load_seconds: float
    trainer_seconds: float
    total_seconds: float
    final_save_seconds: float
    samples_per_second: float
    steps_per_second: float
    rank_peak_allocated_bytes: tuple[int, ...]
    rank_peak_reserved_bytes: tuple[int, ...]
    trainable_parameters: int
    total_parameters: int
    checkpoint_bytes: int
    final_loss: float
    representative_gradient_norm: float
    git_commit: str
    dependency_versions: dict[str, str]
    gpu_names: tuple[str, ...]
    config_digest: str
    probe_row_indices: tuple[int, ...]
```

Validate all numeric values as finite/nonnegative and every tuple against world size. Audit JSONL text/image references once without reading or hashing image contents; rank candidates by token estimate, original image count, helper count, and latent-block count, then persist selected row indices and the small input-file stat witness.

- [ ] **Step 4: Implement fresh-process probe behavior**

`run_stage1_batch_probe.sh` creates a unique output/log/evidence directory per `method/micro_batch/attempt` and launches `run_stage1_training.sh` with `--max-steps 3 --no-final-save` so work includes an optimizer step and a post-step iteration without serializing a model for every candidate. It starts at 8 and ascends powers of two. On nonzero exit it stops immediately, records stderr/stdout paths, and reports that `superpowers:systematic-debugging` must classify the failure before the previous success is accepted. It never silently retries or lowers batch in the same process.

After a confirmed capacity OOM, the operator invokes the script with `--fallback 4` only if batch 8 was the failed initial candidate. The previous successful candidate then runs a fresh `--max-steps 10 --no-final-save` validation exactly once.

- [ ] **Step 5: Run unit tests and shell syntax checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_benchmark.py \
  tests/stage1/test_launcher.py -q
bash -n run_stage1_training.sh
bash -n run_stage1_batch_probe.sh
```

Expected: tests pass and both scripts pass syntax validation without launching GPUs.

- [ ] **Step 6: Commit**

```bash
git add src/stage1/benchmark.py run_stage1_batch_probe.sh \
  run_stage1_training.sh tests/stage1/test_benchmark.py \
  tests/stage1/test_launcher.py
git commit -m "feat: record stage1 benchmark evidence"
```

### Task 14: Close CPU Integration, Regression, and Documentation Coverage

**Files:**
- Create: `tests/stage1/test_integration.py`
- Modify: `README.md`
- Modify: `tests/stage2/test_routing.py`
- Modify: `tests/stage2/test_rollout.py`

**Interfaces:**
- Consumes: all production components from Tasks 1–13.
- Produces: one tiny deterministic end-to-end witness for each Stage 1 method and documented exact full/DoRA/probe/resume/Stage 2-handoff commands.

- [ ] **Step 1: Write end-to-end RED tests before documentation changes**

```python
@pytest.mark.parametrize("method", ["full", "dora"])
def test_tiny_stage1_b2_train_save_resume_end_to_end(tmp_path, method):
    run = tiny_stage1_run(
        method=method,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=2,
        save_steps=1,
        output_dir=tmp_path / method,
    )
    assert run.student_batch_sizes == [2, 2, 2, 2]
    assert run.original_visual_calls == run.micro_batches
    assert run.helper_visual_calls == run.micro_batches
    assert torch.isfinite(torch.tensor(run.losses)).all()
    resumed = tiny_stage1_resume(run.checkpoint_1, max_steps=3)
    assert resumed.initial_global_step == 1
    assert resumed.final_global_step == 3
    assert resumed.optimizer_state_restored
    assert resumed.scheduler_state_restored


def test_fresh_stage2_handoff_resets_training_state(stage1_dora_checkpoint):
    stage2 = tiny_fresh_stage2(stage1_adapter_path=stage1_dora_checkpoint)
    assert stage2.adapter_matches_stage1
    assert stage2.token_rows_match_stage1
    assert stage2.trainer.state.global_step == 0
    assert stage2.optimizer_state_loaded is False
```

Add a full→full and full→DoRA routing regression, legacy Stage 1 `B=1` loss fixture, and legacy `--use_lora` one-step tiny save fixture.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_integration.py \
  tests/stage2/test_routing.py \
  tests/stage2/test_rollout.py -q
```

Expected: at least one lifecycle/routing assertion fails until the integrated factories expose the required state.

- [ ] **Step 3: Make the minimum integration corrections**

Correct only root causes surfaced by the tests. The end-to-end factory must use the public `build_stage1_training_components` and `CustomTrainerStage1` paths, not a second test-only trainer algorithm. Stage 2 handoff must use the public Stage 2 loader. If any failure is unexpected, invoke `superpowers:systematic-debugging` before changing production code.

- [ ] **Step 4: Document exact workflows and limitations**

Update `README.md` with:

- the actual local example data path `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl` and image root `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data`;
- exact full and DoRA launcher commands;
- independent batch probing and the effective-batch formula;
- Stage 1 full and adapter artifact contents;
- Stage 1 resume versus fresh Stage 2 continuation;
- `--load_model_path` plus `--stage1_adapter_path` source separation;
- no merge/stacking/QLoRA;
- Stage 2 `B=1`, no differentiable KV-cache optimization, and no paper-level reproduction claim;
- evidence JSON locations and the rule to reuse successful expensive evidence.

- [ ] **Step 5: Run the complete CPU-feasible Stage 1 and Stage 2 suites once**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1 \
  tests/stage2 \
  -m "not gpu and not distributed" -q
```

Expected: all collected CPU-feasible tests pass. Record this output once; later tasks reuse it unless code changes after this gate.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/stage1/test_integration.py \
  tests/stage2/test_routing.py \
  tests/stage2/test_rollout.py
git commit -m "test: verify stage1 lifecycle integration"
```

### Task 15: Verify Two-Rank ZeRO-3 B>1 Save/Resume

**Files:**
- Create: `tests/stage1/test_distributed_smoke.py`
- Modify: `src/stage1/benchmark.py`
- Runtime evidence: `outputs/stage1-smoke/full/evidence.json`
- Runtime evidence: `outputs/stage1-smoke/dora/evidence.json`

**Interfaces:**
- Consumes: exact runtime/launcher/checkpoint paths from Tasks 9, 10, and 13.
- Produces: one authenticated two-rank real-Qwen/real-CoMT `B=2` save/resume witness for full and one for DoRA. These witnesses are reused; they are not rerun for final reporting.

- [ ] **Step 1: Write the opt-in distributed acceptance wrapper**

```python
@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.parametrize("method", ["full", "dora"])
def test_two_rank_zero3_b2_save_resume_witness(method, real_comt_paths, evidence_dir):
    evidence = run_or_load_distributed_witness(
        method=method,
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        data_path=real_comt_paths.train,
        image_root=real_comt_paths.root,
        micro_batch=2,
        gradient_accumulation_steps=2,
        world_size=2,
        initial_max_steps=3,
        resumed_max_steps=4,
        deepspeed_config="configs/config_stage3.json",
        evidence_dir=evidence_dir / method,
    )
    assert evidence.initial_final_step == 3
    assert evidence.resume_initial_step == 3
    assert evidence.resume_final_step == 4
    assert evidence.batch_size_observed == 2
    assert evidence.finite_loss
    assert evidence.checkpoint_validated
```

The wrapper first calls `load_verified_evidence`. If matching success evidence exists, it validates only the small manifest and checkpoint marker/digest already produced by the run. It never hashes or rereads the full checkpoint tree.

- [ ] **Step 2: Run a zero-cost collection check**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_distributed_smoke.py \
  --collect-only -q
```

Expected: exactly two parameterized distributed cases collect without initializing CUDA.

- [ ] **Step 3: Check for equivalent successful evidence before GPU work**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark check-evidence \
  --evidence-dir outputs/stage1-smoke/full \
  --method full --micro-batch 2 --world-size 2
./venv/ilvr-venv/bin/python -m src.stage1.benchmark check-evidence \
  --evidence-dir outputs/stage1-smoke/dora \
  --method dora --micro-batch 2 --world-size 2
```

Expected: each command prints either `reusable-success` or `missing`. Run only missing cases in Step 4.

- [ ] **Step 4: Run each missing two-rank witness once**

Run the full case only when its Step 3 result is `missing`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
ILVR_REAL_GPU_TESTS=1 \
ILVR_DISTRIBUTED_METHOD=full \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_distributed_smoke.py::test_two_rank_zero3_b2_save_resume_witness \
  -k full -q -s
```

Run the DoRA case only when its Step 3 result is `missing`:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
ILVR_REAL_GPU_TESTS=1 \
ILVR_DISTRIBUTED_METHOD=dora \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1/test_distributed_smoke.py::test_two_rank_zero3_b2_save_resume_witness \
  -k dora -q -s
```

Expected: each missing case passes once and writes its evidence. On any OOM, wrapper, ZeRO, FlashAttention, or gradient failure, stop and invoke `superpowers:systematic-debugging`; do not lower batch or patch the symptom.

- [ ] **Step 5: Record the small evidence summaries and commit the harness**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark summarize \
  outputs/stage1-smoke/full/evidence.json \
  outputs/stage1-smoke/dora/evidence.json
git add tests/stage1/test_distributed_smoke.py src/stage1/benchmark.py
git commit -m "test: verify stage1 zero3 batch resume"
```

Expected: summary reports two ranks, micro-batch 2, finite losses, and successful step-3→step-4 resume for both methods.

### Task 16: Probe Maximum Full and DoRA Micro-Batches

**Files:**
- Runtime evidence: `outputs/stage1-probes/full/`
- Runtime evidence: `outputs/stage1-probes/dora/`

**Interfaces:**
- Consumes: Task 15 smoke witnesses and the approved search policy from Task 13.
- Produces: one independently validated maximum successful per-GPU micro-batch for full and DoRA, with first-failure classification and longer-smoke evidence for each selected maximum.

- [ ] **Step 1: Verify prerequisite smoke evidence and create/reuse the high-cost row audit**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark require-evidence \
  outputs/stage1-smoke/full/evidence.json \
  outputs/stage1-smoke/dora/evidence.json
./venv/ilvr-venv/bin/python -m src.stage1.benchmark audit \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output outputs/stage1-probes/high-cost-rows.json \
  --count 64
```

Expected: prerequisites are accepted and the audit records 64 stable source-row indices. If an unchanged valid audit exists, the command reuses it without rereading image contents.

- [ ] **Step 2: Probe full tuning from batch 8 in fresh processes**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_batch_probe.sh \
  --method full \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output-root outputs/stage1-probes/full \
  --probe-rows outputs/stage1-probes/high-cost-rows.json \
  --start 8 \
  --gradient-accumulation 2 \
  --learning-rate 1e-5 \
  --seed 42
```

Expected: fresh process IDs for 8, 16, 32, … and a stop at the first failure. If 8 fails, invoke `superpowers:systematic-debugging`. A classified `integration_failure` is retried append-only at batch 8 by rerunning the same primary command; only a confirmed `capacity_oom` permits explicitly rerunning with `--start 4 --descending`. No automatic fallback occurs.

- [ ] **Step 3: Classify the first full failure and validate its preceding success**

Invoke `superpowers:systematic-debugging` for the first failed candidate. Reproduce from the recorded command/log, inspect rank-local sequence shapes and memory growth, test the single capacity hypothesis, and record `capacity_oom` only when the process is clean and memory is bounded across iterations. Then run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark validate-maximum \
  --probe-dir outputs/stage1-probes/full \
  --long-smoke-steps 10
```

Expected: a fresh-process longer smoke succeeds at the prior candidate and writes `outputs/stage1-probes/full/maximum.json`. If no candidate failed before the dataset-imposed upper bound, the highest meaningful candidate is recorded without manufacturing an OOM.

- [ ] **Step 4: Probe and classify DoRA independently**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_batch_probe.sh \
  --method dora \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data-path /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data/TRAIN.jsonl \
  --image-root /NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/data \
  --output-root outputs/stage1-probes/dora \
  --probe-rows outputs/stage1-probes/high-cost-rows.json \
  --start 8 \
  --gradient-accumulation 2 \
  --learning-rate 1e-5 \
  --seed 42 \
  --dora-rank 32 --dora-alpha 64 --dora-dropout 0.05
```

Invoke `superpowers:systematic-debugging` at its first failure, then validate the preceding success:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark validate-maximum \
  --probe-dir outputs/stage1-probes/dora \
  --long-smoke-steps 10
```

Expected: `outputs/stage1-probes/dora/maximum.json` records an independently selected batch. If batch 8 has an `integration_failure`, rerun the same primary command for an append-only batch-8 retry; only a confirmed `capacity_oom` permits the explicit `--start 4 --descending` fallback policy.

- [ ] **Step 5: Compare small manifests without rerunning probes**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark compare-maxima \
  outputs/stage1-probes/full/maximum.json \
  outputs/stage1-probes/dora/maximum.json
```

Expected: report lists each selected per-GPU batch, effective global batch, ten-step validation result, peak memory, and first-failure classification. This step only reads small evidence files.

### Task 17: Run the Two Approved Three-Epoch Experiments Once

**Files:**
- Runtime artifact: `outputs/stage1-timed/full-3epoch/`
- Runtime artifact: `outputs/stage1-timed/dora-3epoch/`
- Runtime evidence: `outputs/stage1-timed/full-3epoch/evidence.json`
- Runtime evidence: `outputs/stage1-timed/dora-3epoch/evidence.json`

**Interfaces:**
- Consumes: Task 16 `maximum.json` witnesses.
- Produces: exactly one successful full and one successful DoRA three-epoch run at their independently validated maximum micro-batches, each with final artifact and timing/memory/throughput evidence.

- [ ] **Step 1: Resolve the validated batches and reject stale/colliding outputs**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark prepare-timed-run \
  --method full \
  --maximum outputs/stage1-probes/full/maximum.json \
  --output-dir outputs/stage1-timed/full-3epoch \
  --epochs 3 --learning-rate 1e-5 --gradient-accumulation 2 --world-size 2
./venv/ilvr-venv/bin/python -m src.stage1.benchmark prepare-timed-run \
  --method dora \
  --maximum outputs/stage1-probes/dora/maximum.json \
  --output-dir outputs/stage1-timed/dora-3epoch \
  --epochs 3 --learning-rate 1e-5 --gradient-accumulation 2 --world-size 2
```

Expected: each command prints one exact immutable launch manifest or `reusable-success`. It fails rather than overwriting a nonempty incomplete/colliding directory.

- [ ] **Step 2: Run full training only if no equivalent success exists**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_training.sh \
  --manifest outputs/stage1-timed/full-3epoch/launch-manifest.json
```

Expected: all 3,450 samples train for exactly 3 epochs with LR `1e-5`, accumulation 2, seed 42, FlashAttention 2, ZeRO-3, no intermediate checkpoints, finite loss/gradients, and one final full-model save. Evidence separates model load, Trainer, wall, and final-save times.

- [ ] **Step 3: Validate the full result once**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark validate-timed-run \
  --run-dir outputs/stage1-timed/full-3epoch
```

Expected: validation reads the producer's final small manifest plus only changed/small output metadata, confirms expected optimizer steps from dataset/world-size/batch/accumulation, and seals `evidence.json`. It does not recursively hash or reread the immutable Qwen base or the full checkpoint a second time.

- [ ] **Step 4: Run DoRA training only if no equivalent success exists**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./run_stage1_training.sh \
  --manifest outputs/stage1-timed/dora-3epoch/launch-manifest.json
```

Expected: all 3,450 samples train for exactly 3 epochs with the independently validated DoRA batch, LR `1e-5`, accumulation 2, seed 42, exact r32/alpha64/dropout0.05 targets, frozen visual/base, exactly three active input/output token rows, no intermediate checkpoints, and one final adapter-only save.

- [ ] **Step 5: Validate the DoRA result once**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark validate-timed-run \
  --run-dir outputs/stage1-timed/dora-3epoch
```

Expected: `adapter_config.json`, `adapter_model.safetensors`, `ilvr_token_rows.safetensors`, `stage1_peft_metadata.json`, processor/tokenizer, and sealed `evidence.json` exist; no full base-model weight file exists.

- [ ] **Step 6: Produce the comparison from sealed evidence**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark compare-runs \
  outputs/stage1-timed/full-3epoch/evidence.json \
  outputs/stage1-timed/dora-3epoch/evidence.json \
  --output outputs/stage1-timed/comparison.json
```

Expected: report includes effective batches, optimizer steps, model load/Trainer/wall/save time, throughput, peak allocated/reserved bytes per rank, trainable/total parameters, finite loss/gradient witness, final artifact bytes, git commit, dependencies, and GPU inventory. It states that this is a maximum-throughput comparison and not a common-batch or paper-level reproduction.

### Task 18: Final Requirement Audit, Review, and Verification

**Files:**
- Modify only files required by review findings.
- Inspect: `docs/superpowers/specs/2026-08-26-ilvr-stage1-dora-batched-training-design.md`
- Inspect: `docs/superpowers/plans/2026-08-26-ilvr-stage1-dora-batched-training.md`
- Inspect: the complete branch diff.

**Interfaces:**
- Consumes: all implementation commits and sealed CPU/GPU/experiment evidence.
- Produces: final reviewed branch and a report that distinguishes implementation, distributed smoke, capacity probe, three-epoch experiments, Stage 2 handoff, and non-performed paper reproduction.

- [ ] **Step 1: Invoke the required verification and review skills**

Invoke `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`. The review must compare the actual diff against every binding design requirement, not merely test names.

- [ ] **Step 2: Re-read spec and plan and build a requirement-to-evidence table**

Record, for every requirement, its implementation file/test and latest evidence. Explicit rows include:

```text
B=1 parity
B>1 sample isolation and one student forward
one teacher text forward and one helper visual call per batch
full and DoRA gradients/freeze boundaries
name/shape EMA and ZeRO gathering
exact three input/output latent rows
Stage 1 full and adapter save/resume
Stage 1 DoRA -> fresh Stage 2 DoRA state reset
Stage 2 v2->v3 migration and relocated inference
legacy B=1 and --use_lora
two-rank ZeRO-3 B>1 witnesses
independent capacity maxima
full/DoRA three-epoch evidence
Stage 2 remains B=1 and recurrence/generation unchanged
```

Any missing row is unfinished work, not a reporting caveat.

- [ ] **Step 3: Inspect the complete diff and static red flags**

Run:

```bash
git status --short
git diff --stat "$(git merge-base HEAD feat/vcsd-ilvr-stage2)"..HEAD
git diff --check "$(git merge-base HEAD feat/vcsd-ilvr-stage2)"..HEAD
rg -n \
  "token_id\\s*=\\s*[0-9]{3,}|pixel_values_latent.*model\\(|image_grid_thw_latent.*model\\(|zip\\(.*named_parameters|\\.merge_and_unload\\(" \
  src transformers/src/transformers/models/qwen2_5_vl \
  run_stage1_training.sh run_stage1_batch_probe.sh
```

Expected: no whitespace errors, no hard-coded latent IDs, no official helper-image student forwarding, no positional EMA update, and no adapter merge. Inspect every match manually because declarations/tests may be benign.

- [ ] **Step 4: Run one fresh CPU-feasible verification after final code changes**

If no production/test code changed after Task 14's CPU witness, reuse that witness. Otherwise run exactly once:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD:$PWD/transformers/src:$PWD/src:$PWD/tests/stage1:$PWD/tests/stage2" \
./venv/ilvr-venv/bin/pytest \
  tests/stage1 \
  tests/stage2 \
  -m "not gpu and not distributed" -q
```

Expected: all CPU-feasible Stage 1 and Stage 2 tests pass.

- [ ] **Step 5: Reuse and summarize GPU evidence without rerunning it**

Run:

```bash
./venv/ilvr-venv/bin/python -m src.stage1.benchmark final-report \
  --smoke-full outputs/stage1-smoke/full/evidence.json \
  --smoke-dora outputs/stage1-smoke/dora/evidence.json \
  --maximum-full outputs/stage1-probes/full/maximum.json \
  --maximum-dora outputs/stage1-probes/dora/maximum.json \
  --timed-full outputs/stage1-timed/full-3epoch/evidence.json \
  --timed-dora outputs/stage1-timed/dora-3epoch/evidence.json \
  --output outputs/stage1-timed/final-report.json
```

Expected: all six small manifests validate their configuration digests and success states. This command does not launch CUDA, rerun training, hash model weights, or traverse checkpoint trees.

- [ ] **Step 6: Resolve review findings with TDD and focused verification**

For every valid review finding, write or tighten the smallest failing regression test, confirm the expected failure, implement the root-cause correction, and rerun only that focused test group. Invoke `superpowers:systematic-debugging` for unexpected behavior. Commit each coherent correction:

```bash
git add -A -- src tests README.md \
  run_stage1_training.sh run_stage1_batch_probe.sh run_training.sh eval.py
git diff --cached --check
git commit -m "fix: address stage1 review findings"
```

If review finds no issue, do not run these staging/commit commands and do not create an empty commit.

- [ ] **Step 7: Final branch and report check**

Run:

```bash
git status --short
git log --oneline --decorate -20
```

Expected: worktree clean. The final Korean report lists files created/modified, architecture, exact commands, CPU test counts/results, two-rank evidence, selected batches/effective batches, timings/memory/throughput/save sizes, Stage 2 handoff result, limitations, and skipped work. State explicitly: Stage 1 `B>1` is supported; Stage 2 `B>1` is not; no KV-cache optimization, merge, QLoRA, or paper-level CoMT reproduction was performed. Do not merge or push.
