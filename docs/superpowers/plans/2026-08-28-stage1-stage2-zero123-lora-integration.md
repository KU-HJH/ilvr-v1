# Stage 1/2 ZeRO-1/2/3 and LoRA Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the existing Stage 1/2 distributed implementation into `main`, promote Stage 1 LoRA to the canonical ZeRO-1/2/3 path, start batch probing at micro-batch 2, and verify all 18 Stage/tuning/ZeRO save-resume lifecycles.

**Architecture:** Merge the already-implemented `feat/vcsd-ilvr-stage2` branch instead of rebuilding its Stage 2, checkpoint, sampler, and DeepSpeed layers. Apply the approved Stage 2 PEFT fixes, then make the smallest Stage 1 routing and diagnostic changes needed to treat LoRA like Full and DoRA. Keep performance comparison schemas Full/DoRA-only, while compatibility diagnostics cover Full/LoRA/DoRA.

**Tech Stack:** Python 3.11, PyTorch 2.7.1, DeepSpeed 0.18.0, Transformers/TRL/PEFT, Bash, pytest, two NVIDIA H100 GPUs

**Spec:** `docs/superpowers/specs/2026-08-28-stage1-stage2-zero123-lora-integration-design.md`

## Global Constraints

- Invoke Python, pytest, torchrun, Accelerate, and training only through `./venv/ilvr-venv`.
- Preserve all pre-existing user changes; never stage deleted `src/__pycache__` files or unrelated untracked files.
- Write generated weights only below `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/checkpoints/zero-stage-compatibility/`.
- Write generated results only below `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/results/zero-stage-compatibility/`.
- Write generated logs only below `/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/logs/zero-stage-compatibility/`.
- Do not convert optimizer state between ZeRO stages or world sizes; resume requires the same ZeRO stage, config digest, and world size.
- Do not run full-size 3B/7B training or a performance benchmark in this task.
- Capacity probing starts at micro-batch/GPU 2; only a confirmed CUDA capacity OOM at 2 permits fallback to 1.
- Use one focused unit-test set plus the 18 bounded two-rank lifecycle checks; do not repeat broad test cycles.

---

### Task 1: Integrate the committed Stage 1/2 implementation

**Files:**
- Merge: `feat/vcsd-ilvr-stage2` into `main`
- Preserve: all current untracked files and deleted tracked bytecode

**Interfaces:**
- Consumes: committed feature head `82d10d8`; current design commits on `main`
- Produces: `src/stage1/`, `src/stage2/`, canonical DeepSpeed configs, Stage 1/2 launchers, model-fork changes, and their committed tests in the current checkout

- [ ] **Step 1: Capture the exact pre-merge user-change inventory**

Run:

```bash
git -c safe.directory=/root/workspace/ILVR status --short --untracked-files=all \
  > /tmp/ilvr-pre-merge-status.txt
git -c safe.directory=/root/workspace/ILVR diff --name-only \
  > /tmp/ilvr-pre-merge-tracked.txt
```

Expected: only the known deleted bytecode is tracked-dirty; untracked user files remain listed. These files are evidence only and must not be staged.

- [ ] **Step 2: Merge the feature branch without rewriting either history**

Run:

```bash
git -c safe.directory=/root/workspace/ILVR merge --no-ff feat/vcsd-ilvr-stage2 \
  -m "merge: integrate Stage 1 and Stage 2 distributed training"
```

Expected: a merge commit containing the feature implementation and both design commits as parents. If an untracked-file collision occurs, stop and resolve only the exact collision without moving or deleting unrelated user files.

- [ ] **Step 3: Verify user changes survived the merge**

Run:

```bash
git -c safe.directory=/root/workspace/ILVR status --short --untracked-files=all
git -c safe.directory=/root/workspace/ILVR diff --name-only
```

Expected: deleted bytecode and unrelated untracked files remain; no newly modified training source appears before Task 2.

- [ ] **Step 4: Run the smallest merged-baseline import/routing check**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/test_deepspeed_runtime.py \
  tests/stage1/test_cli.py \
  tests/stage2/test_routing.py
```

Expected: all selected committed-branch tests pass before new LoRA behavior is introduced.

---

### Task 2: Preserve the approved Stage 2 PEFT/checkpoint fixes

**Files:**
- Modify: `tests/stage2/test_peft.py`
- Modify: `tests/stage2/test_peft_checkpointing.py`
- Modify: `src/stage2/peft.py`
- Modify: `src/stage2/checkpointing.py`
- Modify: `src/stage2/trainer.py`

**Interfaces:**
- Consumes: feature-branch `Stage2Trainer`, `Stage2TuningConfig`, and PEFT metadata validation
- Produces: resolved target-selector validation and adapter-only trainer saves that exclude resized base embeddings

- [ ] **Step 1: Add the compact-target and adapter-only regression tests**

Add tests with these assertions:

```python
def test_stage2_peft_accepts_optimized_target_selectors_for_multilayer_decoder():
    base = tiny_stage2_qwen()
    base.model.layers = nn.ModuleList(
        [deepcopy(base.model.layers[0]) for _ in range(3)]
    )
    base.config.num_hidden_layers = 3
    wrapped = build_stage2_tuned_model(base, settings("dora"))
    report = validate_stage2_parameter_boundary(wrapped, settings("dora"))
    assert len(report.resolved_target_names) == 21


def test_stage2_trainer_peft_save_excludes_resized_base_embeddings(
    tmp_path, isolated_trainer_deepspeed_state
):
    source = tmp_path / "original-base"
    source.mkdir()
    base = tiny_stage2_qwen()
    source_config = deepcopy(base.config)
    source_config.vocab_size -= 3
    source_config.save_pretrained(source)
    tuning = Stage2TuningConfig(
        method="dora", r=4, alpha=8, dropout=0.0,
        target_modules=("q_proj",),
    )
    metadata = tiny_trainer_metadata(base, tuning)
    model = build_stage2_tuned_model(base, tuning)
    model.peft_config["default"].base_model_name_or_path = str(source)
    trainer = Stage2Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), report_to=[], use_cpu=True
        ),
        stage2_peft_metadata=metadata,
        deepspeed_runtime_profile=runtime_profile(),
    )
    trainer.save_model(tmp_path)
    saved_keys = set(load_file(tmp_path / "adapter_model.safetensors"))
    assert saved_keys
    assert all("lora_" in key for key in saved_keys)
```

Also add this relocated-base regression:

```python
def test_relocated_stage1_base_accepts_compact_saved_target_selectors(tmp_path):
    metadata, original_adapter_config = write_tiny_adapter(tmp_path)
    adapter_config = deepcopy(original_adapter_config)
    adapter_config.target_modules = {"q_proj", "v_proj"}
    validate_stage2_peft_compatibility(
        relocated_base(), TinyTokenizer(), metadata, adapter_config, tmp_path
    )
```

- [ ] **Step 2: Run the new tests and verify the committed branch fails causally**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_peft.py::test_stage2_peft_accepts_optimized_target_selectors_for_multilayer_decoder \
  tests/stage2/test_peft_checkpointing.py::test_relocated_stage1_base_accepts_compact_saved_target_selectors \
  tests/stage2/test_peft_checkpointing.py::test_stage2_trainer_peft_save_excludes_resized_base_embeddings
```

Expected: failures show compact selectors being compared as resolved paths and PEFT auto-saving resized embeddings.

- [ ] **Step 3: Resolve configured selectors against actual decoder module paths**

In `src/stage2/peft.py`, use PEFT's resolved module list and independently validate configured patterns:

```python
configured_targets = config.target_modules
target_names = tuple(getattr(model.base_model, "targeted_module_names", ()))
configured_matches = {
    name
    for name in modules
    if any(name == pattern or name.endswith(f".{pattern}")
           for pattern in configured_targets)
}
if configured_matches != expected_targets or set(target_names) != expected_targets:
    raise ValueError("loaded adapter targets do not match Stage 2 tuning")
```

Apply the same suffix-to-resolved-name comparison in `validate_stage2_peft_compatibility`; reject strings, empty selectors, duplicate/invalid entries, or non-sequences.

- [ ] **Step 4: Override Stage 2 PEFT `_save` with adapter-only semantics**

In `Stage2Trainer._save`, unwrap the model and delegate Full models to `Trainer._save`. For `PeftModel`, call:

```python
model.save_pretrained(
    output_dir,
    state_dict=state_dict,
    safe_serialization=self.args.save_safetensors,
    save_embedding_layers=False,
)
```

Then save the processing class and `TRAINING_ARGS_NAME` exactly as the base Trainer contract requires.

- [ ] **Step 5: Run the focused Stage 2 PEFT tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage2/test_peft.py \
  tests/stage2/test_peft_checkpointing.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit only the approved Stage 2 fix files**

```bash
git -c safe.directory=/root/workspace/ILVR add \
  src/stage2/checkpointing.py src/stage2/peft.py src/stage2/trainer.py \
  tests/stage2/test_peft.py tests/stage2/test_peft_checkpointing.py
git -c safe.directory=/root/workspace/ILVR commit \
  -m "fix: preserve Stage 2 adapter-only checkpoints"
```

---

### Task 3: Promote Stage 1 LoRA to the canonical launcher

**Files:**
- Modify: `tests/stage1/test_cli.py`
- Modify: `tests/stage1/test_launcher.py`
- Modify: `src/utils_deepseed.py`
- Modify: `run_stage1_training.sh`

**Interfaces:**
- Consumes: `Stage1TuningConfig(method="lora")`, existing Stage 1 PEFT composition, and legacy `--use_lora`
- Produces: canonical `--method full|lora|dora`; explicit `--tuning_method lora`; backward-compatible `--use_lora` alias

- [ ] **Step 1: Replace the Stage 1 LoRA rejection test with explicit acceptance**

Add:

```python
def test_stage1_explicit_lora_is_a_canonical_tuning_method():
    args = parse_args(["--stage", "stage1", "--tuning_method", "lora"])
    tuning = resolve_stage1_tuning_config(args)
    assert tuning.method == "lora"
    assert args.tuning_method_explicit is True
```

Retain the existing test proving `--use_lora` resolves to LoRA and the test rejecting `--use_lora --tuning_method dora`.

- [ ] **Step 2: Add launcher tests for canonical LoRA and its config path**

Add a dry-run test that invokes `run_stage1_training.sh --method lora --deepspeed-config configs/deepspeed_zero2.json` and asserts:

```python
assert command[command.index("--tuning_method") + 1] == "lora"
assert "--use_lora" not in command
assert command[command.index("--deepspeed_config") + 1].endswith(
    "configs/deepspeed_zero2.json"
)
```

- [ ] **Step 3: Run the two new tests and verify they fail**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage1/test_cli.py::test_stage1_explicit_lora_is_a_canonical_tuning_method \
  tests/stage1/test_launcher.py::test_stage1_launcher_routes_canonical_lora
```

Expected: parser says Stage 1 accepts only Full/DoRA and launcher emits legacy `--use_lora`.

- [ ] **Step 4: Allow explicit Stage 1 LoRA in parser validation**

Change the Stage 1 validation to:

```python
if args.tuning_method_explicit and args.tuning_method not in (
    "full", "lora", "dora"
):
    raise ValueError(
        "Stage 1 --tuning_method accepts only full, lora, or dora"
    )
```

Do not weaken the contradiction guard between `--use_lora` and an explicit method.

- [ ] **Step 5: Route canonical LoRA explicitly in the launcher**

Update help and validation to `full|lora|dora`. Emit `--tuning_method "$method"` for all methods, and add rank/alpha/dropout/target arguments when `method` is either `lora` or `dora`. Keep `--use_lora` accepted only as a compatibility input that normalizes to `method=lora` before command construction.

- [ ] **Step 6: Run the Stage 1 CLI and launcher suites**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage1/test_cli.py tests/stage1/test_launcher.py
```

Expected: all tests pass, including legacy alias coverage.

- [ ] **Step 7: Commit canonical Stage 1 LoRA routing**

```bash
git -c safe.directory=/root/workspace/ILVR add \
  src/utils_deepseed.py run_stage1_training.sh \
  tests/stage1/test_cli.py tests/stage1/test_launcher.py
git -c safe.directory=/root/workspace/ILVR commit \
  -m "feat: promote Stage 1 LoRA to canonical training"
```

---

### Task 4: Extend the Stage 1 ZeRO lifecycle diagnostic to LoRA

**Files:**
- Modify: `tests/stage1/zero_compat_diagnostic.py`
- Modify: `tests/stage1/test_distributed_smoke.py`
- Modify: `tests/stage1/test_integration.py`

**Interfaces:**
- Consumes: `build_stage1_training_components`, canonical `deepspeed_zero{1,2,3}.json`, Stage 1 PEFT artifact classifier
- Produces: one bounded Full/LoRA/DoRA save-resume witness per ZeRO stage

- [ ] **Step 1: Add LoRA to every diagnostic parser contract**

Change parser/test method sets from `("full", "dora")` to:

```python
SUPPORTED_DIAGNOSTIC_METHODS = ("full", "lora", "dora")
```

Parameterize parser coverage and the opt-in distributed test over all three values.

- [ ] **Step 2: Add LoRA artifact assertions to the tiny lifecycle**

After `checkpoint-1` is created, inspect based on method:

```python
if args.method in ("lora", "dora"):
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "ilvr_token_rows.safetensors",
        "stage1_peft_metadata.json",
    }
    missing = sorted(name for name in required if not (checkpoint / name).is_file())
    if missing:
        raise AssertionError(f"Stage 1 PEFT checkpoint is incomplete: {missing}")
    if (checkpoint / "model.safetensors").exists():
        raise AssertionError("Stage 1 PEFT checkpoint contains full model weights")
```

Record `artifact_kind` and `adapter_checkpoint_present` in the witness. Assert LoRA metadata reports `method="lora"` and PEFT config `use_dora=False`.

- [ ] **Step 3: Run parser and CPU tiny-integration tests first**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage1/test_distributed_smoke.py::test_stage1_diagnostic_accepts_all_supported_pairs \
  tests/stage1/test_integration.py::test_legacy_use_lora_one_step_saves_adapter_checkpoint
```

Expected: all pass after diagnostic updates; no GPU lifecycle runs in this task.

- [ ] **Step 4: Commit the diagnostic expansion**

```bash
git -c safe.directory=/root/workspace/ILVR add \
  tests/stage1/zero_compat_diagnostic.py \
  tests/stage1/test_distributed_smoke.py \
  tests/stage1/test_integration.py
git -c safe.directory=/root/workspace/ILVR commit \
  -m "test: cover Stage 1 LoRA ZeRO lifecycles"
```

---

### Task 5: Start every Stage 1 batch check at micro-batch 2

**Files:**
- Modify: `src/stage1/benchmark.py`
- Modify: `src/stage1/zero_matrix.py`
- Modify: `run_stage1_batch_probe.sh`
- Modify: `run_stage1_zero_matrix.sh`
- Modify: `tests/stage1/test_benchmark.py`
- Modify: `tests/stage1/test_zero_matrix.py`

**Interfaces:**
- Consumes: existing append-only capacity-attempt records and fixed-EGB-16 matrix identities
- Produces: primary capacity order `2,4,8,...`, fallback `1` after confirmed batch-2 OOM, matrix order `(2,4),(4,2),(8,1),(1,8)`

- [ ] **Step 1: Write ordering and failure-classification tests**

Add/update exact assertions:

```python
assert capacity_candidates(start=2, max_per_gpu=64) == (2, 4, 8, 16, 32, 64)
assert fallback_candidates_after_initial_failure(start=2) == (1,)
with pytest.raises(ValueError, match="initial batch 2"):
    fallback_candidates_after_initial_failure(start=4)

assert [(c.micro_batch, c.gradient_accumulation_steps)
        for c in ZERO_MATRIX_CANDIDATES] == [
    (2, 4), (4, 2), (8, 1), (1, 8)
]
```

Update subprocess probe tests so their first recorded argv has micro-batch 2. Add one test proving an integration failure at 2 cannot authorize batch 1 and one proving a classified capacity OOM at 2 can.

- [ ] **Step 2: Run the focused ordering tests and verify old batch-8 assumptions fail**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage1/test_benchmark.py -k 'capacity or fallback or probe' \
  tests/stage1/test_zero_matrix.py -k 'candidate or plan'
```

Expected: failures identify batch-8 defaults/order and old fallback maps.

- [ ] **Step 3: Change the capacity state machine from initial 8 to initial 2**

Update these exact contracts in `benchmark.py`:

```python
def capacity_candidates(start: int, max_per_gpu: int) -> tuple[int, ...]:
    if start != 2:
        raise ValueError("primary capacity probing must start at batch 2")
    if max_per_gpu < start:
        raise ValueError("max_per_gpu must be at least the initial batch 2")
    candidates = []
    candidate = start
    while candidate <= max_per_gpu:
        candidates.append(candidate)
        candidate *= 2
    return tuple(candidates)

def fallback_candidates_after_initial_failure(start: int) -> tuple[int, ...]:
    if start != 2:
        raise ValueError("fallback is allowed only after the initial batch 2 failure")
    return (1,)
```

In `_validated_capacity_chain`, `_require_candidate`, `_require_initial_oom`, and `_require_fallback`, replace the initial/fallback transition with `2 -> 1`; retain same-candidate retry after `integration_failure` and forbid any launch after the first ascending capacity failure.

- [ ] **Step 4: Update both launcher orders**

Set `run_stage1_batch_probe.sh` defaults to `start=2`, require `max_per_gpu >= 2`, and allow only fallback 1. Set:

```bash
micro_batches=(2 4 8 1)
accumulations=(4 2 1 8)
```

in `run_stage1_zero_matrix.sh`. Keep `_MATRIX_METHODS = ("full", "dora")`; LoRA compatibility is verified by Task 7 and is not added to the performance comparison schema.

- [ ] **Step 5: Update the Python matrix tuple in the same order**

```python
ZERO_MATRIX_CANDIDATES = (
    ZeroBatchCandidate(2, 4),
    ZeroBatchCandidate(4, 2),
    ZeroBatchCandidate(8, 1),
    ZeroBatchCandidate(1, 8),
)
```

Selection must still require exactly these four fixed-EGB-16 identities, now in the new order.

- [ ] **Step 6: Run the complete focused batch/matrix tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/stage1/test_benchmark.py \
  tests/stage1/test_zero_matrix.py \
  tests/stage1/test_launcher.py
```

Expected: all tests pass and subprocess records start with batch 2.

- [ ] **Step 7: Commit the batch-start change**

```bash
git -c safe.directory=/root/workspace/ILVR add \
  src/stage1/benchmark.py src/stage1/zero_matrix.py \
  run_stage1_batch_probe.sh run_stage1_zero_matrix.sh \
  tests/stage1/test_benchmark.py tests/stage1/test_zero_matrix.py
git -c safe.directory=/root/workspace/ILVR commit \
  -m "feat: start Stage 1 batch probes at two"
```

---

### Task 6: Update operator documentation and run one focused unit gate

**Files:**
- Modify: `docs/zero-stage-training.md`
- Modify: `README.md`
- Verify: Stage 1/2 routing, checkpoint, sampler, PEFT, and runtime tests

**Interfaces:**
- Consumes: canonical launcher/config behavior from Tasks 2-5
- Produces: operator examples for Stage 1 LoRA and batch-2-first probing

- [ ] **Step 1: Update the supported-method table and examples**

Document exactly:

```text
Stage 1: Full / LoRA / DoRA -> ZeRO-1 / ZeRO-2 / ZeRO-3
Stage 2: Full / LoRA / DoRA -> ZeRO-1 / ZeRO-2 / ZeRO-3
```

Add a Stage 1 command using `--method lora --deepspeed-config configs/deepspeed_zero2.json`. State that capacity probes begin at micro-batch 2 and that different ZeRO identities cannot share optimizer resume state.

- [ ] **Step 2: Run the focused unit gate once**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q \
  tests/test_deepspeed_runtime.py \
  tests/stage1/test_cli.py \
  tests/stage1/test_launcher.py \
  tests/stage1/test_composition.py \
  tests/stage1/test_ema.py \
  tests/stage1/test_checkpointing.py \
  tests/stage1/test_trainer.py \
  tests/stage1/test_benchmark.py \
  tests/stage1/test_zero_matrix.py \
  tests/stage2/test_routing.py \
  tests/stage2/test_sampler.py \
  tests/stage2/test_peft.py \
  tests/stage2/test_peft_checkpointing.py \
  tests/stage2/integration/test_distributed_contract.py
```

Expected: zero failures. Do not run the repository-wide suite after this focused gate.

- [ ] **Step 3: Commit documentation**

```bash
git -c safe.directory=/root/workspace/ILVR add README.md docs/zero-stage-training.md
git -c safe.directory=/root/workspace/ILVR commit \
  -m "docs: describe LoRA across canonical ZeRO stages"
```

---

### Task 7: Verify all 18 real two-rank DeepSpeed lifecycles

**Files:**
- Generate: canonical checkpoint/result/log directories, one experiment ID per lifecycle
- Verify: `tests/stage1/zero_compat_diagnostic.py`
- Verify: `tests/stage2/integration/tiny_stage2_train.py`

**Interfaces:**
- Consumes: two H100 GPUs, canonical configs, tiny model/data dependencies, Tasks 1-6
- Produces: 18 JSON witnesses with steps 1->2, finite values, artifact identity, ZeRO stage/config digest, world size 2, and tuning method

- [ ] **Step 1: Confirm GPUs and reject a real resource conflict**

Run:

```bash
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
ps -eo pid,etimes,args | rg '/root/workspace/ILVR|accelerate|torchrun|src/main.py' | rg -v 'rg ' || true
```

Expected: GPUs 0 and 1 are available and no foreign training process owns them. If they are busy, wait rather than signal another process.

- [ ] **Step 2: Run the nine Stage 1 lifecycles once**

Run:

```bash
zero_result_root=/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/results/zero-stage-compatibility
zero_log_root=/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/logs/zero-stage-compatibility
witness_paths=()
for zero_stage in 1 2 3; do
  for method in full lora dora; do
    suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
    experiment_id="stage1-zero${zero_stage}-${method}-tiny-lifecycle-${suffix}"
    mkdir -p -- "$zero_log_root"
    mkdir -- "$zero_log_root/$experiment_id"
    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
      ./venv/ilvr-venv/bin/torchrun --standalone --nproc_per_node=2 \
      tests/stage1/zero_compat_diagnostic.py \
      --zero-stage "$zero_stage" --method "$method" \
      --deepspeed "configs/deepspeed_zero${zero_stage}.json" \
      --experiment-id "$experiment_id" \
      >"$zero_log_root/$experiment_id/stdout.log" \
      2>"$zero_log_root/$experiment_id/stderr.log" || exit $?
    witness_paths+=("$zero_result_root/$experiment_id/witness.json")
  done
done
./venv/ilvr-venv/bin/python - "${witness_paths[@]}" <<'PY'
import json
import sys

witnesses = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
observed = {(item["method"], item["zero_stage"]) for item in witnesses}
expected = {(method, zero) for method in ("full", "lora", "dora")
            for zero in (1, 2, 3)}
assert observed == expected
assert all(item["stage"] == "stage1" for item in witnesses)
assert all(item["world_size"] == 2 for item in witnesses)
assert all(item["initial_global_step"] == 1 for item in witnesses)
assert all(item["resumed_global_step"] == 2 for item in witnesses)
print("validated Stage 1 lifecycles:", len(witnesses))
PY
```

Expected: nine zero exits. Each witness records `initial_global_step=1`, `resumed_global_step=2`, a finite representative gradient norm, method, ZeRO identity, world size 2, and the correct Full or PEFT artifact kind.

- [ ] **Step 3: Run the nine Stage 2 lifecycles once**

Run:

```bash
zero_result_root=/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/results/zero-stage-compatibility
zero_log_root=/NHNHOME/WORKSPACE/26msit001_A/PRML_Lab/hjh/workspace/ILVR/logs/zero-stage-compatibility
witness_paths=()
for zero_stage in 1 2 3; do
  for method in full lora dora; do
    suffix="$(date -u +%Y%m%dt%H%M%S)-$$-${RANDOM}"
    experiment_id="stage2-zero${zero_stage}-${method}-tiny-lifecycle-${suffix}"
    mkdir -p -- "$zero_log_root"
    mkdir -- "$zero_log_root/$experiment_id"
    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD/transformers/src:$PWD/src" \
      ./venv/ilvr-venv/bin/torchrun --standalone --nproc_per_node=2 \
      tests/stage2/integration/tiny_stage2_train.py \
      --experiment-id "$experiment_id" \
      --deepspeed "configs/deepspeed_zero${zero_stage}.json" \
      --tuning-method "$method" \
      >"$zero_log_root/$experiment_id/stdout.log" \
      2>"$zero_log_root/$experiment_id/stderr.log" || exit $?
    witness_paths+=("$zero_result_root/$experiment_id/witness.json")
  done
done
./venv/ilvr-venv/bin/python - "${witness_paths[@]}" <<'PY'
import json
import sys

witnesses = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
observed = {(item["tuning_method"], item["deepspeed_zero_stage"])
            for item in witnesses}
expected = {(method, zero) for method in ("full", "lora", "dora")
            for zero in (1, 2, 3)}
assert observed == expected
assert all(item["world_size"] == 2 for item in witnesses)
assert all(item["first_global_step"] == 1 for item in witnesses)
assert all(item["resumed_global_step"] == 2 for item in witnesses)
print("validated Stage 2 lifecycles:", len(witnesses))
PY
```

Expected: nine zero exits. Full witnesses contain full checkpoint topology; LoRA/DoRA witnesses report changed adapter parameters, frozen visual/base parameters, adapter checkpoint presence, and no full model weights.

- [ ] **Step 4: Perform final repository verification**

Run:

```bash
git -c safe.directory=/root/workspace/ILVR diff --check
git -c safe.directory=/root/workspace/ILVR status --short
git -c safe.directory=/root/workspace/ILVR log --oneline --decorate -8
```

Expected: no whitespace errors; only the user's pre-existing deleted bytecode/untracked files remain outside committed task changes. Record the exact 18 witness paths and focused-test pass count in the final handoff.
