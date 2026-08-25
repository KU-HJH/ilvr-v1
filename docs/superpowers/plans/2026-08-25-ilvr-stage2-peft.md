# ILVR Stage 2 LoRA and DoRA Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add adapter-only LoRA and DoRA training, checkpoint resume, and inference to the existing ILVR Stage 2 path without changing its continuous-latent semantics or Stage 1 behavior.

**Architecture:** A new `src/stage2/peft.py` owns tuning configuration, decoder-only target resolution, PEFT construction, wrapper resolution, and trainable-parameter validation. `src/stage2/checkpointing.py` owns the JSON sidecar and fresh-versus-resume decisions, while the existing Qwen Stage 2 forward and `LatentRolloutEngine` remain the only recurrence implementation. `Stage2Trainer` adds only artifact-saving plumbing, and evaluation explicitly resolves `DeepSpeedEngine -> PeftModel -> Qwen` before using the existing inference controller.

**Tech Stack:** Python 3.11; PyTorch 2.7.1; local Transformers 4.50.0.dev0 fork; PEFT 0.17.1; TRL/Transformers Trainer; pytest; DeepSpeed ZeRO-3; Bash.

**Spec:** `docs/superpowers/specs/2026-08-25-ilvr-stage2-peft-design.md`

## Global Constraints

- Run Python, pytest, training, and evaluation through `./venv/ilvr-venv`; never use system Python or `./ilvr-venv`.
- Supported Stage 2 tuning methods are exactly `full`, `lora`, and `dora`; fresh Stage 2 defaults to `full`.
- QLoRA, AdaLoRA, rsLoRA, prompt/prefix tuning, adapter fusion, and arbitrary PEFT methods are out of scope.
- LoRA/DoRA targets are restricted to decoder `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` modules.
- The visual encoder, `embed_tokens`, `lm_head`, and all non-adapter base weights remain frozen in PEFT modes.
- DoRA is represented by `peft.LoraConfig(use_dora=True)`; LoRA uses the same configuration with `use_dora=False`.
- Stage 2 PEFT checkpoints are adapter-only; merge/export is not implemented in this plan.
- The Qwen Stage 2 forward and `LatentRolloutEngine` remain the one authoritative recurrence; do not modify generic `GenerationMixin`.
- Preserve latent recurrence indexing, complete-sequence MRoPE slicing, latent-pad independence, multi-block context, paired-boundary loss masking, and visual-encoder-once behavior.
- Preserve the explicit local `B == 1` safety boundary; multiple distributed ranks and gradient accumulation remain supported.
- Fresh Stage 1-to-Stage 2 training starts a new optimizer/scheduler; Stage 2 PEFT resume restores adapter plus Trainer state.
- A PEFT resume/evaluation uses an explicit compatible Stage 1 base and validates token/config/module shape invariants without hashing full model weights.
- Stage 1 PEFT-only input checkpoints are rejected in this initial implementation; no adapter stacking or implicit merge is allowed.
- Use test-driven development for every behavior change and invoke `superpowers:systematic-debugging` before fixing any unexpected failure.
- Do not push, merge, publish, create a PR, or implement adapter merge/export without explicit permission.
- Before implementation, invoke `superpowers:using-git-worktrees` and confirm work remains on `feat/vcsd-ilvr-stage2`, not `main`/`master`.
- Before completion, invoke `superpowers:verification-before-completion` and `superpowers:requesting-code-review`.

## File Structure

- Create `src/stage2/peft.py`: immutable tuning settings, decoder target discovery, LoRA/DoRA construction/loading, Qwen wrapper resolution, and parameter-boundary reports.
- Modify `src/stage2/checkpointing.py`: versioned PEFT sidecar schema, JSON read/write, base compatibility checks, and PEFT-aware fresh/resume decisions.
- Modify `src/stage2/__init__.py`: export stable Stage 2 PEFT/checkpoint interfaces.
- Modify `src/utils_deepseed.py`: parse the Stage 2 tuning selector, preserve whether it was explicit, and reject invalid combinations before model loading.
- Modify `src/main.py`: construct full or adapter models from the load decision and pass sidecar metadata to the trainer.
- Modify `src/stage2/trainer.py`: save the sidecar next to every PEFT adapter checkpoint/final artifact while leaving loss computation model-owned.
- Modify `eval.py` and `src/evaluate_deepseed.py`: load base plus adapter and resolve the actual Qwen Stage 2 model for controller calls.
- Modify `run_stage2_training.sh`: expose validated PEFT environment variables and retain launcher-owned argument protection.
- Modify `README.md`: document full/LoRA/DoRA commands, checkpoint layout, resume, relocation, non-goals, and verified status.
- Create `tests/stage2/test_peft.py`: tuning settings, target safety, wrapper resolution, adapter construction, gradients, and recurrence parity.
- Create `tests/stage2/test_peft_checkpointing.py`: sidecar round trips, checkpoint compatibility, relocation, adapter-only saving, and resume failures.
- Modify `tests/stage2/test_routing.py`: CLI, Stage 1 regression, model construction, trainer wiring, and fresh/full routing.
- Modify `tests/stage2/test_eval_integration.py`: PEFT evaluation loading and wrapper-chain behavior for both entry points.
- Modify `tests/stage2/test_launcher.py` and `tests/stage2/test_comt_readiness.py`: launcher defaults, PEFT values, and managed-option validation.
- Modify `tests/stage2/integration/tiny_stage2_train.py`: parameterized full/LoRA/DoRA save-resume harness and evidence witness.
- Modify `tests/stage2/integration/test_distributed_contract.py`: CPU harness contract and opt-in two-rank PEFT assertions.

---

### Task 1: Stage 2 Tuning Configuration and CLI Contract

**Files:**
- Create: `src/stage2/peft.py`
- Modify: `src/utils_deepseed.py:12-94`
- Modify: `src/stage2/__init__.py:1-17`
- Create: `tests/stage2/test_peft.py`
- Modify: `tests/stage2/test_routing.py:105-157`

**Interfaces:**
- Consumes: existing `parse_args(argv: list[str] | None) -> argparse.Namespace` and shared `lora_*` arguments.
- Produces: `TuningMethod = Literal["full", "lora", "dora"]`; `ALLOWED_STAGE2_TARGET_SUFFIXES: tuple[str, ...]`; `Stage2TuningConfig`; `normalize_stage2_target_modules(value: str | Sequence[str]) -> tuple[str, ...]`; CLI fields `tuning_method: str | None` and `tuning_method_explicit: bool`.

- [ ] **Step 1: Write failing settings tests**

Add tests that lock down normalization and value validation:

```python
from stage2.peft import Stage2TuningConfig, normalize_stage2_target_modules


def test_stage2_tuning_config_normalizes_the_approved_decoder_targets():
    targets = normalize_stage2_target_modules(
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    config = Stage2TuningConfig(
        method="dora", r=8, alpha=16, dropout=0.1, target_modules=targets
    )
    assert config.method == "dora"
    assert config.use_dora is True
    assert config.target_modules == (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


@pytest.mark.parametrize("value", ["", "q_proj,visual", "q_proj,q_proj"])
def test_stage2_tuning_config_rejects_invalid_target_lists(value):
    with pytest.raises(ValueError):
        normalize_stage2_target_modules(value)
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft.py -q
```

Expected: collection fails because `stage2.peft` does not exist.

- [ ] **Step 3: Implement the immutable tuning configuration**

Create these concrete definitions in `src/stage2/peft.py`:

```python
from dataclasses import dataclass
from typing import Literal, Sequence

TuningMethod = Literal["full", "lora", "dora"]
ALLOWED_STAGE2_TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def normalize_stage2_target_modules(
    value: str | Sequence[str],
) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    targets = tuple(item.strip() for item in raw if item.strip())
    if not targets:
        raise ValueError("Stage 2 PEFT target modules must not be empty")
    if len(set(targets)) != len(targets):
        raise ValueError("Stage 2 PEFT target modules must be unique")
    invalid = sorted(set(targets) - set(ALLOWED_STAGE2_TARGET_SUFFIXES))
    if invalid:
        raise ValueError(f"unsupported Stage 2 PEFT target modules: {invalid}")
    return targets


@dataclass(frozen=True)
class Stage2TuningConfig:
    method: TuningMethod
    r: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method not in ("full", "lora", "dora"):
            raise ValueError(f"unsupported Stage 2 tuning method: {self.method}")
        if self.r <= 0 or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        normalize_stage2_target_modules(self.target_modules)

    @property
    def is_peft(self) -> bool:
        return self.method in ("lora", "dora")

    @property
    def use_dora(self) -> bool:
        return self.method == "dora"
```

Export these names from `src/stage2/__init__.py`.

- [ ] **Step 4: Write failing CLI routing tests**

Replace the old Stage 2 LoRA-rejection assertion and add exact behavior:

```python
def test_fresh_stage2_tuning_defaults_to_full():
    args = parse_args(["--stage", "stage2"])
    assert args.tuning_method == "full"
    assert args.tuning_method_explicit is False


@pytest.mark.parametrize("method", ["lora", "dora"])
def test_stage2_accepts_explicit_peft_methods(method):
    args = parse_args(["--stage", "stage2", "--tuning_method", method])
    assert args.tuning_method == method
    assert args.tuning_method_explicit is True


def test_stage2_resume_leaves_omitted_method_for_sidecar_resolution():
    args = parse_args([
        "--stage", "stage2",
        "--resume_from_checkpoint_path", "checkpoint-7",
    ])
    assert args.tuning_method is None
    assert args.tuning_method_explicit is False


def test_stage2_rejects_legacy_use_lora():
    with pytest.raises(ValueError, match="--tuning_method lora"):
        parse_args(["--stage", "stage2", "--use_lora"])


def test_stage1_cli_contract_does_not_gain_stage2_tuning():
    args = parse_args(["--stage", "stage1", "--use_lora"])
    assert args.use_lora is True
    assert args.tuning_method is None
```

- [ ] **Step 5: Run the CLI tests and verify expected failures**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/test_routing.py -q
```

Expected: new tuning assertions fail because Stage 2 still rejects LoRA and has no selector.

- [ ] **Step 6: Implement Stage 2 CLI resolution without altering Stage 1**

Add `--tuning_method` with choices and an internal `None` parser default. Preserve whether the user supplied it, then derive the fresh-run default:

```python
raw_argv = list(sys.argv[1:] if argv is None else argv)
parser.add_argument("--tuning_method", choices=("full", "lora", "dora"))
# parse raw_argv
args.tuning_method_explicit = any(
    item == "--tuning_method" or item.startswith("--tuning_method=")
    for item in raw_argv
)
if args.stage == "stage2":
    if args.use_lora:
        raise ValueError(
            "Stage 2 does not accept --use_lora; use --tuning_method lora"
        )
    if args.tuning_method is None and args.resume_from_checkpoint_path is None:
        args.tuning_method = "full"
    normalize_stage2_target_modules(args.lora_target_modules)
elif args.tuning_method_explicit:
    raise ValueError("--tuning_method is only valid for Stage 2")
```

Validate positive rank/alpha and finite `[0, 1)` dropout during parsing, before any model load. Retain the existing `per_device_train_batch_size=1` check.

- [ ] **Step 7: Run focused tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft.py tests/stage2/test_routing.py -q
```

Expected: all selected tests pass, including unchanged Stage 1 defaults.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/stage2/peft.py src/stage2/__init__.py src/utils_deepseed.py tests/stage2/test_peft.py tests/stage2/test_routing.py
git commit -m "feat: define stage2 peft tuning contract"
```

---

### Task 2: Decoder-Only Adapter Construction and Wrapper Resolution

**Files:**
- Modify: `src/stage2/peft.py`
- Modify: `tests/stage2/test_peft.py`

**Interfaces:**
- Consumes: `Stage2TuningConfig`, PEFT 0.17.1 `LoraConfig`, `PeftModel`, and existing Qwen `model.layers` structure.
- Produces: `Stage2ParameterReport`; `resolve_stage2_qwen_model(model: nn.Module) -> nn.Module`; `resolve_stage2_target_module_names(model: nn.Module, requested: Sequence[str]) -> tuple[str, ...]`; `build_stage2_tuned_model(model: nn.Module, tuning: Stage2TuningConfig, adapter_path: str | Path | None = None, is_trainable: bool = True) -> nn.Module`; `validate_stage2_parameter_boundary(model: nn.Module, tuning: Stage2TuningConfig, *, expect_trainable: bool = True) -> Stage2ParameterReport`.

- [ ] **Step 1: Write failing target-resolution and wrapper tests**

Use `tiny_stage2_qwen()` so the visual tower contains misleading `gate_proj`,
`up_proj`, and `down_proj` names. Define the test settings helpers in
`tests/stage2/test_peft.py`:

```python
def settings(method: str) -> Stage2TuningConfig:
    return Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )


def lora_settings() -> Stage2TuningConfig:
    return settings("lora")
```

Then add:

```python
@pytest.mark.parametrize("method,use_dora", [("lora", False), ("dora", True)])
def test_stage2_peft_targets_only_decoder_modules(method, use_dora):
    base = tiny_stage2_qwen()
    tuning = Stage2TuningConfig(
        method=method, r=4, alpha=8, dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )
    wrapped = build_stage2_tuned_model(base, tuning)
    qwen = resolve_stage2_qwen_model(wrapped)
    config = wrapped.peft_config["default"]

    assert qwen is base
    assert config.use_dora is use_dora
    assert config.modules_to_save is None
    assert all(name.startswith("model.layers.") for name in config.target_modules)
    assert not any("visual" in name for name in config.target_modules)


def test_wrapper_resolver_handles_deepspeed_then_peft():
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), lora_settings())
    engine = SimpleNamespace(module=wrapped)
    assert resolve_stage2_qwen_model(engine) is wrapped.get_base_model()


def test_wrapper_resolver_rejects_non_stage2_base():
    with pytest.raises(TypeError, match="Stage 2 Qwen interface"):
        resolve_stage2_qwen_model(nn.Linear(2, 2))
```

- [ ] **Step 2: Run the focused tests and verify missing-function failures**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft.py -q
```

Expected: failures identify the missing resolver/construction functions.

- [ ] **Step 3: Implement exact decoder target discovery**

Resolve targets before wrapping and pass full decoder paths to PEFT:

```python
def resolve_stage2_target_module_names(model, requested):
    requested = normalize_stage2_target_modules(requested)
    matches = {
        suffix: tuple(
            name for name, module in model.named_modules()
            if name.startswith("model.layers.")
            and name.rsplit(".", 1)[-1] == suffix
            and isinstance(module, torch.nn.Linear)
        )
        for suffix in requested
    }
    missing = [suffix for suffix, names in matches.items() if not names]
    if missing:
        raise ValueError(f"Stage 2 decoder targets matched no modules: {missing}")
    return tuple(name for suffix in requested for name in matches[suffix])
```

The `model.layers.` prefix is intentional: the tiny and real Qwen outer model expose the language decoder there, while the visual tower is under `visual.`.

- [ ] **Step 4: Implement PEFT construction/loading and explicit unwrapping**

Use these semantics:

```python
def build_stage2_tuned_model(model, tuning, adapter_path=None, is_trainable=True):
    model.visual.requires_grad_(False)
    if tuning.method == "full":
        if adapter_path is not None:
            raise ValueError("full tuning cannot load a PEFT adapter")
        return model
    resolved = resolve_stage2_target_module_names(model, tuning.target_modules)
    if adapter_path is None:
        config = LoraConfig(
            r=tuning.r,
            lora_alpha=tuning.alpha,
            lora_dropout=tuning.dropout,
            target_modules=list(resolved),
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=tuning.use_dora,
            modules_to_save=None,
        )
        wrapped = get_peft_model(model, config)
    else:
        wrapped = PeftModel.from_pretrained(
            model, str(adapter_path), is_trainable=is_trainable
        )
    validate_stage2_parameter_boundary(
        wrapped, tuning, expect_trainable=is_trainable
    )
    return wrapped
```

`resolve_stage2_qwen_model` removes at most one `.module` wrapper, then calls `get_base_model()` only for `PeftModel`, and validates the result has `prepare_stage2_base_inputs_embeds`, `get_rope_index`, `stage2_rollout_engine`, `model`, `lm_head`, and a `config.stage == "stage2"` contract.

- [ ] **Step 5: Write failing trainable-boundary tests**

Add assertions for exact ownership and deliberate corruption:

```python
@pytest.mark.parametrize("method", ["lora", "dora"])
def test_only_adapter_parameters_are_trainable(method):
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), settings(method))
    report = validate_stage2_parameter_boundary(wrapped, settings(method))
    trainable = [name for name, p in wrapped.named_parameters() if p.requires_grad]

    assert report.trainable_parameters > 0
    assert all("lora_" in name for name in trainable)
    assert not any("visual" in name for name in trainable)
    assert not any("embed_tokens" in name or "lm_head" in name for name in trainable)


def test_parameter_boundary_fails_if_visual_is_reenabled():
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), lora_settings())
    next(resolve_stage2_qwen_model(wrapped).visual.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="visual parameter is trainable"):
        validate_stage2_parameter_boundary(wrapped, lora_settings())
```

For DoRA, accept both `lora_` adapter matrices and PEFT's `lora_magnitude_vector` parameters. Reject every other trainable name.

- [ ] **Step 6: Implement and log the parameter report**

Define:

```python
@dataclass(frozen=True)
class Stage2ParameterReport:
    total_parameters: int
    trainable_parameters: int
    trainable_names: tuple[str, ...]
    resolved_target_names: tuple[str, ...]
```

`validate_stage2_parameter_boundary` must inspect the resolved Qwen visual,
embeddings, and LM head directly and return counts without mutating parameters.
With `expect_trainable=True`, require a nonzero adapter trainable set. With
`expect_trainable=False`, require every parameter to be frozen while still
verifying adapter topology. Full tuning validates frozen vision and at least
one trainable language parameter.

- [ ] **Step 7: Run adapter construction tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft.py -q
```

Expected: LoRA and DoRA construction, visual exclusion, wrapper resolution, and corruption failures all pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add src/stage2/peft.py tests/stage2/test_peft.py
git commit -m "feat: build decoder-only stage2 adapters"
```

---

### Task 3: PEFT Latent-Recurrence and Gradient Parity

**Files:**
- Modify: `tests/stage2/test_peft.py`
- Modify: `tests/stage2/test_inference.py`
- Modify: `tests/stage2/test_qwen_forward.py`
- Production change only if a focused failing test proves the wrapper boundary needs correction: `src/stage2/peft.py`

**Interfaces:**
- Consumes: `build_stage2_tuned_model`, `resolve_stage2_qwen_model`, `tiny_stage2_qwen`, existing `LatentRolloutEngine`, and `Stage2InferenceController`.
- Produces: regression evidence that both adapter methods use the unchanged recurrence and remain differentiable.

- [ ] **Step 1: Write a failing adapter-forward participation test**

Define this helper in each PEFT-focused test file that needs it:

```python
def settings(method: str) -> Stage2TuningConfig:
    return Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )
```

Define this deterministic one-block batch in `tests/stage2/test_peft.py`:

```python
def one_block_stage2_batch() -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([[1, 91, 90, 90, 92, 2]])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.tensor([[-100, 91, -100, -100, 92, 2]]),
    }
```

Register a hook on the first adapted decoder `q_proj` and run this `K=2`
Stage 2 training forward:

```python
@pytest.mark.parametrize("method", ["lora", "dora"])
def test_adapter_executes_in_every_recurrent_prefix_and_final_forward(method):
    wrapped = build_stage2_tuned_model(tiny_stage2_qwen(), settings(method))
    qwen = resolve_stage2_qwen_model(wrapped)
    calls = []
    handle = qwen.model.layers[0].self_attn.q_proj.register_forward_hook(
        lambda *unused: calls.append(1)
    )
    try:
        outputs = wrapped(**one_block_stage2_batch(), return_stage2_latent_states=True)
    finally:
        handle.remove()
    assert outputs.loss.isfinite()
    assert len(outputs.stage2_latent_states) == 2
    assert len(calls) == 3  # two recurrent prefixes plus final full forward
```

- [ ] **Step 2: Write failing PEFT gradient-continuity tests**

For each method, retain gradients on returned latent states, backpropagate the normal CE, and assert:

```python
outputs = wrapped(**batch, return_stage2_latent_states=True)
for state in outputs.stage2_latent_states:
    state.retain_grad()
outputs.loss.backward()
assert all(state.grad is not None and torch.isfinite(state.grad).all()
           for state in outputs.stage2_latent_states)
assert all(state.grad.abs().sum() > 0 for state in outputs.stage2_latent_states)
assert any(
    p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    for name, p in wrapped.named_parameters()
    if "lora_" in name
)
assert all(p.grad is None for p in qwen.visual.parameters())
```

Also assert frozen base decoder, `embed_tokens`, and `lm_head` parameters have no gradients.

- [ ] **Step 3: Write failing train/inference and latent-pad parity tests**

Parameterize the existing train/inference parity fixture over `lora` and `dora`, reuse identical adapter weights, and compare every `z1..zK` with `torch.testing.assert_close`. Repeat rollout after replacing the `<latent_pad>` embedding row by a large constant and assert latent states are unchanged.

- [ ] **Step 4: Run the PEFT recurrence tests before production edits**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft.py tests/stage2/test_inference.py tests/stage2/test_qwen_forward.py -q
```

Expected: if Task 2's standard wrapper is sufficient, these tests pass without recurrence changes. Any unexpected mismatch triggers `superpowers:systematic-debugging`; do not edit the Qwen rollout engine speculatively.

- [ ] **Step 5: Make only the minimal wrapper-boundary correction if the tests exposed one**

Permitted correction: ensure calls enter the outer `PeftModel.forward` during training and use the adapter-injected Qwen returned by `get_base_model()` during inference. Forbidden correction: copying recurrence logic or detaching latent states.

- [ ] **Step 6: Re-run the focused recurrence suite**

Run the same command from Step 4.

Expected: all parameterized LoRA/DoRA recurrence, gradient, visual-call-count, multi-block, MRoPE, loss-mask, and train/inference parity assertions pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add tests/stage2/test_peft.py tests/stage2/test_inference.py tests/stage2/test_qwen_forward.py src/stage2/peft.py
git commit -m "test: verify peft latent recurrence semantics"
```

---

### Task 4: Versioned Adapter Sidecar and Load Decisions

**Files:**
- Modify: `src/stage2/checkpointing.py`
- Modify: `src/stage2/__init__.py`
- Create: `tests/stage2/test_peft_checkpointing.py`
- Modify: `tests/stage2/test_checkpointing.py`

**Interfaces:**
- Consumes: `Stage2TuningConfig`, `resolve_stage2_target_module_names`, existing latent token metadata, `PeftConfig.from_pretrained`, and explicit `load_model_path`/`resume_from_checkpoint_path`.
- Produces: `STAGE2_PEFT_METADATA_FILENAME = "stage2_peft_metadata.json"`; `STAGE2_PEFT_SCHEMA_VERSION = 1`; `Stage2PeftMetadata.to_dict() -> dict` and `Stage2PeftMetadata.from_dict(payload: Mapping[str, object]) -> Stage2PeftMetadata`; `build_stage2_peft_metadata(*, tuning: Stage2TuningConfig, base_model_reference: str, model_config, token_ids: LatentTokenIds, resolved_target_names: Sequence[str]) -> Stage2PeftMetadata`; `save_stage2_peft_metadata`; `load_stage2_peft_metadata`; `validate_stage2_peft_compatibility(base_model, tokenizer, metadata, adapter_config, adapter_path: str | Path) -> None`; an extended `Stage2LoadDecision` with `base_model_source`, `processor_source`, `adapter_source`, `trainer_resume_path`, `tuning`, and optional `peft_metadata`; `resolve_stage2_load_decision(*, load_model_path: str, resume_from_checkpoint_path: str | None, load_config: Callable, load_peft_metadata: Callable, latent_size: int, supervise_latent_boundaries: bool, tuning_method: str | None, tuning_method_explicit: bool, lora_r: int, lora_alpha: int, lora_dropout: float, lora_target_modules: str | Sequence[str]) -> Stage2LoadDecision`.

- [ ] **Step 1: Write failing metadata round-trip and malformed-file tests**

Use a complete expected payload:

```python
def sample_metadata():
    return Stage2PeftMetadata(
        schema_version=1,
        stage="stage2",
        tuning_method="dora",
        base_model_reference="/models/stage1",
        latent_size=8,
        latent_start_token_id=91,
        latent_pad_token_id=90,
        latent_end_token_id=92,
        supervise_latent_boundaries=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        requested_target_modules=("q_proj", "v_proj"),
        resolved_target_module_names=(
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.v_proj",
        ),
        peft_version="0.17.1",
    )


def test_stage2_peft_metadata_round_trips(tmp_path):
    save_stage2_peft_metadata(tmp_path, sample_metadata())
    assert load_stage2_peft_metadata(tmp_path) == sample_metadata()


@pytest.mark.parametrize("mutation", ["bad_schema", "bad_stage", "missing_field"])
def test_stage2_peft_metadata_fails_closed(tmp_path, mutation):
    write_mutated_metadata(tmp_path, sample_metadata(), mutation)
    with pytest.raises(ValueError):
        load_stage2_peft_metadata(tmp_path)
```

Define the mutation helper concretely:

```python
def write_mutated_metadata(directory, metadata, mutation):
    payload = metadata.to_dict()
    if mutation == "bad_schema":
        payload["schema_version"] = 999
    elif mutation == "bad_stage":
        payload["stage"] = "stage1"
    elif mutation == "missing_field":
        payload.pop("latent_size")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / STAGE2_PEFT_METADATA_FILENAME).write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )
```

- [ ] **Step 2: Run metadata tests and verify missing-interface failures**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft_checkpointing.py -q
```

Expected: import or symbol failures for the new sidecar API.

- [ ] **Step 3: Implement strict JSON serialization**

Define a frozen dataclass with exactly the fields in Step 1 and strict
`to_dict`/`from_dict` methods. `build_stage2_peft_metadata` copies latent
semantics from the configured Qwen config/token IDs, adapter settings from
`Stage2TuningConfig`, resolved full target names from Task 2, and
`peft.__version__`. `save_stage2_peft_metadata(directory, metadata)` creates the
directory if needed, serializes tuples as JSON lists with sorted keys and a
final newline, writes a sibling temporary file, and uses `Path.replace()` for
atomic publication. `load_stage2_peft_metadata(directory)` requires a regular
file, rejects unknown schema/stage/method values, reconstructs tuples, and
routes values through `Stage2TuningConfig` validation.

- [ ] **Step 4: Write failing fresh/full/PEFT resume-decision tests**

Define the decision-test helpers:

```python
def semantic_args():
    return {
        "latent_size": 8,
        "supervise_latent_boundaries": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": "q_proj,v_proj",
    }


def lora_metadata():
    values = sample_metadata().to_dict()
    values["tuning_method"] = "lora"
    return Stage2PeftMetadata.from_dict(values)


def fake_stage_config(path):
    stage = "stage1" if "stage1" in str(path) else "stage2"
    return SimpleNamespace(stage=stage, latent_size=8,
                           supervise_latent_boundaries=True)


def fake_metadata_loader(path):
    raise AssertionError(f"fresh training must not read PEFT metadata: {path}")
```

Cover these exact results:

```python
def test_fresh_peft_uses_stage1_base_and_no_trainer_resume():
    decision = resolve_stage2_load_decision(
        load_model_path="stage1",
        resume_from_checkpoint_path=None,
        load_config=fake_stage_config,
        load_peft_metadata=fake_metadata_loader,
        tuning_method="lora",
        tuning_method_explicit=True,
        **semantic_args(),
    )
    assert decision.base_model_source == "stage1"
    assert decision.processor_source == "stage1"
    assert decision.adapter_source is None
    assert decision.trainer_resume_path is None
    assert decision.tuning.method == "lora"


def test_peft_resume_keeps_explicit_base_and_loads_adapter_checkpoint():
    decision = resolve_stage2_load_decision(
        load_model_path="relocated-stage1",
        resume_from_checkpoint_path="stage2/checkpoint-9",
        load_config=fake_stage_config,
        load_peft_metadata=lambda path: lora_metadata(),
        tuning_method=None,
        tuning_method_explicit=False,
        **semantic_args(),
    )
    assert decision.base_model_source == "relocated-stage1"
    assert decision.processor_source == "stage2/checkpoint-9"
    assert decision.adapter_source == "stage2/checkpoint-9"
    assert decision.trainer_resume_path == "stage2/checkpoint-9"
    assert decision.tuning.method == "lora"
```

Retain a full-resume test where all model/processor/trainer sources remain the full Stage 2 checkpoint. Add explicit failures for missing sidecar in requested PEFT resume, LoRA/DoRA switching, explicit `full` against a PEFT checkpoint, changed latent settings, and a Stage 1 PEFT-only base.

- [ ] **Step 5: Implement the extended load decision**

Use sidecar presence to distinguish a PEFT Stage 2 checkpoint from a full Stage 2 checkpoint. An omitted resume selector inherits `metadata.tuning_method`; only an explicitly supplied selector is compared as an override. Keep the existing full fresh/resume semantics unchanged.

- [ ] **Step 6: Write failing relocation compatibility tests**

Test a different base path with the same tiny Qwen structure and token IDs succeeds. Parameterize failures for latent token ID mismatch, hidden-size mismatch, missing resolved target, target shape mismatch, method mismatch, rank/alpha/dropout mismatch, and adapter-config `use_dora` mismatch.

- [ ] **Step 7: Implement structural compatibility validation**

`validate_stage2_peft_compatibility(base_model, tokenizer, metadata,
adapter_config, adapter_path)` must:

```text
resolve and compare all three latent token IDs
compare latent_size and supervise_latent_boundaries
require the supported Qwen Stage 2 interface
require every recorded full target path to exist
compare each target's weight shape with adapter tensor/config expectations
compare method, r, alpha, dropout, use_dora, and requested/resolved targets
log relocated source differences without rejecting a path-string change alone
```

Read only the small adapter safetensors header/tensors needed for target-shape
validation. Do not read or hash full base-model weight trees.

- [ ] **Step 8: Run checkpoint test suites**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_checkpointing.py tests/stage2/test_peft_checkpointing.py -q
```

Expected: all legacy full-checkpoint and new PEFT checkpoint decisions pass.

- [ ] **Step 9: Commit Task 4**

```bash
git add src/stage2/checkpointing.py src/stage2/__init__.py tests/stage2/test_checkpointing.py tests/stage2/test_peft_checkpointing.py
git commit -m "feat: define stage2 adapter checkpoint metadata"
```

---

### Task 5: Training Composition and Adapter-Only Trainer Saving

**Files:**
- Modify: `src/main.py:20-182,256-287`
- Modify: `src/stage2/trainer.py`
- Modify: `tests/stage2/test_routing.py:230-468`
- Modify: `tests/stage2/test_peft_checkpointing.py`

**Interfaces:**
- Consumes: extended `Stage2LoadDecision`, `build_stage2_tuned_model`, `validate_stage2_peft_compatibility`, and `Stage2PeftMetadata`.
- Produces: `Stage2TrainingComponents.peft_metadata: Stage2PeftMetadata | None`; `Stage2Trainer(..., stage2_peft_metadata=None)`; adapter-only numbered and final checkpoints with sidecars.

- [ ] **Step 1: Write failing component-construction tests**

Parameterize `build_stage2_training_components` over `full`, `lora`, and `dora`. Record that the base loader uses `decision.base_model_source`, processor uses `decision.processor_source`, fresh PEFT attaches a new adapter, resume passes `decision.adapter_source`, config metadata is applied before wrapping, vision is frozen, and the returned model is a `PeftModel` only for adapter methods.

Add an assertion that `Stage2TrainingComponents.peft_metadata` is `None` for full tuning and complete for LoRA/DoRA, including resolved full decoder paths and the explicit Stage 1 base reference. Construct it only through `build_stage2_peft_metadata` so training and tests share one schema builder.

- [ ] **Step 2: Run routing tests and verify construction failures**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_routing.py -q
```

Expected: failures identify missing decision fields, PEFT construction, and trainer metadata wiring.

- [ ] **Step 3: Implement model composition in `src/main.py`**

Change construction order to:

```python
processor = AutoProcessor.from_pretrained(decision.processor_source)
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    decision.base_model_source,
    attn_implementation=args.attention_implementation,
)
token_ids = resolve_latent_token_ids(processor.tokenizer, base_model.config)
if decision.peft_metadata is not None:
    adapter_config = PeftConfig.from_pretrained(decision.adapter_source)
    validate_stage2_peft_compatibility(
        base_model,
        processor.tokenizer,
        decision.peft_metadata,
        adapter_config,
        decision.adapter_source,
    )
configure_stage2_metadata(base_model.config, token_ids, ...)
configure_stage2_model_backend(base_model, gradient_checkpointing=...)
base_model.visual.requires_grad_(False)
model = build_stage2_tuned_model(
    base_model,
    decision.tuning,
    adapter_path=decision.adapter_source,
    is_trainable=True,
)
```

For resume, validate the unmodified Stage 1 base config/tokenizer and adapter
config against the sidecar before `configure_stage2_metadata` writes Stage 2
values. This prevents configuration from hiding an incompatible relocated
base. Then attach the adapter, validate the resulting parameter shapes, and log
one `Stage2ParameterReport`. Do not pass `peft_config` to `Stage2Trainer`; the
model is already wrapped so resume topology is deterministic.

- [ ] **Step 4: Write failing trainer-save tests**

Instantiate `Stage2Trainer` with a tiny PEFT model and metadata, call `save_model(tmp_path)`, and assert:

```python
assert (tmp_path / "adapter_config.json").is_file()
assert (tmp_path / "adapter_model.safetensors").is_file()
assert (tmp_path / "stage2_peft_metadata.json").is_file()
assert not (tmp_path / "model.safetensors").exists()
assert not (tmp_path / "pytorch_model.bin").exists()
```

Create a numbered `checkpoint-1` through a one-step tiny Trainer run and assert the same sidecar plus `trainer_state.json`, optimizer, scheduler, and RNG artifacts. Assert full tuning writes no PEFT sidecar.

- [ ] **Step 5: Implement thin sidecar saving in `Stage2Trainer`**

Add an initializer that pops `stage2_peft_metadata`, then override only model saving:

```python
class Stage2Trainer(Trainer):
    def __init__(self, *args, stage2_peft_metadata=None, **kwargs):
        self.stage2_peft_metadata = stage2_peft_metadata
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        target = output_dir or self.args.output_dir
        if self.stage2_peft_metadata is not None and self.args.should_save:
            save_stage2_peft_metadata(target, self.stage2_peft_metadata)
```

Do not override `compute_loss` with adapter behavior and do not reimplement `_save_checkpoint`.

- [ ] **Step 6: Pass metadata through `main_train`**

Add `stage2_peft_metadata=components.peft_metadata` to trainer construction. Keep `trainer.train(resume_from_checkpoint=components.trainer_resume_path)`. Final `trainer.save_model` and rank-zero processor saving remain explicit; assert processor/tokenizer IDs are unchanged.

- [ ] **Step 7: Run training-composition and checkpoint tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_routing.py tests/stage2/test_peft_checkpointing.py -q
```

Expected: fresh and resume composition, adapter-only artifacts, sidecars, and unchanged full mode pass.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/main.py src/stage2/trainer.py tests/stage2/test_routing.py tests/stage2/test_peft_checkpointing.py
git commit -m "feat: integrate stage2 adapter training and saving"
```

---

### Task 6: Base-Plus-Adapter Evaluation and Inference Parity

**Files:**
- Modify: `src/stage2/peft.py`
- Modify: `src/stage2/inference.py:111-248` only for explicit resolver use if tests require it
- Modify: `eval.py:42-110,295-407`
- Modify: `src/evaluate_deepseed.py:35-80,206-288`
- Modify: `tests/stage2/test_eval_integration.py`
- Modify: `tests/stage2/test_inference.py`

**Interfaces:**
- Consumes: sidecar loader/validator, `PeftModel.from_pretrained`, and `resolve_stage2_qwen_model`.
- Produces: evaluation CLI `--base_model_dir`; `load_stage2_model_for_inference(base_model, adapter_path, metadata, tokenizer) -> PeftModel`; both eval entry points pass the resolved Qwen into `Stage2InferenceController`.

- [ ] **Step 1: Write failing inference adapter-loader tests**

Define a tokenizer that supplies the existing tiny-model latent IDs:

```python
class TinyLatentTokenizer:
    unk_token_id = 0
    unk_token = "<unk>"
    ids = {
        "<|latent_pad|>": 90,
        "<|latent_start|>": 91,
        "<|latent_end|>": 92,
    }

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)
```

Define the local settings helper explicitly:

```python
def inference_settings(method: str) -> Stage2TuningConfig:
    return Stage2TuningConfig(
        method=method,
        r=4,
        alpha=8,
        dropout=0.0,
        target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES,
    )
```

For each method, set `tuning = inference_settings(method)`,
`artifact_dir = tmp_path / method`, create
`source = build_stage2_tuned_model(tiny_stage2_qwen(), tuning)`, save it to
`artifact_dir`, build/save its sidecar through `build_stage2_peft_metadata`, set
`fresh_base = tiny_stage2_qwen()`, and then assert:

```python
reloaded = load_stage2_model_for_inference(
    base_model=fresh_base,
    adapter_path=artifact_dir,
    metadata=load_stage2_peft_metadata(artifact_dir),
    tokenizer=TinyLatentTokenizer(),
)
assert isinstance(reloaded, PeftModel)
assert reloaded.training is False
assert resolve_stage2_qwen_model(reloaded).config.stage == "stage2"
```

Compare logits and `z1..zK` between the saved source adapter and reload using `torch.testing.assert_close`.

- [ ] **Step 2: Run inference tests and verify loader failures**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_inference.py -q
```

Expected: the new loader symbol is absent.

- [ ] **Step 3: Implement inference-only adapter loading**

In `src/stage2/peft.py`, validate metadata and adapter config against the
unmodified supplied base/tokenizer, call `configure_stage2_metadata` with the
sidecar values, call
`PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)`, set
the wrapper to eval mode, run
`validate_stage2_parameter_boundary(..., expect_trainable=False)`, and return
the wrapper. Never call `merge_and_unload()`.

- [ ] **Step 4: Write failing eval entry-point tests**

Parameterize both `eval` and `evaluate_deepseed` to assert:

```text
full artifact: --model_dir alone loads existing full checkpoint behavior
PEFT artifact: --model_dir adapter + --base_model_dir Stage 1 loads both sources
PEFT artifact without --base_model_dir fails before sample iteration
processor loads from adapter artifact and latent IDs validate against base config
DeepSpeedEngine.module -> PeftModel -> Qwen resolution reaches controller
controller receives Qwen with injected active adapter modules
helper images remain absent and original images remain present
```

Extend `DeepSpeedShapedWrapper` around a real tiny `PeftModel` and assert the controller receives `wrapped.module.get_base_model()`.

- [ ] **Step 5: Run eval integration tests and verify expected failures**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_eval_integration.py -q
```

Expected: PEFT source selection and two-layer unwrapping assertions fail.

- [ ] **Step 6: Implement shared evaluation selection in both entry points**

Add `--base_model_dir`. Detect PEFT through the required sidecar in `model_dir`:

```text
sidecar absent -> load full Qwen from model_dir; reject a stray base_model_dir
sidecar present -> require base_model_dir; load processor from model_dir;
                   load Qwen from base_model_dir; validate; attach adapter
```

Replace `getattr(model, "module", model)` with `resolve_stage2_qwen_model(model)`. Derive device from the outer runtime wrapper first, then resolved Qwen. Pass only resolved Qwen to the existing controller because adapters are already injected into its decoder modules.

Retain the current FlashAttention-to-eager fallback only for backend initialization errors. Do not use it to suppress metadata, adapter, or token mismatches.

- [ ] **Step 7: Run inference/eval parity suites**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_inference.py tests/stage2/test_eval_integration.py tests/stage2/test_generation_regression.py -q
```

Expected: LoRA/DoRA reload parity and generic generation regressions pass.

- [ ] **Step 8: Commit Task 6**

```bash
git add src/stage2/peft.py src/stage2/inference.py eval.py src/evaluate_deepseed.py tests/stage2/test_inference.py tests/stage2/test_eval_integration.py
git commit -m "feat: load stage2 adapters for inference"
```

---

### Task 7: Launcher and User Documentation

**Files:**
- Modify: `run_stage2_training.sh:14-147`
- Modify: `tests/stage2/test_launcher.py`
- Modify: `tests/stage2/test_comt_readiness.py:40-227`
- Modify: `README.md:132-250`

**Interfaces:**
- Consumes: Stage 2 CLI fields from Task 1 and the explicit base/adapter resume contract from Task 4.
- Produces: launcher environment variables `TUNING_METHOD`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`, and `LORA_TARGET_MODULES`; reproducible full/LoRA/DoRA documentation.

- [ ] **Step 1: Write failing launcher default and explicit-value tests**

Add this local helper to the launcher test file that uses it:

```python
def option_value(tokens: list[str], option: str) -> str:
    return tokens[tokens.index(option) + 1]
```

For a fresh run, assert dry-run tokens include:

```python
assert option_value(tokens, "--tuning_method") == "full"
assert option_value(tokens, "--lora_r") == "32"
assert option_value(tokens, "--lora_alpha") == "64"
assert option_value(tokens, "--lora_dropout") == "0.05"
assert option_value(tokens, "--lora_target_modules") == (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)
```

Run a second dry-run with `TUNING_METHOD=dora`, rank 16, alpha 32, dropout 0.1, and a safe target subset. Assert exact propagation. Parameterize failures for unknown method, non-positive rank/alpha, non-finite/out-of-range dropout, empty targets, and visual/out-of-scope target names.

Add a resume case with `RESUME_CHECKPOINT` set and `TUNING_METHOD` absent from
the environment; assert `--tuning_method` is omitted so the checkpoint sidecar
is authoritative. Add another resume case with an explicit
`TUNING_METHOD=lora`; assert it is forwarded so checkpoint compatibility code
can accept or reject the deliberate override.

- [ ] **Step 2: Extend managed-option rejection tests**

Add all five PEFT options to the existing launcher-owned set and assert both `--option value` and `--option=value` passthrough forms fail. Retain tests proving unmanaged quoted arguments survive unchanged.

- [ ] **Step 3: Run launcher tests and verify failures**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py -q
```

Expected: failures show the launcher does not yet own or emit PEFT settings.

- [ ] **Step 4: Implement shell validation and command propagation**

Preserve whether the environment variable was explicitly supplied, then add
fresh-run defaults:

```bash
if [[ -v TUNING_METHOD ]]; then
  TUNING_METHOD_EXPLICIT=true
else
  TUNING_METHOD_EXPLICIT=false
fi
TUNING_METHOD=${TUNING_METHOD:-full}
LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}
```

Validate method with a `case`, rank/alpha using `require_positive_integer`, dropout with the repository venv Python finite/range check, and target tokens against the exact allowlist. Append the five options to `MANAGED_OPTIONS` and `COMMAND`.

Build `TUNING_ARGS` with this exact policy:

```bash
TUNING_ARGS=()
if [[ -z "${RESUME_CHECKPOINT:-}" || "$TUNING_METHOD_EXPLICIT" == true ]]; then
  TUNING_ARGS+=(
    --tuning_method "$TUNING_METHOD"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_target_modules "$LORA_TARGET_MODULES"
  )
fi
```

When a resume omits `TUNING_METHOD`, omit all five tuning arguments so the
sidecar controls method and hyperparameters. When the method is explicit,
forward the complete set so compatibility code can reject any difference.

- [ ] **Step 5: Run launcher tests and Bash syntax check**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py -q
bash -n run_stage2_training.sh
```

Expected: all launcher tests pass and Bash reports no syntax error.

- [ ] **Step 6: Update README with exact commands and limitations**

Document three fresh examples using `TUNING_METHOD=full|lora|dora`; a PEFT resume example with explicit `STAGE1_CHECKPOINT` and `RESUME_CHECKPOINT`; adapter files (`adapter_config.json`, `adapter_model.safetensors`, `stage2_peft_metadata.json`, Trainer state); `--base_model_dir` evaluation; path relocation structural validation; frozen visual/embedding/LM-head boundary; Stage 1 PEFT-only source rejection; no merge/export; `B == 1`; and separate LoRA/DoRA ZeRO-3 verification status.

- [ ] **Step 7: Run documentation-sensitive launcher tests**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py -q
git diff --check
```

Expected: tests pass and no whitespace errors are reported.

- [ ] **Step 8: Commit Task 7**

```bash
git add run_stage2_training.sh README.md tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py
git commit -m "docs: add stage2 lora and dora workflows"
```

---

### Task 8: Tiny Save/Resume Harness for LoRA, DoRA, and ZeRO-3

**Files:**
- Modify: `tests/stage2/integration/tiny_stage2_train.py`
- Modify: `tests/stage2/integration/test_distributed_contract.py`

**Interfaces:**
- Consumes: `build_stage2_tuned_model`, metadata save/load, `Stage2Trainer`, and `configs/config_stage3.json`.
- Produces: `--tuning-method full|lora|dora` harness option and one JSON witness per run containing training state, adapter update, artifact, frozen-parameter, rank-local batch, and world-size evidence.

- [ ] **Step 1: Write failing harness contract tests**

Parameterize the existing opt-in CPU test over `lora` and `dora`. Require witness fields:

```python
assert witness["tuning_method"] == method
assert witness["first_global_step"] == 1
assert witness["resumed_global_step"] == 2
assert witness["optimizer_step_at_resume"] == 1
assert witness["scheduler_step_at_resume"] == 1
assert witness["adapter_parameters_changed"] is True
assert witness["adapter_checkpoint_present"] is True
assert witness["sidecar_present"] is True
assert witness["full_model_weights_absent"] is True
assert witness["visual_trainable_parameters"] == 0
assert witness["base_trainable_parameters"] == 0
assert set(witness["observed_local_batch_sizes"]) == {1}
```

- [ ] **Step 2: Run the CPU harness tests and verify missing-option failures**

Run:

```bash
RUN_STAGE2_TINY_TRAIN=1 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/integration/test_distributed_contract.py::test_tiny_stage2_harness_restores_full_training_state -q
```

Expected: the parameterized PEFT invocations fail because the harness only constructs full models.

- [ ] **Step 3: Extend the harness model/save/reload path**

Add `--tuning-method`. For PEFT methods:

```text
create tiny base A -> attach new adapter -> snapshot trainable adapter tensors
train to step 1 -> assert checkpoint-1 adapter and sidecar
create identical tiny base B -> load adapter trainably from checkpoint-1
construct a new Trainer -> resume checkpoint-1 -> advance to step 2
compare adapter tensors before/after and collect frozen-boundary counts
```

Keep the existing production `Stage2DataCollator`, rank-distinct fixtures, optimizer/scheduler recorder, and full-mode path. The harness writes the witness only on world process zero after `_assert_contract` passes.

- [ ] **Step 4: Run CPU LoRA and DoRA save/resume once**

Run:

```bash
RUN_STAGE2_TINY_TRAIN=1 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/pytest tests/stage2/integration/test_distributed_contract.py::test_tiny_stage2_harness_restores_full_training_state -q
```

Expected: full, LoRA, and DoRA parameterizations pass with finite loss and restored state.

- [ ] **Step 5: Commit the harness before GPU execution**

```bash
git add tests/stage2/integration/tiny_stage2_train.py tests/stage2/integration/test_distributed_contract.py
git commit -m "test: cover stage2 peft save and resume"
```

- [ ] **Step 6: Check for existing equivalent GPU evidence before allocating GPUs**

Inspect only the small witness paths designated for this feature. If a witness from the same commit, PEFT 0.17.1, Transformers 4.50.0.dev0, two ranks, and the same ZeRO-3 config already records a successful method, reuse it and do not rerun that method. Do not hash model/checkpoint trees.

- [ ] **Step 7: Run the two-GPU LoRA ZeRO-3 save/resume validation once**

Create a dedicated temporary directory, record its printed path, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/torchrun --standalone --nproc_per_node=2 tests/stage2/integration/tiny_stage2_train.py --tuning-method lora --deepspeed configs/config_stage3.json --output-dir /tmp/ilvr-stage2-peft-zero3-lora --witness-path /tmp/ilvr-stage2-peft-zero3-lora-witness.json
```

Expected: exit 0; witness reports `world_size=2`, `deepspeed=true`, steps 1→2, restored optimizer/scheduler, adapter update, adapter-only artifact, sidecar, local batches all 1, and frozen visual/base parameters.

- [ ] **Step 8: Run the two-GPU DoRA ZeRO-3 save/resume validation once**

Run:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD/transformers/src:$PWD/src" ./venv/ilvr-venv/bin/torchrun --standalone --nproc_per_node=2 tests/stage2/integration/tiny_stage2_train.py --tuning-method dora --deepspeed configs/config_stage3.json --output-dir /tmp/ilvr-stage2-peft-zero3-dora --witness-path /tmp/ilvr-stage2-peft-zero3-dora-witness.json
```

Expected: the same contract passes with `tuning_method=dora`. If either GPU run fails unexpectedly, invoke `superpowers:systematic-debugging`, preserve the first failure output, form one hypothesis, add a regression test, and rerun only the affected method after the fix.

- [ ] **Step 9: Record verification status without committing generated checkpoints**

Update only the README verification statements if the fresh witnesses passed. Do not add `/tmp` artifacts or model weights to Git. If hardware was unavailable, leave the status explicitly unverified.

- [ ] **Step 10: Commit evidence wording if it changed**

```bash
git add README.md
git diff --cached --quiet
git commit -m "docs: record stage2 peft zero3 verification"
```

Expected: when README changed, the first command reports staged differences and
the commit succeeds. When no wording changed, skip the commit command.

---

### Task 9: Full Regression, Requirement Audit, and Code Review

**Files:**
- Review: all files changed since `be5520f`
- Modify only for review findings: the directly responsible source/test/documentation file

**Interfaces:**
- Consumes: completed Tasks 1-8 and both approved design documents.
- Produces: fresh verification evidence and a reviewed implementation branch; no merge/push/PR.

- [ ] **Step 1: Invoke verification workflow and inspect the actual diff**

Invoke `superpowers:verification-before-completion`, then run:

```bash
git status --short --branch
git diff --check be5520f..HEAD
git diff --stat be5520f..HEAD
git diff be5520f..HEAD -- src tests README.md run_stage2_training.sh
```

Expected: only approved PEFT scope appears; no generic generation or latent-rollout production change appears unless separately re-approved.

- [ ] **Step 2: Run the focused Stage 2 PEFT suite**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_peft.py tests/stage2/test_peft_checkpointing.py tests/stage2/test_checkpointing.py tests/stage2/test_routing.py tests/stage2/test_inference.py tests/stage2/test_eval_integration.py tests/stage2/test_launcher.py tests/stage2/test_comt_readiness.py -q
```

Expected: all focused tests pass.

- [ ] **Step 3: Run Stage 1 and generic-generation regressions**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2/test_routing.py::test_stage1_cli_defaults_are_preserved tests/stage2/test_routing.py::test_stage1_collator_contract_is_unchanged tests/stage2/test_generation_regression.py -q
```

Expected: Stage 1 behavior and generic generation tests pass.

- [ ] **Step 4: Run the complete feasible suite once**

Run:

```bash
PYTHONPATH="$PWD/transformers/src:$PWD/src:$PWD/tests/stage2" ./venv/ilvr-venv/bin/pytest tests/stage2 -q
```

Expected: all non-resource-gated tests pass; only explicitly documented GPU/opt-in tests may skip. Do not rerun this unchanged full suite after it passes.

- [ ] **Step 5: Audit every design requirement against files and tests**

Read both:

```text
docs/superpowers/specs/2026-08-24-ilvr-stage2-reimplementation-design.md
docs/superpowers/specs/2026-08-25-ilvr-stage2-peft-design.md
```

For each PEFT completion criterion, record the implementing function and verifying test in the final report. Explicitly report `B > 1` unsupported, KV cache optimization absent, merge/export absent, DeepSpeed status per method, and paper-level CoMT reproduction not run.

- [ ] **Step 6: Invoke the code-review workflow**

Invoke `superpowers:requesting-code-review`. Review specifically for visual target leakage, wrapper bypass, adapter/full checkpoint confusion, accidental embedding/LM-head training, unsafe resume inference, swallowed compatibility errors, and Stage 1 changes.

- [ ] **Step 7: Process review findings rigorously**

If review returns findings, invoke `superpowers:receiving-code-review`, reproduce each valid finding with a failing test, implement the smallest root-cause fix, run the directly affected focused test, and commit with a message naming the corrected boundary. If there are no findings, make no code change.

- [ ] **Step 8: Final clean-state check and completion commit if needed**

Run:

```bash
git status --short --branch
git log --oneline be5520f..HEAD
```

Expected: worktree is clean and commits are task-scoped. Do not merge, push, publish, or create a PR. Use `superpowers:finishing-a-development-branch` only to present integration choices; the user's standing choice is to leave merge separate.
