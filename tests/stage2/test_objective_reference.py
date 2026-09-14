from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile

import pytest
import torch
from torch.optim import SGD
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from transformers.utils.peft_utils import find_adapter_config_file

from peft_common import resolve_qwen2_5_vl_model
from stage1.checkpointing import save_stage1_peft_artifact
from stage1.peft import build_stage1_tuned_model
from stage1.token_rows import SparseOutputTokenRows, find_sparse_output_token_rows
from stage2.checkpointing import configure_stage2_metadata
from stage2.objective import entropy_transition, text_entropy
from stage2.peft import Stage2TuningConfig, load_stage1_dora_for_fresh_stage2
from tests.stage1.test_checkpointing import (
    LATENT_TOKEN_IDS,
    RESIZED_VOCAB_SIZE,
    TinyStage1Processor,
    sample_metadata,
    stage1_tuning,
    tiny_original_qwen,
)
from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds


@pytest.fixture
def checkpoint_directory():
    root = Path("checkpoints") / "stage2-objective-tests"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="task3-reference-", dir=root))
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


def _api():
    try:
        module = importlib.import_module("stage2.reference")
    except ModuleNotFoundError:
        pytest.fail("stage2.reference is not implemented")
    for name in (
        "FrozenReference",
        "load_frozen_reference",
        "save_objective_identity",
        "validate_objective_resume",
    ):
        assert callable(getattr(module, name, None)), f"missing reference API: {name}"
    return module


def _configure_stage1_tokens(model):
    model.resize_token_embeddings(RESIZED_VOCAB_SIZE)
    model.config.latent_token_id = LATENT_TOKEN_IDS[0]
    model.config.latent_start_id = LATENT_TOKEN_IDS[1]
    model.config.latent_end_id = LATENT_TOKEN_IDS[2]
    return model


def _write_full_stage1(directory: Path) -> Path:
    source = directory / "stage1-full"
    model = _configure_stage1_tokens(tiny_original_qwen())
    with torch.no_grad():
        model.model.embed_tokens.weight[-3:].add_(0.125)
        model.lm_head.weight[-3:].sub_(0.25)
    model.save_pretrained(source, safe_serialization=True)
    TinyStage1Processor().save_pretrained(source)
    return source


def _write_schema2_dora(directory: Path) -> tuple[Path, Path]:
    base_source = (directory / "original-base").resolve()
    adapter_source = directory / "stage1-dora"
    base = tiny_original_qwen()
    base.save_pretrained(base_source, safe_serialization=True)

    metadata = replace(
        sample_metadata(base), base_model_reference=str(base_source.resolve())
    )
    tuned_base = _configure_stage1_tokens(
        Qwen2_5_VLForConditionalGeneration.from_pretrained(base_source)
    )
    model = build_stage1_tuned_model(
        tuned_base, stage1_tuning(), LATENT_TOKEN_IDS
    )
    model.peft_config["default"].base_model_name_or_path = str(base_source.resolve())
    with torch.no_grad():
        for index, parameter in enumerate(
            (value for value in model.parameters() if value.requires_grad), start=1
        ):
            parameter.fill_(0.003 * (index % 11 + 1))
    save_stage1_peft_artifact(
        model, TinyStage1Processor(), adapter_source, metadata
    )
    return base_source, adapter_source


def _dependencies(loads=None):
    loads = loads if loads is not None else []

    def load_model(source, **kwargs):
        loads.append(("model", str(source), dict(kwargs)))
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(source, **kwargs)

    def load_processor(source, **kwargs):
        loads.append(("processor", str(source), dict(kwargs)))
        return TinyStage1Processor.from_pretrained(source)

    return SimpleNamespace(
        find_adapter_config=find_adapter_config_file,
        load_config=Qwen2_5_VLConfig.from_pretrained,
        load_processor=load_processor,
        load_model=load_model,
    )


def _args(reference_path: Path, **overrides):
    values = dict(
        enable_stage2_objective=True,
        lambda_dyn=1.0,
        lambda_vis=0.0,
        reference_model_path=str(reference_path),
        latent_size=2,
        supervise_latent_boundaries=True,
        attention_implementation="eager",
        resume_from_checkpoint_path="checkpoints/never-reference-resumed-student",
        cache_dir=None,
        revision=None,
        token=None,
        local_files_only=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _student_from_full(source: Path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        source, attn_implementation="eager"
    )
    configure_stage2_metadata(
        model.config,
        LatentTokenIds(*LATENT_TOKEN_IDS),
        latent_size=2,
        supervise_latent_boundaries=True,
    )
    return model


def _student_from_dora(base_source: Path, adapter_source: Path, *, dtype=None):
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_source, attn_implementation="eager", torch_dtype=dtype
    )
    model = load_stage1_dora_for_fresh_stage2(
        base,
        TinyStage1Processor.from_pretrained(adapter_source),
        adapter_source,
        Stage2TuningConfig(
            method="dora",
            r=4,
            alpha=8,
            dropout=0.0,
            target_modules=stage1_tuning().target_suffixes,
        ),
        requested_base_model_reference=str(base_source.resolve()),
    )
    configure_stage2_metadata(
        resolve_qwen2_5_vl_model(model).config,
        LatentTokenIds(*LATENT_TOKEN_IDS),
        latent_size=2,
        supervise_latent_boundaries=True,
    )
    return model


def _latent_inputs():
    input_ids = torch.tensor(
        [
            [
                7,
                LATENT_TOKEN_IDS[1],
                LATENT_TOKEN_IDS[0],
                LATENT_TOKEN_IDS[0],
                LATENT_TOKEN_IDS[2],
                8,
            ]
        ]
    )
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _sparse_input_rows(model):
    rows = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("trainable_tokens_delta.default")
    ]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("artifact_kind", ["full", "dora"])
def test_reference_is_independent_frozen_and_stable_through_student_update(
    checkpoint_directory, artifact_kind
):
    """Catches a copied/shared/trainable reference or a DoRA load missing sparse rows."""
    reference_api = _api()
    if artifact_kind == "full":
        reference_path = _write_full_stage1(checkpoint_directory)
        student = _student_from_full(reference_path)
    else:
        base_source, reference_path = _write_schema2_dora(checkpoint_directory)
        student = _student_from_dora(base_source, reference_path)

    rng_before = torch.get_rng_state().clone()
    reference = reference_api.load_frozen_reference(
        _args(reference_path),
        student=student,
        tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
        source_dependencies=_dependencies(),
    )
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert isinstance(reference, reference_api.FrozenReference)
    assert reference.model.training is False
    assert all(not parameter.requires_grad for parameter in reference.model.parameters())
    assert not ({parameter.data_ptr() for parameter in student.parameters()} & {
        parameter.data_ptr() for parameter in reference.model.parameters()
    })
    if artifact_kind == "dora":
        torch.testing.assert_close(
            _sparse_input_rows(reference.model), _sparse_input_rows(student)
        )
        torch.testing.assert_close(
            find_sparse_output_token_rows(reference.model).output_delta,
            find_sparse_output_token_rows(student).output_delta,
        )

    optimizer = SGD((p for p in student.parameters() if p.requires_grad), lr=0.01)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert not (optimizer_ids & {id(parameter) for parameter in reference.model.parameters()})
    before = {
        name: parameter.detach().clone()
        for name, parameter in reference.model.named_parameters()
    }
    student_before = {
        name: parameter.detach().clone()
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    }
    student.train()
    student_loss = student(**_latent_inputs()).logits.square().mean()
    student_loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(parameter, student_before[name])
        for name, parameter in student.named_parameters()
        if parameter.requires_grad
    )
    for name, parameter in reference.model.named_parameters():
        assert not parameter.requires_grad and parameter.grad is None
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    if artifact_kind == "dora":
        reference_rows = find_sparse_output_token_rows(reference.model)
        student_rows = find_sparse_output_token_rows(student)
        torch.testing.assert_close(reference_rows.token_ids, student_rows.token_ids)
        torch.testing.assert_close(reference_rows.output_delta, before[
            next(name for name in before if name.endswith("output_delta"))
        ])
    else:
        assert not isinstance(
            resolve_qwen2_5_vl_model(reference.model).lm_head,
            SparseOutputTokenRows,
        )
        assert not isinstance(resolve_qwen2_5_vl_model(student).lm_head, SparseOutputTokenRows)

    reloaded = reference_api.load_frozen_reference(
        _args(reference_path),
        student=student,
        tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
        source_dependencies=_dependencies(),
    )
    assert reloaded.identity == reference.identity
    for name, parameter in reloaded.model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("artifact_kind", ["full", "dora"])
def test_matched_reference_loads_own_latents_and_has_near_zero_entropy_transition(
    checkpoint_directory, artifact_kind
):
    """Catches reference reuse of student tensors or mismatched Stage 1 reconstruction."""
    reference_api = _api()
    if artifact_kind == "full":
        reference_path = _write_full_stage1(checkpoint_directory)
        student = _student_from_full(reference_path)
    else:
        base_source, reference_path = _write_schema2_dora(checkpoint_directory)
        student = _student_from_dora(base_source, reference_path)
    reference = reference_api.load_frozen_reference(
        _args(reference_path),
        student=student,
        tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
        source_dependencies=_dependencies(),
    )
    student.eval()
    with torch.no_grad():
        student_output = student(**_latent_inputs(), return_stage2_latent_states=True)
        reference_output = reference.model(
            **_latent_inputs(), return_stage2_latent_states=True
        )
    assert len(student_output.stage2_latent_states) == 2
    assert len(reference_output.stage2_latent_states) == 2
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(
            student_output.stage2_latent_states,
            reference_output.stage2_latent_states,
            strict=True,
        )
    )
    assert all(not value.requires_grad for value in reference_output.stage2_latent_states)
    torch.testing.assert_close(student_output.logits, reference_output.logits)
    student_entropy = text_entropy(student_output.logits[:, (0, 4), :], (7, 8))
    reference_entropy = text_entropy(reference_output.logits[:, (0, 4), :], (7, 8))
    transition = entropy_transition(
        student_entropy[:, :1],
        reference_entropy[:, :1],
        student_entropy[:, 1:],
        reference_entropy[:, 1:],
        eps=1e-6,
        gamma_H=0.0,
    )
    torch.testing.assert_close(
        transition["D"], torch.zeros_like(transition["D"]), atol=1e-6, rtol=0
    )


def test_zero_dynamic_weight_returns_before_any_reference_load(checkpoint_directory):
    """Catches the CE/vision-only path paying for or depending on a reference load."""
    reference_api = _api()

    class BombDependencies:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected source dependency access: {name}")

    assert reference_api.load_frozen_reference(
        _args(checkpoint_directory / "missing", lambda_dyn=0.0),
        student=torch.nn.Linear(2, 2),
        tokenizer=object(),
        source_dependencies=BombDependencies(),
    ) is None


def test_frozen_reference_value_rejects_mutable_or_training_model():
    """Catches direct construction that violates the fixed-reference invariant."""
    reference_api = _api()
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="frozen"):
        reference_api.FrozenReference(model, {"source": "fixture"})
    model.requires_grad_(False)
    with pytest.raises(ValueError, match="eval"):
        reference_api.FrozenReference(model, {"source": "fixture"})


@pytest.mark.parametrize("indexed", [False, "linked", "absolute"])
def test_loadable_symlink_weights_change_reference_identity_and_reject_resume(checkpoint_directory, indexed):
    """Catches omitted linked weights, including index-selected nonstandard shard paths."""
    api = _api()
    source = _write_full_stage1(checkpoint_directory)
    weights = source / "model.safetensors"
    external = checkpoint_directory / "linked-weights"
    external.mkdir()
    target = external / "weights.safetensors"
    weights.rename(target)
    if indexed:
        from safetensors import safe_open
        with safe_open(target, framework="pt") as tensors:
            names = list(tensors.keys())
        (source / "linked").symlink_to(external.resolve(), target_is_directory=True)
        shard_path = str(target.resolve()) if indexed == "absolute" else "linked/weights.safetensors"
        (source / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {}, "weight_map": {name: shard_path for name in names}}))
    else:
        weights.symlink_to(target.resolve())
    student = _student_from_full(source)
    args = _args(source)
    def load():
        return api.load_frozen_reference(args, student=student,
            tokenizer=TinyStage1Processor.from_pretrained(source).tokenizer,
            source_dependencies=_dependencies())
    first = load()
    first_weight = first.model.lm_head.weight.detach().clone()
    config = {"enable_stage2_objective": True, "lambda_dyn": 1.0}
    vocabulary = {"token_ids": [7, 8]}
    saved = checkpoint_directory / "resume-symlink"
    api.save_objective_identity(saved, config, first.identity, vocabulary)
    from safetensors.torch import load_file, save_file
    changed = load_file(target)
    changed["lm_head.weight"] = changed["lm_head.weight"] + 0.25
    save_file(changed, target, metadata={"format": "pt"})
    second = load()
    assert not torch.equal(first_weight, second.model.lm_head.weight)
    with pytest.raises(ValueError, match="reference identity mismatch"):
        api.validate_objective_resume(saved, config, second.identity, vocabulary)
    assert first.identity != second.identity


def test_reference_hashing_preserves_loadable_model_with_unused_stale_index(checkpoint_directory):
    """Catches interpreting a stale index that the actual model loader ignores."""
    api = _api()
    source = _write_full_stage1(checkpoint_directory)
    (source / "model.safetensors.index.json").write_text("stale unused index")
    student = _student_from_full(source)
    reference = api.load_frozen_reference(_args(source), student=student,
        tokenizer=TinyStage1Processor.from_pretrained(source).tokenizer,
        source_dependencies=_dependencies())
    torch.testing.assert_close(reference.model.lm_head.weight, student.lm_head.weight)


def test_dora_reference_preserves_student_mixed_precision_policy(checkpoint_directory):
    """Catches collapsing FP32 adapter parameters into the BF16 base dtype."""
    reference_api = _api()
    base_source, reference_path = _write_schema2_dora(checkpoint_directory)
    student = _student_from_dora(base_source, reference_path, dtype=torch.bfloat16)
    student_dtypes = {parameter.dtype for parameter in student.parameters()}
    assert student_dtypes == {torch.float32, torch.bfloat16}

    reference = reference_api.load_frozen_reference(
        _args(reference_path),
        student=student,
        tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
        source_dependencies=_dependencies(),
    )
    assert {parameter.dtype for parameter in reference.model.parameters()} == student_dtypes


def test_reference_identity_hashes_sources_and_never_uses_resume_student(
    checkpoint_directory,
):
    """Catches resume checkpoint substitution and identity omission of load artifacts."""
    reference_api = _api()
    reference_path = _write_full_stage1(checkpoint_directory)
    student = _student_from_full(reference_path)
    loads = []
    reference = reference_api.load_frozen_reference(
        _args(reference_path),
        student=student,
        tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
        source_dependencies=_dependencies(loads),
    )
    encoded = json.dumps(reference.identity, sort_keys=True, allow_nan=False)
    assert reference.identity["source"] == str(reference_path.resolve())
    assert "never-reference-resumed-student" not in encoded
    assert {item["path"] for item in reference.identity["artifacts"]} >= {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }
    assert all("never-reference-resumed-student" not in source for _, source, _ in loads)


def test_dora_rejects_nonlocal_metadata_base_before_source_loader(
    checkpoint_directory,
):
    """Catches Hub resolution or model loading before the local-base policy gate."""
    reference_api = _api()
    _, reference_path = _write_schema2_dora(checkpoint_directory)
    metadata_path = reference_path / "stage1_peft_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["base_model_reference"] = "forbidden/remote-model"
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    student_source = _write_full_stage1(checkpoint_directory)
    student = _student_from_full(student_source)
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append((name, args, kwargs))
            raise AssertionError(f"forbidden source dependency reached: {name}")

        return fail

    dependencies = SimpleNamespace(
        find_adapter_config=lambda source, **kwargs: str(
            Path(source) / "adapter_config.json"
        ),
        load_config=forbidden("load_config"),
        load_processor=forbidden("load_processor"),
        load_model=forbidden("load_model"),
    )
    with pytest.raises(ValueError, match="base.*local.*directory"):
        reference_api.load_frozen_reference(
            _args(reference_path),
            student=student,
            tokenizer=TinyStage1Processor.from_pretrained(student_source).tokenizer,
            source_dependencies=dependencies,
        )
    assert calls == []


def test_objective_sidecar_rejects_missing_and_every_identity_mismatch(
    checkpoint_directory,
):
    """Catches resume with changed objective, reference bytes, vocabulary, or sidecar."""
    reference_api = _api()
    reference_path = _write_full_stage1(checkpoint_directory)
    student = _student_from_full(reference_path)
    args = _args(reference_path)
    reference = reference_api.load_frozen_reference(
        args,
        student=student,
        tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
        source_dependencies=_dependencies(),
    )
    checkpoint = checkpoint_directory / "stage2-resume"
    checkpoint.mkdir()
    config = {
        "enable_stage2_objective": True,
        "lambda_dyn": 1.0,
        "lambda_vis": 0.0,
    }
    vocabulary = {"output_size": RESIZED_VOCAB_SIZE, "token_ids": [7, 8]}

    with pytest.raises(FileNotFoundError, match="stage2_objective.json"):
        reference_api.validate_objective_resume(
            checkpoint, config, reference.identity, vocabulary
        )
    reference_api.save_objective_identity(
        checkpoint, config, reference.identity, vocabulary
    )
    reference_api.validate_objective_resume(
        checkpoint, config, reference.identity, vocabulary
    )

    with pytest.raises(ValueError, match="configuration mismatch"):
        reference_api.validate_objective_resume(
            checkpoint, {**config, "lambda_dyn": 0.5}, reference.identity, vocabulary
        )
    with pytest.raises(ValueError, match="vocabulary identity mismatch"):
        reference_api.validate_objective_resume(
            checkpoint, config, reference.identity, {**vocabulary, "token_ids": [7]}
        )

    processor_config = reference_path / "preprocessor_config.json"
    processor_config.write_text(
        '{"processor_class":"TinyStage1Processor","changed":true}\n',
        encoding="utf-8",
    )
    changed_reference = reference_api.load_frozen_reference(
        args,
        student=student,
        tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
        source_dependencies=_dependencies(),
    )
    with pytest.raises(ValueError, match="reference identity mismatch"):
        reference_api.validate_objective_resume(
            checkpoint, config, changed_reference.identity, vocabulary
        )

    sidecar = checkpoint / "stage2_objective.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["reference_identity"]["source"] = str(checkpoint.resolve())
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reference identity mismatch"):
        reference_api.validate_objective_resume(
            checkpoint, config, reference.identity, vocabulary
        )


def test_reference_rejects_incompatible_tokenizer_or_output_layout(checkpoint_directory):
    """Catches entropy comparison across different token IDs or output widths."""
    reference_api = _api()
    reference_path = _write_full_stage1(checkpoint_directory)
    student = _student_from_full(reference_path)
    incompatible = TinyStage1Processor().tokenizer
    incompatible._ids["<|latent_start|>"] = 124

    with pytest.raises(ValueError, match="tokenizer|latent"):
        reference_api.load_frozen_reference(
            _args(reference_path),
            student=student,
            tokenizer=incompatible,
            source_dependencies=_dependencies(),
        )

    student.resize_token_embeddings(RESIZED_VOCAB_SIZE + 1)
    with pytest.raises(ValueError, match="output.*layout|vocabulary"):
        reference_api.load_frozen_reference(
            _args(reference_path),
            student=student,
            tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
            source_dependencies=_dependencies(),
        )
