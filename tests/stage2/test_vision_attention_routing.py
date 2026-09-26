"""Routing of the split SDPA-language / FlashAttention2-vision backend."""

from types import SimpleNamespace

import main
from utils_deepseed import parse_args
from .helpers import tiny_stage2_qwen
from .test_routing import CompositionProcessor, _stub_stage2_data_pipeline, stage2_args
from .test_objective_reference import checkpoint_directory  # noqa: F401  (fixture)


def test_vision_attention_defaults_to_language_attention_and_accepts_flash():
    """Catches a missing option or a silent backend change for existing launches."""
    assert parse_args([]).vision_attention_implementation == "sdpa"
    args = parse_args(["--vision_attention_implementation", "flash_attention_2"])
    assert args.attention_implementation == "sdpa"
    assert args.vision_attention_implementation == "flash_attention_2"


def test_stage2_training_model_uses_split_loader_and_validates_backends(monkeypatch):
    """Catches the student silently keeping SDPA vision when FA2 vision is requested."""
    from stage2.checkpointing import Stage2LoadDecision
    from stage2.peft import Stage2TuningConfig

    calls = SimpleNamespace(prepared=[], filtered=None, mixed=[], validated=[], plain=[])
    processor = CompositionProcessor()
    model = tiny_stage2_qwen()
    model.config.stage = "stage1"
    model.config.supervise_latent_boundaries = True
    decision = Stage2LoadDecision(
        base_model_source="stage1/full-base",
        processor_source="stage1/full-processor",
        adapter_source=None,
        trainer_resume_path=None,
        tuning=Stage2TuningConfig("full", 4, 8, 0.0, ("q_proj", "v_proj")),
        latent_size=2,
        supervise_latent_boundaries=True,
    )
    monkeypatch.setattr(main, "resolve_stage2_load_decision", lambda **kwargs: decision)
    monkeypatch.setattr(main.AutoProcessor, "from_pretrained", lambda source: processor)
    monkeypatch.setattr(
        main.Qwen2_5_VLForConditionalGeneration, "from_pretrained",
        lambda source, **kwargs: calls.plain.append((source, kwargs)) or model,
    )

    def mixed(source, **kwargs):
        calls.mixed.append((source, kwargs["attention_implementation"],
                            kwargs["vision_attention_implementation"]))
        return model

    monkeypatch.setattr(main, "load_qwen2_5_vl_with_attention", mixed)
    monkeypatch.setattr(
        main, "validate_qwen2_5_vl_attention_backends",
        lambda resolved, **kwargs: calls.validated.append(
            (resolved, kwargs["attention_implementation"],
             kwargs["vision_attention_implementation"])
        ),
    )
    _stub_stage2_data_pipeline(monkeypatch, calls)

    main.build_stage2_training_components(
        stage2_args(vision_attention_implementation="flash_attention_2")
    )

    assert calls.plain == []
    assert calls.mixed == [("stage1/full-base", "sdpa", "flash_attention_2")]
    assert calls.validated == [(model, "sdpa", "flash_attention_2")]


def test_frozen_reference_uses_split_loader_when_vision_backend_differs(
    monkeypatch, checkpoint_directory
):
    """Catches the reference re-encoding images with SDPA vision attention."""
    from stage2 import reference as reference_api
    from .test_objective_reference import (
        TinyStage1Processor, _args, _dependencies, _student_from_full, _write_full_stage1,
    )

    reference_path = _write_full_stage1(checkpoint_directory)
    student = _student_from_full(reference_path)
    dependencies = _dependencies()
    mixed, validated = [], []

    def split_loader(source, **kwargs):
        mixed.append((str(source), kwargs["attention_implementation"],
                      kwargs["vision_attention_implementation"],
                      kwargs["model_load_kwargs"].get("torch_dtype")))
        return dependencies.load_model(
            source, attn_implementation="eager", **kwargs["model_load_kwargs"]
        )

    monkeypatch.setattr(reference_api, "load_qwen2_5_vl_with_attention", split_loader)
    monkeypatch.setattr(
        reference_api, "validate_qwen2_5_vl_attention_backends",
        lambda resolved, **kwargs: validated.append(kwargs["vision_attention_implementation"]),
    )

    reference = reference_api.load_frozen_reference(
        _args(reference_path, attention_implementation="sdpa",
              vision_attention_implementation="flash_attention_2"),
        student=student,
        tokenizer=TinyStage1Processor.from_pretrained(reference_path).tokenizer,
        source_dependencies=dependencies,
    )

    from pathlib import Path
    assert [(Path(source).resolve(), lang, vision) for source, lang, vision, _ in mixed] == [
        (Path(reference_path).resolve(), "sdpa", "flash_attention_2")
    ]
    assert mixed[0][3] == next(student.parameters()).dtype
    assert validated == ["flash_attention_2"]
    assert reference.model.training is False
