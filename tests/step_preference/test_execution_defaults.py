"""Mode defaults and saved settings must select one coherent execution path."""
from dataclasses import asdict

import pytest

from step_preference.config import StepPreferenceConfig
from step_preference.execution import resolve_stage2_execution
from utils_deepseed import parse_args

CONTEXT = ["--stage", "stage2", "--step_preference_enabled",
           "--step_preference_mode", "stage2b", "--stage2a_checkpoint_path", "checkpoint"]
SAVED = dict(target_scope="until_next_latent",
             assistant_plan_rendering="adjacent-text-space-lf-v1",
             native_checkpointing=True, latent_checkpointing=False,
             sidecar_checkpointing=True)


def context(**kwargs):
    return StepPreferenceConfig(enabled=True, mode="stage2b",
                                stage2a_checkpoint_path="checkpoint", **kwargs)


def test_fresh_context_defaults_to_until():
    args = parse_args(CONTEXT)
    assert args.step_preference_target_scope == "until_next_latent"
    assert args.step_preference_config.target_scope == "until_next_latent"
    assert asdict(resolve_stage2_execution(args.step_preference_config,
        native_checkpointing=False)) == dict(
        assistant_plan_rendering="concatenated-v1", native_checkpointing=False,
        latent_checkpointing=True, sidecar_checkpointing=False)


@pytest.mark.parametrize("argv", [["--stage", "stage1"], ["--stage", "stage2"],
    ["--stage", "stage2", "--step_preference_enabled", "--step_preference_mode", "stage2a"],
    CONTEXT + ["--step_preference_objective", "legacy_dpo", "--step_negative_path", "neg",
               "--step_negative_manifest", "manifest"],
    CONTEXT + ["--step_preference_target_scope", "immediate_text"]])
def test_non_until_modes_remain_launchable(argv):
    config = parse_args(argv).step_preference_config
    assert config.target_scope == "immediate_text"
    result = resolve_stage2_execution(config, native_checkpointing=True)
    assert result.native_checkpointing and not result.latent_checkpointing
    assert result.sidecar_checkpointing
    assert result.assistant_plan_rendering == (
        "adjacent-text-space-lf-v1" if config.enabled else "concatenated-v1")


def test_saved_until_execution_keeps_historical_settings():
    result = resolve_stage2_execution(context(), native_checkpointing=False, saved=SAVED)
    assert asdict(result) == {k: v for k, v in SAVED.items() if k != "target_scope"}


@pytest.mark.parametrize("saved", [{}, {k: v for k, v in SAVED.items() if k != "latent_checkpointing"},
    {**SAVED, "native_checkpointing": "false"}, {**SAVED, "sidecar_checkpointing": 1},
    {**SAVED, "assistant_plan_rendering": "invalid"}, {**SAVED, "target_scope": "invalid"},
    {**SAVED, "target_scope": "immediate_text"}, {**SAVED, "latent_checkpointing": True}])
def test_saved_settings_fail_closed(saved):
    with pytest.raises(ValueError):
        resolve_stage2_execution(context(), native_checkpointing=False, saved=saved)


def test_fresh_until_rejects_native_even_with_explicit_latent_off():
    for latent in (None, False, True):
        with pytest.raises(ValueError, match="native"):
            resolve_stage2_execution(context(), native_checkpointing=True,
                                     latent_checkpointing=latent)


def test_explicit_latent_control_and_conflict():
    args = parse_args(CONTEXT + ["--no-stage2_latent_activation_checkpointing"])
    assert not resolve_stage2_execution(args.step_preference_config,
        native_checkpointing=False,
        latent_checkpointing=args.stage2_latent_activation_checkpointing).latent_checkpointing
    with pytest.raises(ValueError, match="native"):
        resolve_stage2_execution(StepPreferenceConfig(), native_checkpointing=True,
                                 latent_checkpointing=True)


def test_unsupported_stage2b_resume_stays_rejected():
    with pytest.raises(ValueError, match="mutually exclusive for Stage 2B"):
        parse_args(CONTEXT + ["--resume_from_checkpoint_path", "saved"])


def test_parser_retains_scope_omission_without_loading_resume_config(monkeypatch):
    from transformers import AutoConfig
    monkeypatch.setattr(AutoConfig, "from_pretrained",
                        lambda *a, **k: pytest.fail("parser must not load sources"))
    args = parse_args(["--stage", "stage2", "--resume_from_checkpoint_path", "saved"])
    assert args._step_preference_target_scope_requested is None
    explicit = parse_args(CONTEXT + ["--step_preference_target_scope", "immediate_text"])
    assert explicit._step_preference_target_scope_requested == "immediate_text"


def assemble(monkeypatch, tmp_path, *, config=None, saved=None, resume=False,
             requested_scope=None, native=False, latent=None, replay_model=None, arg_overrides=None):
    """Use real assembly/backend/collator; replace artifact I/O and length measurement."""
    import json
    import main
    from types import SimpleNamespace
    from stage2.checkpointing import Stage2LoadDecision
    from stage2.peft import Stage2TuningConfig
    from tests.stage2.helpers import tiny_stage2_qwen
    from tests.stage2.test_routing import CompositionProcessor, stage2_args

    model = tiny_stage2_qwen() if replay_model is None else replay_model
    # Stored initializer settings must never become resume settings.
    model.config.stage2_preference_execution = dict(SAVED)
    from PIL import Image
    Image.new("RGB", (56, 56)).save(tmp_path / "input.png")
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(dict(sample_id="sample", image_input="input.png",
        text_input="question", sequence_plan=[
            dict(type="text", content="before"),
            dict(type="latent", helper_image="helper.png"),
            dict(type="text", content="reason"), dict(type="text", content="answer")])) + "\n")
    config = config or StepPreferenceConfig()
    args = stage2_args(step_preference_config=config, data_path=str(source),
        image_root=str(tmp_path), latent_size=2, gradient_checkpointing=native,
        stage2_latent_activation_checkpointing=latent,
        resume_from_checkpoint_path="resume" if resume else None,
        _step_preference_target_scope_requested=requested_scope)
    for key, value in (arg_overrides or {}).items():
        setattr(args, key, value)
    decision = Stage2LoadDecision("initializer", "initializer", None,
        "resume" if resume else None,
        Stage2TuningConfig(method="full", r=4, alpha=8, dropout=0.0,
                           target_modules=("q_proj",)), 2, True)
    monkeypatch.setattr(main, "resolve_stage2_load_decision", lambda **k: decision)
    monkeypatch.setattr(main, "resolve_stage2b_initialization_decision", lambda **k: decision)
    monkeypatch.setattr(main, "_require_stage2b_attention_contract", lambda a: None)
    monkeypatch.setattr(main, "validate_stage2b_peft_base_identity", lambda **k: None)
    monkeypatch.setattr(main, "inspect_stage2a_checkpoint_identity", lambda p: ({}, {}))
    monkeypatch.setattr(main, "build_prepared_example_length_measure", lambda p: lambda e: 10)
    loaded_configs = []
    def load_config(source, **kwargs):
        loaded_configs.append(source)
        return SimpleNamespace(**({} if saved is None else {"stage2_preference_execution": saved}))
    components = main.build_stage2_training_components(args,
        source_dependencies=main.Stage2SourceDependencies(
            find_adapter_config=lambda *a, **k: None, load_config=load_config,
            load_processor=lambda *a, **k: CompositionProcessor(),
            load_model=lambda *a, **k: model))
    return args, components, loaded_configs


@pytest.mark.parametrize("scope,rendering,latent,sidecar", [
    (None, "concatenated-v1", True, False),
    ("immediate_text", "adjacent-text-space-lf-v1", False, True)])
def test_assembly_selects_rendering_and_checkpointing(monkeypatch, tmp_path, scope, rendering, latent, sidecar):
    import main
    from transformers import Qwen2_5_VLConfig
    args, components, reads = assemble(monkeypatch, tmp_path, config=context(target_scope=scope))
    model = components.model
    assert not reads, "fresh initializer execution settings must be ignored"
    assert components.data_collator.preference_target_scope == (scope or "until_next_latent")
    text = components.train_dataset[0].conversation[1]["content"][0]["text"]
    assert ("reasonanswer" if scope is None else "reason \nanswer") in text
    assert not model.is_gradient_checkpointing
    assert model.config.stage2_latent_activation_checkpointing is latent
    assert model.config.stage2_sidecar_activation_checkpointing is sidecar
    assert not main.build_stage2_training_arguments(args).gradient_checkpointing
    saved = Qwen2_5_VLConfig.from_dict(model.config.to_dict()).stage2_preference_execution
    assert saved == dict(target_scope=scope or "until_next_latent",
        assistant_plan_rendering=rendering, native_checkpointing=False,
        latent_checkpointing=latent, sidecar_checkpointing=sidecar)
    assert asdict(resolve_stage2_execution(components.step_preference_config,
        native_checkpointing=True, saved=saved)) == {k:v for k,v in saved.items() if k != "target_scope"}


def test_supported_resume_restores_saved_rendering_and_native_before_data(monkeypatch, tmp_path):
    import main
    saved = {**SAVED, "target_scope": "immediate_text"}
    args, components, reads = assemble(monkeypatch, tmp_path, saved=saved, resume=True)
    assert reads == ["resume"]
    assert "reason \nanswer" in components.train_dataset[0].conversation[1]["content"][0]["text"]
    assert components.model.is_gradient_checkpointing
    assert not components.model.config.stage2_latent_activation_checkpointing
    assert main.build_stage2_training_arguments(args).gradient_checkpointing
    assert args.step_preference_target_scope == "immediate_text"
    assert components.model.config.stage2_preference_execution == saved


@pytest.mark.parametrize("native,latent", [(True, None), (False, True)])
def test_plain_resume_without_execution_dictionary_keeps_existing_behavior(monkeypatch, tmp_path, native, latent):
    args, components, _ = assemble(monkeypatch, tmp_path, resume=True, native=native, latent=latent)
    assert components.model.is_gradient_checkpointing is native
    assert components.model.config.stage2_latent_activation_checkpointing is bool(latent)
    assert "reasonanswer" in components.train_dataset[0].conversation[1]["content"][0]["text"]


@pytest.mark.parametrize("saved,requested,preference", [
    ({**SAVED, "target_scope": "until_next_latent"}, "immediate_text", False),
    ({}, None, False), (None, None, True)])
def test_resume_errors_before_data(monkeypatch, tmp_path, saved, requested, preference):
    import main
    monkeypatch.setattr(main, "load_jsonl_examples", lambda *a: pytest.fail("data read before resume validation"))
    config = StepPreferenceConfig(enabled=preference, mode="stage2a")
    with pytest.raises(ValueError, match="saved|resume"):
        assemble(monkeypatch, tmp_path, config=config, saved=saved, resume=True,
                 requested_scope=requested)


def test_collator_omission_is_mode_aware():
    from stage2.data import Stage2DataCollator
    from tests.stage2.test_routing import CompositionProcessor
    for enabled, negatives, expected in [(True, None, "until_next_latent"),
                                        (False, None, "immediate_text"),
                                        (True, {}, "immediate_text")]:
        assert Stage2DataCollator(CompositionProcessor(), None, 2, True, step_preference_enabled=enabled,
                                 negative_store=negatives).preference_target_scope == expected


def test_old_renderer_manifest_is_rejected_by_until_assembly(monkeypatch, tmp_path):
    import hashlib
    from types import SimpleNamespace
    import stage2.length_manifest as manifests

    path = tmp_path / "manifest.json"
    path.write_bytes(b"sealed old manifest")
    def old_manifest(_path):
        return SimpleNamespace(
            data_sha256=hashlib.sha256((tmp_path / "source.jsonl").read_bytes()).hexdigest(),
            processor_sha256="processor", latent_size=2, supervise_latent_boundaries=True,
            max_sequence_length=8192, assistant_plan_rendering="adjacent-text-space-lf-v1")
    monkeypatch.setattr(manifests, "read_stage2_length_manifest", old_manifest)
    monkeypatch.setattr(manifests, "_processor_sha256", lambda p: "processor")
    with pytest.raises(ValueError, match="manifest configuration"):
        assemble(monkeypatch, tmp_path, config=context(), arg_overrides=dict(
            stage2_length_manifest=str(path),
            stage2_length_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def test_cli_default_through_assembly_replays_only_latent_calls(monkeypatch, tmp_path):
    import torch
    import main
    import transformers.models.qwen2_5_vl.stage2_shared_prefix as shared
    from step_preference.equivalence import _trainer
    from tests.step_preference.test_context_sidecars import two_step_case

    parsed = parse_args(CONTEXT + ["--step_sidecar_attention_backend", "dense"])
    model, batch = two_step_case()
    args, components, _ = assemble(monkeypatch, tmp_path,
        config=parsed.step_preference_config, native=parsed.gradient_checkpointing,
        latent=parsed.stage2_latent_activation_checkpointing, replay_model=model)
    assert not main.build_stage2_training_arguments(args).gradient_checkpointing
    model = components.model.train()
    records, captured = [], []
    phase = ["forward"]
    def observed(_module, _args, kwargs):
        kind = ("sidecar" if kwargs.get("stage2_prefix_key_values") is not None
                else "clean" if kwargs.get("stage2_capture_kv", False) else "latent")
        records.append((phase[0], kind, kwargs["inputs_embeds"].shape[1]))
    original = shared.score_sidecars
    def score(model, options, ids, embeds, positions, kv):
        assert options["checkpoint"] is False
        for key, value in kv:
            assert key.requires_grad and value.requires_grad
            key.retain_grad()
            value.retain_grad()
            captured.extend((key, value))
        return original(model, options, ids, embeds, positions, kv)
    monkeypatch.setattr(shared, "score_sidecars", score)
    monkeypatch.setattr(shared, "checkpoint", lambda *a, **k: pytest.fail("sidecar replay"))
    handle = model.model.register_forward_pre_hook(observed, with_kwargs=True)
    try:
        _, output = _trainer(components.step_preference_config).compute_loss(model, batch, return_outputs=True)
        phase[0] = "backward"
        voff = [r["mean_logp"] for r in output.stage2_sidecar_results if r["kind"] == "visual_off"]
        assert voff
        (-torch.stack(voff).sum()).backward()
    finally:
        handle.remove()
    forward = [length for p, k, length in records if p == "forward" and k == "latent"]
    backward = [(k, length) for p, k, length in records if p == "backward"]
    assert forward and backward
    assert all(k == "latent" for k, _ in backward)
    assert sorted(length for _, length in backward) == sorted(forward)
    assert any(p == "forward" and k == "clean" for p, k, _ in records)
    assert any(p == "forward" and k == "sidecar" for p, k, _ in records)
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in captured)
    assert any(torch.count_nonzero(t.grad).item() for t in captured)
