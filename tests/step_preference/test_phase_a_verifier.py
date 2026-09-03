from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from transformers import Qwen2_5_VLConfig

import main as training_main
from scripts import verify_step_preference_phase_a as verifier
from step_preference.config import StepPreferenceConfig
from step_preference.negatives import StepNegative, write_negative_artifact


COMPONENT_METRICS = {
    "loss/ce": 1.25,
    "loss/dpo_visual": 0.2,
    "loss/dpo_text": 0.3,
    "loss/entropy_visual": 0.04,
    "loss/entropy_text": 0.05,
    "stats/valid_visual_pairs": 1.0,
    "stats/valid_textual_pairs": 0.0,
}


def _write_full_stage2_checkpoint(path) -> None:
    path.mkdir()
    config = Qwen2_5_VLConfig()
    config.stage = "stage2"
    config.latent_size = 2
    config.supervise_latent_boundaries = False
    config.latent_pad_token_id = config.latent_token_id = 100
    config.latent_start_token_id = config.latent_start_id = 101
    config.latent_end_token_id = config.latent_end_id = 102
    config.save_pretrained(path)
    (path / "model.safetensors").write_bytes(b"structural-test-marker")


def _negative_row(**overrides) -> StepNegative:
    values = dict(
        sample_id="sample-1",
        step_id=0,
        chosen_text="chosen",
        rejected_visual_text="visual rejected",
        rejected_textual_text=None,
        valid_visual_negative=True,
        valid_textual_negative=False,
        reference_chosen_score=-0.5,
        reference_visual_rejected_score=-0.8,
        reference_textual_rejected_score=None,
    )
    values.update(overrides)
    return StepNegative(**values)


def _write_negative_fixture(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text('{"sample_id":"sample-1"}\n', encoding="utf-8")
    stage2a = tmp_path / "stage2a"
    stage2a.mkdir()
    negative_path = tmp_path / "negatives.jsonl"
    manifest_path = tmp_path / "negative-manifest.json"
    write_negative_artifact(
        negative_path=negative_path,
        manifest_path=manifest_path,
        negatives=[_negative_row()],
        source_path=source,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selected_row_indices=[0],
        stage2a_checkpoint_path=stage2a,
        checkpoint_files=[],
        tokenizer_config_identity="tokenizer-v1",
        latent_size=2,
        generation_settings={"do_sample": False},
    )
    return negative_path, manifest_path


def _summary_payload(**overrides):
    payload = {
        "schema_version": 1,
        "global_step": 1,
        "optimizer_step": 1,
        "training_loss": 1.84,
        "gradient_norm": 0.75,
        "metrics": dict(COMPONENT_METRICS),
    }
    for name, value in overrides.items():
        if name.startswith("metric__"):
            payload["metrics"][name.removeprefix("metric__").replace("__", "/")] = value
        else:
            payload[name] = value
    return payload


def _write_training_log(path, *payloads) -> None:
    lines = [
        "2026-09-03 12:00:00 - INFO - ordinary training log",
        *(
            "2026-09-03 12:00:01 - INFO - "
            + verifier.PHASE_A_SUMMARY_PREFIX
            + json.dumps(payload, sort_keys=True, separators=(",", ":"))
            for payload in payloads
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _invoke(tmp_path, *, payload=None):
    negative_path, manifest_path = _write_negative_fixture(tmp_path)
    training_log = tmp_path / "stage2b.log"
    _write_training_log(training_log, payload or _summary_payload())
    checkpoint = tmp_path / "checkpoint"
    _write_full_stage2_checkpoint(checkpoint)
    output = tmp_path / "verification.json"
    argv = [
        "--negative_manifest",
        str(manifest_path),
        "--negative_path",
        str(negative_path),
        "--training_log",
        str(training_log),
        "--checkpoint_path",
        str(checkpoint),
        "--output_path",
        str(output),
    ]
    return verifier.main(argv), output, negative_path, training_log, checkpoint


def test_phase_a_verifier_writes_one_deterministic_success_report(tmp_path):
    """Catches any gate being omitted from the machine-readable success result."""
    exit_code, output, negative_path, _log, checkpoint = _invoke(tmp_path)

    assert exit_code == 0
    expected = {
        "checkpoint": {
            "artifact_kind": "full",
            "path": str(checkpoint.resolve()),
        },
        "errors": [],
        "negative_artifact": {
            "negative_path": str(negative_path.resolve()),
            "negative_sha256": hashlib.sha256(negative_path.read_bytes()).hexdigest(),
            "row_count": 1,
            "valid_pairs": 1,
            "valid_textual_pairs": 0,
            "valid_visual_pairs": 1,
        },
        "schema_version": 1,
        "status": "pass",
        "training": {
            "global_step": 1,
            "gradient_norm": 0.75,
            "metrics": COMPONENT_METRICS,
            "optimizer_step": 1,
            "total_loss": 1.84,
            "training_loss": 1.84,
            "valid_pairs": 1.0,
        },
    }
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    assert output.read_text(encoding="utf-8") == (
        json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"
    )


@pytest.mark.parametrize("log_case", ["malformed", "duplicate"])
def test_phase_a_verifier_fails_closed_on_malformed_or_duplicate_summary(
    tmp_path, log_case
):
    """Catches ambiguous or non-JSON Trainer evidence being accepted."""
    negative_path, manifest_path = _write_negative_fixture(tmp_path)
    training_log = tmp_path / "stage2b.log"
    if log_case == "malformed":
        training_log.write_text(
            verifier.PHASE_A_SUMMARY_PREFIX + '{"schema_version":1,broken}\n',
            encoding="utf-8",
        )
    else:
        _write_training_log(training_log, _summary_payload(), _summary_payload())
    checkpoint = tmp_path / "checkpoint"
    _write_full_stage2_checkpoint(checkpoint)
    output = tmp_path / "verification.json"

    exit_code = verifier.main(
        [
            "--negative_manifest",
            str(manifest_path),
            "--negative_path",
            str(negative_path),
            "--training_log",
            str(training_log),
            "--checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(output),
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code != 0
    assert report["status"] == "fail"
    assert len(report["errors"]) == 1


@pytest.mark.parametrize(
    "payload",
    [
        _summary_payload(training_loss=float("nan")),
        _summary_payload(gradient_norm=float("inf")),
        _summary_payload(metric__loss__ce=float("nan")),
        _summary_payload(global_step=0, optimizer_step=0),
        _summary_payload(
            metric__stats__valid_visual_pairs=0.0,
            metric__stats__valid_textual_pairs=0.0,
        ),
    ],
    ids=["nonfinite-total", "nonfinite-gradient", "nonfinite-component", "zero-steps", "zero-trained-pairs"],
)
def test_phase_a_verifier_rejects_nonfinite_or_zero_training_evidence(
    tmp_path, payload
):
    """Catches non-observations being reported as a successful training gate."""
    exit_code, output, *_ = _invoke(tmp_path, payload=payload)

    assert exit_code != 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "fail"


def test_phase_a_verifier_rejects_stale_negative_artifact(tmp_path):
    """Catches pair survival inferred without authenticating current JSONL bytes."""
    exit_code, output, negative_path, training_log, checkpoint = _invoke(tmp_path)
    assert exit_code == 0
    negative_path.write_bytes(negative_path.read_bytes() + b"\n")

    exit_code = verifier.main(
        [
            "--negative_manifest",
            str(tmp_path / "negative-manifest.json"),
            "--negative_path",
            str(negative_path),
            "--training_log",
            str(training_log),
            "--checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(output),
        ]
    )

    assert exit_code != 0
    assert "SHA-256" in json.loads(output.read_text(encoding="utf-8"))["errors"][0]


def test_phase_a_verifier_rejects_invalid_stage2_checkpoint(tmp_path):
    """Catches a config-only or weight-only directory passing checkpoint gating."""
    exit_code, output, _negative, _log, checkpoint = _invoke(tmp_path)
    assert exit_code == 0
    (checkpoint / "model.safetensors").unlink()

    exit_code = verifier.main(
        [
            "--negative_manifest",
            str(tmp_path / "negative-manifest.json"),
            "--negative_path",
            str(tmp_path / "negatives.jsonl"),
            "--training_log",
            str(tmp_path / "stage2b.log"),
            "--checkpoint_path",
            str(checkpoint),
            "--output_path",
            str(output),
        ]
    )

    assert exit_code != 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "fail"


def test_phase_a_verifier_does_not_resolve_away_checkpoint_symlink(tmp_path):
    """Catches a symlink bypassing the existing Stage 2 artifact path contract."""
    exit_code, output, negative_path, training_log, checkpoint = _invoke(tmp_path)
    assert exit_code == 0
    checkpoint_link = tmp_path / "checkpoint-link"
    checkpoint_link.symlink_to(checkpoint, target_is_directory=True)

    exit_code = verifier.main(
        [
            "--negative_manifest",
            str(tmp_path / "negative-manifest.json"),
            "--negative_path",
            str(negative_path),
            "--training_log",
            str(training_log),
            "--checkpoint_path",
            str(checkpoint_link),
            "--output_path",
            str(output),
        ]
    )

    assert exit_code != 0
    assert "symlink" in json.loads(output.read_text(encoding="utf-8"))["errors"][0]


def test_stage2b_summary_producer_logs_only_canonical_scalar_evidence(monkeypatch):
    """Catches private data leakage or omission of the latest complete metric record."""
    logged = []
    trainer = SimpleNamespace(
        state=SimpleNamespace(
            global_step=3,
            log_history=[{"loss/ce": 99.0}, dict(COMPONENT_METRICS)],
        ),
        _observed_optimizer_step=3,
        _representative_gradient_norm=0.625,
        is_world_process_zero=lambda: True,
    )
    args = SimpleNamespace(
        step_preference_config=StepPreferenceConfig(
            enabled=True,
            mode="stage2b",
            stage2a_checkpoint_path="stage2a",
            negative_path="negatives.jsonl",
            negative_manifest="manifest.json",
        )
    )
    monkeypatch.setattr(training_main.logging, "info", logged.append)

    training_main.publish_step_preference_phase_a_summary(
        args,
        trainer=trainer,
        train_result=SimpleNamespace(training_loss=1.75),
    )

    assert len(logged) == 1
    assert logged[0].startswith(verifier.PHASE_A_SUMMARY_PREFIX)
    payload = json.loads(logged[0].removeprefix(verifier.PHASE_A_SUMMARY_PREFIX))
    assert payload == {
        "schema_version": 1,
        "global_step": 3,
        "optimizer_step": 3,
        "training_loss": 1.75,
        "gradient_norm": 0.625,
        "metrics": COMPONENT_METRICS,
    }
    assert "chosen" not in logged[0]
    assert "rejected" not in logged[0]


def test_run_stage2_publishes_phase_a_summary_once_after_training(monkeypatch):
    """Catches the producer helper not being wired to completed Stage 2B training."""
    events = []
    train_result = SimpleNamespace(training_loss=1.75)

    class Trainer:
        def __init__(self, **_kwargs):
            pass

        def train(self, *, resume_from_checkpoint):
            assert resume_from_checkpoint is None
            events.append("train")
            return train_result

    config = StepPreferenceConfig(
        enabled=True,
        mode="stage2b",
        stage2a_checkpoint_path="stage2a",
        negative_path="negatives.jsonl",
        negative_manifest="manifest.json",
    )
    args = SimpleNamespace(
        step_preference_config=config,
        stage2_diagnostics_interval=0,
        save_final_model=False,
    )
    components = SimpleNamespace(
        model=object(),
        processor=object(),
        train_dataset=["sample"],
        data_collator=object(),
        trainer_class=Trainer,
        trainer_resume_path=None,
        peft_metadata=None,
        step_preference_config=config,
    )
    monkeypatch.setattr(
        training_main,
        "bind_deepspeed_runtime_profile",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        training_main,
        "build_stage2_training_components",
        lambda _args: components,
    )
    monkeypatch.setattr(
        training_main,
        "build_stage2_training_arguments",
        lambda _args: object(),
    )
    monkeypatch.setattr(
        training_main,
        "publish_step_preference_phase_a_summary",
        lambda _args, *, trainer, train_result: events.append("summary"),
    )
    monkeypatch.setattr(
        training_main,
        "publish_stage2_training_evidence",
        lambda _args, *, trainer, train_result, profile: events.append("benchmark"),
    )

    training_main.run_stage2(args)

    assert events == ["train", "summary", "benchmark"]
