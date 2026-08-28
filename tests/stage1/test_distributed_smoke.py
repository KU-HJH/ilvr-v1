from __future__ import annotations

import os
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

_platform_python_version = platform.python_version
if _platform_python_version().endswith("+"):
    platform.python_version = lambda: _platform_python_version().rstrip("+")

from transformers.integrations.deepspeed import (
    is_deepspeed_zero3_enabled,
    set_hf_deepspeed_config,
    unset_hf_deepspeed_config,
)

from stage1.benchmark import run_or_load_distributed_witness
from tests.stage1 import zero_compat_diagnostic
from tests.stage1.zero_compat_diagnostic import (
    SUPPORTED_DIAGNOSTIC_METHODS,
    build_zero_compat_parser,
)

platform.python_version = _platform_python_version


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTED_METHOD = os.environ.get("ILVR_DISTRIBUTED_METHOD") or None
if SELECTED_METHOD not in {None, *SUPPORTED_DIAGNOSTIC_METHODS}:
    raise pytest.UsageError(
        "invalid ILVR_DISTRIBUTED_METHOD: expected one of "
        f"{', '.join(SUPPORTED_DIAGNOSTIC_METHODS)}, "
        f"got {SELECTED_METHOD!r}"
    )


def test_stage1_diagnostic_supported_methods_include_lora():
    """Catches accidental removal of LoRA from the diagnostic contract."""
    assert SUPPORTED_DIAGNOSTIC_METHODS == ("full", "lora", "dora")


@pytest.mark.parametrize("stage", (1, 2, 3))
@pytest.mark.parametrize("method", SUPPORTED_DIAGNOSTIC_METHODS)
def test_stage1_diagnostic_accepts_all_supported_pairs(stage, method):
    """Catches a supported ZeRO/method lifecycle becoming unlaunchable."""
    parser = build_zero_compat_parser()
    args = parser.parse_args(
        [
            "--zero-stage",
            str(stage),
            "--method",
            method,
            "--deepspeed",
            f"configs/deepspeed_zero{stage}.json",
            "--experiment-id",
            f"stage1-zero{stage}-{method}-tiny-lifecycle",
        ]
    )
    assert (args.zero_stage, args.method) == (stage, method)


def test_stage1_tiny_parser_preserves_defaults_and_accepts_lifecycle_steps():
    """Catches matrix step controls changing the legacy 1-to-2 lifecycle."""
    parser = build_zero_compat_parser()
    base = [
        "--zero-stage",
        "3",
        "--method",
        "dora",
        "--deepspeed",
        "configs/deepspeed_zero3.json",
        "--experiment-id",
        "stage1-zero3-dora",
    ]

    defaults = parser.parse_args(base)
    requested = parser.parse_args(
        [*base, "--initial-steps", "2", "--resumed-steps", "3"]
    )

    assert (defaults.initial_steps, defaults.resumed_steps) == (1, 2)
    assert (requested.initial_steps, requested.resumed_steps) == (2, 3)
    assert defaults.phase == "lifecycle"


@pytest.mark.parametrize("phase", ("lifecycle", "initial", "resume"))
def test_stage1_tiny_parser_accepts_only_fixed_lifecycle_phases(phase):
    parser = build_zero_compat_parser()
    args = parser.parse_args([
        "--phase", phase, "--zero-stage", "2", "--method", "lora",
        "--deepspeed", "configs/deepspeed_zero2.json",
        "--experiment-id", "stage1-zero2-lora-phase",
    ])
    assert args.phase == phase


def test_stage1_phase_witness_writes_identical_bytes_on_both_nodes(tmp_path):
    witness = {"schema_version": 1, "phase": "initial"}
    paths = [
        SimpleNamespace(result_dir=tmp_path / node) for node in ("remote", "local")
    ]
    for owned_paths in paths:
        zero_compat_diagnostic._publish_witness(
            owned_paths, witness, filename="initial-witness.json",
            is_writer=True, failure_counter=lambda failed: failed,
            synchronize=lambda: None,
        )
    assert (
        paths[0].result_dir.joinpath("initial-witness.json").read_bytes()
        == paths[1].result_dir.joinpath("initial-witness.json").read_bytes()
    )


def test_stage1_nonwriter_propagates_other_node_writer_failure(tmp_path):
    with pytest.raises(RuntimeError, match="machine-local leader"):
        zero_compat_diagnostic._publish_witness(
            SimpleNamespace(result_dir=tmp_path), {"schema_version": 1},
            is_writer=False, writer=lambda *_: pytest.fail("nonwriter wrote"),
            failure_counter=lambda _failed: 1, synchronize=lambda: None,
        )


@pytest.mark.parametrize(
    "steps",
    [
        ("0", "1"),
        ("2", "2"),
        ("2", "4"),
    ],
)
def test_stage1_tiny_parser_rejects_invalid_lifecycle_steps(steps):
    """Catches non-positive or non-contiguous resume controls reaching CUDA."""
    parser = build_zero_compat_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--zero-stage",
                "1",
                "--method",
                "full",
                "--deepspeed",
                "configs/deepspeed_zero1.json",
                "--experiment-id",
                "stage1-zero1-full",
                "--initial-steps",
                steps[0],
                "--resumed-steps",
                steps[1],
            ]
        )


def test_stage1_diagnostic_has_no_arbitrary_generated_path_arguments():
    """Catches callers bypassing canonical artifact ownership."""
    parser = build_zero_compat_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--zero-stage",
                "2",
                "--method",
                "dora",
                "--deepspeed",
                "configs/deepspeed_zero2.json",
                "--experiment-id",
                "stage1-zero2-dora-tiny-lifecycle",
                "--output-dir",
                "/tmp/unowned",
            ]
        )


def test_diagnostic_resume_handoff_clears_live_zero3_state_before_build(
    monkeypatch,
):
    """Catches a first diagnostic Trainer leaking ZeRO-3 into resume setup."""

    class LiveZero3Config:
        def is_zero3(self):
            return True

    live_config = LiveZero3Config()
    marker = object()
    seen_zero3_states = []

    def build_resumed_components(*args, **kwargs):
        del args, kwargs
        seen_zero3_states.append(is_deepspeed_zero3_enabled())
        return marker

    set_hf_deepspeed_config(live_config)
    try:
        assert is_deepspeed_zero3_enabled()
        monkeypatch.setattr(
            zero_compat_diagnostic,
            "build_stage1_training_components",
            build_resumed_components,
        )

        result = zero_compat_diagnostic._build_resumed_stage1_components(
            args=object(), profile=object()
        )

        assert result is marker
        assert seen_zero3_states == [False]
        assert not is_deepspeed_zero3_enabled()
    finally:
        unset_hf_deepspeed_config()


@pytest.fixture
def real_comt_paths():
    return SimpleNamespace(
        train=REPO_ROOT.parent.parent / "data" / "TRAIN.jsonl",
        root=REPO_ROOT.parent.parent / "data",
    )


@pytest.fixture
def evidence_dir():
    return REPO_ROOT / "outputs" / "stage1-smoke"


@pytest.mark.gpu
@pytest.mark.distributed
@pytest.mark.parametrize("method", SUPPORTED_DIAGNOSTIC_METHODS)
def test_two_rank_zero3_b2_save_resume_witness(
    method, real_comt_paths, evidence_dir
):
    if os.environ.get("ILVR_REAL_GPU_TESTS") != "1":
        pytest.skip("set ILVR_REAL_GPU_TESTS=1 to run the real two-rank witness")
    selected_method = SELECTED_METHOD
    if selected_method not in {None, method}:
        pytest.skip(f"this invocation selected {selected_method}, not {method}")

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
