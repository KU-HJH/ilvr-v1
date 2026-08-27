from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from stage1.benchmark import run_or_load_distributed_witness
from zero_compat_diagnostic import build_zero_compat_parser


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTED_METHOD = os.environ.get("ILVR_DISTRIBUTED_METHOD") or None
if SELECTED_METHOD not in {None, "full", "dora"}:
    raise pytest.UsageError(
        "invalid ILVR_DISTRIBUTED_METHOD: expected full or dora, "
        f"got {SELECTED_METHOD!r}"
    )


def test_stage1_diagnostic_accepts_all_supported_pairs():
    """Catches a supported ZeRO/method lifecycle becoming unlaunchable."""
    parser = build_zero_compat_parser()
    for stage in (1, 2, 3):
        for method in ("full", "dora"):
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
@pytest.mark.parametrize("method", ["full", "dora"])
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
