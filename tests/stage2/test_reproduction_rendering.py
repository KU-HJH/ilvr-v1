"""Selectable assistant rendering for the external V-off reproduction."""

import pytest

from step_preference import spans
from step_preference.spans import (
    ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    ASSISTANT_PLAN_RENDERING_CONCATENATED,
    build_assistant_plan,
)
from tests.stage2.test_launcher import dry_run_tokens, option_value, run_launcher
from src.multinode_runtime import validated_stage2_environment
from utils_deepseed import parse_args


PLAN = [
    {"type": "text", "content": "Before."},
    {"type": "latent", "helper_image": "helper.png"},
    {"type": "text", "content": "reason."},
    {"type": "text", "content": "The final answer is: 7"},
]


@pytest.mark.parametrize(
    ("requested", "preference_enabled", "expected"),
    [
        ("auto", False, ASSISTANT_PLAN_RENDERING_CONCATENATED),
        ("auto", True, ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF),
        (ASSISTANT_PLAN_RENDERING_CONCATENATED, True,
         ASSISTANT_PLAN_RENDERING_CONCATENATED),
        (ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF, False,
         ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF),
    ],
)
def test_rendering_resolver_preserves_auto_policy_and_explicit_modes(
    requested, preference_enabled, expected
):
    """Catches explicit rendering being ignored or auto changing legacy behavior."""
    assert spans.resolve_assistant_plan_rendering(requested, preference_enabled) == expected


def test_rendering_modes_preserve_exact_adjacent_text_bytes_and_span_extent():
    """Catches separators being normalized or excluded from until-next-latent targets."""
    concatenated, _, concat_steps = build_assistant_plan(
        PLAN, 2, assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_CONCATENATED
    )
    adjacent, _, adjacent_steps = build_assistant_plan(
        PLAN, 2,
        assistant_plan_rendering=ASSISTANT_PLAN_RENDERING_ADJACENT_TEXT_SPACE_LF,
    )

    assert concatenated.endswith("reason.The final answer is: 7")
    assert adjacent.endswith("reason. \nThe final answer is: 7")
    assert "The final answer is: 7" in concatenated[concat_steps[0].target_text.start :]
    assert "The final answer is: 7" in adjacent[adjacent_steps[0].target_text.start :]


def test_cli_and_launcher_forward_one_resolved_rendering_and_reject_malformed_mode():
    """Catches manifest and training resolving different rendering identities."""
    args = parse_args([
        "--stage", "stage2", "--step_preference_enabled",
        "--stage2_assistant_plan_rendering", ASSISTANT_PLAN_RENDERING_CONCATENATED,
    ])
    assert args.stage2_assistant_plan_rendering == ASSISTANT_PLAN_RENDERING_CONCATENATED

    tokens = dry_run_tokens(run_launcher(env_overrides={
        "STEP_PREFERENCE_ENABLED": "true",
        "STAGE2_ASSISTANT_PLAN_RENDERING": ASSISTANT_PLAN_RENDERING_CONCATENATED,
    }))
    assert "ASSISTANT_PLAN_RENDERING=concatenated-v1" in tokens
    assert option_value(tokens, "--stage2_assistant_plan_rendering") == (
        ASSISTANT_PLAN_RENDERING_CONCATENATED
    )

    malformed = run_launcher(env_overrides={
        "STAGE2_ASSISTANT_PLAN_RENDERING": "not-a-rendering",
    })
    assert malformed.returncode == 2
    assert "STAGE2_ASSISTANT_PLAN_RENDERING" in malformed.stderr

    with pytest.raises(SystemExit):
        parse_args(["--stage2_assistant_plan_rendering", "not-a-rendering"])


def test_multinode_environment_forwards_rendering_request():
    """Catches the coordinator dropping the rendering before either node launches."""
    environment = validated_stage2_environment(
        {
            "STAGE1_CHECKPOINT": "/models/stage1",
            "DATA_PATH": "/data/train.jsonl",
            "IMAGE_ROOT": "/data/images",
            "STAGE2_ASSISTANT_PLAN_RENDERING": "concatenated-v1",
        },
        output_dir="/checkpoints/run",
        log_file="/logs/run.log",
        deepspeed_config="/configs/zero2.json",
    )

    assert environment["STAGE2_ASSISTANT_PLAN_RENDERING"] == "concatenated-v1"
