import math

import pytest

from evaluation.vision_block_metrics import CONDITIONS, summarize


def test_common_denominators_and_token_weighted_nll():
    expected = {(0, 1): ((5, 7),), (1, 1): ((5, 7), (6, 8), (7, 9))}
    rows = []
    for (sample, plan), targets in expected.items():
        for pos, target in targets:
            for condition in CONDITIONS:
                changed = condition == "body_mask" and (sample, pos) == (1, 6)
                rows.append(
                    dict(
                        source_index=sample,
                        plan_index=plan,
                        target_position=pos,
                        target_id=target,
                        condition=condition,
                        argmax_id=99 if changed else target,
                        nll=1.0 if condition == "clean" else 0.5,
                    )
                )
    out = summarize(rows, expected)
    assert (out["sites"], out["tokens"]) == (2, 4)
    got = out["conditions"]["body_mask"]
    assert (got["first_changed"], got["any_changed"], got["tokens_changed"]) == (
        0,
        1,
        1,
    )
    assert got["mean_nll"] == 0.5 and got["delta_nll"] == -0.5


def test_image_profile_metrics_use_full_clean():
    expected = {(0, 1): ((5, 7),), (1, 1): ((9, 8), (10, 9))}
    rows = []
    for condition in ("clean", "image_mask", "image_body_mask"):
        for (source, plan), targets in expected.items():
            for position, target in targets:
                changed = condition != "clean" and source == 1 and position == 10
                rows.append(
                    dict(
                        source_index=source,
                        plan_index=plan,
                        target_position=position,
                        target_id=target,
                        condition=condition,
                        argmax_id=99 if changed else target,
                        nll={
                            "clean": 2.0,
                            "image_mask": 1.0,
                            "image_body_mask": 3.0,
                        }[condition],
                    )
                )

    out = summarize(rows, expected, analysis_profile="image-local-v1")

    assert (out["sites"], out["tokens"]) == (2, 3)
    assert out["conditions"]["image_mask"] == dict(
        first_changed=0,
        any_changed=1,
        tokens_changed=1,
        mean_nll=1.0,
        delta_nll=-1.0,
    )
    assert out["conditions"]["image_body_mask"]["delta_nll"] == 1.0
    with pytest.raises(ValueError):
        summarize(rows[:-1], expected, analysis_profile="image-local-v1")
    with pytest.raises(ValueError):
        summarize(rows, expected)
    with pytest.raises(ValueError):
        summarize(rows, expected, analysis_profile="typo")


def _fixture():
    expected = {(0, 1): ((5, 7),), (1, 1): ((5, 7), (6, 8), (7, 9))}
    rows = [
        {
            "source_index": sample,
            "plan_index": plan,
            "target_position": position,
            "target_id": target_id,
            "condition": condition,
            "argmax_id": target_id,
            "nll": 1.0,
        }
        for (sample, plan), targets in expected.items()
        for position, target_id in targets
        for condition in CONDITIONS
    ]
    return rows, expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(rows[0].copy()),
        lambda rows: rows.__setitem__(slice(0, 4), []),
        lambda rows: rows[0].__setitem__("target_id", 1234),
        lambda rows: rows[0].__setitem__("nll", math.nan),
        lambda rows: rows[0].__setitem__("nll", math.inf),
        lambda rows: rows[0].__setitem__("nll", -0.1),
        lambda rows: rows[0].__setitem__("condition", "unknown"),
        lambda rows: rows[0].__setitem__("target_position", 1234),
    ],
    ids=[
        "missing-row",
        "duplicate-row",
        "missing-site",
        "wrong-target",
        "nan-nll",
        "infinite-nll",
        "negative-nll",
        "unknown-condition",
        "unexpected-position",
    ],
)
def test_malformed_rows_raise_value_error(mutate):
    rows, expected = _fixture()
    mutate(rows)

    with pytest.raises(ValueError):
        summarize(rows, expected)


def test_empty_expected_set_raises_value_error():
    with pytest.raises(ValueError):
        summarize([], {})


def test_nll_mean_is_weighted_by_tokens_not_sites():
    rows, expected = _fixture()
    for row in rows:
        if row["source_index"] == 0:
            row["nll"] = 10.0
        elif row["condition"] != "clean":
            row["nll"] = 2.0

    out = summarize(rows, expected)

    assert out["conditions"]["clean"]["mean_nll"] == 3.25
    assert out["conditions"]["body_mask"]["mean_nll"] == 4.0
    assert out["conditions"]["body_mask"]["delta_nll"] == 0.75


def test_argmax_changes_are_relative_to_corresponding_clean_predictions():
    rows, expected = _fixture()
    for row in rows:
        if row["source_index"] == 1:
            row["argmax_id"] = 99
        if (
            row["source_index"] == 1
            and row["condition"] == "body_mask"
            and row["target_position"] == 6
        ):
            row["argmax_id"] = row["target_id"]

    out = summarize(rows, expected)

    clean = out["conditions"]["clean"]
    body_mask = out["conditions"]["body_mask"]
    assert (clean["first_changed"], clean["any_changed"], clean["tokens_changed"]) == (
        0,
        0,
        0,
    )
    assert (
        body_mask["first_changed"],
        body_mask["any_changed"],
        body_mask["tokens_changed"],
    ) == (0, 1, 1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_index", False),
        ("plan_index", True),
        ("target_position", []),
        ("target_id", True),
        ("argmax_id", False),
        ("condition", 1),
        ("nll", True),
    ],
)
def test_typed_malformed_rows_raise_value_error(field, value):
    rows, expected = _fixture()
    rows[0][field] = value

    with pytest.raises(ValueError):
        summarize(rows, expected)
