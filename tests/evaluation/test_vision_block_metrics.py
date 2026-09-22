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
