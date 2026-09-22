"""Aggregate token-level measurements for vision-block analysis conditions."""

import math


CONDITIONS = ("clean", "body_mask", "boundary_mask", "full_mask")


def summarize(
    rows: list[dict],
    expected: dict[tuple[int, int], tuple[tuple[int, int], ...]],
) -> dict:
    """Summarize condition changes and token-weighted NLL values."""
    if not expected:
        raise ValueError("expected must contain at least one site")

    expected_rows = {}
    for site, targets in expected.items():
        if not targets:
            raise ValueError("each expected site must contain at least one token")
        for position, target_id in targets:
            for condition in CONDITIONS:
                key = (*site, position, condition)
                if key in expected_rows:
                    raise ValueError("expected token positions must be unique per site")
                expected_rows[key] = target_id

    indexed_rows = {}
    required_fields = {
        "source_index",
        "plan_index",
        "target_position",
        "target_id",
        "condition",
        "argmax_id",
        "nll",
    }
    for row in rows:
        if not isinstance(row, dict) or not required_fields.issubset(row):
            raise ValueError("each row must contain all required fields")
        condition = row["condition"]
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition!r}")
        key = (
            row["source_index"],
            row["plan_index"],
            row["target_position"],
            condition,
        )
        if key in indexed_rows:
            raise ValueError(f"duplicate row: {key!r}")
        if key not in expected_rows:
            raise ValueError(f"unexpected row: {key!r}")
        if row["target_id"] != expected_rows[key]:
            raise ValueError(f"target ID does not match expected row: {key!r}")
        nll = row["nll"]
        if not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
            raise ValueError("NLL must be finite and nonnegative")
        indexed_rows[key] = row

    if indexed_rows.keys() != expected_rows.keys():
        raise ValueError("rows must exactly cover every expected condition token")

    token_count = sum(len(targets) for targets in expected.values())
    conditions = {}
    for condition in CONDITIONS:
        condition_rows = [
            indexed_rows[(*site, position, condition)]
            for site, targets in expected.items()
            for position, _ in targets
        ]
        site_changes = {
            site: [
                indexed_rows[(*site, position, condition)]["argmax_id"] != target_id
                for position, target_id in targets
            ]
            for site, targets in expected.items()
        }
        mean_nll = math.fsum(row["nll"] for row in condition_rows) / token_count
        conditions[condition] = {
            "first_changed": sum(site_changes[site][0] for site in expected),
            "any_changed": sum(any(site_changes[site]) for site in expected),
            "tokens_changed": sum(sum(changes) for changes in site_changes.values()),
            "mean_nll": mean_nll,
            "delta_nll": mean_nll,
        }

    clean_mean = conditions["clean"]["mean_nll"]
    for metrics in conditions.values():
        metrics["delta_nll"] -= clean_mean

    return {
        "status": "completed",
        "sites": len(expected),
        "tokens": token_count,
        "conditions": conditions,
    }
