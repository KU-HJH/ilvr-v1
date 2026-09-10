"""CPU-only summaries for paired visual interventions and matched images."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
import random
from typing import Any

from evaluation.scoring import parse_gold_answer


Row = Mapping[str, Any]


def _indexed(rows: Iterable[Row]) -> dict[int, Row]:
    indexed: dict[int, Row] = {}
    for row in rows:
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("each row index must be an integer")
        if index in indexed:
            raise ValueError(f"duplicate index: {index}")
        indexed[index] = row
    return indexed


def _valid_gold(row: Row) -> bool:
    return (
        row.get("invalid_gold_reason") is None
        and row.get("gold_parse_status") == "parsed"
        and isinstance(row.get("gold_normalized"), str)
        and bool(row["gold_normalized"].strip())
    )


def _successful(row: Row) -> bool:
    return row.get("error") in (None, "") and row.get("prediction_parse_status") != "inference_error"


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_core(rows: list[Row], *, categories: bool) -> dict[str, Any]:
    num_samples = len(rows)
    valid_gold_count = sum(_valid_gold(row) for row in rows)
    invalid_gold_count = num_samples - valid_gold_count
    error_count = sum(not _successful(row) for row in rows)
    correct = sum(
        bool(row.get("match"))
        for row in rows
        if _valid_gold(row) and _successful(row)
    )
    times = [float(row["inference_time_sec"]) for row in rows if _finite_number(row.get("inference_time_sec"))]
    blocks = Counter(
        str(row["latent_block_count"])
        if isinstance(row.get("latent_block_count"), int) and not isinstance(row.get("latent_block_count"), bool)
        else "unavailable"
        for row in rows
    )
    result: dict[str, Any] = {
        "num_samples": num_samples,
        "correct": correct,
        "coverage_accuracy": correct / num_samples if num_samples else 0.0,
        "valid_gold_accuracy": correct / valid_gold_count if valid_gold_count else 0.0,
        "valid_gold_count": valid_gold_count,
        "invalid_gold_count": invalid_gold_count,
        "error_count": error_count,
        "mean_inference_time_sec": _mean(times),
        "latent_block_distribution": dict(sorted(blocks.items())),
    }
    if categories:
        buckets: dict[str, list[Row]] = defaultdict(list)
        for row in rows:
            category = row.get("category")
            buckets[category if isinstance(category, str) and category else "unknown"].append(row)
        result["category_summaries"] = {
            name: _aggregate_core(bucket, categories=False)
            for name, bucket in sorted(buckets.items())
        }
    return result


def aggregate_rows(rows: Iterable[Row]) -> dict[str, Any]:
    """Aggregate one condition while keeping source and valid-gold denominators distinct."""
    materialized = list(rows)
    _indexed(materialized)
    return _aggregate_core(materialized, categories=True)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("a percentile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_ci(differences: Sequence[int], seed: int, samples: int) -> list[float | None]:
    if not differences:
        return [None, None]
    rng = random.Random(seed)
    count = len(differences)
    draws = sorted(
        100.0 * sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    )
    return [_percentile(draws, 0.025), _percentile(draws, 0.975)]


def _mcnemar(correct_to_incorrect: int, incorrect_to_correct: int) -> float:
    discordant = correct_to_incorrect + incorrect_to_correct
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(min(correct_to_incorrect, incorrect_to_correct) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_counts(pairs: Sequence[tuple[Row, Row]]) -> dict[str, Any]:
    transitions = {
        "correct_to_correct": 0,
        "correct_to_incorrect": 0,
        "incorrect_to_correct": 0,
        "incorrect_to_incorrect": 0,
    }
    prediction_changes = 0
    token_changes = 0
    for baseline, condition in pairs:
        before = bool(baseline.get("match"))
        after = bool(condition.get("match"))
        key = (
            "correct_to_" if before else "incorrect_to_"
        ) + ("correct" if after else "incorrect")
        transitions[key] += 1
        prediction_changes += baseline.get("prediction_normalized") != condition.get("prediction_normalized")
        token_changes += baseline.get("generated_suffix_token_ids") != condition.get("generated_suffix_token_ids")
    count = len(pairs)
    baseline_correct = transitions["correct_to_correct"] + transitions["correct_to_incorrect"]
    condition_correct = transitions["correct_to_correct"] + transitions["incorrect_to_correct"]
    return {
        "paired_count": count,
        "baseline_accuracy": baseline_correct / count if count else None,
        "condition_accuracy": condition_correct / count if count else None,
        "accuracy_delta_pp": 100.0 * (condition_correct - baseline_correct) / count if count else None,
        "transitions": transitions,
        "prediction_change_count": prediction_changes,
        "prediction_change_rate": prediction_changes / count if count else None,
        "raw_token_change_count": token_changes,
        "raw_token_change_rate": token_changes / count if count else None,
    }


def paired_summary(
    baseline_rows: Iterable[Row],
    condition_rows: Iterable[Row],
    *,
    seed: int = 42,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Compare conditions on identical successful rows with identical valid golds."""
    if not isinstance(bootstrap_samples, int) or isinstance(bootstrap_samples, bool) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    baseline = _indexed(baseline_rows)
    condition = _indexed(condition_rows)
    if set(baseline) != set(condition):
        raise ValueError("paired conditions must have the same index set")

    pairs: list[tuple[Row, Row]] = []
    excluded = {"invalid_gold": 0, "inference_error": 0}
    for index in sorted(baseline):
        before, after = baseline[index], condition[index]
        before_valid, after_valid = _valid_gold(before), _valid_gold(after)
        if before_valid and after_valid and before["gold_normalized"] != after["gold_normalized"]:
            raise ValueError(f"gold mismatch at index {index}")
        if not before_valid or not after_valid:
            excluded["invalid_gold"] += 1
            continue
        if not _successful(before) or not _successful(after):
            excluded["inference_error"] += 1
            continue
        pairs.append((before, after))

    result = _paired_counts(pairs)
    differences = [int(bool(after.get("match"))) - int(bool(before.get("match"))) for before, after in pairs]
    losses = result["transitions"]["correct_to_incorrect"]
    gains = result["transitions"]["incorrect_to_correct"]
    result.update(
        {
            "bootstrap_ci95_pp": _bootstrap_ci(differences, seed, bootstrap_samples),
            "bootstrap_seed": seed,
            "bootstrap_samples": bootstrap_samples,
            "mcnemar_pvalue": _mcnemar(losses, gains) if pairs else None,
            "excluded_counts": excluded,
            "exploratory_multiple_comparison_caveat": True,
        }
    )

    category_pairs: dict[str, list[tuple[Row, Row]]] = defaultdict(list)
    for before, after in pairs:
        before_category = before.get("category")
        after_category = after.get("category")
        if before_category != after_category:
            raise ValueError(f"category mismatch at index {before['index']}")
        name = before_category if isinstance(before_category, str) and before_category else "unknown"
        category_pairs[name].append((before, after))
    result["category_summaries"] = {
        name: _paired_counts(bucket) for name, bucket in sorted(category_pairs.items())
    }

    paired_times = [
        float(after["inference_time_sec"]) - float(before["inference_time_sec"])
        for before, after in pairs
        if _finite_number(before.get("inference_time_sec")) and _finite_number(after.get("inference_time_sec"))
    ]
    result["mean_inference_time_change_sec"] = _mean(paired_times)
    return result


def _normalized_question(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def matched_image_groups(samples: Iterable[Row]) -> list[list[int]]:
    """Return original indices sharing full normalized text and differing valid labels."""
    buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seen: set[int] = set()
    for position, sample in enumerate(samples):
        index = sample.get("index", position)
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("each sample index must be an integer")
        if index in seen:
            raise ValueError(f"duplicate index: {index}")
        seen.add(index)
        question = _normalized_question(sample.get("text_input"))
        gold = sample.get("gold_normalized")
        if gold is None:
            parsed_gold = parse_gold_answer(sample.get("original_final_answer"))
            gold = parsed_gold.normalized if parsed_gold.status == "parsed" else None
        if question and isinstance(gold, str) and gold.strip():
            buckets[question].append((index, gold.strip()))
    groups = [
        [index for index, _ in members]
        for members in buckets.values()
        if len(members) > 1 and len({gold for _, gold in members}) > 1
    ]
    return sorted((sorted(group) for group in groups), key=lambda group: group[0])


def matched_group_summary(rows: Iterable[Row], groups: Iterable[Sequence[int]]) -> dict[str, Any]:
    """Summarize matched-image controls using groups as independent units."""
    indexed = _indexed(rows)
    details: list[dict[str, Any]] = []
    all_members_correct_groups = 0
    total_members = 0
    directed_pair_count = 0
    donor_correct_switches = 0
    prediction_changes = 0

    for raw_group in groups:
        indices = list(raw_group)
        if len(indices) < 2 or len(indices) != len(set(indices)):
            raise ValueError("each matched group requires at least two unique indices")
        missing = [index for index in indices if index not in indexed]
        if missing:
            raise ValueError(f"matched group contains missing indices: {missing}")
        members = [indexed[index] for index in indices]
        if not all(_valid_gold(member) for member in members):
            raise ValueError("matched groups require valid parsed gold labels")
        if len({member["gold_normalized"] for member in members}) < 2:
            raise ValueError("matched groups require differing gold labels")

        correct = sum(bool(member.get("match")) and _successful(member) for member in members)
        member_accuracy = correct / len(members)
        all_correct = correct == len(members)
        group_directed = 0
        group_donor_switches = 0
        group_prediction_changes = 0
        for recipient in members:
            for donor in members:
                if recipient is donor or recipient["gold_normalized"] == donor["gold_normalized"]:
                    continue
                group_directed += 1
                changed = recipient.get("prediction_normalized") != donor.get("prediction_normalized")
                group_prediction_changes += changed
                group_donor_switches += bool(
                    changed
                    and recipient.get("match")
                    and donor.get("match")
                    and _successful(recipient)
                    and _successful(donor)
                )

        details.append(
            {
                "indices": indices,
                "member_count": len(members),
                "member_accuracy": member_accuracy,
                "all_members_correct": all_correct,
                "directed_pair_count": group_directed,
                "donor_correct_switches": group_donor_switches,
                "prediction_change_rate": group_prediction_changes / group_directed,
            }
        )
        total_members += len(members)
        all_members_correct_groups += all_correct
        directed_pair_count += group_directed
        donor_correct_switches += group_donor_switches
        prediction_changes += group_prediction_changes

    group_accuracies = [detail["member_accuracy"] for detail in details]
    if group_accuracies:
        rng = random.Random(42)
        count = len(group_accuracies)
        bootstrap = sorted(
            sum(group_accuracies[rng.randrange(count)] for _ in range(count)) / count
            for _ in range(10000)
        )
        ci: list[float | None] = [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
    else:
        ci = [None, None]
    return {
        "group_count": len(details),
        "group_details": details,
        "member_count": total_members,
        "all_members_correct_groups": all_members_correct_groups,
        "mean_group_member_accuracy": _mean(group_accuracies),
        "group_member_accuracy_ci95": ci,
        "directed_pair_count": directed_pair_count,
        "donor_correct_switches": donor_correct_switches,
        "prediction_change_rate": prediction_changes / directed_pair_count if directed_pair_count else None,
        "uncertainty_unit": "group",
    }
