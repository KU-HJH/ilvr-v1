from collections import Counter

from evaluation.registry import get_scorer
from evaluation.schema import ContractError, EmmaTarget, EvalConfig, PreparedEvaluation, Sample, Score


def score_rows(prepared: PreparedEvaluation, config: EvalConfig) -> list[Score]:
    scorer = get_scorer(config.benchmark, config.profile)
    result = []
    for sample in prepared.samples:
        prediction = prepared.predictions.get(sample.id)
        if prediction is None or prediction.status == 'infrastructure_error':
            status = 'missing' if prediction is None else 'infrastructure_error'
            error = 'prediction not received' if prediction is None else prediction.error
            result.append(Score(
                sample.id, sample.subtask, config.benchmark, config.profile,
                '1', status, None, None, None, None, error,
            ))
        else:
            result.append(scorer(sample, prediction))
    return result


def category_stats(rows: list[Score]) -> dict:
    observed = sum(row.correctness is not None for row in rows)
    correct = sum(row.correctness is True for row in rows)
    complete = observed == len(rows)
    return dict(
        count=len(rows),
        observed=observed,
        correct=correct,
        complete=complete,
        accuracy=correct / len(rows) if complete and rows else None,
    )


def emma_summary(samples: tuple[Sample, ...], scores: list[Score]) -> dict:
    sample_ids = [sample.id for sample in samples]
    score_ids = [score.id for score in scores]
    if (
        len(set(sample_ids)) != len(sample_ids)
        or len(set(score_ids)) != len(score_ids)
        or set(sample_ids) != set(score_ids)
    ):
        raise ContractError('unique matching sample and score populations required')

    scores_by_id = {score.id: score for score in scores}
    question_type_rows = {'multiple_choice': [], 'open_ended': []}
    category_rows = {}
    task_rows = {}
    for sample in samples:
        if not isinstance(sample.target, EmmaTarget):
            raise ContractError('EMMA summary requires EMMA targets')
        score = scores_by_id[sample.id]
        question_type_rows.setdefault(sample.target.question_type, []).append(score)

        raw_category = sample.target.category
        if sample.subtask == 'Coding':
            categories = sorted({
                part.strip() for part in raw_category.split(';') if part.strip()
            })
        else:
            categories = [raw_category] if raw_category.strip() else []
        for category in categories:
            category_rows.setdefault(sample.subtask, {}).setdefault(category, []).append(score)

        if sample.target.task.strip():
            task_rows.setdefault(f'{sample.subtask}_{sample.target.task}', []).append(score)

    diagnostics = ('unsupported_expression', 'non_finite_value', 'math_limit_exceeded')
    return dict(
        question_types={
            question_type: category_stats(rows)
            for question_type, rows in question_type_rows.items()
        },
        categories={
            subject: {
                category: category_stats(rows)
                for category, rows in subject_rows.items()
            }
            for subject, subject_rows in category_rows.items()
        },
        tasks={task: category_stats(rows) for task, rows in task_rows.items()},
        comparison_diagnostics={
            diagnostic: sum(
                score.status in ('parsed', 'invalid_format') and score.error == diagnostic
                for score in scores
            )
            for diagnostic in diagnostics
        },
    )


def summarize(scores: list[Score], config: EvalConfig) -> dict:
    if not scores or len({row.id for row in scores}) != len(scores):
        raise ContractError('nonempty unique score population required')
    if any(row.category not in config.categories for row in scores):
        raise ContractError('unexpected score category')
    total = category_stats(scores)
    return dict(
        N_expected=total['count'],
        observed=total['observed'],
        correct=total['correct'],
        complete=total['complete'],
        accuracy=total['accuracy'],
        coverage=total['observed'] / total['count'],
        status_counts=dict(Counter(row.status for row in scores)),
        categories={
            category: category_stats([row for row in scores if row.category == category])
            for category in config.categories
        },
    )


def evaluate(prepared: PreparedEvaluation, config: EvalConfig) -> tuple[list[Score], dict]:
    scores = score_rows(prepared, config)
    summary = summarize(scores, config)
    summary.update(
        schema_version=1,
        benchmark=config.benchmark,
        profile=config.profile,
        scorer_version='1',
        manifest_sha256=prepared.manifest_sha256,
        run_fingerprint=prepared.run_fingerprint,
        protocol_kind='modified',
        verification=dict(
            cpu_tests='not_recorded_by_scoring_run',
            upstream_environment='not_run',
            gpu_smoke='not_run',
            full_benchmark='not_run',
            paper_protocol_verified=False,
        ),
    )
    if config.benchmark == 'emma':
        summary['emma'] = emma_summary(prepared.samples, scores)
    if config.benchmark == 'zebra_2d':
        summary['zebra_2d'] = dict(
            comparison_kind='conservative_rules',
            ood_verified='unknown',
            paper_sample_identity='unverified',
            paper_judge_equivalent=False,
        )
    return scores, summary
