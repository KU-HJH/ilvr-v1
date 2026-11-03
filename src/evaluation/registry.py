from collections.abc import Callable

from evaluation.benchmarks.vsp import compare_answer, extract_answer, validate_path
from evaluation.schema import ContractError, Prediction, Sample, Score


def score_vsp(sample: Sample, prediction: Prediction) -> Score:
    if prediction.status != 'ok':
        raise ContractError('score_vsp requires ok prediction')
    parsed = validate_path(extract_answer(prediction.response, sample.model_input))
    simulation = compare_answer(parsed, sample.target) if parsed.status == 'parsed' else None
    return Score(
        sample.id,
        sample.subtask,
        'vsp',
        'vsp_validated_v1',
        '1',
        parsed.status,
        parsed.value,
        parsed.method,
        simulation.success if simulation else False,
        simulation,
        parsed.error,
    )


def get_scorer(benchmark: str, profile: str) -> Callable[[Sample, Prediction], Score]:
    if (benchmark, profile) == ('vsp', 'vsp_validated_v1'):
        return score_vsp
    if (benchmark, profile) == ('comt', 'comt_reference_rules_v1'):
        from evaluation.benchmarks.comt import score_comt

        return score_comt
    if (benchmark, profile) == ('visulogic', 'visulogic_vlmeval_rules_v1'):
        from evaluation.benchmarks.visulogic import score_visulogic

        return score_visulogic
    if (benchmark, profile) == ('emma', 'emma_safe_rules_v1'):
        from evaluation.benchmarks.emma import score_emma

        return score_emma
    if (benchmark, profile) == ('zebra_2d', 'zebra_2d_rules_v1'):
        from evaluation.benchmarks.zebra_2d import score_zebra_2d

        return score_zebra_2d
    raise ContractError('unsupported benchmark/profile')
