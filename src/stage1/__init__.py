"""Stage 1 training contracts and helpers.

Exports are loaded on demand so lightweight ``python -m src.stage1.*`` tools
do not import the model/PEFT stack merely to inspect small evidence files.
"""

from importlib import import_module


_EXPORT_MODULES = {
    "Stage1ParameterReport": ".peft",
    "build_stage1_tuned_model": ".peft",
    "validate_stage1_trainable_boundary": ".peft",
    "Stage1RuntimeConfig": ".types",
    "Stage1TuningConfig": ".types",
    "Stage1TuningMethod": ".types",
    "resolve_stage1_tuning_config": ".types",
}

__all__ = [
    "Stage1ParameterReport",
    "Stage1RuntimeConfig",
    "Stage1TuningConfig",
    "Stage1TuningMethod",
    "build_stage1_tuned_model",
    "resolve_stage1_tuning_config",
    "validate_stage1_trainable_boundary",
]


def __getattr__(name: str):
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
