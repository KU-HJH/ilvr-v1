"""Opt-in zero-dropout training policy matching ours1 (temp2) Stage 2 objective runs.

ours1 applies, after faithful artifact restoration, effective dropout zero to the
student: every PEFT adapter dropout becomes ``nn.Identity`` (PEFT DoRA picks a
different projection path for Identity than for ``Dropout(p=0)``), and any other
``nn.Dropout`` / ``attention_dropout`` is set to 0. The adapter configuration on
disk keeps its original dropout value; only the runtime modules change.
"""

from __future__ import annotations

import math
from numbers import Real

from torch import nn


def _adapter_dropouts(model):
    from peft.tuners.lora.layer import LoraLayer

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for adapter, dropout in module.lora_dropout.items():
                if not isinstance(dropout, (nn.Dropout, nn.Identity)):
                    raise ValueError(f"unsupported adapter dropout {name}.{adapter}")
                yield f"{name}.lora_dropout.{adapter}".lstrip("."), module.lora_dropout, adapter, dropout


def apply_zero_dropout_policy(model: nn.Module) -> dict[str, float]:
    """Zero every dropout site in place; return the original probabilities by site."""
    adapters = list(_adapter_dropouts(model))
    adapter_ids = {id(dropout) for *_, dropout in adapters}
    originals: dict[str, float] = {}
    changes = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and id(module) not in adapter_ids:
            changes.append((f"{name}.p", module, "p"))
        if isinstance(getattr(module, "attention_dropout", None), Real) and not isinstance(
            module.attention_dropout, bool
        ):
            changes.append((f"{name}.attention_dropout", module, "attention_dropout"))
        if isinstance(module, (nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
                               nn.FeatureAlphaDropout)) and module.p != 0:
            raise ValueError(f"unsupported stochastic dropout module {name}")
    for name, module, attribute in changes:
        value = getattr(module, attribute)
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"invalid dropout probability {name}: {value}")
    for name, _, _, dropout in adapters:
        originals[f"{name}.p"] = float(getattr(dropout, "p", 0.0))
    for name, module, attribute in changes:
        originals[name] = float(getattr(module, attribute))
        setattr(module, attribute, 0.0)
    for _, container, adapter, dropout in adapters:
        container[adapter] = nn.Identity().train(dropout.training)
    return dict(sorted(originals.items()))


def has_active_dropout(model: nn.Module) -> bool:
    """True when any training-mode dropout site could draw a random mask."""
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.training and module.p > 0:
            return True
    return False
