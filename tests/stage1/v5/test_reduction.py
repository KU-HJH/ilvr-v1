"""Mutation witnesses: configured GAS tail and empty original blocks."""
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from transformers import Trainer, TrainingArguments

# Shared with Task7: four ranks; empty rank and microbatch, unequal block counts.
FOUR_RANK_WINDOWS = (
    ((2, 1, 0, 0), (1, 0, 1, 0), (0, 0, 0, 0), (1, 2, 0, 1)),
    ((1, 0), (1, 1), (0, 0), (0, 2)),
)


def api():
    assert importlib.util.find_spec('stage1.v5_reduction'), 'exact window reduction missing'
    from stage1.v5_reduction import V5Window, v5_auxiliary_scale
    return V5Window, v5_auxiliary_scale


def test_scale_keeps_configured_gas_on_partial_tail_and_empty_blocks():
    Window, scale = api()
    assert scale(Window(10, 4, 4, 4)) == 1.6
    assert scale(Window(5, 2, 4, 4)) == 3.2
    assert scale(Window(0, 2, 4, 4)) == 0.
    for bad in ((-1, 1, 4, 4), (1, 0, 4, 4), (1, 5, 4, 4), (1, 1, 0, 4)):
        with pytest.raises(ValueError): Window(*bad)


@pytest.mark.parametrize('counts', FOUR_RANK_WINDOWS)
def test_real_parent_dispatch_matches_independent_global_gradient(counts, tmp_path):
    """Actual Trainer dispatch/scaling; fake DS boundary records and executes backward.

    CPU cannot create a GPU ZeRO engine. Task7 consumes these literal windows
    for that missing distributed engine witness, rather than treating this as DS.
    """
    Window, scale = api()
    grads, kwargs_seen = [], []
    total = sum(map(sum, counts))
    for rank, microbatches in enumerate(counts):
        parameter = torch.nn.Parameter(torch.tensor(2.))
        model = torch.nn.Linear(1, 1)
        model.weight = parameter
        trainer = Trainer.__new__(Trainer)
        trainer.args = TrainingArguments(str(tmp_path), use_cpu=True, gradient_accumulation_steps=4, report_to=[])
        trainer.state = SimpleNamespace(global_step=0)
        trainer.optimizer = torch.optim.AdamW([parameter], lr=.01)
        trainer.model_accepts_loss_kwargs = False
        trainer.compute_loss_func = None
        trainer.use_apex = False
        trainer.use_cpu_amp = False
        trainer._prepare_inputs = lambda x: x
        trainer.accelerator = SimpleNamespace(distributed_type=DistributedType.DEEPSPEED,
            backward=lambda loss, **kw: (kwargs_seen.append(kw), loss.backward()))
        window = Window(total, len(microbatches), 4, 4)
        trainer.compute_loss = lambda _m, x, **kw: .1 * parameter.square() + scale(window) * x['blocks'] * parameter.pow(3)
        batches, _ = Trainer.get_batch_samples(trainer, iter({'blocks': n} for n in microbatches), 4)
        assert len(batches) == len(microbatches)
        for batch in batches: Trainer.training_step(trainer, model, batch)
        grads.append(parameter.grad.clone())
    # Independent global objective: native configured sixteen denominator,
    # auxiliary actual original-step denominator; empty blocks remain counted.
    p = torch.tensor(2., requires_grad=True)
    objective = .1 * len(counts) * len(counts[0]) / 16 * p.square() + p.pow(3)
    objective.backward()
    torch.testing.assert_close(torch.stack(grads).mean(), p.grad)
    assert kwargs_seen == [{'scale_wrt_gas': False}] * (4 * len(counts[0]))
