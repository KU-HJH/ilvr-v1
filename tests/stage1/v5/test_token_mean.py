"""Independent token arithmetic and actual parent-dispatch reduction witnesses."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from accelerate.utils import DistributedType
from transformers import Trainer, TrainingArguments

from stage1.v5_config import Stage1V5Config
from stage1.v5_metadata import V5BatchMetadata, V5Block
from stage1.v5_objective import compute_v5_loss_sums
from tests.stage1.v5.test_objective import fixture, metadata, literal_support_terms


def token_losses(clean, reference, targets):
    lp = clean.float().log_softmax(-1)
    lm = reference.detach().float().log_softmax(-1)
    r = lp - lm
    gt = r.gather(1, targets[:, None]).squeeze(1)
    return F.softplus(-gt) + (lm.exp() * F.softplus(r)).sum(-1)


@pytest.mark.parametrize('chunk', [1, 2, 128])
def test_unequal_blocks_are_token_sums_with_independent_gradients(chunk):
    clean, ref, off, _ = fixture()
    config = Stage1V5Config(text_kl_weight=0., loss_chunk_positions=chunk,
        lambda_topk_entropy=.1, lambda_topk_mass=.2, preference_topk_k=2,
        topk_include_gt=True)
    sums = compute_v5_loss_sums(clean, ref, off, None, metadata(), config)
    tokens = token_losses(clean, ref, torch.tensor([0, 3, 1]))
    torch.testing.assert_close(sums.dil_sum, tokens.sum())
    torch.testing.assert_close(torch.autograd.grad(sums.dil_sum, clean, retain_graph=True)[0],
                               torch.autograd.grad(tokens.sum(), clean, retain_graph=True)[0])
    assert torch.autograd.grad(sums.dil_sum, ref, allow_unused=True)[0] is None
    # The softplus is tokenwise: averaging the log ratios first is different.
    ratio = clean.log_softmax(-1) - ref.detach().log_softmax(-1)
    gt = ratio.gather(1, torch.tensor([[0], [3], [1]])).squeeze(1)
    assert not torch.isclose(F.softplus(-gt).mean(), F.softplus(-gt.mean()))
    u, mass = literal_support_terms(clean, off, [[0, 1], [1, 2, 3], [0, 2, 1]])
    for actual, expected in ((sums.entropy_sum, u[:2].mean() + u[2]),
                             (sums.mass_sum, mass[:2].mean() + mass[2])):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, off, retain_graph=True)[0],
                                   torch.autograd.grad(expected, off, retain_graph=True)[0])


def test_repartition_identical_tokens_preserves_dil_value_and_gradient():
    clean, ref, _, _ = fixture()
    original = metadata()
    repartitioned = V5BatchMetadata((
        V5Block(0, 1, 0, 1, (2,), (0,), ()),
        V5Block(0, 3, 4, 5, (), (), ()),
        V5Block(1, 1, 0, 1, (2, 3), (3, 1), ())))
    config = Stage1V5Config(text_kl_weight=0.)
    a = compute_v5_loss_sums(clean, ref, None, None, original, config).dil_sum
    b = compute_v5_loss_sums(clean, ref, None, None, repartitioned, config).dil_sum
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(torch.autograd.grad(a, clean, retain_graph=True)[0],
                               torch.autograd.grad(b, clean)[0])


def test_window_has_independent_immutable_token_and_block_denominators():
    from stage1 import v5_reduction as reduction
    window = reduction.V5Window(global_original_steps=5, actual_microbatches=2,
        configured_gas=4, world_size=4, global_dil_tokens=11)
    assert reduction.v5_dil_scale(window) == 16 / 11
    assert reduction.v5_auxiliary_scale(window) == 16 / 5
    assert reduction.v5_dil_scale(replace(window, global_dil_tokens=0)) == 0
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError): replace(window, global_dil_tokens=bad)
    with pytest.raises(TypeError):
        reduction.V5Window(5, 2, 4, 4)  # Missing token count cannot silently mean zero.


# Each entry is (tokens in original block0, tokens in block1). Empty ranks and
# microbatches still execute native terms. The second case is a partial tail.
TOKEN_WINDOWS = (
    (((2, 1), (0, 0), (1, 0), (0, 2)), ((0, 1), (3, 0), (0, 0), (1, 1)),
     ((0, 0), (0, 0), (0, 0), (0, 0)), ((1, 0), (0, 2), (1, 1), (0, 0))),
    (((2, 1), (0, 0)), ((0, 1), (3, 0)), ((0, 0), (0, 0)), ((1, 0), (0, 2))),
    (((0, 0), (0, 0)),) * 4,
)


@pytest.mark.parametrize('counts', TOKEN_WINDOWS)
def test_actual_parent_dispatch_matches_global_token_and_block_gradient(counts, tmp_path):
    from stage1.v5_trainer import CustomTrainerStage1V5
    from stage1.v5_reduction import v5_dil_scale, v5_auxiliary_scale
    tokens = sum(sum(blocks) for rank in counts for blocks in rank)
    blocks = 2 * sum(map(len, counts))
    gradients, values, kwargs_seen = [], [], []

    def batch_for(lengths):
        return {'stage1_v5_metadata': V5BatchMetadata(tuple(
            V5Block(0, i, 4*i, 4*i+1, tuple(range(4*i+2, 4*i+2+n)),
                    (i,) * n, ()) for i, n in enumerate(lengths)))}

    def actual_terms(p, meta):
        n = len(meta.query_keys)
        clean = p * torch.tensor([.2, -.5, .8]).repeat(n, 1)
        ref = torch.tensor([.3, .6, -.4]).repeat(n, 1)
        off = p * torch.tensor([.6, -.2, .1]).repeat(n, 1)
        config = Stage1V5Config(text_kl_weight=0., lambda_topk_entropy=.2,
                               lambda_topk_mass=.3, preference_topk_k=2)
        return compute_v5_loss_sums(clean, ref, off, None, meta, config)

    for rank in counts:
        p = torch.nn.Parameter(torch.tensor(.7))
        model = torch.nn.Linear(1, 1)
        model.weight = p
        trainer = CustomTrainerStage1V5.__new__(CustomTrainerStage1V5)
        trainer.args = TrainingArguments(str(tmp_path), use_cpu=True,
            gradient_accumulation_steps=4, report_to=[])
        trainer.state = SimpleNamespace(global_step=0)
        trainer.optimizer = torch.optim.AdamW([p], lr=.01)
        trainer.model_accepts_loss_kwargs = False
        trainer.compute_loss_func = None
        trainer.use_apex = trainer.use_cpu_amp = False
        trainer._prepare_inputs = lambda x: x
        reductions = []
        def reduce(value, **kw):
            reductions.append((value.tolist(), kw))
            return torch.tensor([blocks, tokens], dtype=torch.int64)
        trainer.accelerator = SimpleNamespace(device=torch.device('cpu'), num_processes=4,
            distributed_type=DistributedType.DEEPSPEED, reduce=reduce,
            backward=lambda loss, **kw: (kwargs_seen.append(kw), loss.backward()))
        def compute(_m, batch, **kw):
            sums = actual_terms(p, batch['stage1_v5_metadata'])
            w = batch['stage1_v5_window']
            return (.1*p.square() + .2*p.pow(3) + .03*p.pow(4)
                + v5_dil_scale(w)*sums.dil_sum
                + v5_auxiliary_scale(w)*(.2*sums.entropy_sum + .3*sums.mass_sum))
        trainer.compute_loss = compute
        batches, _ = trainer.get_batch_samples(iter(batch_for(n) for n in rank), 4)
        assert reductions == [([2*len(rank), sum(map(sum, rank))], {'reduction': 'sum'})]
        assert len({id(b['stage1_v5_window']) for b in batches}) == 1
        values.append(sum(Trainer.training_step(trainer, model, b) for b in batches))
        gradients.append(p.grad.clone())

    p = torch.tensor(.7, requires_grad=True)
    # Independent token formula and original-block support formulas. No product
    # objective or scales are used to construct this oracle.
    dil, aux = p*0, p*0
    for rank in counts:
        for lengths in rank:
            for target, n in enumerate(lengths):
                if not n: continue
                clean = p * torch.tensor([.2, -.5, .8]).repeat(n, 1)
                ref = torch.tensor([.3, .6, -.4]).repeat(n, 1)
                off = p * torch.tensor([.6, -.2, .1]).repeat(n, 1)
                dil = dil + token_losses(clean, ref, torch.full((n,), target)).sum()
                u, m = literal_support_terms(clean, off, [[0, 2]] * n)
                aux = aux + .2*u.mean() + .3*m.mean()
    expected = (sum(map(len, counts))/16 * (.1*p.square()+.2*p.pow(3)+.03*p.pow(4))
                + (dil/tokens if tokens else dil*0) + aux/blocks)
    expected.backward()
    torch.testing.assert_close(torch.stack(gradients).mean(), p.grad)
    torch.testing.assert_close(torch.stack(values).mean(), expected.detach())
    assert kwargs_seen == [{'scale_wrt_gas': False}] * sum(map(len, counts))
