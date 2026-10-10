import importlib
import math
import pytest
import torch
from stage1.v3_config import Stage1V3Config
from stage1.v3_metadata import V3Block


def api():
    return importlib.import_module('stage1.v3_objective')


def test_block_score_is_sum_of_block_token_means_at_temperature_one():
    blocks = (V3Block(0,1,1,3,(4,5),(0,1),()), V3Block(1,1,1,3,(4,),(1,),()))
    logits = torch.tensor([[.8,.2],[.7,.3],[.4,.6]]).log().requires_grad_()
    loss = api().block_score_sum(logits, blocks)
    torch.testing.assert_close(loss, torch.tensor((math.log(.8)+math.log(.3))/2 + math.log(.6)))
    loss.backward()
    assert logits.grad is not None


def test_kl_direction_stop_gradient_sample_reduction_and_full_vocabulary():
    p = torch.tensor([[.8,.15,.05],[.2,.3,.5],[.1,.1,.8]])
    q = torch.tensor([[.1,.2,.7],[.5,.4,.1],[.6,.3,.1]])
    clean, restricted = (2*p.log()).requires_grad_(), (2*q.log()).requires_grad_()
    loss = api().text_kl_loss(clean, restricted, ((0,4),(0,5),(1,4)))
    per_query = 4*(p*(p.log()-q.log())).sum(-1)
    expected = (per_query[:2].mean()+per_query[2])/2
    torch.testing.assert_close(loss, expected)
    assert not torch.isclose(loss, per_query.mean())
    assert not torch.isclose(loss, (4*(q*(q.log()-p.log())).sum(-1)).mean())
    loss.backward()
    assert clean.grad is None
    torch.testing.assert_close(restricted.grad, 2*(q-p)*torch.tensor([.25,.25,.5])[:,None])


def test_empty_loss_zero_with_restricted_graph():
    x = torch.empty(0,3, requires_grad=True)
    result = api().text_kl_loss(x, x, ()) + api().block_score_sum(x, ())
    result.backward()
    assert result.item() == 0 and x.grad.shape == x.shape


@pytest.mark.parametrize('s,planned,ratio,expected', [(0,21,.05,0.),(1,21,.05,.005),(2,21,.05,.01),(99,21,.05,.01),(0,21,0.,.01)])
def test_warmup(s,planned,ratio,expected):
    assert api().text_kl_weight(s,planned,Stage1V3Config(text_kl_warmup_ratio=ratio)) == expected


@pytest.mark.parametrize('s,planned', [(-1,21),(0,0),(True,2),(0,1.5)])
def test_invalid_schedule(s,planned):
    with pytest.raises(ValueError):
        api().text_kl_weight(s,planned,Stage1V3Config())


def test_projection_uses_one_global_dot_not_per_parameter():
    # h.g=-1, n=2; independent parameter projections would incorrectly give (0,2).
    g, h = (torch.tensor([-3.]),torch.tensor([2.])), (torch.ones(1),torch.ones(1))
    projected = api().project_gain(g,h)
    assert all(p.dtype == torch.float32 for p in projected)
    torch.testing.assert_close(torch.cat(projected), torch.tensor([-2.5,2.5]))
    assert sum((a*b).sum().item() for a,b in zip(projected,h)) >= 0


@pytest.mark.parametrize('h,expected', [(0.,-2.),(1e-16,0.),(1e-15,0.),(-1.,-2.)])
def test_zero_tiny_threshold_and_positive_dot(h,expected):
    out = api().project_gain((torch.tensor([-2.]),),(torch.tensor([h]),))[0]
    torch.testing.assert_close(out,torch.tensor([expected]))


def test_threshold_is_inclusive_without_fp32_norm_underflow():
    h = torch.tensor([1e-20]); exact = h.double().square().sum().item()
    out = api().project_gain((torch.tensor([-2.]),),(h,),min_h_norm_sq=exact)[0]
    torch.testing.assert_close(out,torch.zeros(1),atol=1e-6,rtol=0)


@pytest.mark.parametrize('value', [float('inf'),float('nan')])
@pytest.mark.parametrize('which', ['g','h'])
def test_nonfinite_rejected(value,which):
    g,h = torch.ones(2),torch.ones(2)
    (g if which=='g' else h)[0] = value
    with pytest.raises(ValueError, match='nonfinite'):
        api().project_gain((g,),(h,))
