from dataclasses import replace

import pytest
import torch

from stage1.prefix_mask_config import PrefixConditioningConfig
from stage1.prefix_mask_metadata import BlockSources, HelperSnapshot, ImageGroup
from stage1.helper_attention_proxy import helper_attention_proxy
from stage1.prefix_mask_plan import (build_prefix_mask_plan, query_budgets, reverse_kl,
                                     select_image_groups, select_text_keys)
from transformers.models.qwen2_5_vl.stage1_prefix_trace import AttentionRows, TraceRequest


def test_low_attention_fallback_floor_tie_and_exact_threshold():
    cfg = PrefixConditioningConfig()
    assert select_text_keys(tuple(range(35)), torch.full((35,), .001), rho=.4, config=cfg) == (0, 1, 2)
    assert select_text_keys(tuple(range(9)), torch.zeros(9), rho=.4, config=cfg) == ()
    assert select_text_keys(tuple(range(10)), torch.zeros(10), rho=.4, config=cfg) == (0,)
    cfg = replace(cfg, text_low_attention_threshold=.125)
    assert select_text_keys(tuple(range(16)), torch.full((16,), 1/128), rho=.4, config=cfg) == (0, 1, 2, 3)
    weights = torch.full((16,), .10 / 16)
    cfg = replace(cfg, text_low_attention_threshold=float(weights.sum()))
    assert select_text_keys(tuple(range(16)), weights, rho=.4, config=cfg) == (0, 1, 2, 3)
    assert select_text_keys((9, 3, 7, 2), torch.ones(4), rho=.4,
                            config=replace(cfg, text_max_mask_ratio=1.)) == (2, 3)


def test_mass_crossing_and_caps_do_not_renormalize():
    cfg = replace(PrefixConditioningConfig(), text_max_mask_ratio=1.)
    assert select_text_keys((0, 1, 2), torch.tensor([.3, .25, .2]), rho=.5, config=cfg) == (0, 1)
    groups = tuple(ImageGroup(0, i, tuple(range(a, b))) for i, (a, b) in enumerate(((0, 5), (5, 8), (8, 12), (12, 16))))
    assert select_image_groups(groups, torch.tensor([.6, .2, .1, .1]), rho=.5, config=cfg) == (1,)
    cfg = replace(cfg, image_max_mask_ratio=1.)
    assert select_image_groups(groups, torch.tensor([.3, .25, .2, .1]), rho=.5, config=cfg) == (0, 1)
    assert select_image_groups(groups[::-1], torch.ones(4), rho=.25, config=cfg) == (3,)


def test_full_kl_and_per_sample_centering():
    logits = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
    assert torch.equal(reverse_kl(logits, logits, temperature=2.), torch.zeros(2))
    reference = torch.tensor([[0., 1.], [1., 0.]])
    logp, logr = (logits.float() / 2).log_softmax(-1), (reference / 2).log_softmax(-1)
    got = reverse_kl(logits, reference, temperature=2.)
    torch.testing.assert_close(got, 4 * (logp.exp() * (logp - logr)).sum(-1))
    assert got.dtype == torch.float32 and got.grad_fn is None
    kwargs = dict(rho_min=.3, rho_max=.5, epsilon=1e-6, sigmoid_scale=1.)
    torch.testing.assert_close(query_budgets(torch.zeros(2), torch.tensor([0, 0]), **kwargs), torch.tensor([.4, .4]))
    rho = query_budgets(torch.tensor([0., 1., 999.], requires_grad=True), torch.tensor([0, 0, 1]), **kwargs)
    torch.testing.assert_close(rho[:2], query_budgets(torch.tensor([0., 1.]), torch.tensor([0, 0]), **kwargs))
    assert rho[0] > rho[1] and rho.grad_fn is None


def fixture(*, helper=True, helper_mass=1., gt_mass=1.):
    groups = (ImageGroup(0, 0, (0, 1)), ImageGroup(1, 0, (2, 3)))
    source = BlockSources((0, 0), ((0, 30), (0, 31)), tuple(range(4, 14)), groups, (20, 22) if helper else ())
    snapshot = HelperSnapshot(torch.zeros(2, 3), torch.zeros(1, 40, dtype=torch.bool), (source,), source.gt_queries, 2, 3, 'snapshot')
    queries = ((0, 19), (0, 21), (0, 30), (0, 31))
    values = (torch.tensor([.1, .1, .05, .05]) * helper_mass,
              torch.tensor([.2, .2, .15, .15]) * helper_mass,
              torch.tensor([.1, .1, .2, .2] + [.001] * 10) * gt_mass,
              torch.tensor([.1, .1, .2, .2] + [.001] * 10) * gt_mass)
    rows = AttentionRows(TraceRequest(queries, ((0, 1, 2, 3),) * 2 + (tuple(range(14)),) * 2), values)
    return snapshot, rows


def test_proxy_averages_actual_prediction_rows_and_ignores_targets():
    snap, rows = fixture()
    rows = replace(rows, values=tuple(value.requires_grad_() for value in rows.values))
    pi = helper_attention_proxy(rows, snap.sources[0])
    torch.testing.assert_close(pi, torch.tensor([.3, .2]))
    assert pi.grad_fn is None
    altered = replace(snap, latent_values=torch.ones(2, 3))
    torch.testing.assert_close(pi, helper_attention_proxy(rows, altered.sources[0]))
    missing, rows = fixture(helper=False)
    assert torch.equal(helper_attention_proxy(rows, missing.sources[0]), torch.zeros(2))


@pytest.mark.parametrize('options,reason', [({'helper': False}, 'no_selected_helper'),
    ({'helper_mass': 1e-5}, 'low_proxy_mass'), ({'gt_mass': 1e-9}, 'low_image_score')])
def test_image_skips_preserve_text_and_record_kl(options, reason):
    snap, rows = fixture(**options)
    plan, report = build_prefix_mask_plan(snap, rows, torch.tensor([0., 2.]), config=PrefixConditioningConfig())
    assert all(row.image_keys == () and row.text_keys == (4,) for row in plan.rows)
    assert [row['image_skip_reason'] for row in report['queries']] == [reason, reason]
    assert [row['kl'] for row in report['queries']] == [0., 2.]


def test_plan_identity_order_budgets_and_audit_fields():
    snap, rows = fixture()
    cfg = replace(PrefixConditioningConfig(), image_max_mask_ratio=.5)
    plan, report = build_prefix_mask_plan(replace(snap, sources=snap.sources[::-1]), rows, torch.tensor([0., 2.]), config=cfg)
    assert tuple(row.query for row in plan.rows) == snap.gt_queries
    assert (plan.snapshot_id, plan.teacher_version, plan.parameter_version) == ('snapshot', 2, 3)
    assert all(row.image_keys == (2, 3) for row in plan.rows)
    first = report['queries'][0]
    assert first['text_cap'] == 3 and first['image_cap'] == 2
    assert first['text_selected_count'] == 1 and first['image_selected_count'] == 2
    assert first['image_group_ids'] == ((1, 0),)
    assert first['text_fallback_reason'] == 'low_text_attention'
    assert first['text_raw_mass'] == pytest.approx(.01)
    assert first['image_achieved_mass'] == pytest.approx(4/7)
    assert first['rho_text'] > report['queries'][1]['rho_text']


def test_duplicate_queries_and_missing_observations_rejected():
    snap, rows = fixture()
    with pytest.raises(ValueError, match='GT'):
        build_prefix_mask_plan(replace(snap, gt_queries=(snap.gt_queries[0],) * 2), rows, torch.zeros(2), config=PrefixConditioningConfig())
    with pytest.raises(ValueError, match='GT'):
        build_prefix_mask_plan(replace(snap, sources=snap.sources * 2), rows, torch.zeros(2), config=PrefixConditioningConfig())
    with pytest.raises(ValueError, match='missing'):
        helper_attention_proxy(AttentionRows(TraceRequest((), ()), ()), snap.sources[0])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_inputs_rejected(bad):
    cfg = PrefixConditioningConfig()
    with pytest.raises(ValueError):
        reverse_kl(torch.tensor([[bad, 0.]]), torch.zeros(1, 2), temperature=2.)
    with pytest.raises(ValueError):
        query_budgets(torch.tensor([bad]), torch.tensor([0]), rho_min=.3, rho_max=.5, epsilon=1e-6, sigmoid_scale=1.)
    with pytest.raises(ValueError):
        select_text_keys((0,), torch.tensor([bad]), rho=.4, config=cfg)
    with pytest.raises(ValueError):
        select_image_groups((ImageGroup(0, 0, (0,)),), torch.tensor([bad]), rho=.4, config=cfg)
    snap, rows = fixture()
    with pytest.raises(ValueError):
        helper_attention_proxy(replace(rows, values=(torch.full((4,), bad),) + rows.values[1:]), snap.sources[0])


def test_empty_modalities_and_zero_image_scores():
    cfg = PrefixConditioningConfig()
    assert select_text_keys((), torch.empty(0), rho=.4, config=cfg) == ()
    assert select_image_groups((), torch.empty(0), rho=.2, config=cfg) == ()
    groups = tuple(ImageGroup(0, i, (i,)) for i in range(4))
    assert select_image_groups(groups, torch.zeros(4), rho=.2, config=cfg) == ()
    assert select_image_groups(groups, torch.full((4,), 2.5e-9), rho=.2, config=cfg) == ()


def test_skipped_group_mass_remains_in_denominator():
    cfg = replace(PrefixConditioningConfig(), image_max_mask_ratio=.5)
    groups = (ImageGroup(0, 0, (0, 1, 2, 3, 4)),) + tuple(ImageGroup(0, i, (i + 4,)) for i in range(1, 4))
    assert select_image_groups(groups, torch.tensor([.7, .15, .1, .05]), rho=.4, config=cfg) == (1, 2, 3)


def test_proxy_floor_is_strict_and_budget_clamps_negative_kl():
    snap, rows = fixture()
    cfg = replace(PrefixConditioningConfig(), helper_proxy_mass_floor=.5, image_max_mask_ratio=.5)
    plan, report = build_prefix_mask_plan(snap, rows, torch.tensor([-1., 0.]), config=cfg)
    assert all(row.image_keys == (2, 3) for row in plan.rows)
    assert all(row['rho_text'] == pytest.approx(.4) for row in report['queries'])
    assert all(row['image_skip_reason'] is None for row in report['queries'])


def test_empty_snapshot_is_a_valid_empty_plan():
    snap = HelperSnapshot(torch.empty(0, 3), torch.zeros(1, 1, dtype=torch.bool), (), (), 0, 0, 'empty')
    plan, report = build_prefix_mask_plan(snap, AttentionRows(TraceRequest((), ()), ()), torch.empty(0), config=PrefixConditioningConfig())
    assert plan.rows == () and report['queries'] == ()
