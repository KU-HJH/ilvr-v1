"""Breaks caught: wrong donor scope, rank direction, ties, ownership or fallback."""
import importlib
import pytest
import torch


def select(*args, **kwargs):
    assert importlib.util.find_spec('stage1.v5_donors'), 'reverse-topk donor selector missing'
    return importlib.import_module('stage1.v5_donors').select_reverse_topk_donors(*args, **kwargs)


def test_global_reverse_cosine_excludes_current_image():
    result = select((torch.tensor([[-10., 0.]]), torch.tensor([[-1., 0.], [0., 1.], [1., 0.]]),
                     torch.tensor([[0., -1.], [-2., 0.]])),
        ('current', 'other', 'third'), (0, 1, 2), 0, (), torch.tensor([[1., 0.]]), count=4, source_index=7)
    assert result.donor_kind == 'helper'
    assert result.image_indices == (1, 2, 1, 2)
    assert result.candidate_indices == (0, 1, 1, 0)
    torch.testing.assert_close(result.values, torch.tensor([[-1., 0.], [-2., 0.], [0., 1.], [0., -1.]]))
    assert not result.values.requires_grad


def test_current_identity_is_excluded_across_steps_and_eligible_duplicates_deduplicate():
    x = torch.tensor([[-1., 0.], [0., 1.]], requires_grad=True)
    result = select((x*3, x, x*7, x*9), ('current','other','current','other'),
        (0,1,2,3), 0, (), torch.tensor([[1.,0.]]), count=2, source_index=2)
    assert result.image_indices == (1,1)
    torch.testing.assert_close(result.values,x.detach())
    assert not result.values.requires_grad


def test_single_helper_fallback_preserves_raw_values_and_eight_distinct_indices():
    x = torch.tensor([[1.,float(i)] for i in range(8)], dtype=torch.bfloat16, requires_grad=True)
    result = select((x,), ('current',), (0,), 0, (x,), torch.tensor([[1.,0.]]), count=8, source_index=7)
    assert result.donor_kind == 'original'
    assert result.image_indices == (0,)*8
    assert result.candidate_indices == tuple(range(7,-1,-1))
    assert result.values.dtype == torch.bfloat16 and not result.values.requires_grad
    torch.testing.assert_close(result.values,x.detach().flip(0))


@pytest.mark.parametrize('mutation',['underfilled_helper','underfilled_original','missing_original','identity_count',
    'owner_count','invalid_owner','missing_current','empty_identity','nan_query','inf_feature','zero_count','bool_count','shape'])
def test_invalid_donors_fail_with_source_block_identity(mutation):
    helpers=(torch.tensor([[1.,0.],[0.,1.]]), torch.tensor([[-1.,0.],[0.,1.]]))
    ids=('current','other'); mapping=(0,1); originals=(); query=torch.tensor([[1.,0.]]); count=2
    if mutation=='underfilled_helper': helpers=(helpers[0],helpers[1][:1]); originals=(helpers[0],)
    elif mutation=='underfilled_original': helpers=helpers[:1];ids=ids[:1];mapping=(0,); originals=(helpers[0][:1],)
    elif mutation=='missing_original': helpers=helpers[:1];ids=ids[:1];mapping=(0,)
    elif mutation=='identity_count': ids=('current',)
    elif mutation=='owner_count': mapping=(0,)
    elif mutation=='invalid_owner': mapping=(0,-1)
    elif mutation=='missing_current': mapping=(1,2)
    elif mutation=='empty_identity': ids=('current','')
    elif mutation=='nan_query': query.fill_(float('nan'))
    elif mutation=='inf_feature': helpers[1][0,0]=float('inf')
    elif mutation=='zero_count': count=0
    elif mutation=='bool_count': count=True
    elif mutation=='shape': helpers=(helpers[0],torch.ones(2,3))
    with pytest.raises((ValueError,FloatingPointError),match='source_index=7.*block=0'):
        select(helpers,ids,mapping,0,originals,query,count=count,source_index=7)


def test_no_cross_sample_state_or_cache():
    def run(other):
        return select((torch.ones(2,2),other),('current','other'),(0,1),0,(),torch.tensor([[1.,0.]]),count=2,source_index=0)
    first=run(torch.tensor([[-1.,0.],[0.,1.]]))
    second=run(torch.tensor([[1.,0.],[0.,-1.]]))
    torch.testing.assert_close(first.values,torch.tensor([[-1.,0.],[0.,1.]]))
    torch.testing.assert_close(second.values,torch.tensor([[0.,-1.],[1.,0.]]))
