import importlib
import pytest
import torch
from stage1.v3_config import Stage1V3Config
from stage1.v3_metadata import V3Block


def select(blocks, rows, student=None, reference=None, **config):
    student = torch.zeros(2, 40, 3) if student is None else student
    reference = torch.zeros_like(student) if reference is None else reference
    return importlib.import_module('stage1.v3_selection').select_text_keys(
        blocks, rows, student, reference, Stage1V3Config(**config))


def block(row=0, targets=(22,), candidates=tuple(range(2, 12))):
    return V3Block(row, 1, 18, 21, targets, tuple(0 for _ in targets), candidates)


@pytest.mark.parametrize('weights,expected', [
    ([.03]*10, (2,3,4)), ([.009]*10, (2,)),
    ([.025]*4+[0.]*6, (2,3)), ([0.]*10, (2,)),
    ([.2,.15,.05]+[0.]*7, (2,)),
])
def test_selector_mass_boundary_ties_and_crossing(weights, expected):
    assert select((block(),), {(0,21):torch.tensor(weights)}) == {(0,21):expected}


@pytest.mark.parametrize('count', [0,1,3])
def test_floor_cap_zero_and_empty_candidates(count):
    assert select((block(candidates=tuple(range(count))),), {(0,21):torch.zeros(count)}) == {(0,21):()}


def test_selector_uses_full_vocabulary_student_to_reference_kl_and_sample_centering():
    b0, b1 = block(targets=(22,23)), block(row=1)
    student = torch.zeros(2,40,3, requires_grad=True)
    reference = torch.zeros_like(student)
    with torch.no_grad():
        student[0,22] = 2*torch.tensor([.8,.1,.1]).log()
        reference[0,22] = 2*torch.tensor([.05,.9,.05]).log()
    # rho near .5 for identical query, near .3 for divergent query;
    # singleton sample has rho=.4 regardless of the other sample.
    rows = {key:torch.tensor([.21,.21,.21,.07,.05,.05,.05,.05,.05,.05], requires_grad=True)
            for key in ((0,21),(0,22),(1,21))}
    chosen = select((b0,b1), rows, student, reference)
    assert chosen == {(0,21):(2,3,4), (0,22):(2,3), (1,21):(2,3)}
    assert student.grad is None and all(t.grad is None for t in rows.values())


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-.01])
def test_selector_rejects_invalid_attention(bad):
    with pytest.raises(ValueError, match='attention'):
        select((block(),), {(0,21):torch.tensor([bad]+[.01]*9)})


@pytest.mark.parametrize('first_two,expected_low', [(.41, (2,3,4)), (.43, (2,3))])
def test_entropy_selector_uses_temperature_and_full_vocabulary_without_reference(first_two, expected_low):
    # At T=2, H(.8,.1,.1)=.6390 and H(uniform)=1.0986, giving
    # rho_low=.4134 and rho_high=.3866. T=1 instead gives rho_low>.43.
    student = torch.zeros(2,40,3, requires_grad=True)
    with torch.no_grad():
        student[0,21] = 2*torch.tensor([.8,.1,.1]).log()
    weights = torch.tensor([first_two/2]*2+[.2]+[(1-first_two-.2)/7]*7, requires_grad=True)
    rows = {key:weights for key in ((0,21),(0,22),(1,21))}
    chosen = importlib.import_module('stage1.v3_selection').select_text_keys(
        (block(targets=(22,23)),block(row=1)), rows, student, None,
        Stage1V3Config(text_selector='entropy'))
    assert chosen == {(0,21):expected_low, (0,22):(2,3), (1,21):(2,3)}
    assert student.grad is None and weights.grad is None


def test_entropy_selector_does_not_replace_full_vocabulary_with_gt_probability():
    student = torch.zeros(2,40,3)
    student[0,21] = 2*torch.tensor([.8,.1,.1]).log()
    student[0,22] = 2*torch.tensor([.45,.1,.45]).log()
    # Same probability for GT id1, different entropy. rho straddles .4.
    b = V3Block(0,1,18,21,(22,23),(1,1),tuple(range(2,12)))
    rows = {key:torch.tensor([.2,.2,.2]+[.4/7]*7) for key in ((0,21),(0,22))}
    assert importlib.import_module('stage1.v3_selection').select_text_keys(
        (b,), rows, student, None, Stage1V3Config(text_selector='entropy')) == {
            (0,21):(2,3,4), (0,22):(2,3)}


@pytest.mark.parametrize('weights,expected', [([.03]*10,(2,3,4)),([.009]*10,(2,)),([0.]*10,(2,))])
def test_entropy_selector_preserves_equal_entropy_caps_and_low_mass_fallback(weights, expected):
    student = torch.zeros(2,40,3)
    # Extreme finite logits yield zero entropy; inverse-log epsilon stays finite.
    student[...,0] = 10000
    assert importlib.import_module('stage1.v3_selection').select_text_keys(
        (block(),), {(0,21):torch.tensor(weights)}, student, None,
        Stage1V3Config(text_selector='entropy')) == {(0,21):expected}


@pytest.mark.parametrize('bad', [float('nan'),float('inf'),-float('inf')])
def test_entropy_selector_rejects_nonfinite_student_scores(bad):
    student = torch.zeros(2,40,3)
    student[0,21,0] = bad
    with pytest.raises(ValueError, match='entropy|finite'):
        importlib.import_module('stage1.v3_selection').select_text_keys(
            (block(),), {(0,21):torch.tensor([.03]*10)}, student, None,
            Stage1V3Config(text_selector='entropy'))


@pytest.mark.parametrize('bad', ['unknown',None,True])
def test_selector_config_rejects_unknown_modes(bad):
    with pytest.raises(ValueError, match='selector'):
        Stage1V3Config(text_selector=bad)
