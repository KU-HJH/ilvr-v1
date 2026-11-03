"""Projection must remove only the masked-descending component."""
import pytest
import torch


@pytest.mark.parametrize('g,h,expected,projected',[
    ([-2.,3.],[1.,0.],[0.,3.],True),([2.,3.],[1.,0.],[2.,3.],False),
    ([-2.,3.],[0.,0.],[-2.,3.],False),([0.,0.],[1.,2.],[0.,0.],False)])
def test_projection_geometry(g,h,expected,projected):
    from stage1.vision_gain import project_gain_direction
    result=project_gain_direction((torch.tensor(g),),(torch.tensor(h),),min_h_norm_sq=1e-30)
    torch.testing.assert_close(result.direction[0],torch.tensor(expected))
    assert result.projected is projected and not result.skipped


def test_projection_invariants_and_degenerate_norm():
    from stage1.vision_gain import project_gain_direction
    gen=torch.Generator().manual_seed(42)
    for _ in range(20):
        g,h=torch.randn(13,generator=gen),torch.randn(13,generator=gen)
        u=project_gain_direction((g,),(h,),min_h_norm_sq=1e-30).direction[0]
        assert h.double().dot(u.double()) >= -2e-6
        torch.testing.assert_close(g.dot(u),u.dot(u),atol=2e-6,rtol=1e-5)
        assert (g+h).dot(u) >= -2e-6
    result=project_gain_direction((torch.ones(1),),(torch.tensor([1e-18]),),min_h_norm_sq=1e-30)
    assert result.skipped and result.zero_direction


@pytest.mark.parametrize('alpha,guard',[(-1.,1e-30),(float('nan'),1e-30),(float('inf'),1e-30),(1.,0.)])
def test_invalid_config_rejected(alpha,guard):
    from stage1.vision_gain import Stage1VisionGainConfig
    with pytest.raises(ValueError):
        Stage1VisionGainConfig(alpha,guard)
