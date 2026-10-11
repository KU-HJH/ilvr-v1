"""Real CUDA placement contracts; no simulated optimizer or model training."""
import pytest
import torch

from stage1.v3_config import Stage1V3Config
from stage1.v3_gradients import V3Gradients
from stage1.v3_update import V3GainController


pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA relocation')]


def gradients(r, rv, count, device):
    zero = torch.zeros((), device=device)
    return V3Gradients((torch.tensor(r, dtype=torch.float32, device=device),),
                       (torch.tensor(rv, dtype=torch.float32, device=device),), count, zero, zero)


@pytest.mark.parametrize('empty_first', [False, True])
def test_idle_cpu_controller_follows_parameter_placement_and_preserves_sums(empty_first):
    # Regresses buffers created before DeepSpeed moves the same parameter to CUDA.
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.bfloat16))
    controller = V3GainController((parameter,), Stage1V3Config())
    parameter.data = parameter.data.cuda()
    if empty_first:
        controller.accumulate(gradients([0., 0.], [0., 0.], 0, parameter.device))
    controller.accumulate(gradients([2., -1.], [1., 1.], 2, parameter.device))
    controller.accumulate(gradients([4., -2.], [2., 2.], 3, parameter.device))
    controller.prepare()
    assert controller.state == 'PREPARED'
    assert controller.global_block_count == 5
    torch.testing.assert_close(controller.direction[0],
                               torch.tensor([.9, -.9], device=parameter.device))
    assert all(value.device == parameter.device and value.dtype == torch.float32
               for value in controller._r + controller._rv)
    torch.testing.assert_close(controller._r[0], torch.tensor([6., -3.], device=parameter.device))
    torch.testing.assert_close(controller._rv[0], torch.tensor([3., 3.], device=parameter.device))


def test_relocation_during_accumulation_rejects_without_erasing_prior_contributions():
    # Repairing an open window would silently replace earlier sums and counts.
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.bfloat16))
    controller = V3GainController((parameter,), Stage1V3Config())
    controller.accumulate(gradients([2., -1.], [1., 1.], 2, parameter.device))
    parameter.data = parameter.data.cuda()
    controller.accumulate(gradients([4., -2.], [2., 2.], 3, parameter.device))
    with pytest.raises(RuntimeError, match='invalid or nonfinite auxiliary'):
        controller.prepare()
    assert controller.state == 'FAILED'
    assert controller._count == 5
    torch.testing.assert_close(controller._r[0], torch.tensor([2., -1.]))
    torch.testing.assert_close(controller._rv[0], torch.tensor([1., 1.]))
