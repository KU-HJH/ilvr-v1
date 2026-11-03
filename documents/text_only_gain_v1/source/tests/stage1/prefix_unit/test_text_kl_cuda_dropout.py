"""CUDA tensor primitives only: never instantiate or execute a model."""
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from stage1.text_kl import joint_positive_kl_gradients
from transformers.models.qwen2_5_vl.stage1_vision_gain_packed import _PairedDropout


@pytest.mark.parametrize('gpu', [0, 1])
def test_native_cuda_dropout_recompute_matches_both_vjp_lanes_and_rng(gpu):
    if torch.cuda.device_count() <= gpu:
        pytest.skip('requires both project GPUs')
    device = torch.device(f'cuda:{gpu}')
    with torch.cuda.device(device), torch.random.fork_rng(devices=[gpu]):
        torch.manual_seed(42)
        x = torch.linspace(-1., 1., 512, device=device, dtype=torch.bfloat16).reshape(1, 16, 32)
        x.requires_grad_()
        indices = torch.tensor([4, 1, 4], device=device)
        pair = _PairedDropout(SimpleNamespace(p=.05, forward=lambda t: F.dropout(t, p=.05, training=True)))
        pair.native_randomness = True
        calls = []
        def expression(t):
            calls.append(1)
            clean = pair.clean(t.sin())
            return pair.replay(indices, clean[:, indices].cos())
        initial_rng = torch.cuda.get_rng_state(device)
        h = checkpoint(expression, x, use_reentrant=False)
        after_forward = torch.cuda.get_rng_state(device)
        positive, kl, negative = h.float().sum(), h.float().square().sum(), -h.float().sum()
        r_gradient, kl_gradient = joint_positive_kl_gradients(positive, kl, (x,))
        torch.autograd.grad(negative, x)
        assert len(calls) == 3, 'KL must not add another checkpoint traversal'
        assert torch.equal(after_forward, torch.cuda.get_rng_state(device))
        reference = x.detach().clone().requires_grad_()
        torch.cuda.set_rng_state(initial_rng, device)
        # Existing native (non-vmap) dropout is the numerical/RNG reference.
        native = _PairedDropout(SimpleNamespace(p=.05, forward=lambda t: F.dropout(t, p=.05, training=True)))
        clean = native.clean(reference.sin())
        want = native.replay(indices, clean[:, indices].cos())
        want_r = torch.autograd.grad(want.float().sum(), reference, retain_graph=True)[0]
        want_kl = torch.autograd.grad(want.float().square().sum(), reference)[0]
        torch.testing.assert_close(h, want, rtol=0, atol=0)
        torch.testing.assert_close(r_gradient[0], want_r, rtol=0, atol=0)
        torch.testing.assert_close(kl_gradient[0], want_kl, rtol=0, atol=0)
        assert torch.equal(after_forward, torch.cuda.get_rng_state(device))
