from contextlib import contextmanager

from accelerate import Accelerator
import pytest
import torch

from evidence_window.config import ReferenceConfig
from evidence_window.reference import (
    AdapterDisabledReference,
    FrozenCopyReference,
    build_reference_provider,
)


class AdapterToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.adapter = torch.nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.adapter_enabled = True

    @contextmanager
    def disable_adapter(self):
        previous = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = previous

    def forward(self, x):
        result = self.base(x)
        if self.adapter_enabled:
            result = result + self.adapter(x)
        return result


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.forward_calls = 0

    def forward(self, **inputs):
        self.forward_calls += 1
        return self.module(**inputs)


def test_adapter_disabled_reference_ignores_adapter_update_through_active_wrapper():
    policy = AdapterToy()
    wrapped_policy = ModuleWrapper(policy)
    provider = AdapterDisabledReference(wrapped_policy)
    inputs = {"x": torch.ones(1, 2)}

    before = provider.forward(wrapped_policy, inputs)
    with torch.no_grad():
        policy.adapter.weight.add_(1)
    after = provider.forward(wrapped_policy, inputs)

    assert torch.equal(before, after)
    assert policy.adapter_enabled
    assert wrapped_policy.forward_calls == 2
    assert not before.requires_grad


def test_adapter_disabled_reference_restores_adapter_after_forward_error():
    policy = AdapterToy()

    class FailingWrapper(ModuleWrapper):
        def forward(self, **inputs):
            assert not policy.adapter_enabled
            raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        AdapterDisabledReference(policy).forward(FailingWrapper(policy), {"x": torch.ones(1, 2)})

    assert policy.adapter_enabled


def test_adapter_disabled_reference_rejects_trainable_non_adapter_parameter():
    policy = AdapterToy()
    policy.base.weight.requires_grad_(True)

    with pytest.raises(ValueError, match="non-adapter.*base.weight"):
        AdapterDisabledReference(policy)


def test_adapter_disabled_reference_rejects_model_without_disable_context():
    with pytest.raises(ValueError, match="disable_adapter"):
        AdapterDisabledReference(torch.nn.Linear(2, 2))


def test_frozen_copy_is_no_grad_and_unchanged_after_policy_step():
    policy = torch.nn.Linear(2, 2, bias=False)
    provider = FrozenCopyReference.create(policy, Accelerator(cpu=True))
    snapshot = [parameter.detach().clone() for parameter in provider.model.parameters()]

    loss = policy(torch.ones(1, 2)).sum()
    loss.backward()
    torch.optim.SGD(policy.parameters(), lr=1).step()

    assert not provider.model.training
    assert all(not parameter.requires_grad for parameter in provider.model.parameters())
    assert all(parameter.grad is None for parameter in provider.model.parameters())
    assert all(
        torch.equal(before, after)
        for before, after in zip(snapshot, provider.model.parameters())
    )
    assert provider.forward(policy, {"input": torch.ones(1, 2)}).grad_fn is None


def test_build_reference_provider_selects_mode():
    accelerator = Accelerator(cpu=True)
    adapter_policy = AdapterToy()

    assert (
        build_reference_provider(ReferenceConfig(mode="none"), adapter_policy, accelerator)
        is None
    )
    assert isinstance(
        build_reference_provider(
            ReferenceConfig(mode="adapter_disabled"), adapter_policy, accelerator
        ),
        AdapterDisabledReference,
    )
    assert isinstance(
        build_reference_provider(
            ReferenceConfig(mode="frozen_copy"), torch.nn.Linear(2, 2), accelerator
        ),
        FrozenCopyReference,
    )


def test_build_reference_provider_rejects_unknown_mode():
    class BadReferenceConfig:
        mode = "ema"

    with pytest.raises(ValueError, match="unsupported reference mode"):
        build_reference_provider(
            BadReferenceConfig(), torch.nn.Linear(2, 2), Accelerator(cpu=True)
        )
