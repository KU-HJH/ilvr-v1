from types import MethodType

import torch

from evidence_window.config import EvidenceWindowConfig


def test_disabled_matches_recorded_legacy_teacher_latent_branch(
    tiny_trainer_factory,
    legacy_latent_case,
):
    trainer, model = tiny_trainer_factory()
    trainer.evidence_window_config = EvidenceWindowConfig.disabled()

    torch.manual_seed(1234)
    loss, outputs = trainer.compute_loss(
        model,
        dict(legacy_latent_case["inputs"]),
        return_outputs=True,
    )

    assert torch.allclose(
        loss,
        legacy_latent_case["recorded_legacy_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        outputs.logits.sum(),
        legacy_latent_case["recorded_logits_sum"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert model.forward_count == legacy_latent_case["recorded_legacy_forward_count"]
    assert model.visual.forward_count == 1


def test_disabled_matches_recorded_legacy_ce_fallback(
    tiny_trainer_factory,
    legacy_no_latent_case,
):
    trainer, model = tiny_trainer_factory()
    trainer.evidence_window_config = EvidenceWindowConfig.disabled()

    def no_latents(_self, inputs, k, p):
        del k, p
        return None, torch.zeros_like(inputs["input_ids"], dtype=torch.bool)

    trainer._teacher_build_latents = MethodType(no_latents, trainer)
    loss, outputs = trainer.compute_loss(
        model,
        dict(legacy_no_latent_case["inputs"]),
        return_outputs=True,
    )

    assert torch.allclose(
        loss,
        legacy_no_latent_case["recorded_ce_loss"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        outputs.logits.sum(),
        legacy_no_latent_case["recorded_logits_sum"],
        atol=1e-6,
        rtol=1e-6,
    )
    assert model.forward_count == legacy_no_latent_case["recorded_legacy_forward_count"]
    assert model.visual.forward_count == 1
    assert trainer._ew_engine is None
    assert trainer._reference is None
