"""Explicit CE-only trainer for ILVR Stage 2."""

from trl import SFTTrainer

from evidence_window.stage2_config import Stage2EvidenceWindowConfig


class CustomTrainerStage2(SFTTrainer):
    """Train the Stage-2 policy without Stage-1 teacher supervision."""

    def __init__(
        self,
        *args,
        evidence_window_config: Stage2EvidenceWindowConfig | None = None,
        **kwargs,
    ):
        self.evidence_window_config = (
            Stage2EvidenceWindowConfig.disabled()
            if evidence_window_config is None
            else evidence_window_config
        )
        if not isinstance(
            self.evidence_window_config,
            Stage2EvidenceWindowConfig,
        ):
            raise TypeError(
                "evidence_window_config must be a Stage2EvidenceWindowConfig"
            )
        self.evidence_window_config.validate()
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch
        inputs.pop("ew_sample_indices", None)
        inputs.pop("ew_plan_types", None)
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


__all__ = ["CustomTrainerStage2"]
