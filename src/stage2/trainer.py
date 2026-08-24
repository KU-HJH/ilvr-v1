"""Thin Hugging Face Trainer adapter for Stage 2 model-native loss."""

from transformers import Trainer


class Stage2Trainer(Trainer):
    """Delegate Stage 2 loss computation entirely to the model."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & inputs.keys()
        if forbidden:
            raise ValueError(
                f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}"
            )
        outputs = model(**inputs, collect_stage2_diagnostics=True)
        self._latest_stage2_diagnostics = outputs.stage2_diagnostics
        return (outputs.loss, outputs) if return_outputs else outputs.loss
