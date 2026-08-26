"""Thin Hugging Face Trainer adapter for Stage 2 model-native loss."""

import tempfile
from pathlib import Path

from transformers import Trainer

from .checkpointing import save_stage2_peft_metadata
from stage1.token_rows import save_sparse_token_rows


class Stage2Trainer(Trainer):
    """Add Stage 2 observability and PEFT sidecars to ordinary Trainer flows."""

    def __init__(self, *args, stage2_peft_metadata=None, **kwargs):
        self.stage2_peft_metadata = stage2_peft_metadata
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forbidden = {"pixel_values_latent", "image_grid_thw_latent"} & inputs.keys()
        if forbidden:
            raise ValueError(
                f"helper-image tensors leaked into Stage 2: {sorted(forbidden)}"
            )
        outputs = model(**inputs, collect_stage2_diagnostics=True)
        diagnostics = getattr(outputs, "stage2_diagnostics", None)
        self._latest_stage2_diagnostics = (
            None
            if diagnostics is None
            else {name: value.detach() for name, value in diagnostics.items()}
        )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs, start_time=None):
        diagnostics = getattr(self, "_latest_stage2_diagnostics", None)
        self._latest_stage2_diagnostics = None
        if diagnostics is not None:
            logs = dict(logs)
            logs["stage2/latent_block_count"] = float(
                diagnostics["block_count"].item()
            )
            norms = diagnostics["latent_l2_norms"]
            if norms.numel():
                logs.update(
                    {
                        "stage2/latent_l2_norm_mean": float(norms.mean().item()),
                        "stage2/latent_l2_norm_min": float(norms.min().item()),
                        "stage2/latent_l2_norm_max": float(norms.max().item()),
                    }
                )
            adjacent = diagnostics["adjacent_cosine_similarities"]
            if adjacent.numel():
                logs["stage2/adjacent_cosine_mean"] = float(adjacent.mean().item())
        super().log(logs, start_time=start_time)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        metadata = self.stage2_peft_metadata
        if metadata is None:
            return
        target = Path(output_dir or self.args.output_dir)
        if metadata.token_row_source == "sparse_adapter":
            if self.is_deepspeed_enabled:
                # The sparse ZeRO gather is collective. Non-writing ranks use a
                # private temporary destination while rank zero publishes.
                if self.args.should_save:
                    save_sparse_token_rows(self.model, target)
                else:
                    with tempfile.TemporaryDirectory(
                        prefix="stage2-token-rows-"
                    ) as temporary:
                        save_sparse_token_rows(self.model, Path(temporary))
            elif self.args.should_save:
                save_sparse_token_rows(self.model, target)
        if self.args.should_save:
            save_stage2_peft_metadata(target, metadata)
