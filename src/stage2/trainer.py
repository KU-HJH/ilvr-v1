"""Thin Hugging Face Trainer adapter for Stage 2 model-native loss."""

from transformers import Trainer

from .checkpointing import save_stage2_peft_metadata


class Stage2Trainer(Trainer):
    """Add only Stage 2 PEFT sidecars to ordinary Trainer saves."""

    def __init__(self, *args, stage2_peft_metadata=None, **kwargs):
        self.stage2_peft_metadata = stage2_peft_metadata
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call=_internal_call)
        target = output_dir or self.args.output_dir
        if self.stage2_peft_metadata is not None and self.args.should_save:
            save_stage2_peft_metadata(target, self.stage2_peft_metadata)
