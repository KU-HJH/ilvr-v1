"""Invocation-only Stage1-v3 stop after a complete numbered checkpoint."""
from transformers import TrainerCallback


def validate_planned_checkpoint_stop(
    step, *, is_v3, resume=None, final_save=False, save_only_model=False,
    planned_steps=None,
):
    if step is None:
        return
    if type(step) is not int or step <= 0:
        raise ValueError("stage1_v3_stop_after_checkpoint must be a positive integer")
    if not is_v3:
        raise ValueError("stage1_v3_stop_after_checkpoint is only valid for Stage1-v3")
    if resume is not None and resume is not False:
        raise ValueError("stage1_v3_stop_after_checkpoint is fresh-run only")
    if final_save:
        raise ValueError("stage1_v3_stop_after_checkpoint requires --no_final_save")
    if save_only_model:
        raise ValueError("stage1_v3_stop_after_checkpoint requires a full checkpoint")
    if planned_steps is not None and (
        type(planned_steps) is not int or not step < planned_steps
    ):
        raise ValueError("stage1_v3_stop_after_checkpoint must precede the full prepared-loader budget")


class PlannedCheckpointStop(TrainerCallback):
    def __init__(self, step):
        self.step = step
        self.completed_step = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        # Trainer calls this only after Stage1's save, sidecars, rank agreement,
        # barrier and checkpoint rotation have returned successfully.
        if state.global_step == self.step:
            self.completed_step = state.global_step
            control.should_training_stop = True
        return control
