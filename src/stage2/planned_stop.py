"""Optional fresh-run checkpoint boundary for a same-budget DIL resume probe."""
from transformers import TrainerCallback


def validate_planned_checkpoint_stop(step, *, config, max_steps, resume=None, stage='stage2'):
    if step is None:
        return
    if type(step) is not int or step <= 0:
        raise ValueError('stage2_stop_after_checkpoint must be a positive integer')
    if stage != 'stage2' or not (config.is_context_preference and config.is_stage2b
            and config.preference_loss_type in ('dil_bce_rf', 'dil_bce_rf_topk', 'dil_bce_rf_topk_reverse')):
        raise ValueError('stage2_stop_after_checkpoint requires enabled context-preference DIL Stage2B')
    if resume:
        raise ValueError('stage2_stop_after_checkpoint is fresh-run only; omit it when resuming')
    if type(max_steps) is not int or not step < max_steps:
        raise ValueError('stage2_stop_after_checkpoint must be below resolved max_steps')


class PlannedCheckpointStop(TrainerCallback):
    """Request a full save, then stop only after Trainer's save has returned."""

    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        # Stage2Trainer._save_checkpoint publishes runtime/sampler sidecars and
        # completes its rank barriers before Trainer invokes this callback.
        if state.global_step == self.step:
            control.should_training_stop = True
        return control
