"""The isolated speed experiment must report completed steps without changing control."""
import json
from types import SimpleNamespace


def test_probe_reports_completed_step_and_preserves_control(monkeypatch, capsys):
    from stage2.speed_probe import InsertionSpeedProbe
    import stage2.speed_probe as probe
    clock = iter([10.0, 12.5])
    monkeypatch.setattr(probe.time, 'perf_counter', lambda: next(clock))
    monkeypatch.setattr(probe.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(probe.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(probe.torch.cuda, 'max_memory_allocated', lambda: 100)
    monkeypatch.setattr(probe.torch.cuda, 'max_memory_reserved', lambda: 200)
    monkeypatch.setenv('RANK', '2')
    owner = SimpleNamespace(_representative_gradient_norm=3.0,
        _step_preference_metric_totals={'loss/ce': 1.0}, _step_preference_metric_batches=2)
    callback = InsertionSpeedProbe(owner)
    control=SimpleNamespace(should_training_stop=False)
    assert callback.on_step_begin(None, None, control) is control
    owner._step_preference_metric_totals={'loss/ce': 5.0}
    owner._step_preference_metric_batches=4
    assert callback.on_step_end(None, SimpleNamespace(global_step=1), control) is control
    row=json.loads(capsys.readouterr().out.split('INSERTION_SPEED_PROBE ', 1)[1])
    assert row == {'rank': 2, 'step': 1, 'seconds': 2.5,
                   'peak_allocated_bytes': 100, 'peak_reserved_bytes': 200,
                   'gradient_norm': 3.0, 'metrics': {'loss/ce': 2.0}}
    assert not control.should_training_stop


def test_probe_counts_actual_on_decoder_calls_and_removes_hooks(monkeypatch, capsys):
    import stage2.speed_probe as probe
    from tests.step_preference.insertion_helpers import native_case, preference
    from tests.step_preference.test_v_off_insertion import scorer
    model, ids, base, grid, options, _ = native_case()
    monkeypatch.setattr(probe.torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(probe.torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(probe.torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(probe.torch.cuda, 'max_memory_reserved', lambda: 0)
    monkeypatch.setenv('RANK', '0')
    owner = SimpleNamespace(model=model, _representative_gradient_norm=0.,
        _step_preference_metric_totals={}, _step_preference_metric_batches=0)
    callback = probe.InsertionSpeedProbe(owner)
    callback.on_train_begin(None, None, None)
    callback.on_step_begin(None, None, None)
    values, _ = scorer(model, options, ids, base, grid)
    preference({(r['kind'], r['step_id']): r['mean_logp'] for r in values}).backward()
    callback.on_step_end(None, SimpleNamespace(global_step=1), None)
    row = json.loads(capsys.readouterr().out.split('INSERTION_SPEED_PROBE ', 1)[1])
    # Two ON branches, each with two latent recurrences + a final text forward.
    assert row['decoder_calls'] == {'on': [6, 6], 'clean_or_off': [1, 1]}
    callback.on_train_end(None, None, None)
    assert not any(layer._forward_pre_hooks for layer in model.model.layers)
