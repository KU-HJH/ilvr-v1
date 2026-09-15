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
