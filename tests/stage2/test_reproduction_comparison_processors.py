"""Comparison workers must choose the observed slow processor explicitly."""
from types import SimpleNamespace
import pytest
from transformers import AutoProcessor


@pytest.mark.parametrize("worker", ["dataset", "model"])
def test_comparison_worker_pins_slow_local_processor(worker, monkeypatch):
    from scripts import verify_voff_reproduction as dataset
    from scripts import verify_voff_model_path as model

    class ProcessorReached(Exception):
        pass

    def load_processor(path, **kwargs):
        assert kwargs == {"local_files_only": True, "use_fast": False}
        raise ProcessorReached

    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_processor)
    if worker == "dataset":
        monkeypatch.setattr(dataset, "_parent_scaling_characterization", lambda _: {})
        invoke = lambda: dataset._dataset_worker(SimpleNamespace(worker="production", processor="local-fixture"))
    else:
        # Stop at processor loading, before any model or numerical comparison.
        monkeypatch.setattr(model, "inventory", lambda _: {})
        original_import = model.importlib.import_module
        monkeypatch.setattr(model.importlib, "import_module", lambda name, *args, **kwargs: SimpleNamespace()
            if name.startswith(("transformers.models.voff_supplied_qwen.", "voff_supplied_step."))
            else original_import(name, *args, **kwargs))
        # run() registers isolated evidence packages; restore the registry afterward.
        monkeypatch.setitem(model.sys.modules, "transformers.models.voff_supplied_qwen", None)
        monkeypatch.setitem(model.sys.modules, "voff_supplied_step", None)
        invoke = model.run
    with pytest.raises(ProcessorReached):
        invoke()
