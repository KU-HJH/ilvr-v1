"""Comparison workers need fast tokenizer offsets with the saved image processor."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from transformers import AutoProcessor


@pytest.mark.parametrize("worker", ["dataset", "model"])
def test_comparison_worker_preserves_real_tokenizer_offsets(worker, monkeypatch):
    from scripts import verify_voff_reproduction as dataset
    from scripts import verify_voff_model_path as model

    class ProcessorReached(Exception):
        pass

    checkpoint = Path(__file__).resolve().parents[2] / "checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431"
    real_load_processor = AutoProcessor.from_pretrained

    def load_processor(path, **kwargs):
        processor = real_load_processor(path, **kwargs)
        assert processor.tokenizer.is_fast
        assert processor.tokenizer("before after", return_offsets_mapping=True)["offset_mapping"] == [(0, 6), (6, 12)]
        assert type(processor.image_processor).__name__ == "Qwen2VLImageProcessor"
        raise ProcessorReached

    monkeypatch.setattr(AutoProcessor, "from_pretrained", load_processor)
    if worker == "dataset":
        monkeypatch.setattr(dataset, "_parent_scaling_characterization", lambda _: {})
        invoke = lambda: dataset._dataset_worker(SimpleNamespace(worker="production", processor=str(checkpoint)))
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
