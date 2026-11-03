"""CPU-only prefix tests: real model forwards are forbidden."""
import pytest


@pytest.fixture(autouse=True)
def forbid_model_forwards(monkeypatch):
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen

    def forbidden(*args, **kwargs):
        raise AssertionError('real model forward forbidden in prefix_unit')

    for name in ('Qwen2_5_VLForConditionalGeneration', 'Qwen2_5_VLModel',
                 'Qwen2_5_VisionTransformerPretrainedModel'):
        monkeypatch.setattr(getattr(qwen, name), 'forward', forbidden)
