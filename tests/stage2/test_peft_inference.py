from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from peft_common import resolve_qwen2_5_vl_model
from stage1.token_rows import (
    SparseOutputTokenRows,
    find_sparse_output_token_rows,
)
from stage2.peft import (
    load_stage2_model_for_inference,
    resolve_stage2_qwen_model,
)
from tests.stage2.test_stage1_checkpoint_handoff import (
    tiny_tag_free_original_qwen,
    write_sparse_stage2_artifact,
)


@pytest.fixture(scope="module")
def sparse_inference_artifact(tmp_path_factory):
    root = tmp_path_factory.mktemp("sparse-stage2-inference")
    stage1_dir = root / "stage1"
    stage2_dir = root / "relocated-stage2"
    original_base_dir = root / "original-base"
    processor, _, _, trained = write_sparse_stage2_artifact(
        stage1_dir, stage2_dir
    )
    tiny_tag_free_original_qwen().save_pretrained(original_base_dir)
    config_path = original_base_dir / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload["architectures"] = []
    config_path.write_text(
        json.dumps(config_payload, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(stage1_dir)
    return SimpleNamespace(
        root=root,
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        original_base_dir=original_base_dir,
        processor=processor,
        trained=trained,
    )


def _reload_sparse_artifact(artifact):
    return load_stage2_model_for_inference(
        artifact.original_base_dir,
        artifact.stage2_dir,
        artifact.processor,
    )


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_v3_inference_loads_sparse_rows_without_stage1_directory(
    sparse_inference_artifact,
):
    """Dropping artifact-owned sparse rows must change the latent-token logits."""

    fixed_hidden = torch.linspace(-0.4, 0.9, 16).reshape(2, 8)
    expected_logits = resolve_qwen2_5_vl_model(
        sparse_inference_artifact.trained
    ).lm_head(fixed_hidden).detach()

    reloaded = _reload_sparse_artifact(sparse_inference_artifact)

    actual_logits = resolve_qwen2_5_vl_model(reloaded).lm_head(fixed_hidden)
    torch.testing.assert_close(actual_logits, expected_logits)
    assert not any(parameter.requires_grad for parameter in reloaded.parameters())


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_v3_reload_preserves_two_block_logits_and_latent_states_at_b1(
    sparse_inference_artifact,
):
    """Dropping adapter, row, or recurrence state must break reload parity."""

    source = sparse_inference_artifact.trained.eval()
    reloaded = _reload_sparse_artifact(sparse_inference_artifact)
    scaffold = torch.tensor(
        [[1, 126, 125, 125, 127, 2, 126, 125, 125, 127, 3]]
    )
    attention_mask = torch.ones_like(scaffold)

    with torch.inference_mode():
        expected = source(
            input_ids=scaffold,
            attention_mask=attention_mask,
            return_stage2_latent_states=True,
        )
        actual = reloaded(
            input_ids=scaffold,
            attention_mask=attention_mask,
            return_stage2_latent_states=True,
        )

    torch.testing.assert_close(actual.logits, expected.logits)
    assert len(actual.stage2_latent_states) == len(
        expected.stage2_latent_states
    ) == 4
    for actual_state, expected_state in zip(
        actual.stage2_latent_states,
        expected.stage2_latent_states,
        strict=True,
    ):
        torch.testing.assert_close(actual_state, expected_state)


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_inference_rejects_malformed_adapter_before_large_base_load(
    sparse_inference_artifact, tmp_path
):
    """A corrupt adapter payload must fail before original weights are loaded."""

    artifact_dir = tmp_path / "corrupt-stage2"
    shutil.copytree(sparse_inference_artifact.stage2_dir, artifact_dir)
    (artifact_dir / "adapter_model.safetensors").write_bytes(b"corrupt")
    model_loads = []

    with pytest.raises(ValueError, match="adapter safetensors"):
        load_stage2_model_for_inference(
            sparse_inference_artifact.original_base_dir,
            artifact_dir,
            sparse_inference_artifact.processor,
            load_model=lambda *args, **kwargs: model_loads.append((args, kwargs)),
        )

    assert model_loads == []


def test_inference_source_provenance_rejects_model_only_selectors(
    sparse_inference_artifact,
):
    """Mixing model options into immutable Hub selectors must fail closed."""

    with pytest.raises(ValueError, match="source provenance"):
        load_stage2_model_for_inference(
            sparse_inference_artifact.original_base_dir,
            sparse_inference_artifact.stage2_dir,
            sparse_inference_artifact.processor,
            source_provenance_kwargs={"torch_dtype": torch.bfloat16},
            discover_adapter_config=lambda *args, **kwargs: pytest.fail(
                "adapter discovery must not run after selector rejection"
            ),
        )


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_inference_shares_source_selectors_and_resizes_original_base_once(
    sparse_inference_artifact,
):
    """Divergent source selectors or a second resize must fail this test."""

    source_selectors = {
        "cache_dir": "/cache/models",
        "revision": "tested-revision",
        "token": "tested-token",
        "local_files_only": True,
    }
    model_options = {"attn_implementation": "eager"}
    loaded_base = tiny_tag_free_original_qwen()
    resize_calls = []
    original_resize = loaded_base.resize_token_embeddings
    observed = []

    def record_resize(size, *args, **kwargs):
        resize_calls.append(size)
        return original_resize(size, *args, **kwargs)

    def discover(source, **kwargs):
        observed.append(("discover", source, kwargs))
        return None

    def load_config(source, **kwargs):
        observed.append(("config", source, kwargs))
        return tiny_tag_free_original_qwen().config

    def load_model(source, **kwargs):
        observed.append(("model", source, kwargs))
        return loaded_base

    loaded_base.resize_token_embeddings = record_resize
    load_stage2_model_for_inference(
        sparse_inference_artifact.original_base_dir,
        sparse_inference_artifact.stage2_dir,
        sparse_inference_artifact.processor,
        source_provenance_kwargs=source_selectors,
        model_load_kwargs=model_options,
        discover_adapter_config=discover,
        load_config=load_config,
        load_model=load_model,
    )

    assert [name for name, _, _ in observed] == ["discover", "config", "model"]
    assert all(
        source == str(sparse_inference_artifact.original_base_dir)
        for _, source, _ in observed
    )
    assert observed[0][2] == observed[1][2] == source_selectors
    assert observed[2][2] == {**source_selectors, **model_options}
    assert resize_calls == [128]


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_resolution_supports_direct_and_deepspeed_wrapped_models(
    sparse_inference_artifact,
):
    """Losing the active sparse head through a DeepSpeed wrapper must fail."""

    model = _reload_sparse_artifact(sparse_inference_artifact)
    direct_rows = find_sparse_output_token_rows(model)
    wrapped_rows = find_sparse_output_token_rows(SimpleNamespace(module=model))

    assert wrapped_rows is direct_rows
    assert tuple(direct_rows.token_ids.tolist()) == (125, 126, 127)


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_resolution_rejects_zero_or_multiple_active_adapters(
    sparse_inference_artifact,
):
    """A detached or stacked sparse adapter must never reach generation."""

    model = _reload_sparse_artifact(sparse_inference_artifact)
    qwen = model.get_base_model()
    with pytest.raises(ValueError, match="exactly one active Stage 2 adapter"):
        resolve_stage2_qwen_model(qwen)

    model.peft_config["extra"] = model.peft_config["default"]
    with pytest.raises(ValueError, match="exactly one active Stage 2 adapter"):
        resolve_stage2_qwen_model(model)


@pytest.mark.filterwarnings("ignore:Could not find a config file.*:UserWarning")
def test_sparse_resolution_rejects_malformed_or_duplicate_row_wrappers(
    sparse_inference_artifact,
):
    """Malformed or duplicate sparse state must fail before generation."""

    malformed = _reload_sparse_artifact(sparse_inference_artifact)
    find_sparse_output_token_rows(malformed).token_ids = torch.tensor(
        [125, 125, 127]
    )
    with pytest.raises(ValueError, match="latent token IDs must be unique"):
        resolve_stage2_qwen_model(malformed)

    duplicate = _reload_sparse_artifact(sparse_inference_artifact)
    qwen = duplicate.get_base_model()
    qwen.unexpected_sparse_rows = SparseOutputTokenRows(
        nn.Linear(8, 128, bias=False), (125, 126, 127)
    )
    with pytest.raises(ValueError, match="exactly one sparse output-token row"):
        resolve_stage2_qwen_model(duplicate)
