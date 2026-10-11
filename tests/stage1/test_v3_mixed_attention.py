"""Causal regression for the helper-image dense SDPA allocation path."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLSdpaAttention, Qwen2_5_VLVisionFlashAttention2,
)


@pytest.mark.gpu
def test_v3_composition_and_frozen_teacher_keep_concrete_mixed_backends(tmp_path,monkeypatch):
    # Regresses parent attn_implementation flattening vision back onto SDPA.
    # No training or CUDA allocation: real tiny module construction and EMA copy.
    import stage1.composition as composition
    import stage1.v3_resume as resume
    from stage1.ema import Stage1EmaTeacher
    from tests.stage1.test_composition import stage1_args,fake_dependencies
    from tests.stage2.test_attention_backend import _tiny_qwen_config
    from deepspeed_runtime import load_deepspeed_runtime_profile

    args=stage1_args(tmp_path,'--stage1_variant','v3')
    deps,processor,_,_=fake_dependencies([])
    config=_tiny_qwen_config();config.vocab_size=len(processor.tokenizer)
    def load_model(source,**kwargs):
        prepared=kwargs.pop('config')
        if 'attn_implementation' in kwargs:
            prepared._attn_implementation=kwargs['attn_implementation']
            prepared.vision_config._attn_implementation=kwargs['attn_implementation']
        return Qwen2_5_VLForConditionalGeneration(prepared).to(torch.bfloat16)
    monkeypatch.setattr(resume,'attach_v3_identity',lambda *a:None)
    monkeypatch.setattr(composition,'build_stage1_training_arguments',lambda args:SimpleNamespace(output_dir=args.save_model_path))
    components=composition.build_stage1_training_components(args,
        dependencies=replace(deps,load_config=lambda *a,**kw:config,load_model=load_model,
            collator_factory=lambda *a,**kw:object()),
        deepspeed_runtime_profile=load_deepspeed_runtime_profile(args.deepspeed_config,world_size=4))
    teacher=Stage1EmaTeacher(components.model,decay=1.,method='full')
    for model in (components.model,teacher.model):
        assert all(type(block.attn) is Qwen2_5_VLVisionFlashAttention2 for block in model.visual.blocks)
        assert all(type(layer.self_attn) is Qwen2_5_VLSdpaAttention for layer in model.model.layers)
        assert model.config.latent_size==8 and model.config.use_cache is False
