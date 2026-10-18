"""Original ILVR inference, isolated from Stage 2 and continuous diagnostics.

The native sampler/model are mechanical snapshots, not reconstructed rollouts.
Do not normalize their cache, EOS, or incomplete-block behavior.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import logging

import torch
from transformers import LogitsProcessorList

GENERATION_SNAPSHOT = 'transformers/src/transformers/generation/_ilvr_original_utils.py'
MODELING_SNAPSHOT = 'transformers/src/transformers/models/qwen2_5_vl/_ilvr_original_modeling.py'
ORIGINAL_COMMIT = '13a67ed2ee975d6ca56b4e9eba18d68b352b166b'


def original_model_class():
    from transformers.models.qwen2_5_vl._ilvr_original_modeling import Qwen2_5_VLForConditionalGeneration
    return Qwen2_5_VLForConditionalGeneration


from .original_template import LatentTemplateLogitsProcessor


def load_original_model(model_dir, cache_dir, *, load_model=None):
    """Original eval.load_model: whole-model FA2, then whole-model eager."""
    loader = load_model or original_model_class().from_pretrained
    try:
        model = loader(model_dir, device_map='auto', torch_dtype=torch.bfloat16,
                       cache_dir=cache_dir, attn_implementation='flash_attention_2')
    except Exception as error:
        logging.warning('flash_attention_2 initialization failed (%s); falling back to eager attention.',error)
        model = loader(model_dir, device_map='auto', torch_dtype=torch.bfloat16,
                       cache_dir=cache_dir, attn_implementation='eager')
    model.eval()
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    return model


@dataclass(frozen=True)
class OriginalGenerationResult:
    sequences: torch.Tensor
    latent_start_count: int
    latent_block_count: int
    stop_reason: str


def generate_original(model, inputs, generation_kwargs, *, tokenizer=None, static_inference_cache=False):
    """Call the pinned source's real generate/prepare/forward/_sample chain."""
    if type(static_inference_cache) is not bool:
        raise TypeError('static_inference_cache must be a boolean')
    from peft_common import resolve_qwen2_5_vl_model
    qwen=resolve_qwen2_5_vl_model(model)
    if not isinstance(qwen, original_model_class()):
        raise TypeError('original protocol requires the pinned original ILVR model')
    # Original source hardcodes these three IDs in _sample. Reject incompatible vocabularies.
    start, pad, end = 151666, 151665, 151667
    if tokenizer is not None:
        ids = [int(tokenizer(token,return_tensors='pt')['input_ids'][0,0])
               for token in ('<|latent_start|>','<|latent_pad|>','<|latent_end|>')]
        if ids != [start,pad,end]:
            raise ValueError('original ILVR sampler requires original latent token IDs')
    kwargs = dict(generation_kwargs)
    # These are continuous-controller options, never native generation kwargs.
    kwargs.pop('max_sequence_length',None)
    kwargs.pop('generation_limit_policy',None)
    generator=model if callable(getattr(model,'generate',None)) else qwen
    from .static_inference import static_inference_cache as static_cache
    with static_cache(model) if static_inference_cache else nullcontext():
        sequences = generator.generate(**inputs, **kwargs, tokenizer=tokenizer,
            logits_processor=LogitsProcessorList([LatentTemplateLogitsProcessor(
                start,end,pad,int(getattr(qwen.config,'latent_size',8)))]))
    if not torch.is_tensor(sequences):
        sequences = sequences.sequences
    suffix = sequences[0,inputs['input_ids'].shape[1]:].tolist()
    eos = kwargs.get('eos_token_id',qwen.generation_config.eos_token_id)
    eos = [eos] if isinstance(eos,int) else eos or []
    return OriginalGenerationResult(sequences,suffix.count(start),suffix.count(end),
        'eos' if suffix and suffix[-1] in eos else 'max_new_tokens')


def execution_identity(model, protocol, generation_kwargs):
    """Resolved native configuration and concrete classes, independent of mirror paths."""
    kwargs=dict(generation_kwargs)
    kwargs.pop('max_sequence_length',None)
    kwargs.pop('generation_limit_policy',None)
    if protocol=='original':
        resolved,_=model._prepare_generation_config(None,**kwargs)
        generation=resolved.to_dict()
    else:
        # Continuous controllers own their config; retain their requested fields plus native defaults.
        native=getattr(model,'generation_config',None)
        generation={'controller':kwargs,'model_generation_config':native.to_dict() if hasattr(native,'to_dict') else vars(native) if native else {}}
    config=model.config.to_dict() if hasattr(model.config,'to_dict') else vars(model.config).copy()
    config.pop('_name_or_path',None)
    return dict(evaluation_protocol=protocol,source_commit=ORIGINAL_COMMIT if protocol=='original' else None,
        generation_config=generation,model_config=config,
        attention_backend=dict(language=[type(layer.self_attn).__name__ for layer in model.model.layers],
                               vision=[type(block.attn).__name__ for block in model.visual.blocks]))


def config_resolution(model_dir, base_model_dir, model):
    """Record original adapter-to-base resolution without treating a saved training flag as residue."""
    import json
    from pathlib import Path
    artifact=Path(model_dir)
    peft=(artifact/'adapter_config.json').is_file()
    source=Path(base_model_dir) if peft and base_model_dir else artifact
    def read(path):
        return json.loads(path.read_text()) if path.is_file() else None
    return dict(rule='original adapter_config redirects to base before config/generation_config loading' if peft else 'original full-model from_pretrained',
        artifact_config=read(artifact/'config.json'),source=str(source),
        source_config=read(source/'config.json'),source_generation_config=read(source/'generation_config.json'),
        loaded_model_use_cache=model.config.use_cache,
        loaded_generation_config=model.generation_config.to_dict(),
        representation='unmerged PEFT plus sparse input/output rows' if peft else 'full model')
