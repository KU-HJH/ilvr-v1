"""Auxiliary mapping must neither search ahead nor alter ordinary training input."""
from dataclasses import replace
import importlib

import pytest
import torch

from .v3_helpers import SourceProcessor, source_collator, source_example


def mapping_api():
    return importlib.import_module('stage1.v3_metadata')


def test_immediate_item_mapping_preserves_original_tensors():
    processor = SourceProcessor()
    examples = [source_example(['same text 한국어', None, 'same text 한국어', None, 'same text 한국어']),
                source_example(['short', None, 'x'], source_index=18)]
    off = source_collator(processor=processor)(examples)
    on = source_collator(enabled=True, processor=processor)(examples)
    for key, value in off.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, on[key]), key
    assert off['stage1_batch_metadata'].v3 is None
    blocks = on['stage1_batch_metadata'].v3.blocks
    assert [(b.batch_index, b.source_item_index) for b in blocks] == [(0, 2), (0, 4), (1, 2)]
    assert [processor.tokenizer.decode(b.target_ids) for b in blocks] == [
        'same text 한국어', 'same text 한국어', 'x']
    assert processor.tokenizer.decode(on['input_ids'][0, list(blocks[0].candidate_positions)].tolist()) == 'same text 한국어'
    for block in blocks:
        assert block.query_positions == tuple(t - 1 for t in block.target_positions)
        assert block.query_positions[0] == block.end
        assert block.target_ids == tuple(on['input_ids'][block.batch_index, list(block.target_positions)].tolist())
        assert all(p < block.start for p in block.candidate_positions)
        assert not set(block.candidate_positions) & set((on['input_ids'][block.batch_index] == 96).nonzero().flatten().tolist())
    assert blocks[0].target_positions[-1] < blocks[1].start
    assert set(blocks[0].target_positions) <= set(blocks[1].candidate_positions)


@pytest.mark.parametrize('items,source_indices', [
    (['pre', None, '', 'later'], []),
    (['pre', None, None, 'later'], [3]),
    (['pre', None], []),
    (['no blocks'], []),
    (['pre', None, '<|im_end|>'], []),
])
def test_only_the_immediate_text_item_is_eligible(items, source_indices):
    batch = source_collator(enabled=True)([source_example(items)])
    assert [b.source_item_index for b in batch['stage1_batch_metadata'].v3.blocks] == source_indices


def test_crossing_tokens_are_excluded_from_auxiliary_only():
    processor = SourceProcessor()
    example = source_example(['a', 'b prefix', None, 'a', 'b later', None, 'target'])
    ordinary = source_collator(processor=processor)([example])
    batch = source_collator(enabled=True, processor=processor)([example])
    crossing = (batch['input_ids'][0] == 98).nonzero().flatten().tolist()
    assert len(crossing) == 2
    assert torch.equal(batch['labels'], ordinary['labels'])
    assert all(batch['labels'][0, p].item() == 98 for p in crossing)
    blocks = batch['stage1_batch_metadata'].v3.blocks
    assert [b.source_item_index for b in blocks] == [6]  # 'a' owns no complete token
    assert processor.tokenizer.decode(blocks[0].target_ids) == 'target'
    assert not set(crossing) & set(blocks[0].candidate_positions)


def test_minus100_special_and_padding_are_not_auxiliary_targets():
    processor = SourceProcessor()
    example = source_example(['pre', None, 'x<|im_end|>y'])
    batch = source_collator(processor=processor)([example])
    metadata = batch['stage1_batch_metadata']
    target = metadata.samples[0].latent_spans[0].end + 1
    batch['labels'][0, target] = -100
    result = mapping_api().build_v3_metadata(processor, [example], batch, metadata)
    assert processor.tokenizer.decode(result.blocks[0].target_ids) == 'y'
    batch['labels'].fill_(-100)
    assert mapping_api().build_v3_metadata(processor, [example], batch, metadata).blocks == ()


@pytest.mark.parametrize('damage', ['source', 'roundtrip', 'truncation', 'identity'])
def test_unexplained_mapping_fails_with_source_identity(damage, monkeypatch):
    processor = SourceProcessor()
    example = source_example(['pre', None, 'target'])
    batch = source_collator(processor=processor)([example])
    metadata = batch['stage1_batch_metadata']
    if damage == 'source':
        example.conversation[-1]['content'][-1]['text'] = 'different'
    elif damage == 'identity':
        example = replace(example, source_id='wrong-source')
    elif damage == 'truncation':
        sample = replace(metadata.samples[0], unpadded_length=metadata.samples[0].latent_spans[0].end + 2)
        metadata = replace(metadata, samples=(sample,))
    else:
        decode = processor.tokenizer.decode
        monkeypatch.setattr(processor.tokenizer, 'decode', lambda ids, **kwargs: decode(ids, **kwargs) + 'x')
    with pytest.raises(ValueError, match='source(_index|_id)?[= -].*17'):
        mapping_api().build_v3_metadata(processor, [example], batch, metadata)


@pytest.mark.parametrize('change', [dict(schema_version=2), dict(objective='other'),
    dict(temperature=0.), dict(gain_alpha=float('nan')), dict(text_kl_beta=-1.),
    dict(rho_min=.6), dict(text_cap=2.), dict(image_cap=.1), dict(schema_version=True)])
def test_v3_config_rejects_invalid_or_unimplemented_semantics(change):
    config = importlib.import_module('stage1.v3_config').Stage1V3Config
    with pytest.raises(ValueError):
        config(**change).validate()


def test_v3_config_has_approved_coefficients_and_immutable_identity():
    config = importlib.import_module('stage1.v3_config').Stage1V3Config()
    config.validate()
    assert config.to_dict() == dict(schema_version=1, objective='text_only_gain_v1', text_selector='kl',
        gain_alpha=.001, min_h_norm_sq=1e-30, text_kl_beta=.01, text_kl_warmup_ratio=.05,
        temperature=2., rho_min=.3, rho_max=.5, text_cap=.3, low_mass_threshold=.1,
        low_mass_ratio=.1, image_cap=0.)
    with pytest.raises(AttributeError):
        config.gain_alpha = 1.


@pytest.mark.parametrize('mutable_field', ['targets', 'blocks'])
def test_frozen_metadata_rejects_mutable_sequences(mutable_field):
    api = mapping_api()
    block = api.V3Block(0, 2, 2, 5, (6,), (11,), (1,))
    with pytest.raises(ValueError, match='tuple'):
        if mutable_field == 'targets':
            replace(block, target_ids=[11])
        else:
            api.V3BatchMetadata([block])


def test_byte_level_unicode_offsets_map_complete_source_items():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    from stage1.data import Stage1DataCollator
    from transformers.models.qwen2_5_vl.stage2_latent_rollout import LatentTokenIds

    special = ['<|pad|>', '<|latent_pad|>', '<|latent_start|>', '<|latent_end|>',
        '<|im_start|>', '<|im_end|>', '<|vision_start|>', '<|image_pad|>', '<|vision_end|>']
    vocabulary = {s: i for i, s in enumerate([*special, *sorted(pre_tokenizers.ByteLevel.alphabet()), 'ab'])}
    tokenizer = Tokenizer(models.BPE(vocabulary, [('a', 'b')]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    processor = SourceProcessor()
    processor.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
        pad_token='<|pad|>', additional_special_tokens=special[1:])
    ids = LatentTokenIds(*(processor.tokenizer.convert_tokens_to_ids(s) for s in special[1:4]))
    example = source_example(['한국어🙂', None, '日本語🙂', 'a', 'b'])
    batch = Stage1DataCollator(processor, ids, 3)([example])
    result = mapping_api().build_v3_metadata(processor, [example], batch, batch['stage1_batch_metadata'])
    assert len(result.blocks) == 1
    block = result.blocks[0]
    assert processor.tokenizer.decode(block.target_ids) == '日本語🙂'
    assert processor.tokenizer.decode(batch['input_ids'][0, list(block.candidate_positions)].tolist()) == '한국어🙂'
