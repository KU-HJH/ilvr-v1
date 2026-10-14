"""Lossless source/processor fixtures; no downloaded tokenizer or model needed."""
from PIL import Image
import torch

from stage1.data import PreparedStage1Example, Stage1DataCollator
from tests.stage1.helpers import TOKENS


class SourceTokenizer:
    """Character vocabulary with one deliberately boundary-crossing merge, 'ab'."""
    def __init__(self):
        self.vocabulary = {'<|pad|>': 0, '<|latent_pad|>': 90,
            '<|latent_start|>': 91, '<|latent_end|>': 92, '<|im_start|>': 93,
            '<|im_end|>': 94, '<|vision_start|>': 95, '<|image_pad|>': 96,
            '<|vision_end|>': 97, 'ab': 98}
        self.all_special_ids = tuple(range(90, 98)) + (0,)
        self.pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.vocabulary[token]

    def decode(self, ids, **kwargs):
        inverse = {value: key for key, value in self.vocabulary.items()}
        return ''.join(inverse[int(value)] for value in ids)

    def __call__(self, text, *, add_special_tokens=False, return_tensors='pt',
                 return_offsets_mapping=False):
        ids, offsets, position = [], [], 0
        pieces = sorted(self.vocabulary, key=len, reverse=True)
        while position < len(text):
            piece = next((p for p in pieces if text.startswith(p, position)), text[position])
            if piece not in self.vocabulary:
                self.vocabulary[piece] = max(self.vocabulary.values()) + 1
            ids.append(self.vocabulary[piece])
            offsets.append((position, position + len(piece)))
            position += len(piece)
        result = {'input_ids': torch.tensor([ids], dtype=torch.long)}
        if return_offsets_mapping:
            result['offset_mapping'] = torch.tensor([offsets], dtype=torch.long)
        return result


class SourceProcessor:
    """Render items in order and expand each image to four input image tokens."""
    def __init__(self):
        self.tokenizer = SourceTokenizer()

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt=False):
        return ''.join('<|im_start|>' + turn['role'] + '\n' + ''.join(
            '<|vision_start|><|image_pad|><|vision_end|>' if item['type'] == 'image'
            else item['text'] for item in turn['content']) + '<|im_end|>\n'
            for turn in conversation)

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        rows = [self.tokenizer(t.replace('<|image_pad|>', '<|image_pad|>' * 4), return_tensors='pt', add_special_tokens=False)['input_ids'][0]
                for t in text]
        ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
        count = sum(len(group) for group in images) if images else 0
        assert sum(t.count('<|image_pad|>') for t in text) == count
        return dict(input_ids=ids, attention_mask=ids.ne(0).long(),
            pixel_values=torch.arange(count * 4, dtype=torch.float32).reshape(-1, 1),
            image_grid_thw=torch.tensor([[1, 2, 2]] * count, dtype=torch.long).reshape(-1, 3))


def source_example(items, *, source_index=17):
    """None denotes a helper image; strings remain separate text items."""
    original = Image.new('RGB', (56, 56), 'blue')
    helpers = tuple(Image.new('RGB', (56, 56), 'red') for item in items if item is None)
    helper_iter = iter(helpers)
    content = [dict(type='image', image=next(helper_iter)) if item is None
               else dict(type='text', text=item) for item in items]
    return PreparedStage1Example(source_index, f'source-{source_index}', [
        dict(role='user', content=[dict(type='image', image=original),
                                   dict(type='text', text='same text 한국어')]),
        dict(role='assistant', content=content)], (original,), helpers)


def source_collator(*, enabled=False, processor=None):
    kwargs = {'stage1_variant': 'v5'} if enabled else {}
    return Stage1DataCollator(processor or SourceProcessor(), TOKENS, latent_size=3, **kwargs)


from dataclasses import replace
import importlib
import pytest


def mapping_api():
    return importlib.import_module('stage1.v5_metadata')


def test_original_blocks_include_empty_and_adjacent_items_without_changing_tensors():
    processor = SourceProcessor()
    examples = [source_example(['same text 한국어', None, 'same text 한국어',
        None, '', 'later', None, 'a', 'b later']),
        source_example(['short'], source_index=18)]
    standard_batch = source_collator(processor=processor)(examples)
    v5_batch = source_collator(enabled=True, processor=processor)(examples)
    assert torch.equal(v5_batch['input_ids'], standard_batch['input_ids'])
    assert torch.equal(v5_batch['labels'], standard_batch['labels'])
    for key, value in standard_batch.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, v5_batch[key]), key
    assert v5_batch['stage1_batch_metadata'] == standard_batch['stage1_batch_metadata']
    assert 'stage1_v5_metadata' not in standard_batch
    metadata = v5_batch['stage1_v5_metadata']
    assert metadata.original_step_count == 3
    assert metadata.blocks[1].target_positions == ()
    assert metadata.blocks[2].target_positions == ()  # 'ab' crosses source items
    assert metadata.blocks[0].query_positions[0] == metadata.blocks[0].end
    assert processor.tokenizer.decode(metadata.blocks[0].target_ids) == 'same text 한국어'
    assert processor.tokenizer.decode(v5_batch['input_ids'][0, list(metadata.blocks[0].candidate_positions)].tolist()) == 'same text 한국어'
    assert metadata.query_keys == tuple((0, p) for p in metadata.blocks[0].query_positions)
    assert metadata.target_ids == metadata.blocks[0].target_ids
    assert [b.source_item_index for b in metadata.blocks] == [2, 4, 7]
    crossing = set(v5_batch['input_ids'][0].eq(98).nonzero().flatten().tolist())
    assert crossing
    assert all(not crossing.intersection(b.candidate_positions + b.target_positions) for b in metadata.blocks)
    assert all(v5_batch['labels'][0, p] == 98 for p in crossing)


@pytest.mark.parametrize('items,indices,texts', [
    (['pre', None, '', 'later'], [2], ['']),
    (['pre', None, None, 'later'], [2, 3], ['', 'later']),
    (['pre', None], [2], ['']),
    (['no blocks'], [], []),
    (['pre', None, '<|im_end|>'], [2], ['']),
])
def test_immediate_item_never_searches_ahead_and_no_helpers_have_no_blocks(items, indices, texts):
    processor = SourceProcessor()
    batch = source_collator(enabled=True, processor=processor)([source_example(items)])
    result = batch['stage1_v5_metadata']
    assert [b.source_item_index for b in result.blocks] == indices
    assert [processor.tokenizer.decode(b.target_ids) for b in result.blocks] == texts
    assert result.original_step_count == len(indices)


def test_masked_special_and_padding_tokens_never_become_targets():
    processor = SourceProcessor()
    examples = [source_example(['pre', None, 'x<|im_end|>y']), source_example(['longer ' * 20], source_index=18)]
    batch = source_collator(processor=processor)(examples)
    metadata = batch['stage1_batch_metadata']
    assert metadata.samples[0].unpadded_length < metadata.sequence_length
    batch['labels'][0, metadata.samples[0].latent_spans[0].end + 1] = -100
    result = mapping_api().build_v5_metadata(processor, examples, batch, metadata)
    assert processor.tokenizer.decode(result.target_ids) == 'y'
    batch['labels'].fill_(-100)
    result = mapping_api().build_v5_metadata(processor, examples, batch, metadata)
    assert result.original_step_count == 1 and result.blocks[0].target_positions == ()


@pytest.mark.parametrize('damage', ['source', 'roundtrip', 'truncation', 'identity', 'padding'])
def test_mapping_rejects_mismatched_source_offsets_and_padding(damage, monkeypatch):
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
    elif damage == 'padding':
        batch['attention_mask'][0, 0] = 0
    else:
        decode = processor.tokenizer.decode
        monkeypatch.setattr(processor.tokenizer, 'decode', lambda ids, **kwargs: decode(ids, **kwargs) + 'x')
    with pytest.raises(ValueError, match='source_index=17'):
        mapping_api().build_v5_metadata(processor, [example], batch, metadata)


def test_metadata_accepts_empty_steps_but_rejects_mutability_or_duplicate_query_ownership():
    api = mapping_api()
    block = api.V5Block(0, 2, 2, 5, (6,), (11,), (1,))
    assert api.V5BatchMetadata((replace(block, target_positions=(), target_ids=()),)).original_step_count == 1
    for change in ({'target_ids': [11]}, {'target_positions': [6]},
                   {'candidate_positions': (2, 1)}, {'target_positions': (5,)}):
        with pytest.raises(ValueError):
            replace(block, **change)
    with pytest.raises(ValueError):
        api.V5BatchMetadata([block])
    with pytest.raises(ValueError):
        api.V5BatchMetadata((block, block))


def test_byte_level_unicode_offsets_preserve_complete_source_items():
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
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
    batch = Stage1DataCollator(processor, ids, 3, stage1_variant='v5')([example])
    metadata = batch['stage1_v5_metadata']
    assert processor.tokenizer.decode(metadata.target_ids) == '日本語🙂'
    assert processor.tokenizer.decode(batch['input_ids'][0, list(metadata.blocks[0].candidate_positions)].tolist()) == '한국어🙂'
