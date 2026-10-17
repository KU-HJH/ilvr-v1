"""Real Qwen pixel preprocessing and a frozen three-call collation oracle."""
from dataclasses import replace

from PIL import Image
import torch
from transformers import Qwen2VLImageProcessor

from stage1.data import PreparedStage1Example, Stage1DataCollator
from stage1.types import Stage1SampleOccurrence
from tests.stage1.helpers import TOKENS
from tests.stage1.v5.test_metadata import SourceProcessor
from tests.stage1.v5.frozen_throughput.stage1_data import Stage1DataCollator as OriginalCollator


class PixelProcessor(SourceProcessor):
    """Portable source tokenizer with actual Qwen resize/normalize/patch output."""
    def __init__(self):
        super().__init__()
        self.image_processor = Qwen2VLImageProcessor(min_pixels=3136, max_pixels=12845056)

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        flat = [image for group in images for image in group] if images else []
        vision = dict(self.image_processor(images=flat, return_tensors='pt')) if flat else {}
        grids = iter(vision.get('image_grid_thw', ()))
        rows = []
        for rendered in text:
            pieces = rendered.split('<|image_pad|>')
            expanded = pieces[0]
            for piece in pieces[1:]:
                # Qwen merges four raw patch rows into each language image token.
                expanded += '<|image_pad|>' * (int(next(grids).prod()) // 4) + piece
            rows.append(self.tokenizer(expanded)['input_ids'][0])
        ids = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=0)
        return dict(input_ids=ids, attention_mask=ids.ne(0).long(), **vision)


class CountingProcessor:
    def __init__(self, processor):
        self.processor = processor
        self.calls = 0
        self.image_counts = []

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, **kwargs):
        self.calls += 1
        self.image_counts.append(sum(len(group) for group in kwargs['images']) if kwargs['images'] else 0)
        return self.processor(**kwargs)


def examples_at(root, *, scale=1):
    """Two originals/two helpers, a no-helper row, and one helper; unequal grids."""
    root.mkdir(parents=True, exist_ok=True)
    examples = []
    for index, (original_sizes, helper_sizes) in enumerate([
        (((56, 84), (112, 56)), ((84, 112), (140, 56))),
        (((84, 56),), ()),
        (((56, 112),), ((112, 84),)),
    ]):
        paths = []
        for image_index, size in enumerate((*original_sizes, *helper_sizes)):
            size = tuple(dimension * scale for dimension in size)
            path = root / f'{index}-{image_index}.png'
            # Nonuniform image catches patch reordering as well as role ownership.
            image = Image.new('RGB', size, (30 + index * 70, 40 + image_index * 35, 90))
            image.paste((220, 12, 35), (7, 3, size[0] // 2, size[1] // 2))
            image.save(path)
            paths.append(str(path))
        originals = tuple(paths[:len(original_sizes)])
        helpers = tuple(paths[len(original_sizes):])
        assistant = [dict(type='text', text='reasoning 한국어 ')]
        for helper in helpers:
            assistant.extend([dict(type='image', image=helper), dict(type='text', text='target 日本語🙂')])
        examples.append(PreparedStage1Example(index, f'source-{index}', [
            dict(role='user', content=[*[dict(type='image', image=p) for p in originals],
                                      dict(type='text', text='question')]),
            dict(role='assistant', content=assistant)], originals, helpers))
    return examples


def with_occurrences(examples):
    return [replace(example, occurrence=Stage1SampleOccurrence(
        epoch=0, view_slot=example.source_index, permutation_offset=index,
        canonical_slot_id=example.source_index, source_index=example.source_index,
        rank=index, rank_local_sample_ordinal=0, rank_local_microbatch_ordinal=0,
        row_ordinal_within_microbatch=0)) for index, example in enumerate(examples)]


def assert_batches_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert actual[key].dtype == value.dtype, key
            assert actual[key].shape == value.shape, key
            assert torch.equal(actual[key], value), key
        else:
            assert actual[key] == value, key


def make_collator(processor, *, original=False, variant='v5', token_ids=TOKENS):
    return (OriginalCollator if original else Stage1DataCollator)(
        processor, token_ids, 3, stage1_variant=variant)
