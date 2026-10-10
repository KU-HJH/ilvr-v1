"""Lossless source/processor fixtures; no downloaded tokenizer or model needed."""
from PIL import Image
import torch

from stage1.data import PreparedStage1Example, Stage1DataCollator
from .helpers import TOKENS


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
    kwargs = {'v3_enabled': True} if enabled else {}
    return Stage1DataCollator(processor or SourceProcessor(), TOKENS, latent_size=3, **kwargs)
