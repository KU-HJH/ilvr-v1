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


def replay_qwen(dtype, p):
    """Two-layer real Qwen decoder, with a stable dropout at each Q projection."""
    from copy import deepcopy
    import torch
    from torch import nn
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    from tests.stage1.test_peft import tiny_qwen
    from stage1.equivalence import SampleStableStage1Dropout
    base = tiny_qwen.__wrapped__()
    config = deepcopy(base.config)
    config.num_hidden_layers = 2
    config._attn_implementation = 'sdpa'
    decoder = Qwen2_5_VLModel(config).to(dtype)
    sites = []
    for i,layer in enumerate(decoder.layers):
        site = f'model.layers.{i}.self_attn.q_proj.lora_dropout.default'
        layer.self_attn.q_proj = nn.Sequential(SampleStableStage1Dropout(site,p),layer.self_attn.q_proj)
        sites.append(site)
    return decoder, base.lm_head.to(dtype), tuple(sites)


def product_v3_trainer(tmp_path,*,checkpointed=False,empty=False,image=False):
    from dataclasses import replace
    import torch
    from transformers import Trainer,TrainingArguments
    from tests.stage1.test_peft import tiny_qwen,tuning
    from tests.stage1.test_trainer import make_trainer,make_batch,CountingTargetBuilder
    from stage1.peft import build_stage1_tuned_model
    from stage1.ema import Stage1EmaTeacher
    from stage1.v3_config import Stage1V3Config
    from stage1.v3_metadata import V3Block,V3BatchMetadata
    from stage1.v3_update import V3GainController
    from stage1.equivalence import stage1_checkpoint_context_fn
    from peft_common import resolve_qwen2_5_vl_model
    model=build_stage1_tuned_model(tiny_qwen.__wrapped__(),tuning('dora',dropout=.05),(125,126,127))
    ids=torch.tensor([[1]*12+[125,126,126,127,2,3,4]])
    if image: ids[0,0]=80
    mask=ids.eq(126)
    batch=make_batch(mask)
    batch.update(input_ids=ids,labels=ids.clone(),attention_mask=torch.ones_like(ids),
                 position_ids=torch.arange(19).view(1,1,-1).expand(3,1,-1),
                 pixel_values=torch.arange(12,dtype=torch.float32).view(1,-1)/12 if image else None,
                 image_grid_thw=torch.tensor([[1,1,1]]) if image else None)
    blocks=() if empty else (V3Block(0,1,12,15,(16,18),(2,4),tuple(range(1,12))),)
    batch['stage1_batch_metadata']=replace(batch['stage1_batch_metadata'],v3=V3BatchMetadata(blocks))
    builder=CountingTargetBuilder(torch.full((2,8),.25),mask,((2,),))
    trainer=make_trainer(model,builder,sim_weight=1.,ce_weight=.1)
    Trainer.__init__(trainer,model=model,args=TrainingArguments(output_dir=str(tmp_path),use_cpu=True,
        report_to=[],gradient_accumulation_steps=4,remove_unused_columns=False,seed=42))
    trainer.ema_teacher=Stage1EmaTeacher(model,decay=1.,method='dora')
    trainer.state.global_step=1
    trainer.v3_config=Stage1V3Config()
    trainer.v3_planned_steps=21
    trainer.v3_gain_controller=V3GainController(tuple(p for p in model.parameters() if p.requires_grad),trainer.v3_config)
    if checkpointed:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            'use_reentrant':False,'context_fn':stage1_checkpoint_context_fn})
    return trainer,batch
