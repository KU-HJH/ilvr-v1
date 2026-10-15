"""Local random Qwen/DoRA fixtures; no pretrained downloads or GPU required."""
from dataclasses import dataclass

import torch

from stage1.peft import build_stage1_tuned_model
from stage1.types import (Stage1BatchMetadata, Stage1LatentSpan,
                          Stage1SampleMetadata, Stage1TuningConfig)
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


class HistoryTokenizer:
    pad_token_id = 0
    unk_token_id = 0
    all_special_ids = [0, 7, 31, 32, 33, 80, 81, 82, 83, 125, 126, 127]
    _ids = {'<|latent_pad|>': 125, '<|latent_start|>': 126,
            '<|latent_end|>': 127, '<|vision_start|>': 81,
            '<|vision_end|>': 83}

    def __len__(self):
        return 128

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, 0)

    def __call__(self, text, *, return_tensors='pt', add_special_tokens=False):
        if text == '<|im_start|>assistant':
            ids = [31, 32]
        elif text == '<|im_start|>assistant\n':
            ids = [31, 32, 33]
        else:
            ids = [self._ids[text]]
        return {'input_ids': torch.tensor([ids], dtype=torch.long)}


@dataclass(frozen=True)
class HistoryCase:
    teacher: torch.nn.Module
    tokenizer: HistoryTokenizer
    inputs: dict[str, torch.Tensor]
    metadata: Stage1BatchMetadata


def make_history_case(*, layers: int = 1, dtype=torch.float32,
                      attention_backend: str = 'sdpa', nonzero_dora: bool = True
                      ) -> HistoryCase:
    """Match checkpointing's tiny-original/Stage1 tuning conventions with GQA."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(29)
        config = Qwen2_5_VLConfig(
            vocab_size=125, hidden_size=8, intermediate_size=16,
            num_hidden_layers=layers, num_attention_heads=2,
            num_key_value_heads=1,
            vision_config={
                'depth': 1, 'hidden_size': 8, 'intermediate_size': 16,
                'num_heads': 2, 'patch_size': 2, 'spatial_merge_size': 1,
                'temporal_patch_size': 1, 'window_size': 2,
                'out_hidden_size': 8, 'fullatt_block_indexes': [0],
            },
            image_token_id=80, vision_start_token_id=81, video_token_id=82,
            stage='stage1', latent_size=2,
            attn_implementation=attention_backend,
            rope_scaling={'type': 'mrope', 'mrope_section': [1, 1, 0]},
        )
        config._name_or_path = 'tiny/original-qwen'
        base = Qwen2_5_VLForConditionalGeneration(config)
        base.resize_token_embeddings(128, mean_resizing=False)
        base.config.latent_token_id = base.model.config.latent_token_id = 125
        base.config.latent_start_id = base.model.config.latent_start_id = 126
        base.config.latent_end_id = base.model.config.latent_end_id = 127
        tuning = Stage1TuningConfig(method='dora', rank=4, alpha=8, dropout=.05)
        teacher = build_stage1_tuned_model(base, tuning, (125, 126, 127))
        if nonzero_dora:
            with torch.no_grad():
                for name, parameter in teacher.named_parameters():
                    if 'lora_B' in name:
                        parameter.normal_(mean=.03, std=.1)
                    elif 'lora_magnitude_vector' in name:
                        parameter.mul_(1.1)
        teacher.to(dtype=dtype).eval().requires_grad_(False)
        ids = torch.tensor([[1, 81, 80, 80, 80, 80, 83, 10, 31, 32, 33,
                             20, 21, 126, 125, 125, 127,
                             22, 23, 126, 125, 125, 127, 24, 7]])
        inputs = {
            'input_ids': ids, 'attention_mask': torch.ones_like(ids),
            'pixel_values': torch.randn(4, 12, dtype=dtype),
            'image_grid_thw': torch.tensor([[1, 2, 2]]),
            'pixel_values_latent': torch.randn(8, 12, dtype=dtype),
            'image_grid_thw_latent': torch.tensor([[1, 2, 2], [1, 2, 2]]),
        }
    sample = Stage1SampleMetadata(
        source_index=0, source_id='history-tiny', original_grid_range=(0, 1),
        helper_grid_range=(0, 2),
        latent_spans=(Stage1LatentSpan(13, 16, (14, 15)),
                      Stage1LatentSpan(19, 22, (20, 21))),
        helper_to_block=(0, 1), unpadded_length=25,
        original_pixel_range=(0, 4), helper_pixel_range=(0, 8),
    )
    return HistoryCase(teacher, HistoryTokenizer(), inputs,
                       Stage1BatchMetadata((sample,), 1, 25))


def load_baseline_target_builder(base_commit: str) -> type:
    """Execute an inspected, pinned Git blob in a test-only isolated namespace."""
    import re
    import subprocess
    from pathlib import Path
    from types import ModuleType

    if re.fullmatch(r'[0-9a-f]{40}', base_commit) is None:
        raise ValueError('baseline must be a full lowercase Git commit identity')
    root = Path(__file__).resolve().parents[2]
    source = subprocess.run(
        ['git', 'show', f'{base_commit}:src/stage1/targets.py'], cwd=root,
        check=True, capture_output=True, text=True).stdout
    namespace = ModuleType(f'_stage1_targets_baseline_{base_commit}')
    exec(compile(source, f'git:{base_commit}:src/stage1/targets.py', 'exec'),
         namespace.__dict__)
    return namespace.Stage1TeacherTargetBuilder
