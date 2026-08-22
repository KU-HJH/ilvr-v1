"""Two-rank tiny-Qwen witness for the Stage-2 ZeRO-3 latent call ledger."""

from __future__ import annotations

import math
import tempfile

import deepspeed
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from trl import SFTConfig
from transformers import PreTrainedTokenizerFast

from evidence_window.stage2_config import Stage2EvidenceWindowConfig
from trainer_stage2 import CustomTrainerStage2
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)


def _tiny_config() -> Qwen2_5_VLConfig:
    config = Qwen2_5_VLConfig(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        attention_dropout=0.0,
        use_cache=False,
        pad_token_id=0,
        latent_token_id=9,
        vision_start_token_id=29,
        image_token_id=30,
        video_token_id=31,
        stage="stage2",
        rope_scaling={"rope_type": "default", "mrope_section": [2, 2, 2]},
        vision_config={
            "depth": 1,
            "hidden_size": 24,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "spatial_merge_size": 1,
            "temporal_patch_size": 1,
            "out_hidden_size": 24,
            "fullatt_block_indexes": [0],
        },
    )
    config._attn_implementation = "eager"
    return config


class _Rows(Dataset):
    rows = (
        torch.tensor([1, 9, 2, 3, 4, 5, 6], dtype=torch.long),
        torch.tensor([1, 9, 2, 9, 3, 9, 4], dtype=torch.long),
    )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {"input_ids": self.rows[index]}


def _collate(features):
    input_ids = torch.stack([feature["input_ids"] for feature in features])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def _zero3_config() -> dict:
    return {
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": False},
        "fp16": {"enabled": False},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": False,
        },
        "steps_per_print": 0,
    }


def _tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {"<pad>": 0, "<unk>": 1}
    vocabulary.update({f"t{index}": index for index in range(2, 32)})
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="<unk>")),
        pad_token="<pad>",
        unk_token="<unk>",
    )


def main() -> None:
    try:
        torch.manual_seed(707)
        model = Qwen2_5_VLForConditionalGeneration(_tiny_config())
        for parameter in model.visual.parameters():
            parameter.requires_grad_(False)

        decoder_calls = []
        requested_steps = []
        observed_local_latents = []
        backward_witness = []

        def record_outer(_module, _args, kwargs):
            requested_steps.append(kwargs.get("stage2_latent_prefix_steps"))
            observed_local_latents.append(
                int(kwargs["input_ids"].eq(9).sum().item())
            )

        def record_decoder(_module, _args, kwargs):
            decoder_calls.append(int(kwargs["inputs_embeds"].shape[1]))

        def record_backward(_module, _grad_input, grad_output):
            tensors = [value for value in grad_output if torch.is_tensor(value)]
            backward_witness.append(
                bool(tensors) and all(torch.isfinite(value).all().item() for value in tensors)
            )

        model.register_forward_pre_hook(record_outer, with_kwargs=True)
        model.model.register_forward_pre_hook(record_decoder, with_kwargs=True)
        model.lm_head.register_full_backward_hook(record_backward)
        initial_lm_head = model.lm_head.weight.detach().cpu().clone()

        with tempfile.TemporaryDirectory(prefix="stage2-zero3-ledger-") as output_dir:
            args = SFTConfig(
                output_dir=output_dir,
                max_steps=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1e-2,
                logging_steps=1,
                save_strategy="no",
                report_to=[],
                disable_tqdm=True,
                remove_unused_columns=False,
                dataset_kwargs={"skip_prepare_dataset": True},
                max_seq_length=7,
                use_cpu=False,
                deepspeed=_zero3_config(),
                ddp_find_unused_parameters=False,
            )
            trainer = CustomTrainerStage2(
                model=model,
                args=args,
                train_dataset=_Rows(),
                data_collator=_collate,
                processing_class=_tokenizer(),
                evidence_window_config=Stage2EvidenceWindowConfig.disabled(),
                latent_token_ids={
                    "start": 8,
                    "end": 10,
                    "pad": 9,
                    "padding": 0,
                    "special": (0, 8, 9, 10),
                },
            )
            assert trainer.accelerator.state.deepspeed_plugin.zero_stage == 3
            result = trainer.train()
            assert math.isfinite(result.training_loss)

            unwrapped = trainer.accelerator.unwrap_model(trainer.model)
            with deepspeed.zero.GatheredParameters(
                [unwrapped.lm_head.weight], modifier_rank=None
            ):
                updated_lm_head = unwrapped.lm_head.weight.detach().cpu().clone()

            assert len(observed_local_latents) == 1
            local_latents = observed_local_latents[0]
            record = (
                local_latents,
                len(decoder_calls),
                tuple(requested_steps),
                bool(backward_witness) and all(backward_witness),
                not torch.equal(initial_lm_head, updated_lm_head),
            )
            records = [None] * trainer.accelerator.num_processes
            torch.distributed.all_gather_object(records, record)

            assert {item[0] for item in records} == {1, 3}
            assert len({item[1] for item in records}) == 1
            assert all(item[2] == (3,) for item in records)
            assert all(item[3] for item in records)
            assert all(item[4] for item in records)
            print(
                "STAGE2_ZERO3_LATENT_LEDGER_OK "
                f"rank={trainer.accelerator.process_index} record={record}",
                flush=True,
            )
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
