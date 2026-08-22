"""Helper-free records and collation for Stage-2 training."""

import torch
from qwen_vl_utils import process_vision_info

from task_deepseed import interleaved_latent_cot_preprocess_function
from utils_deepseed import (
    generate_labels_with_latent_template,
    place_input_image,
    place_output_image,
    process_batch,
    remove_assistant_images,
    replace_visual_spectial_tokens,
)


def build_stage2_record(sample, row_index, *, validate_user_images: bool = True):
    """Build one Stage-2 row without loading or checking helper-image assets."""
    conversations = interleaved_latent_cot_preprocess_function(
        sample,
        validate_user_images=validate_user_images,
        validate_helper_images=False,
    )
    plan_types = []
    for step in sample["sequence_plan"]:
        step_type = step["type"]
        if step_type == "text" and plan_types and plan_types[-1] == "text":
            continue
        plan_types.append(step_type)
    return {
        "conversations": conversations,
        "ew_sample_index": row_index,
        "ew_plan_types": tuple(plan_types),
    }


def collate_fn_stage2(records, processor, args):
    """Collate Stage-2 rows while decoding only user-provided images."""
    conversations = [record["conversations"] for record in records]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    texts = [place_input_image(text) for text in texts]
    texts = [place_output_image(text) for text in texts]
    texts = replace_visual_spectial_tokens(texts)
    texts = [
        text.replace(
            "<|latent_start|><|image_pad|><|latent_end|>",
            "<|latent_start|><|latent_pad|><|latent_end|>",
        )
        for text in texts
    ]

    user_conversations = remove_assistant_images(conversations)
    image_inputs, _ = process_vision_info(user_conversations)
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0, 0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0, 0].item()
    latent_end_idx = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0, 0].item()
    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0, 0].item()

    new_input_ids, new_attention_mask = process_batch(
        batch["input_ids"],
        batch["attention_mask"],
        start_token=latent_start_idx,
        end_token=latent_end_idx,
        replacement_token=latent_token_idx,
        replacement_length=args.latent_size,
        pad_token=pad_token_idx,
    )
    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask

    answer_start_token_pattern = processor.tokenizer(
        "<|im_start|>assistant", return_tensors="pt"
    )["input_ids"][0]
    batch["labels"] = generate_labels_with_latent_template(
        batch["input_ids"],
        answer_start_token_pattern,
        pad_token_idx,
        int(latent_start_idx),
        int(latent_end_idx),
        int(latent_token_idx),
        latent_ce_ratio=getattr(args, "latent_ce_ratio", 0.0),
    )
    batch["ew_sample_indices"] = torch.tensor(
        [record["ew_sample_index"] for record in records], dtype=torch.long
    )
    batch["ew_plan_types"] = tuple(record["ew_plan_types"] for record in records)
    return batch
