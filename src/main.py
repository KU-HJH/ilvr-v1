import os
import re
import json
import logging
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
from torch.nn import functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from qwen_vl_utils import process_vision_info

from utils_deepseed import *
from task_deepseed import *
from trainer import CustomTrainerStage1
from evidence_window.config import EvidenceWindowConfig
from evidence_window.stage2_config import (
    Stage2EvidenceWindowConfig,
    read_deepspeed_zero_stage,
)
from evidence_window.artifacts import ArtifactIdentity
from stage2_data import build_stage2_record, collate_fn_stage2
from trainer_stage2 import CustomTrainerStage2

# ==============================================================
# Collate Functions
# ==============================================================

def collate_fn_stage1(examples, processor, args):
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(t) for t in texts]
    texts = [place_output_image(t) for t in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    if 'pixel_values' in user_batch:
        batch['pixel_values'] = user_batch['pixel_values']
        batch['image_grid_thw'] = user_batch['image_grid_thw']
    
    if 'pixel_values' in assistant_batch:
        batch['pixel_values_latent'] = assistant_batch['pixel_values']
        batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    pad_token_idx    = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0,0].item()

    new_input_ids, new_attention_mask = process_batch(
        batch["input_ids"], batch["attention_mask"],
        start_token=latent_start_idx, end_token=latent_end_idx,
        replacement_token=latent_token_idx, replacement_length=args.latent_size,
        pad_token=pad_token_idx
    )
    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask
    
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_with_latent_template(
        batch["input_ids"],
        answer_start_token_pattern,
        pad_token_idx,
        int(latent_start_idx),
        int(latent_end_idx),
        int(latent_token_idx),
        latent_ce_ratio=0,
    )
    batch["labels"] = labels

    image_out_mask = mask_latent_output_tokens_all_segments(
        batch["input_ids"], latent_start_idx, latent_end_idx, latent_token_idx
    )
    batch["image_out_mask"] = image_out_mask
    return batch


def select_stage_components(stage):
    """Return the trainer and collator belonging to one ILVR stage."""
    if stage == "stage1":
        return CustomTrainerStage1, collate_fn_stage1
    if stage == "stage2":
        return CustomTrainerStage2, collate_fn_stage2
    raise ValueError("stage must be stage1 or stage2")


def resolve_policy_model_path(args):
    """Resolve the base model for Stage 1 or required Stage-1 checkpoint for Stage 2."""
    if args.stage == "stage1":
        return args.model
    if args.stage != "stage2":
        raise ValueError("stage must be stage1 or stage2")

    checkpoint = getattr(args, "load_model_path", None)
    if not checkpoint or not Path(checkpoint).exists():
        raise ValueError(
            "Stage 2 requires an existing Stage 1 checkpoint via --load_model_path"
        )
    return str(checkpoint)


def resolve_evidence_window_config(args):
    """Parse only the evidence-window configuration version valid for a stage."""
    if args.stage == "stage1":
        config = (
            EvidenceWindowConfig.disabled()
            if args.ew_config is None
            else EvidenceWindowConfig.from_json(args.ew_config)
        )
        if not config.is_disabled:
            raise ValueError(
                "Stage 1 accepts only no EW config or a disabled version-1 EW config"
            )
        return config
    if args.stage == "stage2":
        return (
            Stage2EvidenceWindowConfig.disabled()
            if args.ew_config is None
            else Stage2EvidenceWindowConfig.from_json(args.ew_config)
        )
    raise ValueError("stage must be stage1 or stage2")


def resolve_stage2_reference_identity(
    args,
    ew_config,
    *,
    tokenizer_special_tokens,
):
    """Build the training-side expectation for immutable reference targets."""
    if not isinstance(ew_config, Stage2EvidenceWindowConfig):
        return None
    if ew_config.reference.mode != "precomputed" or ew_config.is_disabled:
        return None
    required = {
        "reference_dataset_sha256": getattr(args, "reference_dataset_sha256", None),
        "reference_checkpoint_id": getattr(args, "reference_checkpoint_id", None),
        "reference_checkpoint_witness": getattr(
            args, "reference_checkpoint_witness", None
        ),
        "reference_tokenizer_id": getattr(args, "reference_tokenizer_id", None),
        "reference_parser_version": getattr(args, "reference_parser_version", None),
        "reference_model_config_id": getattr(args, "reference_model_config_id", None),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "precomputed reference requires explicit identity argument(s): "
            + ", ".join(sorted(missing))
        )
    return ArtifactIdentity(
        format_version=1,
        dataset_sha256=required["reference_dataset_sha256"],
        checkpoint_id=required["reference_checkpoint_id"],
        checkpoint_witness=required["reference_checkpoint_witness"],
        tokenizer_id=required["reference_tokenizer_id"],
        tokenizer_special_tokens=tuple(int(value) for value in tokenizer_special_tokens),
        parser_version=required["reference_parser_version"],
        contribution_reduction=ew_config.preference.contribution_reduction,
        intervention=ew_config.loo.intervention,
        attention_backend="sdpa",
        max_evidence_edges_per_window=ew_config.loo.max_evidence_edges_per_window,
        model_config_id=required["reference_model_config_id"],
    )


def load_stage2_online_reference_model(
    args,
    ew_config,
    *,
    tokenizer_size,
    latent_token_ids,
):
    """Independently load the fixed checkpoint under Stage-2 model semantics.

    The returned model is intentionally not derived from the policy object.
    ``OnlineFrozenStage2Reference.create`` freezes and prepares it for ZeRO-1/2.
    """
    if not isinstance(ew_config, Stage2EvidenceWindowConfig):
        raise TypeError("ew_config must be a Stage2EvidenceWindowConfig")
    if ew_config.reference.mode != "online_frozen" or ew_config.is_disabled:
        raise ValueError("online reference loading requires enabled online_frozen mode")
    checkpoint = ew_config.reference.checkpoint
    assert checkpoint is not None
    reference_config = Qwen2_5_VLConfig.from_pretrained(
        checkpoint,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    reference_config.compress_strategy = args.compress_strategy
    reference_config.latent_size = args.latent_size
    reference_config.stage = "stage2"
    reference_config.latent_token_id = int(latent_token_ids["pad"])
    reference_config.latent_start_id = int(latent_token_ids["start"])
    reference_config.latent_end_id = int(latent_token_ids["end"])
    reference_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint,
        config=reference_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    reference_model.resize_token_embeddings(tokenizer_size)
    return reference_model


def prepare_training_dataset(args, raw_rows):
    """Apply the stage-specific preprocessing contract to raw training rows."""
    if args.stage == "stage1":
        preprocess_function = task_preporcess_config[args.task]
        return [preprocess_function(sample) for sample in raw_rows]
    if args.stage == "stage2":
        return [
            build_stage2_record(sample, row_index)
            for row_index, sample in enumerate(raw_rows)
        ]
    raise ValueError("stage must be stage1 or stage2")

    

# ==============================================================
# Main Training Function
# ==============================================================
def main_train():
    seed_everything(seed=42)
    args = get_args()
    CustomTrainer, collate_fn_raw = select_stage_components(args.stage)
    model_path = resolve_policy_model_path(args)
    ew_config = resolve_evidence_window_config(args)
    stage2_zero_stage = None
    if args.stage == "stage2":
        stage2_zero_stage = read_deepspeed_zero_stage(args.deepspeed_config)
        # This runs before processor, policy, or optional reference allocation.
        ew_config.validate_runtime(stage2_zero_stage)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ],
    )
    logging.info('==' * 20)
    logging.info(args)
    logging.info('==' * 20)

    cache_dir = args.cache_dir
    os.environ['HF_HOME'] = cache_dir
    
    logging.info(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
    
    new_tokens = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]
    processor.tokenizer.add_tokens(new_tokens, special_tokens=True)


    logging.info(f"Loading model ({args.stage}) from: {model_path}")
    config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
    grad_checkpointing = True
    
    
    config.compress_strategy = args.compress_strategy
    config.latent_size = args.latent_size
    config.stage = args.stage

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    assistant_termination_idx = processor.tokenizer(
        "<|im_end|>", return_tensors="pt"
    )["input_ids"][0,0].item()
    pad_token_idx = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_token_ids = {
        "start": int(latent_start_idx),
        "end": int(latent_end_idx),
        "pad": int(latent_token_idx),
        "padding": int(pad_token_idx),
        "special": tuple(int(x) for x in processor.tokenizer.all_special_ids),
        "assistant_termination": (int(assistant_termination_idx),),
    }
    reference_identity = resolve_stage2_reference_identity(
        args,
        ew_config,
        tokenizer_special_tokens=processor.tokenizer.all_special_ids,
    )
    config.latent_token_id = int(latent_token_idx)
    config.latent_start_id = int(latent_start_idx)
    config.latent_end_id   = int(latent_end_idx)

    # Materialize the independent fixed checkpoint before SFTTrainer can apply
    # PEFT or other policy mutations. ZeRO compatibility was checked above.
    online_reference_model = None
    if (
        args.stage == "stage2"
        and not ew_config.is_disabled
        and ew_config.reference.mode == "online_frozen"
    ):
        online_reference_model = load_stage2_online_reference_model(
            args,
            ew_config,
            tokenizer_size=len(processor.tokenizer),
            latent_token_ids=latent_token_ids,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir if args.stage == 'stage1' else None,
        trust_remote_code=True
    )
    
    model.resize_token_embeddings(len(processor.tokenizer))

    for param in model.visual.parameters():
        param.requires_grad = False
    
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
            
    logging.info(f"Moving model to CUDA device: {torch.cuda.current_device()} ...")
    

    raw_train_dataset = load_jsonl_dataset(args.data_path)
    train_dataset = prepare_training_dataset(args, raw_train_dataset)
        
    collate_fn = partial(collate_fn_raw, processor=processor, args=args)

    peft_config = None
    if getattr(args, "use_lora", False):
        logging.info("Enabling LoRA...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_args = SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warm_up_steps,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        optim="adamw_torch_fused" if args.stage == 'stage1' else "adamw_torch",
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_checkpointing=grad_checkpointing,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=[],
        logging_dir='./logs/',
        logging_strategy='steps',
        max_seq_length=32768 if args.stage == 'stage1' else args.max_seq_length_train,
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False if args.stage == 'stage2' else None,
    )
    
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        evidence_window_config=ew_config,
    )
    if args.stage == "stage1":
        trainer_kwargs.update(
            tokenizer=processor.tokenizer,
            sim_weight=getattr(args, "sim_weight", 1.0),
            ema_tau=getattr(args, "ema_tau", 0.999),
            coverage_p=getattr(args, "coverage_p", 0.9),
            image_pool_k=getattr(args, "image_pool_k", 8),
            helper_group_L=getattr(args, "helper_group_L", 256),
            ce_weight=getattr(args, "ce_weight", 1.0),
            latent_token_ids=latent_token_ids,
        )
    else:
        trainer_kwargs["processing_class"] = processor.tokenizer
        trainer_kwargs["latent_token_ids"] = latent_token_ids
        if reference_identity is not None:
            trainer_kwargs["reference_identity"] = reference_identity
        if online_reference_model is not None:
            trainer_kwargs["reference_model_factory"] = (
                lambda model=online_reference_model: model
            )
            trainer_kwargs["deepspeed_zero_stage"] = stage2_zero_stage
    trainer = CustomTrainer(**trainer_kwargs)
    

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logging.info(f"last checkpoint: {last_checkpoint}，continue。")
        else:
            logging.info(f"output log {training_args.output_dir} exists but no checkpoint， train from start。")
    else:
        logging.info("no checkpoint，train from start。")

    configured_zero_stage = (
        stage2_zero_stage
        if stage2_zero_stage is not None
        else read_deepspeed_zero_stage(args.deepspeed_config)
    )
    logging.info("start training (DeepSpeed ZeRO-%s Mode)...", configured_zero_stage)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_model_path = training_args.output_dir

    if trainer.is_world_process_zero():
        logging.info(f"training finish，save model: {final_model_path}")
        processor.save_pretrained(final_model_path)

    trainer.save_model(final_model_path)

    if trainer.is_world_process_zero():
        logging.info("all saved。")
        
    logging.info("finish。")

if __name__ == "__main__":
    main_train()
