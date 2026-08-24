import os
import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from functools import partial

import torch
from torch.nn import functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoConfig,
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
from stage2.checkpointing import (
    configure_stage2_metadata,
    resolve_latent_token_ids,
    resolve_stage2_load_decision,
)
from stage2.data import (
    Stage2DataCollator,
    build_prepared_example_length_measure,
    filter_stage2_examples,
    load_jsonl_examples,
    prepare_stage2_example,
)
from stage2.trainer import Stage2Trainer


_STAGE2_GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}


@dataclass(frozen=True)
class TrainingPath:
    trainer_class: type
    collator_factory: object


def select_training_path(stage: str) -> TrainingPath:
    if stage == "stage1":
        return TrainingPath(CustomTrainerStage1, collate_fn_stage1)
    if stage == "stage2":
        return TrainingPath(Stage2Trainer, Stage2DataCollator)
    raise ValueError(f"Unsupported training stage: {stage}")


@dataclass(frozen=True)
class Stage2TrainingComponents:
    model: object
    processor: object
    train_dataset: object
    data_collator: Stage2DataCollator
    trainer_class: type[Stage2Trainer]
    trainer_resume_path: str | None


def configure_stage2_model_backend(model, *, gradient_checkpointing: bool) -> None:
    """Apply Stage 2 cache/checkpoint settings outside the latent recurrence."""
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=dict(_STAGE2_GRADIENT_CHECKPOINTING_KWARGS)
        )
    else:
        model.gradient_checkpointing_disable()


def validate_stage2_distributed_settings(
    per_device_train_batch_size: int,
) -> None:
    """Enforce the rank-local batch contract used by latent rollout."""
    if per_device_train_batch_size != 1:
        raise ValueError(
            "Stage 2 train_micro_batch_size_per_gpu must resolve to 1"
        )


def build_stage2_training_components(args) -> Stage2TrainingComponents:
    decision = resolve_stage2_load_decision(
        load_model_path=args.load_model_path,
        resume_from_checkpoint_path=args.resume_from_checkpoint_path,
        load_config=AutoConfig.from_pretrained,
        latent_size=args.latent_size,
        supervise_latent_boundaries=args.supervise_latent_boundaries,
    )
    processor = AutoProcessor.from_pretrained(decision.processor_source)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        decision.model_source,
        attn_implementation=args.attention_implementation,
    )
    token_ids = resolve_latent_token_ids(processor.tokenizer, model.config)
    configure_stage2_metadata(
        model.config,
        token_ids,
        latent_size=decision.latent_size,
        supervise_latent_boundaries=decision.supervise_latent_boundaries,
    )
    configure_stage2_model_backend(
        model,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.visual.requires_grad_(False)

    raw_examples = load_jsonl_examples(Path(args.data_path))
    selected_examples = (
        raw_examples
        if args.max_train_samples is None
        else raw_examples[: args.max_train_samples]
    )
    prepared = [
        prepare_stage2_example(
            item,
            image_root=Path(args.image_root),
            latent_size=decision.latent_size,
        )
        for item in selected_examples
    ]
    kept, _length_stats = filter_stage2_examples(
        examples=prepared,
        measure_length=build_prepared_example_length_measure(processor),
        max_length=args.max_seq_length_train,
    )
    collator = Stage2DataCollator(
        processor,
        token_ids,
        decision.latent_size,
        decision.supervise_latent_boundaries,
    )
    return Stage2TrainingComponents(
        model=model,
        processor=processor,
        train_dataset=kept,
        data_collator=collator,
        trainer_class=Stage2Trainer,
        trainer_resume_path=decision.trainer_resume_path,
    )


def build_stage2_training_arguments(args) -> SFTConfig:
    validate_stage2_distributed_settings(args.per_device_train_batch_size)
    return SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warm_up_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        optim="adamw_torch",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_length_train,
        max_steps=args.max_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            dict(_STAGE2_GRADIENT_CHECKPOINTING_KWARGS)
            if args.gradient_checkpointing
            else None
        ),
        deepspeed=args.deepspeed_config,
        ddp_find_unused_parameters=False,
    )

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

    

# ==============================================================
# Main Training Function
# ==============================================================
def main_train():
    seed_everything(seed=42)
    args = get_args()

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

    if args.stage == "stage2":
        components = build_stage2_training_components(args)
        trainer = components.trainer_class(
            model=components.model,
            args=build_stage2_training_arguments(args),
            train_dataset=components.train_dataset,
            data_collator=components.data_collator,
            processing_class=components.processor,
        )
        trainer.train(resume_from_checkpoint=components.trainer_resume_path)
        if args.save_final_model:
            trainer.save_model(args.save_model_path)
            if trainer.is_world_process_zero():
                components.processor.save_pretrained(args.save_model_path)
        return

    cache_dir = args.cache_dir
    os.environ['HF_HOME'] = cache_dir
    
    logging.info(f"Loading processor from: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir, trust_remote_code=True)
    
    new_tokens = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]
    processor.tokenizer.add_tokens(new_tokens, special_tokens=True)


    logging.info(f"Loading model (Stage 1) from: {args.model}")
    model_path = args.model
    config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
    grad_checkpointing = True
    
    
    config.compress_strategy = args.compress_strategy
    config.latent_size = args.latent_size
    config.stage = args.stage

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    config.latent_token_id = int(latent_token_idx)
    config.latent_start_id = int(latent_start_idx)
    config.latent_end_id   = int(latent_end_idx)

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
    

    preprocess_function = task_preporcess_config[args.task]
    train_dataset = load_jsonl_dataset(args.data_path)
    train_dataset = [preprocess_function(sample) for sample in train_dataset]

    
    CustomTrainer = CustomTrainerStage1
    collate_fn_raw = collate_fn_stage1
    
        
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
        deepspeed="configs/config_stage3.json",
        ddp_find_unused_parameters=False if args.stage == 'stage2' else None,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        peft_config=peft_config,
        sim_weight=getattr(args, "sim_weight", 1.0),
        ema_tau=getattr(args, "ema_tau", 0.999),
        coverage_p=getattr(args, "coverage_p", 0.9),
        image_pool_k=getattr(args, "image_pool_k", 8),
        helper_group_L=getattr(args, "helper_group_L", 256),
        ce_weight=getattr(args, "ce_weight", 1.0),
    )
    

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logging.info(f"last checkpoint: {last_checkpoint}，continue。")
        else:
            logging.info(f"output log {training_args.output_dir} exists but no checkpoint， train from start。")
    else:
        logging.info("no checkpoint，train from start。")

    logging.info("start training (DeepSpeed ZeRO-3 Mode)...")
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
