import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoConfig,
    AutoProcessor,
)
from trl import SFTConfig
from peft import PeftConfig

from qwen_vl_utils import process_vision_info

from utils_deepseed import *
from task_deepseed import *
from stage2.checkpointing import (
    Stage2PeftMetadata,
    build_stage2_peft_metadata,
    configure_stage2_metadata,
    resolve_latent_token_ids,
    resolve_stage2_load_decision,
    validate_stage2_peft_compatibility,
)
from stage2.data import (
    Stage2DataCollator,
    build_prepared_example_length_measure,
    filter_stage2_examples,
    load_jsonl_examples,
    prepare_stage2_example,
)
from stage2.trainer import Stage2Trainer
from stage2.peft import (
    build_stage2_tuned_model,
    resolve_stage2_target_module_names,
    validate_stage2_parameter_boundary,
)


_STAGE2_GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}


@dataclass(frozen=True)
class TrainingPath:
    trainer_class: type
    collator_factory: object


def select_training_path(stage: str) -> TrainingPath:
    if stage == "stage1":
        from stage1.trainer import CustomTrainerStage1

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
    peft_metadata: Stage2PeftMetadata | None


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
        tuning_method=args.tuning_method,
        tuning_method_explicit=args.tuning_method_explicit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )
    processor = AutoProcessor.from_pretrained(decision.processor_source)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        decision.base_model_source,
        attn_implementation=args.attention_implementation,
    )
    token_ids = resolve_latent_token_ids(processor.tokenizer, base_model.config)
    if decision.peft_metadata is not None:
        adapter_config = PeftConfig.from_pretrained(decision.adapter_source)
        validate_stage2_peft_compatibility(
            base_model,
            processor.tokenizer,
            decision.peft_metadata,
            adapter_config,
            decision.adapter_source,
        )
    configure_stage2_metadata(
        base_model.config,
        token_ids,
        latent_size=decision.latent_size,
        supervise_latent_boundaries=decision.supervise_latent_boundaries,
    )
    configure_stage2_model_backend(
        base_model,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    base_model.visual.requires_grad_(False)
    fresh_resolved_target_names = (
        resolve_stage2_target_module_names(
            base_model, decision.tuning.target_modules
        )
        if decision.peft_metadata is None and decision.tuning.is_peft
        else None
    )
    model = build_stage2_tuned_model(
        base_model,
        decision.tuning,
        adapter_path=decision.adapter_source,
        is_trainable=True,
    )
    parameter_report = validate_stage2_parameter_boundary(model, decision.tuning)
    logging.info(
        "Stage 2 parameters: total=%d trainable=%d resolved_targets=%s",
        parameter_report.total_parameters,
        parameter_report.trainable_parameters,
        parameter_report.resolved_target_names,
    )
    peft_metadata = decision.peft_metadata
    if peft_metadata is None and decision.tuning.is_peft:
        peft_metadata = build_stage2_peft_metadata(
            tuning=decision.tuning,
            base_model_reference=decision.base_model_source,
            model_config=base_model.config,
            token_ids=token_ids,
            resolved_target_names=fresh_resolved_target_names,
        )

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
        peft_metadata=peft_metadata,
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

    

def run_stage1(args):
    """Load the Stage 1 runtime only after Stage 1 was selected."""

    from stage1.composition import run_stage1 as run_composed_stage1

    return run_composed_stage1(args)


def run_stage2(args):
    """Run the existing Stage 2 path without importing Stage 1 Trainer code."""

    components = build_stage2_training_components(args)
    trainer = components.trainer_class(
        model=components.model,
        args=build_stage2_training_arguments(args),
        train_dataset=components.train_dataset,
        data_collator=components.data_collator,
        processing_class=components.processor,
        stage2_peft_metadata=components.peft_metadata,
    )
    trainer.train(resume_from_checkpoint=components.trainer_resume_path)
    if args.save_final_model:
        trainer.save_model(args.save_model_path)
        if trainer.is_world_process_zero():
            components.processor.save_pretrained(args.save_model_path)


def route_training(args, *, stage1_handler=None, stage2_handler=None):
    """Select exactly one public stage handler and return its observable result."""

    if args.stage == "stage1":
        return (stage1_handler or run_stage1)(args)
    if args.stage == "stage2":
        return (stage2_handler or run_stage2)(args)
    raise ValueError(f"Unsupported stage: {args.stage}")


# ==============================================================
# Main Training Function
# ==============================================================
def main_train():
    args = get_args()
    seed_everything(seed=42 if args.stage == "stage2" else args.seed)

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

    return route_training(args)


def __getattr__(name):
    """Retain the historical public trainer import without eager Stage 1 load."""

    if name == "CustomTrainerStage1":
        from stage1.trainer import CustomTrainerStage1

        return CustomTrainerStage1
    raise AttributeError(name)

if __name__ == "__main__":
    main_train()
