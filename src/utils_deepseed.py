import torch
import logging
import numpy as np
import json
import random
import os
import argparse
import math
import sys
from pathlib import Path
from datasets import Dataset
from transformers import LogitsProcessor
from stage2.peft import normalize_stage2_target_modules
from peft_common import normalize_decoder_target_suffixes
from artifact_paths import resolve_run_artifact_paths
from step_preference.config import StepPreferenceConfig

from transformers import LogitsProcessor
import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_output_path", type=str, default="evaluation_results.json", help="Path to save evaluation results JSON file.")
    parser.add_argument("--model", type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument("--stage", type=str, default="stage1")
    parser.add_argument("--query_mode", choices=("legacy", "adaptive"), default="legacy")
    parser.add_argument("--history_text_gamma", type=float, default=.9)
    parser.add_argument("--history_attention_layers", choices=("last", "last4", "all"), default="last")
    parser.add_argument("--stage1_variant", choices=("standard", "v5"), default="standard")
    parser.add_argument("--stage1_match_v5_execution", action="store_true")
    parser.add_argument("--stage1_ce_entropy_top_percent", type=float, default=None)
    parser.add_argument("--stage1_loss_type", choices=("dil-bce-orig", "dil_bce_orig"))
    parser.add_argument("--lambda_preference", type=float)
    parser.add_argument("--lambda_topk_entropy", type=float)
    parser.add_argument("--lambda_entropy", type=float)
    parser.add_argument("--preference_topk_k", type=int)
    parser.add_argument("--preference_topk_mode", choices=("independent", "sync_topk"))
    parser.add_argument("--topk_include_gt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lambda_topk_mass", type=float)
    parser.add_argument("--lambda_topk_chosen_mass", type=float)
    parser.add_argument("--preference_entropy_top_percent", type=float)
    parser.add_argument("--loss_chunk_positions", type=int)
    parser.add_argument("--stage1_v5_selection", choices=("all", "clean", "voff", "gap", "intersection"))
    parser.add_argument("--stage1_v5_text_kl_weight", type=float)
    from stage1.v5_config import V5_VOFF_MODES
    parser.add_argument("--stage1_v5_voff_mode", choices=V5_VOFF_MODES)
    parser.add_argument("--stage1_v5_debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--task", type=str, default="vsp-spatial-reasoning", choices=["vsp-spatial-reasoning", "vsp-spatial-planning", "blink-jigsaw", "sat","vsp-spatial-planning-cot",'zebra-cot'])
    parser.add_argument("--test_data_path", type=str, default=None, help="Path to the test data jsonl file for evaluation after training.")
    parser.add_argument("--latent_size", type=int, default=8)
    parser.add_argument("--stage2_reproduction_probe_dir", type=str, default="")
    parser.add_argument("--stage2_sampler_policy", choices=("auto", "latent_balanced", "reference_random", "historical_balanced_2rank"), default="auto")
    parser.add_argument("--stage2_reference_sampler_probe_dir", type=str, default="")
    parser.add_argument("--stage2_diagnostics_interval", type=int, default=0)
    parser.add_argument("--step_preference_enabled", action="store_true")
    parser.add_argument("--step_preference_packing", action="store_true")
    parser.add_argument("--step_shared_prefix_enabled", action="store_true")
    parser.add_argument("--step_sidecar_attention_backend", choices=("dense", "flex"), default="flex")
    parser.add_argument("--step_sidecar_query_budget", type=int, default=1024)
    parser.add_argument(
        "--step_preference_mode", choices=("stage2a", "stage2b"), default="stage2a"
    )
    parser.add_argument(
        "--step_preference_objective",
        choices=("context_preference", "legacy_dpo"),
        default="context_preference",
    )
    parser.add_argument("--step_negative_path")
    parser.add_argument("--step_negative_manifest")
    parser.add_argument("--step_max_packed_sequence_length", type=int, default=4608)
    parser.add_argument("--step_packed_admission_manifest")
    parser.add_argument("--step_packed_admission_manifest_sha256")
    parser.add_argument("--step_packed_admission_verification")
    parser.add_argument("--step_packed_admission_verification_sha256")
    parser.add_argument("--step_packed_capacity_evidence_path")
    parser.add_argument("--step_packed_capacity_sample_id")
    parser.add_argument("--step_packed_gpu_snapshot")
    parser.add_argument("--step_packed_gpu_snapshot_sha256")
    parser.add_argument(
        "--step_preference_beta",
        "--step_dpo_beta",
        dest="step_dpo_beta",
        type=float,
        default=None,
    )
    parser.add_argument("--step_preference_weight", type=float, default=1.0)
    parser.add_argument("--step_preference_loss_type",
                        choices=("dpo", "dil_bce_rf", "dil_bce_rf_topk", "dil_bce_rf_topk_reverse", "dil_bce_rf_topk_gt"), default="dpo")
    parser.add_argument("--step_preference_topk", type=int)
    parser.add_argument("--step_preference_lambda_topk_mass", type=float, default=0.0)
    parser.add_argument("--step_preference_entropy_top_percent", type=float)
    parser.add_argument("--step_preference_entropy_source",
                        choices=("clean", "voff"), default="clean")
    parser.add_argument(
        "--step_preference_target_scope",
        choices=("immediate_text", "until_next_latent"), default="immediate_text",
        help="Preference target: next text entry (default), or all text until the next latent block.",
    )
    parser.add_argument("--step_preference_margin", type=float, default=None)
    parser.add_argument("--step_preference_prefix_mode",
        choices=("clean_prefix", "v_off_prefix", "v_off_insertion", "evidence_replacement"), default="clean_prefix")
    parser.add_argument("--step_preference_branch_mode",
        choices=("independent", "cumulative"), default="independent")
    parser.add_argument("--step_preference_block_original_image", action="store_true")
    parser.add_argument(
        "--step_visual_only_context_preference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use only the visual-off comparison for context preference "
            "(default for context-preference Stage 2B)."
        ),
    )
    parser.add_argument(
        "--step_visual_off_suppress_latent_boundaries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Suppress latent start/end controls in the visual-off comparison "
            "(default when visual-only context preference is active)."
        ),
    )
    parser.add_argument("--step_dpo_weight", type=float, default=1.0)
    parser.add_argument("--step_entropy_weight", type=float, default=0.1)
    parser.add_argument("--step_entropy_margin", type=float, default=0.1)
    parser.add_argument("--stage2a_ratio", type=float, default=0.5)
    parser.add_argument("--stage2_total_steps", type=int)
    parser.add_argument("--stage2a_checkpoint_path")
    parser.add_argument("--stage2b_direct_from_stage1", action="store_true")
    parser.add_argument(
        "--stage2b_preflight_only", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--save_steps", type=int, default=612)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--compress_strategy", type=str, default='average', choices=['average'])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--data_path", type=str, default='PathToJsonlData')
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--save_model_path", type=str, default='./checkpoints/model_stage1')
    parser.add_argument("--load_model_path", type=str, default='./checkpoints/model_stage1')
    parser.add_argument("--cache_dir", type=str, default='./cache')
    parser.add_argument("--ce_weight", type=float, default=None)
    parser.add_argument("--warm_up_steps", type=int, default=10)
    parser.add_argument("--sim_weight", type=float, default=1.0)
    parser.add_argument("--ema_tau", type=float, default=0.999)
    parser.add_argument("--coverage_p", type=float, default=0.9)
    parser.add_argument("--image_pool_k", type=int, default=8)  
    parser.add_argument("--latent_ce_ratio", type=float, default=0.0, help="stage2")
    parser.add_argument(
        "--textual_states",
        choices=("all", "previous"),
        default="previous",
    )
    parser.add_argument("--use_lora", action="store_true", help="")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str,
        default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    parser.add_argument(
        "--tuning_method",
        choices=("full", "lora", "dora"),
        default=None,
    )
    parser.add_argument("--stage1_adapter_path", type=str, default=None)
    
    parser.add_argument(
        "--resume_from_checkpoint_path",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume training from. If None, training starts from scratch."
    )
    parser.add_argument("--helper_group_L", type=int, default=256,help="")
    parser.add_argument(
        "--latent_num_segments_train", type=int, default=8,
        help="Stage-2"
    )
    parser.add_argument("--max_seq_length_train", type=int, default=8192,
        help="Stage-2")  # 改默认 8192
    parser.add_argument("--stage2_length_manifest")
    parser.add_argument("--stage2_length_manifest_sha256")
    parser.add_argument(
        "--stage2_assistant_plan_rendering",
        choices=("auto", "concatenated-v1", "adjacent-text-space-lf-v1"),
        default="auto",
    )
    parser.add_argument("--stage2_launcher_log_capture", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--supervise_latent_boundaries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_final_model", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_strategy", choices=("no", "steps", "epoch"), default="steps"
    )
    parser.add_argument("--final_save", action="store_true", default=True)
    parser.add_argument("--no_final_save", action="store_false", dest="final_save")
    parser.add_argument("--image_root", required=False)
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--stage2_stop_after_checkpoint", type=int)
    parser.add_argument("--benchmark_run_config")
    parser.add_argument("--benchmark_evidence_path")
    parser.add_argument("--benchmark_source_data_path")
    parser.add_argument("--benchmark_smoke_resume_witness")
    parser.add_argument("--training_evidence_path")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--stage2_latent_activation_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--attention_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument(
        "--vision_attention_implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
    )
    parser.add_argument("--deepspeed_config")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # global_pad_to 已存在，保留
# 其余不变

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    _resolve_stage1_v5_args(parser, args, raw_argv)
    if args.stage1_ce_entropy_top_percent is not None:
        from stage1.entropy_ce import validate_ce_entropy_percent
        args.stage1_ce_entropy_top_percent = validate_ce_entropy_percent(args.stage1_ce_entropy_top_percent)
        if args.stage != 'stage1' or args.stage1_variant != 'standard' or not args.stage1_match_v5_execution:
            raise ValueError('CE entropy selection requires matched standard Stage1')
    if args.stage1_match_v5_execution and (args.stage != 'stage1' or args.stage1_variant != 'standard'):
        raise ValueError('matched execution requires standard Stage1')
    if args.vision_attention_implementation is None:
        args.vision_attention_implementation = args.attention_implementation
    if args.log_file is None:
        try:
            paths = resolve_run_artifact_paths(Path(args.save_model_path).name)
        except ValueError:
            pass
        else:
            args.log_file = str(paths.log_dir / "train.log")
    args.tuning_method_explicit = any(
        item == "--tuning_method" or item.startswith("--tuning_method=")
        for item in raw_argv
    )
    args.lora_target_modules_explicit = any(
        item == "--lora_target_modules"
        or item.startswith("--lora_target_modules=")
        for item in raw_argv
    ) or bool(args.stage1_v5_saved_contract is not None
              and 'lora_target_modules' in args.stage1_v5_saved_contract['execution'])
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if args.lora_r <= 0 or args.lora_alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not math.isfinite(args.lora_dropout) or not 0.0 <= args.lora_dropout < 1.0:
        raise ValueError("LoRA dropout must be finite and in [0, 1)")
    if args.per_device_train_batch_size < 1:
        raise ValueError("per_device_train_batch_size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.save_total_limit < 1:
        raise ValueError("save_total_limit must be at least 1")
    if args.stage2_diagnostics_interval < 0:
        raise ValueError("Stage 2 diagnostics interval must be non-negative")
    if args.stage == "stage1" and args.resume_from_checkpoint_path and args.ce_weight is None:
        metadata_path = Path(args.resume_from_checkpoint_path) / "stage1_peft_metadata.json"
        if metadata_path.is_file():
            args.ce_weight = json.loads(metadata_path.read_text(encoding="utf-8"))["ce_weight"]
    if args.stage1_v5_config is None:
        args.step_preference_config = StepPreferenceConfig.from_args(args)
        args.step_dpo_beta = args.step_preference_config.beta
        args.step_preference_beta = args.step_preference_config.beta
        args.step_preference_margin = args.step_preference_config.preference_margin
    else:
        # Compatibility for existing Stage2 checks below; v5 never enters its
        # phase/default resolver and all v5 consumers use stage1_v5_config.
        args.step_preference_config = StepPreferenceConfig(lambda_ce=args.ce_weight)
    args.ce_weight = args.step_preference_config.lambda_ce
    from stage2.planned_stop import validate_planned_checkpoint_stop
    validate_planned_checkpoint_stop(
        args.stage2_stop_after_checkpoint, config=args.step_preference_config,
        max_steps=(args.step_preference_config.phase_steps
                   if args.step_preference_config.phase_steps is not None else args.max_steps),
        resume=args.resume_from_checkpoint_path, stage=args.stage,
    )
    if (args.stage2_stop_after_checkpoint is not None
            and args.step_preference_config.phase_steps is not None
            and args.max_steps != args.step_preference_config.phase_steps):
        raise ValueError("planned stop --max_steps must match the resolved phase budget")
    args.step_visual_only_context_preference = (
        args.step_preference_config.visual_only_context_preference
    )
    args.step_visual_off_suppress_latent_boundaries = (
        args.step_preference_config.visual_off_suppress_latent_boundaries
    )
    capacity_fields = (args.step_packed_capacity_evidence_path, args.step_packed_capacity_sample_id,
                       args.step_packed_gpu_snapshot, args.step_packed_gpu_snapshot_sha256)
    if any(value is not None for value in capacity_fields):
        if not all(type(value) is str and value for value in capacity_fields):
            raise ValueError("capacity-only options must be supplied together")
        if not args.step_preference_config.is_stage2b or not args.step_preference_config.packing_enabled:
            raise ValueError("capacity-only options require packed Stage 2B")
        from deepspeed_runtime import load_deepspeed_runtime_profile
        profile = load_deepspeed_runtime_profile(args.deepspeed_config, world_size=2)
        if profile.zero_stage != 2 or args.max_steps != 2:
            raise ValueError("capacity-only options require ZeRO-2 and exactly two optimizer steps")
    if args.stage2b_preflight_only and not args.step_preference_config.is_stage2b:
        raise ValueError("--stage2b_preflight_only requires Stage 2B")
    if args.stage == "stage2":
        manifest_fields = (
            args.stage2_length_manifest,
            args.stage2_length_manifest_sha256,
        )
        if any(manifest_fields) and not all(manifest_fields):
            raise ValueError(
                "--stage2_length_manifest and --stage2_length_manifest_sha256 "
                "must be provided together"
            )
        if args.dataloader_num_workers < 0:
            raise ValueError("Stage 2 dataloader_num_workers must be non-negative")
        if args.use_lora:
            raise ValueError(
                "Stage 2 does not accept --use_lora; use --tuning_method lora"
            )
        if args.stage1_adapter_path and args.resume_from_checkpoint_path:
            raise ValueError(
                "--stage1_adapter_path and --resume_from_checkpoint_path are mutually exclusive"
            )
        if (
            args.step_preference_config.direct_from_stage1
            and not args.stage1_adapter_path
        ):
            raise ValueError(
                "--stage2b_direct_from_stage1 requires --stage1_adapter_path"
            )
        if (
            args.tuning_method is None
            and args.resume_from_checkpoint_path is None
            and not args.step_preference_config.is_stage2b
        ):
            args.tuning_method = "full"
        normalize_stage2_target_modules(args.lora_target_modules)
        if args.per_device_train_batch_size != 1:
            raise ValueError("Stage 2 requires per_device_train_batch_size=1")
    elif args.stage == "stage1":
        if args.dataloader_num_workers < 0:
            raise ValueError("Stage 1 dataloader_num_workers must be non-negative")
        if args.step_preference_enabled:
            raise ValueError("step preference training is only valid for Stage 2")
        if args.use_lora and args.tuning_method_explicit:
            raise ValueError("--use_lora and --tuning_method cannot be combined")
        if args.tuning_method_explicit and args.tuning_method not in (
            "full", "lora", "dora"
        ):
            raise ValueError(
                "Stage 1 --tuning_method accepts only full, lora, or dora"
            )
        normalize_decoder_target_suffixes(
            args.lora_target_modules, stage_label="Stage 1 PEFT target modules"
        )
    elif args.tuning_method_explicit:
        raise ValueError("--tuning_method is only valid for Stage 1 or Stage 2")
    return args


def _resolve_stage1_v5_args(parser, args, raw_argv):
    """Keep explicit alias presence for later saved-objective resolution."""
    from stage1.v5_config import V5_OPTION_ALIASES, resolve_v5_config
    public = {
        'stage1_loss_type', 'lambda_preference', 'lambda_topk_entropy',
        'lambda_entropy', 'preference_topk_k', 'preference_topk_mode',
        'topk_include_gt', 'lambda_topk_mass', 'lambda_topk_chosen_mass',
        'preference_entropy_top_percent', 'loss_chunk_positions',
        'stage1_v5_selection', 'stage1_v5_text_kl_weight', 'stage1_v5_voff_mode',
    }
    # Resolve saved variant/objective BEFORE any eager fresh-option validation.
    # Omitted entropy support and execution options inherit their exact identity.
    from stage1.v5_resume import load_v5_contract, validate_v5_resume, V5_CONTRACT_FILENAME
    saved = None
    resume = getattr(args, 'resume_from_checkpoint_path', None)
    present = set()
    for item in raw_argv:
        option = item.partition('=')[0]
        action = parser._option_string_actions.get(option)
        if action is None and option.startswith('--'):
            matches = parser._get_option_tuples(option)
            if len(matches) == 1:
                action = matches[0][0]
        if action is not None:
            present.add(action.dest)
    if args.stage == 'stage1' and resume:
        if (Path(resume) / V5_CONTRACT_FILENAME).exists():
            saved = load_v5_contract(Path(resume))
            validate_v5_resume(saved, {})  # Reject historical training before inheritance/model work.
            if 'stage1_variant' in present and args.stage1_variant != 'v5':
                raise ValueError('explicit stage1_variant conflicts with saved v5 contract')
            args.stage1_variant = 'v5'
            inherited = {**saved['coefficients'], **saved['execution']}
            inherited.update(max_steps=saved['schedule'].get('configured_max_steps', args.max_steps),
                             epochs=saved['schedule'].get('num_train_epochs', args.epochs))
            if 'deepspeed_profile' in saved['execution']:
                inherited['deepspeed_config'] = saved['execution']['deepspeed_profile']['config_path']
            for name, value in inherited.items():
                if name == "stage1_v5_debug":
                    continue  # Runtime instrumentation never inherits objective semantics.
                name = {'warmup_steps': 'warm_up_steps'}.get(name, name)
                if hasattr(args, name):
                    if name in present and getattr(args, name) != value:
                        # Node-local locations are authenticated by mandatory
                        # reference restore and runtime config-content agreement.
                        if name not in ('model', 'deepspeed_config'):
                            raise ValueError(f'explicit {name} conflicts with saved v5 contract')
                        if name == 'model' and not Path(args.model).is_dir():
                            raise ValueError('v5 requires a local original full base; no HF-name fallback')
                    if name not in present:
                        setattr(args, name, value)
        elif args.stage1_variant == 'v5':
            load_v5_contract(Path(resume))  # Explicitly reject legacy relabeling.
    args.stage1_v5_saved_contract = saved
    explicit = {}
    query_explicit = {}
    query_fields = {'query_mode', 'history_text_gamma', 'history_attention_layers'}
    is_v5 = args.stage1_variant == 'v5'
    for index, item in enumerate(raw_argv):
        option, equal, inline = item.partition('=')
        action = parser._option_string_actions.get(option)
        if action is None:
            # argparse accepts unique abbreviations. Track the registered name
            # so an abbreviation cannot evade path/alias validation.
            matches = parser._get_option_tuples(option) if option.startswith('--') else ()
            if len(matches) != 1:
                continue
            action, option = matches[0][:2]
        if action.dest in query_fields:
            value = inline if equal else raw_argv[index + 1]
            value = action.type(value) if action.type else value
            if action.dest in query_explicit and query_explicit[action.dest] != value:
                raise ValueError(f'conflicting explicit values for {action.dest}')
            query_explicit[action.dest] = value
        spelling = option.removeprefix('--')
        name = action.dest if spelling == 'no-topk_include_gt' else spelling
        if name in public and not (is_v5 and args.stage == 'stage1'):
            raise ValueError(f'{option} is only valid for --stage stage1 --stage1_variant v5')
        if is_v5 and name in public | set(V5_OPTION_ALIASES):
            if isinstance(action, argparse.BooleanOptionalAction):
                value = not option.startswith('--no-')
            else:
                value = action.type(inline if equal else raw_argv[index + 1]) if action.type else (
                    inline if equal else raw_argv[index + 1])
            if name in explicit and explicit[name] != value:
                raise ValueError(f'conflicting explicit values for {name}')
            explicit[name] = value
        if (is_v5 and action.dest.startswith(('step_', 'stage2'))
                and name not in V5_OPTION_ALIASES):
            # Only approved one-to-one aliases participate in the v5 objective.
            # Reject explicit legacy controls, including negative boolean forms,
            # rather than leaving requested semantics unused on the namespace.
            raise ValueError(f'{option} is unsupported for Stage1-v5')
    if is_v5 and args.stage != 'stage1':
        raise ValueError('stage1_variant=v5 requires --stage stage1')
    args.stage1_v5_explicit_values = explicit
    args.stage1_v5_config = (validate_v5_resume(saved, explicit) if saved is not None
                            else resolve_v5_config(explicit)) if is_v5 else None
    if is_v5:
        config = args.stage1_v5_config
        args.stage1_loss_type = config.loss_type
        for name in public - {'stage1_loss_type', 'stage1_v5_selection', 'stage1_v5_text_kl_weight',
                              'stage1_v5_voff_mode'}:
            setattr(args, name, getattr(config, name))
        args.stage1_v5_selection = config.selection
        args.stage1_v5_text_kl_weight = config.text_kl_weight
        args.stage1_v5_voff_mode = config.voff_mode
        args.step_dpo_beta = args.step_preference_beta = config.beta
        args.step_preference_margin = config.margin
        args.step_preference_weight = config.lambda_preference
        args.step_preference_topk = config.preference_topk_k
        args.step_preference_lambda_topk_mass = config.lambda_topk_mass
        args.step_preference_entropy_top_percent = config.preference_entropy_top_percent
        args.step_entropy_weight = config.lambda_entropy

    from stage1.query import resolve_query_config
    from stage1.v5_resume import resolve_v5_query_resume
    query = (resolve_v5_query_resume(saved, query_explicit) if saved is not None
             else resolve_query_config(query_explicit))
    if query.mode == 'adaptive':
        if (args.stage != 'stage1' or not is_v5 or args.tuning_method != 'dora'
                or args.per_device_train_batch_size != 1):
            raise ValueError('adaptive query requires Stage1-v5 DoRA with batch size 1')
        if 'textual_states' in present:
            raise ValueError('adaptive query does not accept explicit textual_states')
    elif query_fields.intersection(query_explicit) - {'query_mode'}:
        raise ValueError('legacy query does not accept history controls')
    args.stage1_query_explicit_values = query_explicit
    args.stage1_query_config = query
    args.query_mode, args.history_text_gamma, args.history_attention_layers = (
        query.mode, query.text_gamma, query.attention_layers)


def get_args():
    return parse_args()

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_jsonl_dataset(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        data = data[:]
    return Dataset.from_list(data)

def place_input_image(
    text,
    image_pad="<|vision_start|><|image_pad|><|vision_end|>",
    image_placeholder="<image>",
    sep_token="<|im_start|>assistant"
) -> str:
    
    if sep_token is not None and sep_token in text:
        t1, t2 = text.split(sep_token, 1)
        if image_placeholder in t1:
            t1 = t1.replace(image_pad, "")
            t1 = t1.replace(image_placeholder, image_pad)
        return t1 + sep_token + t2
    else:
        if image_placeholder in text:
            text = text.replace(image_pad, "")
            text = text.replace(image_placeholder, image_pad)
        return text

def place_output_image(
    text,
    image_pad="<|vision_start|><|image_pad|><|vision_end|>",
    latent_placeholder="<output_image>",
    sep_token="<|im_start|>assistant"
) -> str:
    if latent_placeholder in text:
        text = text.replace(image_pad + '<think>', '<think>')
        text = text.replace(latent_placeholder, image_pad)
    return text

def remove_user_images(examples):
    new_examples = []
    for example in examples:
        new_example = []
        for turn in example:
            new_turn = dict(turn)
            if turn.get("role") == "user":
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            new_example.append(new_turn)
        new_examples.append(new_example)
    return new_examples

def remove_assistant_images(examples):
    new_examples = []
    for example in examples:
        new_example = []
        for turn in example:
            new_turn = dict(turn)
            if turn.get("role") == "assistant":
                new_turn["content"] = [
                    item for item in turn.get("content", [])
                    if item.get("type") != "image"
                ]
            new_example.append(new_turn)
        new_examples.append(new_example)
    return new_examples

def replace_visual_spectial_tokens(texts):
    # 更健壮：找不到分隔符也不崩
    update_texts = []
    for text in texts:
        parts = text.split("<|im_start|>assistant", 1)
        if len(parts) == 2:
            prev, after = parts
            update_texts.append(
                prev + "<|im_start|>assistant" +
                after.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>")
            )
        else:
            update_texts.append(
                text.replace("<|vision_start|><|image_pad|><|vision_end|>", "<|latent_start|><|image_pad|><|latent_end|>")
            )
    return update_texts

def replace_subsequent_image_parts_2d(
    input_ids: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int,
    pad_token: int = 0
) -> torch.Tensor:
    batch_size, _ = input_ids.shape
    replaced_sequences = []
    max_len = 0
    for i in range(batch_size):
        seq_1d = input_ids[i]
        new_seq_1d = replace_subsequent_image_parts_1d(
            seq_1d, start_token, end_token, replacement_token, replacement_length
        )
        replaced_sequences.append(new_seq_1d)
        max_len = max(max_len, new_seq_1d.size(0))
    new_input_ids = input_ids.new_full((batch_size, max_len), fill_value=int(pad_token))
    for i, seq_1d in enumerate(replaced_sequences):
        length = seq_1d.size(0)
        new_input_ids[i, :length] = seq_1d
    return new_input_ids



def replace_subsequent_image_parts_1d(
    seq: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int
) -> torch.Tensor:
    start_positions = (seq == start_token).nonzero().squeeze(-1)
    end_positions   = (seq == end_token).nonzero().squeeze(-1)
    new_seq_pieces = []
    prev_end = 0
    for (s_pos, e_pos) in zip(start_positions, end_positions):
        new_seq_pieces.append(seq[prev_end : s_pos + 1])
        replacement_span = torch.tensor(
            [replacement_token] * replacement_length,
            dtype=seq.dtype, device=seq.device
        )
        new_seq_pieces.append(replacement_span)
        new_seq_pieces.append(torch.tensor([end_token], dtype=seq.dtype, device=seq.device))
        prev_end = e_pos + 1
    if prev_end < len(seq):
        new_seq_pieces.append(seq[prev_end:])
    new_seq = torch.cat(new_seq_pieces, dim=0)
    return new_seq


def normalize_stage1_latent_blocks_1d(
    seq: torch.Tensor,
    *,
    start_token: int,
    end_token: int,
    latent_pad_token: int,
    latent_size: int,
    source_index: int,
) -> torch.Tensor:
    """Replace visual block interiors while preserving prebuilt latent scaffolds.

    A real processor expands assistant image placeholders between latent boundary
    tokens. Tests and already-tokenized inputs may instead contain latent-pad
    tokens, which are kept so the Stage 1 parser can validate their exact count.
    """
    if seq.ndim != 1:
        raise ValueError(
            f"source_index={source_index}: observed tensor rank {seq.ndim}; "
            "expected a one-dimensional token row"
        )
    if latent_size <= 0:
        raise ValueError(
            f"source_index={source_index}: observed latent_size={latent_size}; "
            "expected a positive latent_size"
        )

    pieces = []
    cursor = 0
    open_start = None
    values = seq.tolist()
    for position, raw_token in enumerate(values):
        token = int(raw_token)
        if open_start is None:
            if token == start_token:
                open_start = position
            elif token == end_token:
                raise ValueError(
                    f"source_index={source_index}: observed latent_end at position "
                    f"{position} without latent_start; expected ordered complete boundaries"
                )
            elif token == latent_pad_token:
                raise ValueError(
                    f"source_index={source_index}: observed latent_pad at position "
                    f"{position} outside a latent block; expected pads only inside a block"
                )
            continue

        if token == start_token:
            raise ValueError(
                f"source_index={source_index}: observed nested latent_start at position "
                f"{position}; expected ordered, nonnested latent boundaries"
            )
        if token != end_token:
            continue

        interior = seq[open_start + 1 : position]
        interior_is_latent = bool(interior.numel()) and bool(
            torch.all(interior == latent_pad_token)
        )
        interior_has_latent = bool(
            interior.numel() and torch.any(interior == latent_pad_token)
        )
        if interior_has_latent and not interior_is_latent:
            observed = interior.tolist()
            raise ValueError(
                f"source_index={source_index}: observed mixed latent/visual block "
                f"tokens {observed}; expected either a visual placeholder or only "
                "latent_pad slots"
            )

        pieces.append(seq[cursor : open_start + 1])
        if interior_is_latent:
            pieces.append(interior)
        else:
            pieces.append(
                seq.new_full((latent_size,), fill_value=int(latent_pad_token))
            )
        pieces.append(seq[position : position + 1])
        cursor = position + 1
        open_start = None

    if open_start is not None:
        raise ValueError(
            f"source_index={source_index}: observed incomplete latent block starting "
            f"at position {open_start}; expected a complete latent_start/latent_end pair"
        )

    if not pieces:
        return seq.clone()
    pieces.append(seq[cursor:])
    return torch.cat(pieces)

def process_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_token: int,
    end_token: int,
    replacement_token: int,
    replacement_length: int,
    pad_token: int = 0
):
    batch_size, _ = input_ids.shape
    processed_sequences = []
    for b in range(batch_size):
        real_tokens = input_ids[b][attention_mask[b] == 1]
        updated_seq = replace_subsequent_image_parts_1d(
            real_tokens, start_token, end_token, replacement_token, replacement_length
        )
        processed_sequences.append(updated_seq)
    new_max_len = max(seq.size(0) for seq in processed_sequences)
    new_input_ids = input_ids.new_full((batch_size, new_max_len), fill_value=int(pad_token))
    new_attention_mask = input_ids.new_zeros((batch_size, new_max_len))
    for b in range(batch_size):
        seq_len_b = processed_sequences[b].size(0)
        new_input_ids[b, :seq_len_b] = processed_sequences[b]
        new_attention_mask[b, :seq_len_b] = 1
    return new_input_ids, new_attention_mask

def find_subsequence(row: torch.Tensor, pattern: torch.Tensor) -> int:
    seq_len = row.size(0)
    pat_len = pattern.size(0)
    for start_idx in range(seq_len - pat_len + 1):
        if torch.all(row[start_idx : start_idx + pat_len] == pattern):
            return start_idx
    return -1

def generate_labels_after_multi_token_start(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
    img_token_idx: int = 151655,
) -> torch.Tensor:
    batch_size, _ = input_ids.shape
    labels = input_ids.clone()
    for b in range(batch_size):
        row = labels[b]
        start_idx = find_subsequence(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
        else:
            end_of_subseq = start_idx + start_sequence.size(0)
            row[:end_of_subseq] = -100
        row[row == pad_token_idx] = -100
        row[row == img_token_idx] = -100
    return labels

def generate_labels_after_latent_tokens(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int = 0,
) -> torch.Tensor:
    batch_size, _ = input_ids.shape
    labels = input_ids.clone()
    for b in range(batch_size):
        row = labels[b]
        start_idx = find_subsequence(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
        else:
            row[:start_idx] = -100
        row[row == pad_token_idx] = -100
    return labels

def mask_image_output_tokens(
    input_ids: torch.Tensor,
    image_start_token: int,
    image_token: int
) -> torch.Tensor:
   
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids)
    for i in range(batch_size):
        seq = input_ids[i]
        first_start_pos = -1
        for j in range(seq_len):
            if seq[j] == image_start_token:
                first_start_pos = j
                break
        if first_start_pos == -1:
            continue
        for k in range(first_start_pos + 1, seq_len):
            if seq[k] == image_token:
                mask[i, k] = 1
    return mask

def mask_latent_output_tokens_all_segments(
    input_ids: torch.Tensor,
    latent_start_id: int,
    latent_end_id: int,
    latent_pad_id: int
) -> torch.Tensor:
    B, T = input_ids.shape
    mask = torch.zeros_like(input_ids)
    for b in range(B):
        seq = input_ids[b]
        starts = (seq == latent_start_id).nonzero().squeeze(-1).tolist()
        ends   = (seq == latent_end_id).nonzero().squeeze(-1).tolist()
        e_ptr = 0
        for s in starts:
            while e_ptr < len(ends) and ends[e_ptr] <= s:
                e_ptr += 1
            if e_ptr >= len(ends):
                break
            e = ends[e_ptr]
            if e - s > 1:
                seg = seq[s+1:e]
                mask[b, s+1:e] = (seg == latent_pad_id).to(mask.dtype)
    return mask

def generate_labels_with_latent_template(
    input_ids: torch.Tensor,
    start_sequence: torch.Tensor,
    pad_token_idx: int,
    latent_start_id: int,
    latent_end_id: int,
    latent_pad_id: int,
    latent_ce_ratio: float = 1.0,
) -> torch.Tensor:
    B, _ = input_ids.shape
    labels = input_ids.clone()

    def _find_subseq(row: torch.Tensor, pat: torch.Tensor) -> int:
        for i in range(0, row.size(0) - pat.size(0) + 1):
            if torch.all(row[i:i+pat.size(0)] == pat):
                return i
        return -1

    for b in range(B):
        row = labels[b]
        start_idx = _find_subseq(row, start_sequence)
        if start_idx == -1:
            row[:] = -100
            continue
        end_of_subseq = start_idx + start_sequence.size(0)
        row[:end_of_subseq] = -100
        row[row == pad_token_idx] = -100

        seq = input_ids[b]
        start_pos = (seq == latent_start_id).nonzero().squeeze(-1)
        end_pos   = (seq == latent_end_id).nonzero().squeeze(-1)

        in_latent = torch.zeros_like(seq, dtype=torch.bool)
        for s in start_pos.tolist():
            e_candidates = end_pos[end_pos > s]
            if len(e_candidates) == 0:
                continue
            e = int(e_candidates[0].item())
            if e - s > 1:
                in_latent[s+1:e] = True

        outside_latent = ~in_latent
        row[(seq == latent_pad_id) & outside_latent] = -100

        latent_zone = in_latent & (seq == latent_pad_id)
        if latent_ce_ratio < 1.0:
            keep = torch.rand_like(seq.float()) < latent_ce_ratio
            latent_zone = latent_zone & keep.bool()
        drop_zone = (seq == latent_pad_id) & in_latent & ~latent_zone
        row[drop_zone] = -100

    return labels


class LatentTemplateLogitsProcessor(LogitsProcessor):
    def __init__(self, latent_start_id: int, latent_end_id: int, latent_pad_id: int, K: int):
        self.latent_start_id = int(latent_start_id)
        self.latent_end_id   = int(latent_end_id)
        self.latent_pad_id   = int(latent_pad_id)
        self.K               = int(K)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        B = input_ids.size(0)
        for b in range(B):
            seq = input_ids[b]
            start_positions = (seq == self.latent_start_id).nonzero().squeeze(-1)
            end_positions   = (seq == self.latent_end_id).nonzero().squeeze(-1)

            def _last_pos(t):
                return int(t[-1].item()) if t.numel() > 0 else -1

            last_s = _last_pos(start_positions)
            last_e = _last_pos(end_positions)
            inside = (last_s != -1) and (last_e < last_s)

            if not inside:
                scores[b, self.latent_pad_id] = -float("inf")
                scores[b, self.latent_end_id] = -float("inf")
                continue

            i = last_s + 1
            n_pad = 0
            L = seq.size(0)
            while i < L and int(seq[i].item()) == self.latent_pad_id:
                n_pad += 1
                i += 1

            if n_pad < self.K:
                scores[b, :] = -float("inf")
                scores[b, self.latent_pad_id] = 0.0
            else:
                scores[b, :] = -float("inf")
                scores[b, self.latent_end_id] = 0.0
        return scores


if __name__ == "__main__":
    pass
