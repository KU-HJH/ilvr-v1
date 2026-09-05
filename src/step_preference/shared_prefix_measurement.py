"""Small runtime witnesses shared by the existing verification/benchmark CLIs."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace

import torch

from stage2.peft import resolve_stage2_qwen_model
from step_preference.benchmark import Stage2BTrainingStep, benchmark_training_path
from step_preference.equivalence import _tiny_model
from step_preference.negatives import StepNegative
from step_preference.spans import ReasoningStepSpan


class RuntimeCounters:
    """Observe calls and row counts only; never retain hidden states or K/V."""

    def __init__(self, model):
        self.policy = model
        self.base = resolve_stage2_qwen_model(model)
        self.phase = "forward"
        self.stage = "ordinary"
        self.in_rollout = False
        self.counts = Counter()
        self.handles = []

    def add(self, name, value=1):
        self.counts[f"{name}_{self.phase}"] += int(value)

    def reset(self):
        self.counts.clear()
        self.phase = "forward"

    def snapshot(self):
        return dict(self.counts)

    def __enter__(self):
        def policy(_module, _args):
            self.add("policy_calls")

        def decoder(_module, _args, kwargs):
            self.add("decoder_calls")
            if self.in_rollout:
                self.add("latent_decoder_calls")
            if kwargs.get("stage2_prefix_key_values") is not None:
                self.add("sidecar_chunks")

        def layer(_module, args, kwargs):
            # Checkpointed clean layers are called positionally, bypassing decoder.
            capture = kwargs.get("stage2_capture_kv", args[9] if len(args) > 9 else False)
            prefix = kwargs.get("stage2_prefix_key_value", args[10] if len(args) > 10 else None)
            self.stage = "sidecar" if prefix is not None else "clean" if capture else "ordinary"

        def rows(kind):
            def observe(_module, args):
                self.add(f"{self.stage}_{kind}_rows", args[0].numel() // args[0].shape[-1])
                if kind == "lm_head" and self.stage == "sidecar":
                    # One selected head call ends a sidecar chunk; the clean
                    # CE/selected head calls follow after all chunks return.
                    self.stage = "clean"
            return observe

        self.handles = [
            self.policy.register_forward_pre_hook(policy),
            self.base.model.register_forward_pre_hook(decoder, with_kwargs=True),
            self.base.model.layers[0].register_forward_pre_hook(layer, with_kwargs=True),
            self.base.model.layers[0].self_attn.q_proj.register_forward_pre_hook(rows("projected")),
            self.base.model.layers[0].mlp.register_forward_pre_hook(rows("mlp")),
            self.base.lm_head.register_forward_pre_hook(rows("lm_head")),
        ]
        self.original_rollout = self.base.stage2_rollout_engine.rollout

        def rollout(*args, **kwargs):
            self.add("rollout_calls")
            self.in_rollout = True
            try:
                return self.original_rollout(*args, **kwargs)
            finally:
                self.in_rollout = False

        self.base.stage2_rollout_engine.rollout = rollout
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.base.stage2_rollout_engine.rollout = self.original_rollout


class MeasuredTrainingStep(Stage2BTrainingStep):
    def __init__(self, model, batch, config, counters, *, autocast_dtype=torch.bfloat16):
        super().__init__(model, batch, config)
        self.counters = counters
        self.autocast_dtype = autocast_dtype

    def __call__(self):
        if self.counters is not None:
            self.counters.reset()
        precision = torch.autocast(self.batch["input_ids"].device.type, dtype=self.autocast_dtype) if self.autocast_dtype else nullcontext()
        with precision:
            loss = self.trainer.compute_loss(self.model, self.batch)
        if self.counters is not None:
            self.counters.phase = "backward"
        loss.backward()
        self.last_loss = float(loss.detach())


def batch_accounting(batch):
    spans = batch["reasoning_step_spans"]
    chosen = sum(s.target_text_end - s.target_text_start for s in spans)
    rejected = 0
    valid_pairs = 0
    for step in batch["step_preference"]["steps"]:
        for modality in ("visual", "textual"):
            if getattr(step["negative"], f"valid_{modality}_negative"):
                rejected += len(step[f"{modality}_rejected"]["completion_token_ids"])
                valid_pairs += 1
    mask = batch["attention_mask"]
    return dict(input_lengths=mask.sum(-1).tolist(), padding_tokens=int((mask == 0).sum()),
                chosen_target_tokens=chosen, entropy_scored_tokens=2*chosen,
                rejected_scored_tokens=rejected, valid_pairs=valid_pairs,
                valid_scored_tokens=3*chosen+rejected,
                reference_cache_entries_supplied=len(spans)+valid_pairs,
                reference_cache_validity="requires separate fixed-input audit; synthetic for tiny inputs")


def measure_paths(model, batch, configs, runtime, *, warmups=3, measurements=10,
                  autocast_dtype=torch.bfloat16):
    records = []
    accounting = batch_accounting(batch)
    for mode, config in configs.items():
        model.zero_grad(set_to_none=True)
        torch.manual_seed(307)
        # Exclude Python observation-hook overhead from measured iterations.
        timed_step = MeasuredTrainingStep(model, batch, config, None, autocast_dtype=autocast_dtype)
        measured = benchmark_training_path(timed_step, runtime, warmups=warmups, measurements=measurements)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(307)
        with RuntimeCounters(model) as counters:
            step = MeasuredTrainingStep(model, batch, config, counters, autocast_dtype=autocast_dtype)
            step()
            metrics = step.trainer._step_preference_metric_totals
            batches = step.trainer._step_preference_metric_batches
            reference = {k: v/batches for k, v in metrics.items() if "reference_" in k}
            records.append(dict(mode=mode,
                backend=config.sidecar_attention_backend if config.shared_prefix_enabled else "flex" if config.packing_enabled else config.attention_implementation,
                ordinary_language_backend=config.attention_implementation,
                dtype=str(autocast_dtype or next(model.parameters()).dtype),
                parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                dropout=sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)}),
                gradient_checkpointing=bool(model.is_gradient_checkpointing),
                warmups=warmups, measurements=measurements, counters_scope="one separate untimed complete iteration",
                elapsed_forward_backward_ms=measured.median_ms,
                valid_scored_tokens_per_second=accounting["valid_scored_tokens"]*measured.iterations_per_second,
                runtime_counters=counters.snapshot(), reference_counters=reference,
                actual_sidecar_entropy_reduction_rows=counters.counts.get("sidecar_lm_head_rows_forward", 0),
                loss=step.last_loss, **accounting, **measured.to_dict()))
    return records


def tiny_measurement_case(*, steps, total_target_tokens, device):
    """Deterministic two-layer fixture with fresh latent blocks and valid rejects."""
    if steps < 1 or total_target_tokens < steps or total_target_tokens % steps:
        raise ValueError("target tokens must divide evenly among positive steps")
    torch.manual_seed(731)
    config = _tiny_model().config
    config.num_hidden_layers = 2
    config._attn_implementation = "sdpa"
    model = type(_tiny_model())(config).to(device).train()
    model.visual.requires_grad_(False)
    ids, spans = [11], []
    target_length = total_target_tokens // steps
    previous_start, previous_end = 0, 1
    for index in range(steps):
        latent_control = len(ids)
        ids.extend([91, 90, 90, 92])
        start = len(ids)
        ids.extend([12 + index]*target_length)
        spans.append(ReasoningStepSpan(index, previous_start, previous_end,
            latent_control+1, latent_control+3, start, len(ids), None))
        previous_start, previous_end = start, len(ids)
    clean = torch.tensor([ids], device=device)
    joined = []
    for span in spans:
        positions = tuple(range(span.target_text_start, span.target_text_end))
        rejected = clean[:, :span.target_text_end].clone()
        rejected[:, span.target_text_start:] = 41
        negative = StepNegative(sample_id="tiny-measurement", step_id=span.step_id,
            chosen_text="chosen", rejected_visual_text="reject", rejected_textual_text=None,
            valid_visual_negative=True, valid_textual_negative=False,
            reference_chosen_score=-1., reference_visual_rejected_score=-2.,
            reference_textual_rejected_score=None)
        joined.append(dict(span=span, negative=negative, textual_rejected=None,
            visual_rejected=dict(model_inputs=dict(input_ids=rejected, attention_mask=torch.ones_like(rejected)),
                completion_token_positions=positions, completion_token_ids=(41,)*len(positions))))
    return model, dict(input_ids=clean, attention_mask=torch.ones_like(clean), labels=clean.clone(),
        sample_ids=("tiny-measurement",), reasoning_step_spans=tuple(spans), step_preference=dict(steps=tuple(joined)))


def tiny_sweep_cases(*, device, step_counts=(1, 2, 4), target_tokens=8):
    for series in ("fixed", "growing"):
        for steps in step_counts:
            total = target_tokens if series == "fixed" else target_tokens*steps
            model, batch = tiny_measurement_case(steps=steps, total_target_tokens=total, device=device)
            yield dict(series=series, steps=steps, model=model, batch=batch)


def add_shared_measurement_arguments(parser):
    parser.add_argument("--shared-prefix", action="store_true")
    parser.add_argument("--sidecar-attention-backend", choices=("dense", "flex"), default="flex")
    parser.add_argument("--sidecar-query-budget", type=int, default=1024)
    parser.add_argument("--tiny-only", action="store_true")
    parser.add_argument("--tuning-method", choices=("full", "lora", "dora"), default="full")
    parser.add_argument("--preserve-dropout", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--requested-sample-id")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--measurements", type=int, default=10)


def validate_measurement_arguments(parser, args):
    if args.sidecar_query_budget <= 0:
        parser.error("--sidecar-query-budget must be positive")
    if args.tiny_only and not args.shared_prefix:
        parser.error("--tiny-only requires --shared-prefix")
    if not args.tiny_only:
        for name in ("checkpoint_path", "data_path", "image_root", "negative_path", "negative_manifest"):
            if getattr(args, name) is None:
                parser.error(f"--{name} is required unless --shared-prefix --tiny-only")


def run_shared_measurement_cli(args, *, benchmark, git_commit):
    from artifact_paths import require_canonical_artifact_path
    from stage2.peft import ALLOWED_STAGE2_TARGET_SUFFIXES, Stage2TuningConfig, build_stage2_tuned_model
    from step_preference.benchmark import CudaBenchmarkRuntime
    from step_preference.equivalence import (_config, load_production_context,
        run_shared_prefix_comparison, write_evidence_exclusive)

    output = require_canonical_artifact_path(args.output_path, kind="result")
    if args.preserve_dropout and not benchmark:
        raise ValueError("numerical comparison requires dropout disabled; use benchmark for stochastic training")
    report = dict(schema_version=3, git_commit=git_commit, status="pass", records=[],
        stochastic_limitation="Naive resamples clean prefixes; active-dropout objectives are not numerical equivalence oracles.")
    if args.tiny_only:
        cases = tiny_sweep_cases(device=args.device) if benchmark else [dict(
            series="tiny-two-layer", steps=2,
            **dict(zip(("model", "batch"), tiny_measurement_case(steps=2, total_target_tokens=8, device=args.device))))]
    else:
        context = load_production_context(checkpoint_path=args.checkpoint_path, data_path=args.data_path,
            image_root=args.image_root, negative_path=args.negative_path, negative_manifest=args.negative_manifest,
            device=args.device, tuning_method=args.tuning_method, gradient_checkpointing=args.gradient_checkpointing,
            requested_sample_id=args.requested_sample_id, preserve_dropout=args.preserve_dropout)
        report["identity"] = dict(context.identity)
        cases = [dict(series="production", steps=len(context.batch["reasoning_step_spans"]),
                      model=context.model, batch=context.batch)]
    for case in cases:
        model, batch = case["model"], case["batch"]
        if args.tiny_only and args.tuning_method != "full":
            model = build_stage2_tuned_model(model, Stage2TuningConfig(method=args.tuning_method,
                r=32, alpha=64, dropout=0.05 if args.preserve_dropout else 0.,
                target_modules=ALLOWED_STAGE2_TARGET_SUFFIXES), is_trainable=True)
        if args.tiny_only and args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        naive = _config(packing=False) if args.tiny_only else context.naive_config
        naive = replace(naive, attention_implementation="sdpa")
        shared = replace(naive, shared_prefix_enabled=True, sidecar_attention_backend=args.sidecar_attention_backend,
                         sidecar_query_budget=args.sidecar_query_budget)
        if benchmark:
            # Each independent case gets fresh compiler caches, with all
            # specialization/compilation confined to its excluded warmups.
            torch._dynamo.reset()
            if args.tiny_only:
                # Match the BF16 model storage used by the actual ZeRO2 run;
                # all three routes use this same model and parameter dtype.
                model.to(dtype=torch.bfloat16)
            configs = dict(naive=naive, shared=shared, legacy_packed=replace(naive,
                packing_enabled=True, max_packed_sequence_length=4608))
            for mode, config in configs.items():
                try:
                    records = measure_paths(model, batch, {mode: config}, CudaBenchmarkRuntime(args.device),
                        warmups=args.warmups, measurements=args.measurements)
                except torch.cuda.OutOfMemoryError as error:
                    records = [dict(mode=mode, status="incomplete", error=str(error), reason="OOM")]
                    report["status"] = "incomplete"
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                for record in records:
                    report["records"].append(dict(series=case["series"], steps=case["steps"],
                        tuning_method=args.tuning_method, adapter_rank=32 if args.tuning_method != "full" else None,
                        adapter_alpha=64 if args.tuning_method != "full" else None, **record))
        else:
            if args.tiny_only:
                strict = run_shared_prefix_comparison(model, batch, naive_config=naive, backend="dense",
                    query_budget=args.sidecar_query_budget, strict=True)
                report["records"].append(dict(series=case["series"], precision="FP32 strict dense", **strict))
                model.to(dtype=torch.bfloat16)
            diagnostic = run_shared_prefix_comparison(model, batch, naive_config=naive,
                backend=args.sidecar_attention_backend, query_budget=args.sidecar_query_budget,
                autocast_dtype=torch.bfloat16)
            report["records"].append(dict(series=case["series"], precision="BF16 diagnostic", **diagnostic))
            if any(record["status"] == "fail" for record in report["records"]):
                report["status"] = "fail"
    write_evidence_exclusive(output, report)
    return 0 if report["status"] == "pass" else 1
