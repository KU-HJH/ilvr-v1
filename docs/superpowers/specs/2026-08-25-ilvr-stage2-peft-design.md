# ILVR Stage 2 LoRA and DoRA Support Design

Date: 2026-08-25

## 1. Objective

Extend the clean Stage 2 implementation so training and inference support three
explicit tuning methods:

```text
full
lora
dora
```

LoRA and DoRA adapt only the Qwen language decoder. The visual encoder remains
fully frozen, as it is in the existing Stage 2 path. The continuous latent
recurrence, loss semantics, multimodal positions, data transformation, and
batch-size-one correctness boundary must not change.

This design extends the approved Stage 2 design in
`2026-08-24-ilvr-stage2-reimplementation-design.md`; it does not replace it.

## 2. Repository Context

The current branch already has one authoritative Stage 2 recurrence in the
Qwen model path, a Stage 2 CoMT collator, checkpoint metadata validation, a thin
Stage 2 trainer, and a model-specific inference controller.

The relevant current behavior is:

- `src/main.py` implements Stage 1 LoRA using `peft.LoraConfig` and
  `get_peft_model`, but the Stage 2 construction path does not attach PEFT.
- `src/utils_deepseed.py` exposes the legacy Stage 1 `--use_lora` switch and
  shared LoRA hyperparameters, and explicitly rejects Stage 2 LoRA.
- `src/stage2/checkpointing.py` distinguishes a fresh Stage 1-to-Stage 2 handoff
  from resuming a Stage 2 Trainer checkpoint.
- `src/stage2/inference.py` and `src/evaluate_deepseed.py` currently expect the
  Qwen Stage 2 methods directly on the model (or one DeepSpeed `.module` layer).
- the frozen visual encoder is evaluated once per Stage 2 batch before the
  differentiable language-model recurrence.
- `peft==0.17.1` is already a repository dependency and provides DoRA through
  `LoraConfig(use_dora=True)`.

PEFT support therefore requires model construction, wrapper resolution,
checkpoint, evaluation, launcher, and tests. It must not introduce a second
latent rollout implementation.

## 3. Binding Scope

### 3.1 Supported methods

- `full`: existing Stage 2 full language-model fine-tuning behavior.
- `lora`: standard PEFT LoRA.
- `dora`: PEFT LoRA configuration with `use_dora=True`.

The initial implementation does not support QLoRA, AdaLoRA, rsLoRA,
prompt/prefix tuning, arbitrary PEFT methods, adapter fusion, or mixed adapter
types.

### 3.2 Batch and recurrence scope

Stage 2 remains explicitly limited to `per_device_train_batch_size == 1` and a
rollout batch dimension of one. This PEFT work does not generalize recurrent
scheduling to `B > 1`. Multiple latent blocks within one sample remain fully
supported.

### 3.3 Merge/export scope

Stage 2 PEFT checkpoints are adapter-only checkpoints. Merging adapters into a
standalone base model, exporting a merged model, publishing, pushing, or making
a PR is separate future work.

## 4. Command-Line Interface

Add one Stage 2 selector:

```text
--tuning_method full|lora|dora
```

Its effective default for a fresh run is `full`, preserving existing Stage 2
behavior. On resume, an omitted selector inherits the checkpoint sidecar's
method. Argument parsing must retain whether the user explicitly supplied the
selector so an implicit fresh-run default is not mistaken for a conflicting
resume override.

Both `lora` and `dora` use the existing hyperparameters:

```text
--lora_r
--lora_alpha
--lora_dropout
--lora_target_modules
```

The Stage 2 default target list is constrained to:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

Stage 1 retains its existing `--use_lora` behavior and compatibility. For
Stage 2, `--use_lora` is invalid even when it appears consistent with
`--tuning_method`; the CLI must fail with a message directing the caller to
`--tuning_method lora`. This avoids two sources of truth.

Invalid method names, empty target lists, targets outside the approved decoder
set, and contradictory Stage 2 flags fail before model loading or GPU work.

The Stage 2 launcher exposes `TUNING_METHOD`, forwards the shared LoRA
hyperparameters, prints the resolved setting in dry-run output, and rejects
duplicate managed arguments in passthrough input using its existing policy.

## 5. Model and Wrapper Architecture

### 5.1 Construction order

For fresh LoRA or DoRA Stage 2 training:

```text
load Stage 1 Qwen base model and processor
        -> validate/configure Stage 2 latent metadata
        -> freeze the entire visual module
        -> attach PEFT adapter to approved language-decoder modules
        -> verify trainable-parameter boundary
        -> construct Stage2Trainer with fresh optimizer/scheduler state
```

For `full`, construction remains the current Stage 2 path.

### 5.2 Standard PEFT wrapper

Use the standard `PeftModel` produced by `get_peft_model` rather than adding
LoRA layers manually. LoRA uses `LoraConfig(use_dora=False)` and DoRA uses
`LoraConfig(use_dora=True)`. Other configuration fields are identical unless a
later approved design changes them.

The adapter task type must match causal language modeling, while preserving the
Qwen multimodal wrapper's existing forward contract.

### 5.3 One authoritative recurrence

The PEFT wrapper must delegate model forward calls to the existing Qwen Stage 2
forward. The underlying Qwen model continues to own:

- original-image encoding exactly once per batch;
- complete MRoPE computation and recurrent prefix slicing;
- `LatentRolloutEngine` invocation;
- final CE computation and Stage 2 diagnostics.

No latent recurrence is added to `PeftModel`, `Stage2Trainer`, generic
`GenerationMixin`, or evaluation scripts.

### 5.4 Explicit unwrapping resolver

Introduce one Stage 2 model resolver used anywhere code needs Qwen-specific
methods or config:

```text
DeepSpeedEngine.module
        -> PeftModel.get_base_model()
        -> Qwen2_5_VLForConditionalGeneration
```

The resolver handles any absent wrapper layer without relying on fragile
attribute forwarding. It returns both the training-facing model when needed and
the authoritative Qwen base for Stage 2 methods. Unsupported wrapper topology
or a base lacking the Stage 2 API raises a descriptive error.

Training still passes the outer `PeftModel` to Trainer/DeepSpeed so adapters are
the parameters optimized and checkpointed. The base resolver is for explicit
Stage 2 method/config access, not for bypassing PEFT forward hooks.

## 6. Trainable-Parameter Boundary

### 6.1 Visual encoder

Every parameter under the actual Qwen `model.visual` component must have
`requires_grad=False` for all three tuning methods. LoRA/DoRA injection must
never select visual modules. Visual encoding remains under `torch.no_grad()` and
occurs once per batch.

### 6.2 LoRA and DoRA targets

Adapters may be attached only to decoder modules whose terminal names are:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Target resolution must additionally prove each matched module belongs to the
language decoder, rather than trusting only a terminal-name match. Each
requested target must match at least one permitted module. The resolved module
names are recorded in adapter metadata.

### 6.3 Frozen base parameters

For `lora` and `dora`:

- base decoder parameters are frozen;
- visual parameters are frozen;
- `embed_tokens` is frozen;
- `lm_head` is frozen;
- only adapter-owned parameters are trainable, including DoRA magnitude
  parameters where applicable.

No `modules_to_save` exception is introduced for embeddings or the LM head.
Latent special tokens already exist in the Stage 1 tokenizer/checkpoint, so the
Stage 2 PEFT flow must not resize or newly train token embeddings.

After adapter attachment, fail fast unless the nonzero trainable set is exactly
the expected PEFT parameter set and contains no visual, embedding, LM-head, or
unrelated base parameters. Log total and trainable parameter counts once.

For `full`, the existing policy remains: the visual module is frozen and the
language model is trainable.

## 7. Gradient Semantics

PEFT changes which weights receive gradients, not the latent computation graph.
The downstream text CE must remain connected through `zK ... z1` into the
adapter parameters active in all recurrent prefix forwards and the final
forward.

Required properties are:

- no latent `.detach()` or graph break;
- finite, nonzero gradients on representative adapter parameters;
- no gradients on visual or frozen base parameters;
- identical recurrence indexing and latent-pad independence to full tuning;
- adapter effects participate in every recurrent step, not only the final CE
  forward.

## 8. Checkpoint Layout and Metadata

### 8.1 Adapter-only checkpoints

Trainer checkpoints for LoRA/DoRA save adapter weights/config using PEFT's
standard format plus normal Trainer state required for resume. They do not copy
or merge full Stage 1 base weights into every checkpoint.

Every PEFT Stage 2 checkpoint includes a repository-owned metadata sidecar. Its
schema records at least:

```text
format/schema version
stage = "stage2"
tuning_method = "lora" | "dora"
base_model_reference
latent_size
latent_start_token_id
latent_pad_token_id
latent_end_token_id
supervise_latent_boundaries
lora_r
lora_alpha
lora_dropout
requested target modules
resolved target module names
PEFT library version
```

The sidecar is written for ordinary numbered Trainer checkpoints and the final
adapter artifact. The tokenizer and processor metadata needed to preserve
latent-token identity are saved alongside the adapter artifact, without adding
tokens or changing their IDs.

### 8.2 Base-model relocation

`base_model_reference` is a locator and provenance hint, not an immutable path
lock. Resume/evaluation may use an explicit replacement Stage 1 base path.
Relocation is accepted only after structural validation:

- supported Qwen model/config family;
- tokenizer/config latent token IDs match exactly;
- `latent_size` and boundary-supervision metadata match;
- language hidden size and relevant architecture fields are compatible;
- all recorded adapter target module names exist with compatible shapes;
- adapter type and hyperparameters match the checkpoint sidecar and PEFT
  config.

This design intentionally does not hash full base-model weights. Validation
therefore proves structural and token compatibility, not byte-identical base
weights. The selected base path and any difference from the recorded reference
are logged clearly.

## 9. Fresh Training and Resume Semantics

### 9.1 Fresh Stage 1 to Stage 2

```text
explicit Stage 1 checkpoint
        -> load base weights/config/tokenizer/processor/latent metadata
        -> attach a new randomly initialized LoRA or DoRA adapter
        -> start new optimizer, scheduler, global step, and Trainer state
```

Stage 1 optimizer/scheduler state and any Stage 1 adapter state are not resumed.
If the specified Stage 1 source is itself a PEFT-only artifact, the initial
implementation fails clearly rather than implicitly stacking or merging
adapters.

### 9.2 Resume an existing Stage 2 PEFT run

Resume requires two explicit conceptual inputs:

```text
compatible Stage 1 base model source
Stage 2 Trainer checkpoint containing the trainable adapter and Trainer state
```

The loader validates sidecar metadata first, constructs the same LoRA/DoRA
adapter topology in trainable mode, loads adapter weights, and then lets Trainer
restore optimizer, scheduler, scaler, RNG, and global-step state through normal
checkpoint semantics.

The resume checkpoint determines `tuning_method` and adapter hyperparameters.
An omitted CLI selector inherits those values; an explicitly conflicting CLI
value fails rather than silently overriding them. Resuming an adapter checkpoint
as `full`, switching LoRA to DoRA, or switching DoRA to LoRA is unsupported.

The existing distinction remains intact:

```text
Stage 1 -> fresh Stage 2: reset training state
Stage 2 PEFT checkpoint -> resumed Stage 2 PEFT: restore training state
```

## 10. Evaluation and Inference

Evaluation accepts a compatible base model plus a Stage 2 adapter artifact.
It loads the adapter in inference mode, resolves the underlying Qwen Stage 2
interface explicitly, and runs the existing model-specific no-cache Stage 2
generation controller.

Inference must not merge adapters implicitly. It must not add Stage 2 behavior
to generic `GenerationMixin`. The same wrapped forward path must affect normal
text logits and each continuous latent rollout step.

Train/inference recurrence parity tests run with the same adapter weights and
assert matching latent states within numerical tolerance.

## 11. Trainer and Diagnostics

`Stage2Trainer` remains thin. It may add checkpoint-sidecar saving and PEFT
parameter diagnostics, but it does not own rollout logic.

Existing diagnostics remain observability-only:

- CE loss;
- latent block count;
- latent state L2 norms;
- adjacent latent cosine similarity.

PEFT mode adds one-time parameter-count and resolved-target reporting. It must
not add an auxiliary loss or modify Stage 2 semantics.

## 12. Error Handling

Fail before expensive training for:

- unsupported or contradictory tuning flags;
- invalid/empty/out-of-scope target modules;
- visual, embedding, or LM-head parameters becoming trainable in PEFT mode;
- no trainable adapter parameters;
- unsupported wrapper topology;
- missing Stage 2 PEFT sidecar on resume;
- adapter/sidecar mismatch;
- latent token, model architecture, or target-shape incompatibility;
- attempting adapter resume without an explicit compatible base source;
- attempting to resume with a different tuning method;
- `B > 1`, according to the existing Stage 2 safety rule.

Do not hide these errors with broad fallbacks. Unexpected PEFT, numerical,
gradient, CUDA, or DeepSpeed failures require the systematic-debugging workflow
before a fix.

## 13. Test-Driven Implementation and Acceptance

Every behavior change begins with a focused failing test. CPU/tiny-model tests
come before GPU integration.

### 13.1 CLI and routing

- default Stage 2 method is `full`;
- `lora` and `dora` parse with shared hyperparameters;
- Stage 1 `--use_lora` remains unchanged;
- Stage 2 rejects `--use_lora` and contradictory settings;
- invalid methods/targets fail before model load;
- launcher dry-run renders the selected method and settings.

### 13.2 Adapter construction and parameter safety

- LoRA constructs `LoraConfig(use_dora=False)`;
- DoRA constructs `LoraConfig(use_dora=True)`;
- only approved decoder modules are adapted;
- every requested target resolves;
- visual, embeddings, LM head, and base decoder remain frozen;
- adapter parameters are the only nonzero trainable set;
- `full` retains the existing trainable boundary.

### 13.3 Recurrence correctness under PEFT

Run the existing Stage 2 acceptance invariants with a tiny PEFT-wrapped Qwen:

- exact `K` latent count and off-by-one recurrence;
- gradient continuity into representative LoRA and DoRA parameters;
- latent-pad embedding independence;
- MRoPE prefix-slice parity;
- one visual encoder call and frozen visual gradients;
- multi-block conditioning;
- loss masking and structural validation;
- explicit `B > 1` failure;
- train/inference latent parity.

A focused spy/regression test proves adapter computation is active in recurrent
prefix forwards, not only in the final full-sequence forward.

### 13.4 Checkpoint tests

- fresh Stage 2 adapter construction resets optimizer/scheduler state;
- each checkpoint and final artifact contains adapter files and sidecar metadata;
- the checkpoint does not contain a merged/full base-model weight artifact;
- compatible base relocation succeeds;
- token, architecture, shape, target, method, and hyperparameter mismatches fail;
- resume restores adapter weights and Trainer state;
- inference loads base plus adapter without merge.

### 13.5 Stage 1 and full-tuning regressions

- existing Stage 1 routing and LoRA behavior remain unchanged;
- existing Stage 2 `full` tests continue to pass;
- generic generation regression tests remain unchanged;
- the full feasible suite passes once after focused validation.

### 13.6 Required two-GPU ZeRO-3 smoke tests

Both LoRA and DoRA receive a tiny two-GPU DeepSpeed ZeRO-3 save/resume
validation. Each validation must demonstrate:

```text
initial short training segment
        -> numbered adapter checkpoint with sidecar
        -> new process resumes from that checkpoint
        -> global step advances
        -> adapter parameters update
        -> optimizer/scheduler state is restored
        -> visual and frozen base parameters remain untrained
```

These are integration smoke tests, not paper reproduction. They run only after
CPU/tiny-model correctness. If hardware is unavailable, the final report must
say they were not verified; implementation completion may not claim verified
ZeRO-3 adapter resume without fresh evidence.

In accordance with repository policy, do not repeat an already successful
expensive validation or hash full checkpoint/model trees merely to create
publication metadata.

## 14. Documentation and Launcher Changes

Update the Stage 2 training documentation and launcher examples with:

- full, LoRA, and DoRA invocations;
- explicit Stage 1 base source and Stage 2 adapter resume source;
- adapter-only checkpoint layout;
- relocation validation behavior;
- the fact that merge/export is separate;
- visual freeze and target-module boundary;
- `B == 1` limitation;
- verified versus unverified DeepSpeed status.

## 15. Expected Implementation Boundaries

Expected production changes are concentrated in:

- `src/utils_deepseed.py`: Stage 2 tuning CLI and early validation;
- `src/main.py`: tuning routing and PEFT construction;
- `src/stage2/checkpointing.py`: PEFT sidecar, load, compatibility, and resume
  decisions;
- a small Stage 2 PEFT/resolution module under `src/stage2/` if separation keeps
  construction and unwrapping independently testable;
- `src/stage2/trainer.py`: only checkpoint-sidecar plumbing if required;
- `src/stage2/inference.py`, `eval.py`, and `src/evaluate_deepseed.py`: explicit
  wrapper resolution and base-plus-adapter loading;
- `run_stage2_training.sh` and Stage 2 documentation;
- focused tests under `tests/stage2/` plus the two-GPU integration smoke path.

No production change is expected in the latent rollout engine or generic
`transformers/generation/utils.py`. If implementation evidence shows such a
change is unavoidable, stop and return to design approval before modifying
those semantics.

## 16. Completion Criteria

The PEFT extension is complete only when:

1. `full`, `lora`, and `dora` route explicitly and Stage 1 is unchanged;
2. LoRA/DoRA adapt only approved language-decoder projections;
3. the visual encoder, embeddings, LM head, and base weights remain frozen in
   PEFT modes;
4. the differentiable continuous recurrence and train/inference parity pass for
   both adapter methods;
5. adapter-only save, base relocation validation, evaluation loading, and
   Trainer resume semantics pass;
6. tiny two-GPU ZeRO-3 save/resume validation passes independently for LoRA and
   DoRA, or is reported explicitly as unverified due to resources;
7. focused Stage 2, Stage 1 regression, and one proportionate full test suite
   pass with fresh evidence;
8. merge/export remains unimplemented and clearly documented;
9. `B > 1` remains explicitly unsupported;
10. verification-before-completion and code-review workflows find no unresolved
    requirement or correctness issue.
