# ILVR Stage 2 Clean Reimplementation Design

Date: 2026-08-24

Status: Approved in chat; written specification awaiting user review

## 1. Purpose

This document specifies a clean reimplementation of ILVR Stage 2 for the
repository's local Qwen2.5-VL fork. The implementation follows the paper-level
continuous-latent semantics and the user-approved requirements. Existing
experimental Stage 2 code is a behavioral reference only and is not the
authoritative design.

The target training transition is:

```text
Stage 1 checkpoint
        -> self-generated continuous latent reasoning
        -> text-only cross-entropy supervision
        -> Stage 2 checkpoint
```

The initial reproduction target is CoMT.

## 2. Repository audit summary

The Stage 2 worktree starts from commit `13a67ed` on branch
`feat/vcsd-ilvr-stage2`, with no branch-specific production changes.

The audit established the following facts:

- `src/main.py` accepts a stage argument but always chooses the Stage 1
  collator and `CustomTrainerStage1`.
- `load_model_path` is defined but does not select the model or processor.
- `CustomTrainerStage1` owns EMA teacher, helper-image encoding, feature
  selection, similarity loss, and EMA updates. None of these belong in Stage 2.
- The local Qwen forward contains an experimental recurrent Stage 2 branch.
  It is mixed into the large model forward, lacks structural validation and an
  explicit batch-size contract, controls gradient checkpointing internally,
  and uses mutable `rope_deltas` state.
- The local generic `GenerationMixin._sample` contains ILVR-specific state,
  hard-coded latent IDs, and a batch-global latent-mode controller. The former
  generic sampler remains in the file as `_sample_backup`.
- `eval.py` and `src/evaluate_deepseed.py` depend on the modified generic
  generation path and a latent-template logits processor.
- `configs/config_stage2.json` describes DeepSpeed ZeRO stage 2; it is not an
  ILVR Stage 2 configuration.
- The available Stage 1 checkpoint stores the processor, tokenizer, weights,
  `stage=stage1`, `latent_size=8`, and legacy latent token-ID metadata. Its
  tokenizer contains the three latent special tokens with IDs 151665-151667.
- The worktree uses `./venv/ilvr-venv`, but that shared environment's editable
  Transformers installation points at the main checkout. Worktree commands
  must place the worktree's `transformers/src` and `src` first on `PYTHONPATH`.
- The local CoMT training file contains 3,450 samples and 10,701 latent/helper
  steps. Samples contain one to six latent blocks. No latent step is missing a
  helper-image field.

The Stage 1 checkpoint weights are large immutable inputs. The audit inspected
their small configuration/tokenizer metadata only and did not reread or hash
the weight shards.

## 3. Scope

### 3.1 In scope

- A single authoritative continuous-latent recurrence implementation.
- Exact final-`last_hidden_state` recurrence with no graph break.
- Full-sequence MRoPE computation followed by exact prefix slicing.
- Multiple ordered latent blocks in one sample.
- Per-rank micro-batch size one with explicit rejection of `B > 1`.
- CoMT Stage 2 transformation, length filtering, structural validation, and
  loss masking.
- Stage 1 checkpoint initialization and explicit Stage 2 resume behavior.
- A thin Stage 2 trainer and explicit Stage 1/Stage 2 routing.
- A model-specific no-cache Stage 2 inference controller.
- Train/inference recurrence parity.
- Safe removal or bypass of Stage 2 behavior in generic generation after
  regression coverage exists.
- Lightweight diagnostics that do not affect the objective.
- Correctness-first integration followed by FlashAttention, gradient
  checkpointing, DeepSpeed ZeRO-3, and distributed compatibility work.

### 3.2 Out of scope

- Qwen3-VL or other model families.
- Datasets other than CoMT.
- RL, GRPO, or additional objectives.
- Adaptive latent length, projection, normalization, quantization, sampling,
  or learned latent lookup.
- Visual-tower fine-tuning.
- Per-rank recurrent batches larger than one.
- Differentiable KV-cache optimization or custom kernels.
- Complex block-aware truncation.
- Generic Transformers generation redesign.
- LoRA support for the new Stage 2 path in the initial implementation.
- Paper-wide or paper-level reproduction claims without an actual run.

Data parallelism and gradient accumulation may produce an effective batch size
larger than one. Every rank still receives `per_device_train_batch_size=1`.

## 4. Authoritative mathematical contract

For a block with scaffold

```text
... <latent_start> L1 L2 ... LK <latent_end> ...
```

the recurrence is:

```text
z1     = final last_hidden_state after consuming <latent_start>
z(i+1) = final last_hidden_state after consuming zi
```

For latent slot position `p`, the recurrent forward consumes positions
`[0, p)`, and its last output state replaces the embedding at `p`. This fixes
the off-by-one contract:

```text
latent_1 is produced by the prefix ending at <latent_start>
latent_2 is produced by the prefix ending at latent_1
```

The replacement is exactly:

```python
latent = hidden_state
```

The implementation must not detach, project, normalize, quantize, sample, or
look up a token embedding for the generated latent state.

After every latent slot in every block has been populated, the language model
runs once over the completed embedding sequence. The normal language-model
head and shifted cross entropy compute the loss. Downstream text loss must
backpropagate through every latent state and every recurrent prefix forward.

## 5. Architecture

### 5.1 Stage 2 token metadata

`src/stage2/checkpointing.py` will expose one token-metadata resolver and
validator. Its canonical fields are:

- `latent_pad_token_id`
- `latent_start_token_id`
- `latent_end_token_id`

For Stage 1 compatibility it will also recognize:

- `latent_token_id`
- `latent_start_id`
- `latent_end_id`

If canonical and legacy fields both exist, their values must match. Tokenizer
IDs and config IDs must also match. Any mismatch raises a descriptive error.
The Stage 2 loader never blindly adds tokens. A fresh Stage 2 load requires the
Stage 1 checkpoint tokenizer to contain all three tokens already.

Before saving a Stage 2 checkpoint, the model config will contain:

- `stage = "stage2"`
- `latent_size`
- all three canonical token-ID fields
- matching legacy token-ID aliases for Stage 1 compatibility
- `supervise_latent_boundaries`

### 5.2 Latent span representation and validation

The rollout module will use an immutable `LatentSpan` value containing:

- the `<latent_start>` position
- the ordered tuple of exactly `K` latent-slot positions
- the `<latent_end>` position

A pure parser validates one token sequence and returns ordered spans. It rejects:

- unequal start/end counts
- an end without a preceding start
- nested starts
- missing ends
- any non-latent-pad token inside a block
- any block with a slot count different from `latent_size`
- overlapping, reversed, or incomplete blocks

The parser is shared by the collator and model-facing Stage 2 validation so bad
data cannot bypass checks through direct model invocation.

### 5.3 `LatentRolloutEngine`

The authoritative engine will live at:

```text
transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py
```

It is model-specific but independent of Trainer and generic generation. Its
conceptual API is:

```python
rollout(
    language_model,
    base_inputs_embeds,
    attention_mask,
    full_position_ids,
    latent_spans,
    collect_debug_state=False,
) -> LatentRolloutResult
```

`LatentRolloutResult` contains the final input embeddings, ordered latent
states, and optional detached diagnostic metadata. The production result does
not alter the latent tensors to collect diagnostics.

The engine enforces:

- batch dimension exactly one
- `use_cache=False`
- ordered processing of every block and slot
- complete sequence context across blocks
- block-local slot numbering only
- exact slices of the supplied full position IDs
- differentiable functional embedding replacement

The engine calls only the Qwen language model (`model.model`), never the visual
encoder or language-model head. It does not compute MRoPE, loss, sampling, or
checkpoint state.

### 5.4 Qwen Stage 2 forward integration

`Qwen2_5_VLForConditionalGeneration.forward` remains the model-facing training
entry point. The existing experimental Stage 2 recurrence is removed rather
than retained as a second implementation.

For `config.stage == "stage2"`, the wrapper will:

1. validate `B == 1` and token metadata;
2. build token embeddings;
3. encode original user images exactly once under `torch.no_grad()` using the
   frozen visual tower;
4. merge image features into the base token embeddings;
5. parse and validate latent spans;
6. call `get_rope_index` once for the complete scaffold;
7. invoke `LatentRolloutEngine` with explicit full position IDs;
8. run the final full language-model forward;
9. compute logits and the normal shifted CE loss;
10. return optional Stage 2 diagnostics without changing the objective.

Stage 2 never consumes `pixel_values_latent`, `image_grid_thw_latent`,
teacher-provided `latent_hidden_states`, or `image_out_mask`.

The Stage 1 path, including its current helper-image behavior, remains intact.
Stage 2 will not mutate `self.rope_deltas` during recurrent training. Existing
normal Stage 1/generation behavior may retain that state until the model-specific
generation cleanup is covered by tests.

### 5.5 MRoPE contract

The complete token scaffold, attention mask, original image grid metadata, and
normal Qwen `get_rope_index` implementation define one `full_position_ids`
tensor. Every recurrent call receives:

```python
full_position_ids[..., :prefix_end]
```

The engine cannot recompute position IDs. Tests compare every supplied prefix
against the corresponding slice of the complete multimodal position tensor.
Training and inference construct the same scaffold for a latent block, so the
same prefix receives the same MRoPE positions.

### 5.6 Visual and gradient semantics

Before Stage 2 training, every `model.visual` parameter is set to
`requires_grad=False`. Stage 2 visual encoding is performed once per batch and
may run under `torch.no_grad()`.

The language model, recurrent prefix forwards, inserted latent states, final
full forward, and loss are never placed in `no_grad`. Tests verify that:

- visual parameters remain frozen and have no gradients;
- expected language-model parameters receive finite, non-zero gradients;
- retained latent debug tensors receive gradients from downstream text CE;
- perturbing the latent-pad vocabulary embedding does not alter rollout states
  or downstream logits under an otherwise identical setup.

### 5.7 CoMT Stage 2 data path

`src/stage2/data.py` will own the Stage 2 CoMT transformation and collator.
Stage 1 preprocessing in `src/task_deepseed.py` will remain unchanged.

For each raw CoMT sample:

- original/user image paths are resolved from an explicit `image_root` argument;
- user text and original images form the user message;
- assistant text steps are preserved in order;
- every `latent` step is replaced at that exact location by one block of
  `<latent_start>`, `K` latent-pad tokens, and `<latent_end>`;
- helper-image field presence is validated, but helper images are neither
  opened nor passed to the processor;
- unknown trajectory step types fail immediately.

Preprocessing applies no truncation. An initial length-audit pass uses the real
processor and original image to measure the final model sequence length. A
sample exceeding `max_seq_length_train` is removed as a whole. Logging includes:

- total and retained sample counts
- filtered count and percentage
- min, median, p90, p95, p99, and maximum lengths
- latent-block count distribution

The collator emits only model-relevant tensors, including original
`pixel_values` and `image_grid_thw`. It must never emit helper-image tensors.
It runs structural validation after tokenization and rejects `B > 1`.

### 5.8 Loss masking

The only new Stage 2 setting is:

```python
supervise_latent_boundaries: bool = True
```

Prompt/user positions, padding, and continuous latent slots always receive
label `-100`. Assistant text is supervised. Boundary behavior is atomic:

```text
True:  latent_start supervised, latent slots ignored, latent_end supervised
False: latent_start ignored,    latent slots ignored, latent_end ignored
```

There are no independent start/end flags. Existing Stage 1 label behavior and
`latent_ce_ratio` remain unchanged; Stage 2 does not repurpose that argument.

### 5.9 Training routing and trainer

`src/main.py` will explicitly build one of two paths:

```text
stage1 -> existing Stage 1 processor setup, collator, and CustomTrainerStage1
stage2 -> Stage 2 checkpoint loader, CoMT data path, collator, and Stage2Trainer
```

`src/stage2/trainer.py` will define a thin `transformers.Trainer` subclass. It
will use the model-provided CE loss and only collect/log detached Stage 2
diagnostics. It will not implement recurrence, visual encoding, label
construction, or a new loss objective.

For the correctness milestones:

- `per_device_train_batch_size=1`
- `use_cache=False`
- gradient checkpointing disabled
- DeepSpeed disabled
- eager attention for tiny M1 correctness tests
- SDPA for the first realistic single-device M2 smoke test

Later compatibility milestones enable each training-system feature separately
without changing the rollout contract.

### 5.10 Stage 1 initialization and Stage 2 resume

Fresh Stage 2 initialization uses:

```text
--stage stage2
--load_model_path <Stage 1 checkpoint>
```

The model, config, tokenizer, processor, and latent metadata are loaded from
that checkpoint. A new Trainer creates a new optimizer, scheduler, and Stage 2
global state. Stage 1 optimizer/scheduler state is never loaded.

Resuming an existing Stage 2 run uses:

```text
--stage stage2
--resume_from_checkpoint_path <Stage 2 trainer checkpoint>
```

The resume checkpoint must validate as `stage2`. Normal Trainer restoration of
model, optimizer, scheduler, scaler, RNG, and global step is then allowed.
Stage 2 does not implicitly resume from whatever happens to be present in the
output directory. Stage 1 retains its current behavior unless a focused
regression-preserving change is required.

### 5.11 Stage 2 inference controller

`src/stage2/inference.py` will implement a Qwen-specific, no-cache controller.
It does not call generic `GenerationMixin` for Stage 2 recurrence.

The controller maintains:

```text
NORMAL_TEXT -> LATENT_MODE -> force latent_end -> NORMAL_TEXT
```

In normal mode it performs repository-supported greedy or temperature/top-p
text selection. When the model naturally emits `<latent_start>`, the controller:

1. constructs the complete provisional block scaffold containing `K` latent
   slots and the end boundary;
2. computes the complete block's position specification;
3. invokes the shared `LatentRolloutEngine`;
4. records exactly `K` structural latent-pad IDs without vocabulary sampling;
5. forces `<latent_end>`;
6. resumes normal text generation with all previous context and latent states.

The original image is encoded once and its merged embedding is retained while
the no-cache language-model prefix is recomputed. Multiple blocks are supported.
EOS and normal stopping criteria are not evaluated inside latent mode. Once a
start token has been emitted, the complete block is always appended, even if
this exceeds the nominal `max_new_tokens` budget by up to `K + 1` structural
tokens. A hard model sequence-length overflow remains an explicit error.

This historical limit policy is superseded for evaluation by design commitment
v29 in `2026-09-01-stage1-batch-equivalence-design.md`: both caps are strict,
and a truncated scaffold is returned without running an incomplete recurrence.

The initial controller supports `B=1` only. It does not implement beam search,
arbitrary Hugging Face generation features, or KV caching.

### 5.12 Generic generation cleanup

No new Stage 2 behavior will be added to generic generation. Cleanup occurs
only after regression tests cover ordinary local-Qwen greedy/sampling behavior
and the new Stage 2 controller.

The cleanup will remove:

- hard-coded latent IDs
- generic latent-mode state
- `latent_hidden_states` accumulation in generic output
- the Stage 2-specific tokenizer plumbing into `_sample`
- the duplicate `_sample_backup` once the normal sampler is authoritative

The local Qwen fork also uses `generate_mode` to select normal FlashAttention
instead of its training/Ulysses path. To preserve ordinary local-Qwen
generation after generic cleanup, `Qwen2_5_VL.prepare_inputs_for_generation`
will set `generate_mode=True` as model-specific plumbing. It will not remain a
Stage 2 mechanism in generic generation.

### 5.13 Diagnostics

Optional Stage 2 diagnostics are observability only:

- CE loss
- latent block count
- mean/min/max latent L2 norm
- adjacent latent-state cosine similarity
- early latent-state gradient norm when explicitly enabled for debugging
- dataset filtering counts and length percentiles

Diagnostics are detached only after the mathematical outputs are produced.
They do not add loss terms or permanent per-step verbose logs. The rollout
result keeps latent states inspectable so later zero/random/detached ablations
can be added without restructuring the engine.

## 6. Error handling

The implementation raises explicit errors for:

- missing or mismatched special tokens
- Stage 1/Stage 2 checkpoint type mismatch
- invalid latent size
- malformed or incomplete CoMT trajectories
- malformed tokenized latent blocks
- `B > 1`
- helper-image tensors appearing in a Stage 2 batch
- unexpected visual token/feature count mismatch
- impossible MRoPE or attention-mask shapes
- hard maximum sequence-length overflow during inference

It will not use broad exception handlers or fallback paths that change the
continuous-latent semantics.

## 7. Expected file changes

New production files:

- `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`
- `src/stage2/__init__.py`
- `src/stage2/data.py`
- `src/stage2/checkpointing.py`
- `src/stage2/trainer.py`
- `src/stage2/inference.py`
- `run_stage2_training.sh`

Expected modifications:

- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- `transformers/src/transformers/generation/utils.py` during M3 only
- `src/main.py`
- `src/utils_deepseed.py`
- `eval.py`
- `src/evaluate_deepseed.py`
- `README.md`

New focused tests will live under `tests/stage2/`. Stage 1 trainer and task
preprocessing files are intentionally left unchanged unless a test exposes a
strictly necessary compatibility edit.

All Python, training, evaluation, and test commands use
`./venv/ilvr-venv/bin/` and explicitly prefer the worktree's local sources:

```text
PYTHONPATH=<worktree>/transformers/src:<worktree>/src
```

## 8. TDD and acceptance coverage

Every production behavior begins with a test that fails for the intended
reason. Focused suites map to the required acceptance behaviors:

| Requirement | Focused coverage |
| --- | --- |
| Exact latent count | One and multiple blocks each produce exactly `K` states |
| Exact recurrence | Deterministic toy LM verifies every off-by-one state |
| Gradient continuity | Downstream CE reaches every retained latent and LM parameter |
| Pad independence | Large latent-pad embedding perturbation leaves rollout unchanged |
| MRoPE parity | Every recurrent prefix equals the complete-position slice |
| Visual call count | Counting visual stub observes one call per Stage 2 batch |
| Frozen visual tower | No visual gradients; expected LM gradients exist |
| Multi-block semantics | Block B changes when block A's generated states change |
| Loss masking | Slots always ignored; both boundaries change together |
| Train/inference parity | Shared prefix yields equal `z1..zK` within tolerance |
| Structural validation | Missing, nested, misordered, incomplete, and wrong-K blocks fail |
| B > 1 safety | Descriptive error for a two-sample local batch |
| Stage 1 regression | Stage routing and focused Stage 1 collator behavior remain stable |

Tests use a tiny deterministic language model for mathematical properties and
the real local Qwen position-index function for multimodal position parity. No
large checkpoint is needed for these unit tests.

## 9. Milestones

### M0: Audit and design

Complete the repository audit, approve this design, write an implementation
plan, and preserve all approval gates.

### M1: Rollout correctness

Implement span validation, the no-cache rollout engine, exact indexing,
multi-block behavior, MRoPE slicing, gradient continuity, visual-once behavior,
and `B=1` enforcement. Use tiny tests; do not enable DeepSpeed, FlashAttention,
or gradient checkpointing.

### M2: Training integration

Implement CoMT transformation/filtering/collation, loss labels, Stage 1
checkpoint handoff, Stage 2 metadata, routing, thin trainer diagnostics, visual
freeze, and explicit resume behavior. Run a minimal single-device Stage 2
training smoke test if hardware permits.

### M3: Inference parity

Implement the model-specific controller, update both evaluation entry points,
verify train/inference latent parity, add ordinary-generation regression
coverage, and retire the Stage 2-specific generic generation path safely.

### M4: Training-system compatibility

Test gradient checkpointing, FlashAttention behavior, DeepSpeed ZeRO-3,
distributed rank-local `B=1`, checkpoint save/load, and Stage 2 resume one at a
time. Any unexpected failure follows systematic debugging before a fix.

### M5: CoMT readiness

Record the concrete Stage 2 command and configuration, data/filter counts,
effective batch settings, learning rate, latent size, boundary setting,
checkpoint origin, visual freeze, and optimizer reset. Run only experiments
supported by available resources and distinguish implementation, smoke, and
paper-level reproduction evidence.

## 10. Verification and review

Before any milestone is called complete, fresh focused verification must
support that claim. Final completion requires:

- rereading this approved design and the implementation plan
- checking every requirement against the actual diff
- feasible full and focused test suites
- Stage 1 regression coverage
- Stage 2 training smoke evidence when resources permit
- train/inference parity evidence
- inspection of the final diff
- Superpowers code-review workflow
- explicit reporting of unexecuted resource-heavy checks and remaining limits

An expensive model/checkpoint validation is performed at most once when needed.
Existing equivalent evidence is reused, and no large checkpoint tree is
rehashed for publication or bookkeeping.

## 11. Known limitations of the approved initial design

- Per-rank batch dimension greater than one is unsupported and fails clearly.
- The naive prefix recurrence is intentionally slow.
- No KV-cache optimization is implemented.
- FlashAttention, gradient checkpointing, and DeepSpeed are compatibility
  milestones, not assumptions of M1 correctness.
- Full CoMT filtering rates are unknown until the approved implementation runs
  the real processor length audit.
- A started inference latent block may finish beyond the nominal generation
  budget, but it will never be emitted incompletely.
- Paper-level reproduction is not established by unit or smoke tests.
