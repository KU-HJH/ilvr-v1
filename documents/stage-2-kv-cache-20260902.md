# Stage 2 Differentiable KV-Cache Guidance and Specification

## 1. Status, intent, and authority

- Date: 2026-09-03
- Status: approved design commitment v3; specification only
- Target repository: the Stage 2 feature worktree for ILVR
- Target runtime: the current two-node NVIDIA H100 NVL environment
- Understood as: provide a standalone handoff for a later session to implement
  and validate an opt-in training-only KV-cache candidate without changing the
  current `no_cache` default

This document authorizes design and implementation work for a candidate named
`segmented_kv`. It does not authorize a production training run merely because
the code exists or a timing threshold is met. Production use remains
fail-closed until every gate in Sections 10 and 11 passes on the active H100
target.

The current `no_cache` recurrence remains the semantic oracle, default backend,
resume meaning for legacy checkpoints, and fallback supported by the multi-node
Stage 2 specification. Small deterministic correctness gates use it with
gradient checkpointing disabled. Production-scale performance gates use the
same recurrence with gradient checkpointing enabled as the strongest feasible
operational baseline. B200 measurements are historical and must not be used as
H100 authorization evidence.

This specification refines the deferred KV notes in
`docs/superpowers/specs/2026-09-02-stage2-efficiency-design.md`. It does not
change that document's completed no-cache optimization or schema-v1 evidence.

### 1.1 Reviewed implementation baseline

The specification was derived from feature commit
`13564855b42d3ca78dad462d4e2930073bcd4c60`. The next session must start from
that commit or a reviewed descendant and must record its own full commit. The
reviewed runtime was:

| Component | Observed value |
|---|---|
| GPU | 2× NVIDIA H100 NVL per node, 95,830 MiB each |
| Driver | 580.173.02 |
| Python environment | `./venv/ilvr-venv` |
| PyTorch / CUDA | 2.7.1+cu128 / 12.8 |
| Transformers | 4.50.0.dev0, vendored |
| PEFT | 0.17.1 |
| Accelerate | 1.10.1 |
| FlashAttention | 2.8.3 |

The reviewed SHA256 values were:

| Runtime file | SHA256 |
|---|---|
| `stage2_latent_rollout.py` | `bcc49ba378e790e2acd65efdfac3bc27b530cdf351c7eae19307f4b4e063f25e` |
| `modeling_qwen2_5_vl.py` | `d86ecedf8374d77cf571ef0edf1ed2b2b3567935e23d0d0e8dd0ee8d8b99b913` |
| `cache_utils.py` | `234e05f0b75281c045ba0988ca787d66e9e12b748945afab373bb29c985e42c2` |
| `src/main.py` | `349766b22cbfd5fca0033edc22bb35e74e00dc909b2f1866e5829f15863fe929` |
| `efficiency_probe.py` | `4cabe88269c55f4b6b1a70762942851445c26990d9e570f1dc5c324b7a143ae8` |

These hashes identify the reviewed baseline, not required future values. A
future implementation must update the recorded runtime hashes and rerun all
gates; it must not force modified code to match the historical digests.

### 1.2 Observed capacity correction

On 2026-09-02, the compatible single-H100 `no_cache` probe reached BF16
FlashAttention execution but OOMed during the first p50 warm-up with gradient
checkpointing disabled. The sealed p50 representative has sequence length 648,
32 latent positions, and recurrence work 18,272; p95 work is 68,179 and maximum
work is 388,431. At failure, the process used 92.86 GiB and only 224.56 MiB was
free, while reserved-but-unallocated memory was 1.38 GiB. This is a real
capacity failure, not an allocator-fragmentation result.

A separate `no_cache` Stage 2 run with gradient checkpointing and ZeRO-2
completed 846/846 optimizer steps and wrote its final adapter. That run used
DoRA dropout 0.05 and is not candidate authorization, but it directly shows
that the checkpointed oracle is an operational baseline. Design commitment v3
therefore separates semantic-oracle equivalence from operational-stack
performance instead of requiring an impossible production-scale
no-checkpoint oracle.

## 2. Result in one paragraph

Implement a sibling differentiable rollout engine that creates a fresh
Transformers `DynamicCache` for each rank-local sample, evaluates the initial
prefix once, and then advances only the embedding segment not yet represented
in the cache. All cached key/value tensors and generated continuous latents
remain attached to autograd. After the last latent, discard the runtime cache,
assemble the complete embedding sequence, and run the existing mandatory
full-sequence decoder and shifted cross-entropy path with `use_cache=False`.
Training defaults to `no_cache`; only a fresh compatible training run may
explicitly select `segmented_kv` with valid authorization evidence. Evaluation
and inference always use `no_cache`.

## 3. Non-negotiable Stage 2 method

The candidate may change only how repeated causal-prefix computation is reused.
It must preserve all of the following:

1. The input scaffold, latent token IDs, latent-block order, and latent size are
   unchanged.
2. For latent slot position `p`, the generated continuous latent is the final
   decoder hidden state after consuming exactly positions `[0, p)`.
3. Every generated latent remains attached to autograd and influences later
   latents and the final text loss.
4. No latent tensor or cached key/value tensor is detached, copied through CPU,
   quantized, normalized, projected, or replaced with a latent-pad embedding.
5. Image features, the three-axis multimodal RoPE positions, attention mask,
   causal ordering, label mask, and optional latent-boundary supervision are
   unchanged.
6. After rollout, the existing complete embedding sequence is assembled once.
7. The final full-sequence decoder pass remains mandatory, uses
   `use_cache=False`, and computes the existing shifted assistant-token
   cross-entropy objective.
8. The visual encoder stays frozen and the existing full/LoRA/DoRA trainable
   parameter boundary stays unchanged.
9. Rank-local micro-batch remains one. This work does not change effective
   batch size, sampler membership, accumulation, learning rate, or distributed
   placement.
10. The existing `LatentRolloutEngine` implementation and its tests remain an
    independent oracle. Candidate code must not replace or silently modify it.

Numerical closeness is evidence only within the predeclared tolerances. It does
not excuse a changed recurrence, detached graph, missing final pass, different
mask, or different supervised rows.

## 4. Deliberate compatibility boundary

### 4.1 Why DoRA dropout 0.05 is excluded

The canonical `no_cache` recurrence recomputes earlier prefixes for every
latent slot. With active adapter or attention dropout, those recomputations draw
new dropout masks. A cache reuses the earlier key/value realization instead.
This changes stochastic correlations in the computation graph; it is not merely
BF16 rounding drift.

Therefore `segmented_kv` is valid only for a fresh run in which every active
training-time stochastic module in the language decoder is deterministic:

- DoRA/LoRA dropout is exactly `0.0`;
- model attention dropout is exactly `0.0`;
- every active `torch.nn.Dropout` reachable from the decoder has `p == 0`;
- no stochastic router or other module changes a reused prefix.

The preflight must inspect the active wrapped model after PEFT composition. It
must not trust launcher arguments alone. A DoRA `0.05` checkpoint or run cannot
switch to `segmented_kv`. The implementation must never lower dropout
automatically.

Version 1 supports only a model that resolves through the existing Stage 2
resolver to vendored `Qwen2_5_VLForConditionalGeneration` and its vendored Qwen
language decoder. “Pinned module graph” means that resolved class/config plus
the exact runtime-file digests recorded by qualification; it is not a separate
historical graph fingerprint. Qualification records the new digests, while a
production request must match the accepted comparison's digests exactly.
Preflight walks the resolved decoder and rejects: any
`torch.nn.modules.dropout._DropoutNd` with positive `p`; `torch.nn.RReLU`;
positive `attention_dropout` or other configured decoder dropout; any active
PEFT configuration whose `lora_dropout` is positive; positive `num_experts`;
any qualified module name ending in `.router`; or any class name containing
`Router`, `MoE`, or `MixtureOfExperts` case-insensitively. These are the complete
version-1 operational checks. Any other model class or newly supported
stochastic module requires a new algorithm version and evidence; preflight does
not claim to recognize arbitrary third-party stochastic code.

The latent recurrence and loss method remain the same for an explicitly
dropout-zero run, but dropout-zero is a different training configuration. This
specification makes no claim that its regularization or final model quality is
equivalent to the canonical DoRA `0.05` configuration.

### 4.2 Why gradient checkpointing is excluded

The vendored Qwen decoder currently forces `use_cache=False` during checkpointed
training. Replaying a forward that mutates a cache can also append entries twice
or observe state from a later call. Version 1 therefore rejects
`segmented_kv` whenever decoder gradient checkpointing is enabled. It must not
toggle checkpointing inside rollout.

This restriction applies to the candidate, not to the `no_cache` operational
baseline. The candidate's small-model semantic oracle also disables
checkpointing so that the algorithm comparison has no execution confound. A
separate deterministic test must show that checkpointed and non-checkpointed
`no_cache` produce equivalent outputs, gradients, and optimizer updates before
the checkpointed oracle may serve as the production-scale baseline.

If dropout-zero `segmented_kv` DDP without checkpointing does not fit the p50,
p95, and maximum-work examples, this candidate is unsupported. Do not reduce
sequence length, latent size, adapter rank, or batch size to make the cache path
pass.

### 4.3 Initially supported execution subset

Version 1 supports only:

- training mode;
- rank-local batch size one;
- fresh `DynamicCache` state per sample;
- all-one attention masks;
- deterministic decoder execution;
- gradient checkpointing disabled for `segmented_kv`;
- eager FP32 as the structural oracle backend;
- BF16 FlashAttention 2 only after its separate H100 integration gate passes;
- ordinary DDP only after the single-device gates pass.

Evaluation, generation, inference, `torch.compile`, StaticCache,
SlidingWindowCache, offloaded or quantized cache, ZeRO-3, FSDP, tensor
parallelism, sequence parallelism, and cross-sample cache reuse are outside
version 1.

## 5. Selected architecture

### 5.1 Components

Add one sibling engine, provisionally
`DifferentiableSegmentedKVRolloutEngine`, in a separate model-local module:

```text
transformers/src/transformers/models/qwen2_5_vl/stage2_kv_rollout.py
```

The new engine may reuse immutable Stage 2 structural types and the span parser.
Its cursor advancement and cache logic must not call the candidate through the
no-cache oracle or vice versa. Keeping the algorithms independent prevents a
shared implementation defect from making a comparison pass itself.

The Qwen model continues to construct `LatentRolloutEngine()` as
`stage2_rollout_engine` and adds a default-null
`stage2_training_rollout_engine`. `_forward_stage2` selects the latter only when
`self.training` is true and the training composition path installed it;
otherwise it selects the existing oracle. The composition path in `src/main.py`
installs the candidate only after model/PEFT construction and successful
active-model preflight. Inference loaders never install it, and Trainer
evaluation on a training model automatically selects the oracle.

Keep `model.config.use_cache = False`. The candidate passes `use_cache=True`
only to its internal recurrence calls. The final Stage 2 call still passes
`use_cache=False` explicitly.

### 5.2 Exact recurrence data flow

Let ordered latent positions be `p[0], p[1], ...`. Set `cursor = 0`, create one
empty `DynamicCache`, and execute:

```text
for p in ordered_latent_positions:
    delta = embeddings for [cursor, p)
    outputs = decoder(
        inputs_embeds=delta,
        past_key_values=cache,
        use_cache=True,
        attention_mask=full_attention_mask[:, :p],
        position_ids=full_position_ids[..., cursor:p],
        cache_position=arange(cursor, p),
        return_dict=True,
    )
    assert outputs.past_key_values is cache
    assert cache.get_seq_length() == p
    latent = outputs.last_hidden_state[:, -1, :]
    record replacement (p, latent)
    cursor = p
```

Construct each delta from immutable base-embedding slices and the ordered
replacement map. For `[cursor, p)`, substitute every generated latent whose
position lies in that half-open interval, concatenate the pieces out of place,
and assert the resulting length is `p - cursor`. Never write a latent into the
base tensor in place. The parser already guarantees strictly increasing valid
positions; the candidate additionally rejects an empty initial prefix or an
empty delta.

The assignment `cursor = p`, not `p + 1`, is mandatory. The generated latent at
position `p` has not yet been consumed by the decoder. It must be the first item
in the next delta. Consecutive latent positions therefore advance by one token.
For separate latent blocks, the delta includes the previous generated latent,
the latent-end token, intervening ordinary or image scaffold embeddings, and
the next latent-start token up to but excluding the new latent slot.

The initial delta is the complete prefix `[0, p[0])`. Position IDs are sliced
from the already computed three-axis Stage 2 tensor; they are never regenerated
as scalar positions. `cache_position` is an absolute cache index and must equal
the delta's source positions. The attention mask covers the full key prefix,
not only the current query delta.

After the last latent:

1. do not extend the cache through the rest of the sequence;
2. do not expose the cache in `LatentRolloutResult`;
3. assemble the complete final inputs with all graph-attached latents;
4. release the cache when the forward graph is freed;
5. let the unchanged `_forward_stage2` final decoder/loss path run with
   `use_cache=False`.

A sample with no latent spans returns the same assembled inputs as the oracle
without allocating a cache.

### 5.3 Cache implementation rules

Use only the pinned Transformers `DynamicCache` in version 1. Its current update
path grows per-layer key/value tensors with `torch.cat`, which preserves tensor
autograd history. The implementation must still prove, at runtime in tests,
that every cached layer tensor has the expected sequence length and that
backward reaches the trainable projections that produced it.

The upstream Transformers documentation describes its cache classes as an
inference facility and explicitly warns that enabling cache during training may
cause unexpected errors. This candidate is therefore a repository-specific,
evidence-gated training experiment, not a claim of upstream training support.

Do not use StaticCache: the vendored Qwen class declares static cache unsupported
and static update paths use indexed in-place writes that are outside this
design's autograd contract. Do not detach, clone to CPU, convert to a legacy
tuple, crop, offload, quantize, serialize, or reuse a cache.

Dynamic concatenation can retain intermediate cache versions until backward.
The candidate is expected to reduce decoder computation, but reduced peak
memory is not assumed and must be measured.

## 6. User-facing and runtime contract

Add one explicit selector:

```text
--stage2-recurrence-backend {no_cache,segmented_kv}
```

The launcher environment equivalent is:

```text
STAGE2_RECURRENCE_BACKEND=no_cache|segmented_kv
```

Add a second required selector whenever the backend is `segmented_kv`:

```text
--stage2-kv-run-purpose {qualification,production}
STAGE2_KV_RUN_PURPOSE=qualification|production
```

A qualification run also requires exactly one gate selector:

```text
--stage2-kv-qualification-gate {single_h100_capacity,ddp_adoption,save_resume}
STAGE2_KV_QUALIFICATION_GATE=single_h100_capacity|ddp_adoption|save_resume
```

`single_h100_capacity` executes only Section 10.4,
`ddp_adoption` executes the four child jobs and rank evidence in Sections 10.5
and 11, and `save_resume` executes only the candidate save/resume witness.
Unit/FP32/BF16 test commands remain direct development probes outside the
ordinary launcher. Missing, unknown, or multiple gates fail before model
loading. Both backends may use `ddp_adoption`; `single_h100_capacity` and
`save_resume` require `segmented_kv`. Production rejects this selector.

A production candidate additionally requires both:

```text
--stage2-kv-comparison-path results/.../stage2-kv-comparison.json
--stage2-kv-comparison-sha256 64-lowercase-hex
STAGE2_KV_COMPARISON_PATH=results/.../stage2-kv-comparison.json
STAGE2_KV_COMPARISON_SHA256=64-lowercase-hex
```

The path must be normalized relative to the repository, resolve to a regular
non-symlink file inside `./results/`, match the supplied digest, parse as the
strict schema-v2 comparison, and have `accepted=true`. Qualification rejects
either comparison option so development evidence cannot accidentally change
the run purpose.

There are exactly three execution states:

1. Unit and single-H100 development probes instantiate the engine through test
   helpers and never enter the ordinary training launcher.
2. A `qualification` run may select `segmented_kv` without prior acceptance
   evidence, but must use a fresh run ID under `zero-stage-compatibility`, may
   write only qualification artifacts, and must satisfy every static/runtime
   preflight that can be checked before its selected gate. It cannot execute a
   different gate or be reused as production training.
3. A `production` run must supply an immutable schema-v2 comparison with
   `accepted=true`; its decoded model, execution, data, batch, runtime, and
   candidate algorithm identities must exactly match the requested run.

Omitting the purpose is valid only for an ordinary fresh `no_cache` run. A
qualification purpose is valid for either backend only with an allowed gate.
Selecting `segmented_kv` without a purpose, supplying `production` with
`no_cache`, using qualification artifacts as production authorization, or
requesting production before acceptance fails before model loading. This
separation lets qualification create evidence without authorizing itself.

Default and omitted values resolve to `no_cache`. Unknown values fail before
model loading. `segmented_kv` must fail before distributed preparation if any
precondition in Section 4 is violated. There is no warning-and-fallback path;
an operator who wants the oracle must request `no_cache` in a new run.

The launcher dry run prints the resolved recurrence backend, algorithm version,
dropout values, checkpointing value, and attention backend. It must not load the
model or create a result/checkpoint directory.

The backend selector records the training recurrence. It is not a generation
option or a general Qwen configuration default. Evaluation performed inside a
`segmented_kv` training run still uses `no_cache`; the recurrence sidecar
continues to identify the training backend. “Candidate requested outside
training” means a direct candidate-engine call while `model.training` is false,
or selecting the candidate for a non-Stage-2 launcher. Merely evaluating a
training model with a configured candidate is permitted and routes to the
oracle without invoking the candidate.

## 7. Identity, checkpoint, and resume contract

Create `stage2-recurrence.json` in each numbered checkpoint and final adapter
artifact. Unknown, missing, duplicate, boolean-as-integer, non-finite, or
wrong-type fields fail. The top-level object has exactly these keys:

| Key | Exact type and meaning |
|---|---|
| `schema_version` | integer `1` |
| `backend` | string `no_cache` or `segmented_kv` |
| `algorithm_version` | matching string `no-cache-v1` or `segmented-kv-v1` |
| `git_commit` | 40-character lowercase hexadecimal string |
| `runtime_sha256` | object defined below |
| `model_identity` | object defined below |
| `execution` | object defined below |
| `data_identity` | object defined below |
| `batch_geometry` | object defined below |
| `global_step` | non-negative integer matching the saved Trainer state as defined below |

`runtime_sha256` has exactly these string-to-64-lowercase-hex entries:

```text
transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py
transformers/src/transformers/models/qwen2_5_vl/stage2_kv_rollout.py
transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
transformers/src/transformers/cache_utils.py
src/main.py
src/task_deepseed.py
src/artifact_paths.py
src/peft_common.py
src/stage2/efficiency_probe.py
src/stage2/kv_evidence.py
src/stage2/checkpointing.py
src/stage2/trainer.py
src/stage2/peft.py
src/stage2/sampler.py
run_stage2_training.sh
transformers/src/transformers/trainer.py
```

Every listed path must be tracked at `git_commit`, resolve inside the repository,
and have no tracked worktree modification on either node. Reject an untracked
module that can shadow a listed import through `PYTHONPATH`. Both nodes compare
the complete key set and values before rendezvous; a matching commit alone is
insufficient.

Use the exact qualification `PYTHONPATH=transformers/src:src`. For each imported
runtime module, resolve `importlib.util.find_spec(...).origin` and require its
real path to equal the corresponding listed repository file. Before import,
inspect every earlier path entry for a same-named module file or package and
reject it, whether tracked or untracked. Verify listed-file tracking and dirtiness
with Git pathspecs, not by scanning unrelated working-tree changes.

`model_identity` has exactly:

- `base_reference`: nonempty string;
- `base_tree_sha256`: 64-lowercase-hex string;
- `adapter_reference`: nonempty string or JSON null;
- `adapter_tree_sha256`: 64-lowercase-hex string or JSON null, null exactly
  when `adapter_reference` is null;
- `tuning_method`: `full`, `lora`, or `dora`;
- `peft_version`: nonempty string.

`model_identity` always names the validated initialization inputs, never the
newly trained output directory. `base_reference` and `adapter_reference` are
normalized POSIX paths relative to the repository's canonical `checkpoints/`
link, prefixed with `checkpoints/`; validation first proves that each resolved
root lies inside the real checkpoint-storage target. The adapter fields are
null only for a base-only initialization. Consequently both benchmark controls
and both nodes have identical logical model identity even if their checkpoint
links resolve to different physical mount paths, while their saved output
adapters may differ numerically.

For each input root, recursively enumerate regular files without following
symlinks. Reject any symlink or path that resolves outside the root. For each
file create one JSON object with exactly `path` (root-relative POSIX path),
`size` (non-negative byte count), and `sha256` (digest of the file bytes).
Serialize each object independently with the canonical JSON rules below plus
one newline, order the serialized records by UTF-8 bytes of `path`, concatenate
them without an enclosing array or extra separator, and SHA256 the resulting
JSONL bytes. The empty tree is invalid. No file is excluded.

`execution` has exactly:

- `attention_implementation`: `eager`, `sdpa`, or `flash_attention_2`;
- `precision`: `fp32` or `bf16`;
- `decoder_dropout`, `attention_dropout`, and `adapter_dropout`: finite JSON
  numbers in `[0, 1)`;
- `gradient_checkpointing`: boolean;
- `latent_size`: positive integer;
- `supervise_latent_boundaries`: boolean.

`data_identity` has exactly three 64-lowercase-hex strings:
`manifest_sha256`, `dataset_sha256`, and `sampler_schedule_sha256`.
`batch_geometry` has exactly positive integers `world_size`,
`per_device_batch`, `gradient_accumulation`, and `effective_batch`, and must
satisfy their multiplication identity.

Serialize every sidecar and artifact-tree record with UTF-8,
`sort_keys=True`, compact separators, `ensure_ascii=False`,
`allow_nan=False`, and one terminal newline. SHA256 always means the digest of
the exact serialized bytes. Parsers reject duplicate keys before constructing
the object.

A legacy checkpoint without this sidecar means `no_cache` and
`no-cache-v1`. It never means `segmented_kv`. It may resume only when the
operator supplies the literal CLI or environment value `no_cache`; omission is
accepted for a fresh oracle run but rejected for legacy resume. Every
pre-existing checkpoint, artifact, sampler, optimizer, and runtime compatibility
check must pass. The next
successful numbered save writes a full sidecar describing the current no-cache
runtime; it does not change the backend. A legacy checkpoint requested with
`segmented_kv` fails before model preparation.

When a sidecar exists, resume requires byte-compatible semantic identity: every
field except `global_step` compares exactly, and `global_step` matches both the
checkpoint directory and Trainer state. Switching either direction is a fresh
run from a separately validated initialization artifact, not an optimizer
resume.

For a numbered directory named `checkpoint-N`, `global_step` must equal positive
integer `N` and the `global_step` in `trainer_state.json`. For a final adapter
directory, it equals the Trainer's terminal global step passed directly to the
save hook and, when `trainer_state.json` is present, must match that file. A
final adapter directory name is never parsed as a step. Zero is permitted only
for an explicit pre-step qualification artifact, not a numbered checkpoint.

KV tensors are runtime activations. Never write them to checkpoints, results,
logs, or adapters. The existing checkpoint policy remains unchanged:
model-bearing state under `./checkpoints/`, metadata/results under `./results/`,
and logs under `./logs/`.

For distributed saving, all ranks first agree on the recurrence identity. Only
the designated writer publishes the common sidecar; every rank must validate
the same digest before the save is considered complete.

## 8. Existing evidence and authorization boundary

`src/stage2/efficiency_probe.py` schema version 1 already records:

- whether median recurrence-prefix decoder time is at least 30% of median step
  time; and
- `segmented_kv_authorized=false` unconditionally.

Do not reinterpret, patch, or forge an existing schema-v1 result as KV
authorization. The observed no-checkpoint p50 OOM is a negative result and must
remain immutable. Add a separately versioned schema-v2 prerequisite result for
the production-feasible operational oracle: dropout zero, checkpointing
enabled, BF16 FlashAttention 2, rank-local batch one, and the sealed p50, p95,
and maximum representatives. It must record
`comparison_role="operational_oracle"`, `gradient_checkpointing=true`, the
three capacity outcomes, and `segmented_kv_authorized=false`.

The schema-v2 prerequisite passes only if all three operational-oracle steps
complete with finite loss and gradients. Continue to record the 30% prefix-time
classification, but treat it as descriptive rather than an implementation
kill-switch: checkpoint recomputation occurs during backward and is not fully
represented by the top-level recurrence-forward timer. Final adoption remains
controlled by the predeclared end-to-end 15% speed gate.

Write this immutable result as `stage2-efficiency-v2.json`. Its top level has
exactly `schema_version`, `experiment_id`, `comparison_role`,
`gradient_checkpointing`, `segmented_kv_authorized`, `manifest_sha256`,
`device`, `representatives`, and `overall_evidence`. The fixed values are
schema version `2`, role `operational_oracle`, checkpointing `true`, and
authorization `false`. `device` has exactly `name`, `total_memory_bytes`,
`torch_version`, and `cuda_version`. `representatives` contains exactly three
ordered p50/p95/maximum objects, each with `selection`, `raw_index`,
`sequence_length`, `latent_positions`, `scaffold_sha256`, `recurrence_work`,
`completed`, `finite`, `peak_allocated_bytes`, `peak_reserved_bytes`, and
nullable `evidence`. A failed representative is still written with
`completed=false`, `finite=false`, and `evidence=null`; no later
representative runs after failure, and each skipped later object uses the same
false/null state with both peak-byte fields zero. `overall_evidence` is the
existing complete efficiency-evidence object when all three representatives
finish and is null otherwise.

`raw_index` is the zero-based physical row number in the validated training
JSONL. `retained_indices` is the ordered integer array sealed by the validated
Stage 2 length manifest; each index must address exactly one prepared example
and its recorded sequence length and latent-block count. Re-run the production
Stage 2 collator on each retained prepared example, parse the ordered latent
positions from its actual one-row `input_ids`, and hash the contiguous int64
CPU bytes of that row as `scaffold_sha256`.

Derive representatives reproducibly. For retained example `i`, define

```text
recurrence_work[i] = sequence_length[i]
                   + sum(all ordered latent-slot positions in example i)
```

This equals the no-cache final decoder length plus all recurrence decoder input
lengths. For sorted work values `x[0..n-1]`, compute a linear quantile at
probability `p` with `h=(n-1)*p`, `lo=floor(h)`, `hi=ceil(h)`, and
`q=x[lo] + (h-lo)*(x[hi]-x[lo])`. Use `p=0.50` and `p=0.95`. For each target,
select the example minimizing `(absolute distance to target, raw_index)`.
Select maximum work by `(-recurrence_work, raw_index)`. Record the three raw
indices, their scaffold/manifest digests, lengths, latent positions, and work
values before loading model weights.

Sampler schedule v2 is also byte-defined here. Its base seed is the normalized
integer Stage 2 training `seed`; every qualification in this specification fixes
it to `42`. For each epoch, bucket retained
rows by latent-block count. Derive an operation seed as the unsigned
little-endian integer from the first eight SHA256 bytes of the NUL-separated
UTF-8 fields `schema-v2`, dataset digest, decimal base seed, decimal epoch,
decimal bucket ID, and operation label, masked to 63 bits. Use only a CPU
`torch.Generator` and `torch.randperm` under the pinned PyTorch version. Shuffle
with label `bucket-shuffle`; divide the result into consecutive windows of 64
and sort each window by descending recurrence work then ascending raw index.

Place every retained row exactly once, then pad each bucket to a multiple of
eight by cycling from `seed("padding-offset") % bucket_size`. An occurrence ID
is `(epoch, bucket_id, post_shuffle_slot, padding_ordinal)`, where original
occurrences use ordinal zero and replays use consecutive one-based ordinals.
Chunk in order into eight-slot frames, shuffle frame indices with label
`frame-shuffle`, and never shuffle slots within a frame. Each canonical
occurrence record has exactly `epoch`, `bucket_id`, `post_shuffle_slot`,
`padding_ordinal`, `raw_index`, and `is_padding`; serialize the ordered records
with the Section 7 canonical JSONL rule and SHA256 their concatenation as the
schedule digest. Empty buckets emit no frames. Global frame slot `s` is assigned
by `s = microstep * world_size + global_rank`; gathered slots in that order must
reconstruct the frame.

For a multi-epoch training schedule, concatenate complete epochs in increasing
zero-based epoch order and digest the occurrence JSONL in that same order. The
qualification/adoption schedule contains exactly epoch `0`; its `N` and
`schedule_sha256` therefore refer only to that epoch. Use its logical frames of
effective batch eight, require `N >= 13`, and compute:

```text
cursor_seed = SHA256("stage2-kv-benchmark-v1\0" + schedule_sha256_ascii)
benchmark_cursor = big_endian_uint64(cursor_seed[0:8]) % (N - 12)
```

Frames `cursor..cursor+2` are warm-up and frames `cursor+3..cursor+12` are
measured. Seal the cursor and exact ordered occurrence IDs before either
backend runs. Both backends use those same frames and initial weights.

The distributed adoption geometry is fixed:

| Setting | Value |
|---|---|
| Rank-0/master node | `10.1.102.41`, machine rank 0 |
| Worker node | `10.1.102.40`, machine rank 1 |
| Interface | `bond1.102` |
| GPUs | 2 H100 NVL per node, 4 global processes |
| Per-device batch | 1 |
| Gradient accumulation | 2 |
| Effective batch | 8 |
| DoRA dropout | 0.0 |
| Seed / scheduled epochs | 42 / exactly epoch 0 |
| Oracle gradient checkpointing | enabled |
| Candidate gradient checkpointing | disabled |

Single-device gates use `CUDA_VISIBLE_DEVICES=0` on one H100 and the same model,
precision, attention backend, and examples.

For each distributed control, define a fresh `RUN_ID`, set the repository's
real artifact/data paths, and export these identical values on both nodes:

`STAGE1_CHECKPOINT` and the matching Stage 2 manifest are operator-bound inputs,
not placeholders the implementation may guess. The checkpoint must resolve
inside the canonical `./checkpoints/` target and pass the existing Stage 2 load
decision, PEFT metadata, sparse-token-row, and artifact-tree validators. The
manifest must be a regular file under `./results/`, match the dataset/image root
and execution geometry, and pass its supplied SHA256. Both nodes must agree on
the logical references and digests before output creation or rendezvous.

```bash
export STAGE1_CHECKPOINT=/absolute/checkpoints/stage1-source
export DATA_PATH=/root/workspace/ILVR/data/TRAIN.jsonl
export IMAGE_ROOT=/root/workspace/ILVR/data
export OUTPUT_DIR="./checkpoints/zero-stage-compatibility/$RUN_ID"
export TUNING_METHOD=dora
export LORA_R=32 LORA_ALPHA=64 LORA_DROPOUT=0.0
export LORA_TARGET_MODULES=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
export LATENT_SIZE=8 SUPERVISE_LATENT_BOUNDARIES=true
export ATTENTION_IMPLEMENTATION=flash_attention_2
export GRADIENT_ACCUMULATION_STEPS=2
export STAGE2_DIAGNOSTICS_INTERVAL=0
export MASTER_ADDR=10.1.102.41 MASTER_PORT=29520
export NCCL_SOCKET_IFNAME='=bond1.102'
export GLOO_SOCKET_IFNAME=bond1.102
export NCCL_SOCKET_FAMILY=AF_INET
export CUDA_VISIBLE_DEVICES=0,1
```

For the oracle export `STAGE2_RECURRENCE_BACKEND=no_cache` and
`GRADIENT_CHECKPOINTING=true`. For the candidate use a different fresh `RUN_ID`
and export `STAGE2_RECURRENCE_BACKEND=segmented_kv` and
`GRADIENT_CHECKPOINTING=false`. Both controls export
`STAGE2_KV_RUN_PURPOSE=qualification` and
`STAGE2_KV_QUALIFICATION_GATE=ddp_adoption`. Backend and checkpointing are the
only permitted cross-control differences. Launch the target wrapper on `.41`:

```bash
STAGE2_DISTRIBUTED_MODE=ddp \
STAGE2_NUM_MACHINES=2 STAGE2_NUM_PROCESSES=4 \
STAGE2_MACHINE_RANK=0 STAGE2_MAIN_PROCESS_IP=10.1.102.41 \
STAGE2_MAIN_PROCESS_PORT=29520 \
./run_stage2_training.sh
```

Launch it on `.40` with only the machine rank changed:

```bash
STAGE2_DISTRIBUTED_MODE=ddp \
STAGE2_NUM_MACHINES=2 STAGE2_NUM_PROCESSES=4 \
STAGE2_MACHINE_RANK=1 STAGE2_MAIN_PROCESS_IP=10.1.102.41 \
STAGE2_MAIN_PROCESS_PORT=29520 \
./run_stage2_training.sh
```

These are target commands: the next session must first implement and test the
multi-node DDP and recurrence-selector wrapper arguments. The topology and
effective-batch rules are consistent with the separate multi-node document,
but its `no_cache` and dropout `0.05` defaults do not apply to this operational
dropout-zero comparison.

Create the separately versioned operational-comparison result in Section 11.1
only after candidate implementation. Its acceptance field is derived only by
the independent validator and is true only when every correctness, capacity,
distributed, save/resume, and performance gate is true. A false or absent
field is final for that experiment ID.

## 9. Implementation sequence for the next session

Use the repository venv at `./venv/ilvr-venv`. Reuse the existing Stage 2
components and make the smallest changes in this order:

1. Confirm the exact Stage 2 feature commit and preserve all unrelated dirty
   worktree changes.
2. Produce the schema-v2 checkpointed `no_cache` operational-oracle profile and
   require finite p50, p95, and maximum optimizer steps.
3. Add failing CPU/eager tests for cache cursor, multiple blocks, graph
   continuity, and final no-cache behavior.
4. Implement the sibling cache engine without changing `LatentRolloutEngine`.
5. Pass the tiny FP32 semantic gate, including checkpointed/non-checkpointed
   `no_cache` equivalence, before running any CUDA candidate test.
6. Pass the single-H100 BF16/FlashAttention correctness gate.
7. Immediately run candidate p50, p95, and maximum-work full
   forward/backward admission probes. Stop before integration on any failure.
8. Add active-model preflight, training-only routing, the launcher selector,
   and dry-run coverage.
9. Add recurrence sidecar save/load/resume validation and neutral evidence
   collection.
10. Pass the four-rank DDP correctness, save, and resume gates.
11. Run the operational `no_cache` versus `segmented_kv` comparison.
12. Open production authorization only if the immutable result satisfies every
    adoption condition.

After a failure, preserve the log, make the smallest causal fix, add one
regression test, and retry with a fresh run ID. Do not add generalized recovery
or retry infrastructure.

## 10. Correctness verification

### 10.1 Structural correctness gate

Use a tiny real Qwen decoder in FP32 eager mode with deterministic weights,
dropout zero, one sample, two latent blocks, and ordinary tokens between blocks.
Run cloned models through the untouched `no_cache` oracle and the candidate.
Also run a separate hand-unrolled toy causal decoder that does not import
`DynamicCache`, Qwen cache helpers, or candidate cursor helpers. Its explicit
prefix equations independently witness which positions each latent is allowed
to consume.

Before treating the checkpointed oracle as the production baseline, clone the
same deterministic `no_cache` state into checkpointing-off and checkpointing-on
executions. Compare final embeddings, supervised logits, scalar loss, every
trainable parameter gradient, input-embedding gradients, and one optimizer
update with the FP32 tolerances below. This establishes execution equivalence;
it does not make checkpointing part of the candidate algorithm.

Use this exact Qwen fixture:

| Field | Value |
|---|---|
| Torch seed | 1234, set before model composition |
| Vocabulary / hidden / intermediate | 128 / 32 / 64 |
| Decoder layers / attention heads / KV heads | 1 / 4 / 2 |
| Attention / dtype | eager / FP32 |
| Attention and adapter dropout | 0.0 |
| Config cache / tied embeddings | false / false |
| Stage / latent size | `stage2` / 2 |
| Latent pad/start/end IDs | 90 / 91 / 92 |
| RoPE scaling | `{"type":"mrope","mrope_section":[2,1,1]}` |
| Input IDs | `[1,91,90,90,92,2,91,90,90,92,3]` |
| Latent positions | `[2,3,7,8]` |
| Attention mask | eleven ones |
| Labels | `[-100,-100,-100,-100,92,-100,-100,-100,-100,92,3]` |
| Optimizer | AdamW, LR `1e-3`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay 0 |

Use image token ID 80, vision-start ID 81, and video ID 82. The unused but
required vision config is: depth 1, hidden size 32, intermediate size 64, heads
4, patch size 2, spatial merge size 1, temporal patch size 1, window size 2,
output hidden size 32, and full-attention block indices `[0]`. Pass no pixel
inputs. Save one initialized state dictionary, then load clones from those
exact bytes for oracle and candidate.

Run three separate parameter-boundary cases rather than one mixed model:

1. full decoder tuning with the visual encoder frozen;
2. LoRA with rank 2, alpha 4, dropout 0, and the seven canonical decoder
   projection suffixes;
3. DoRA with the same rank, alpha, dropout, and target suffixes.

For the toy independent oracle, use hidden size 4 and the same token scaffold
and latent positions. Its causal layer must be hand-written from cumulative
prefix operations and must not import Qwen attention or any cache class.

Compare:

- decoder call count and candidate delta lengths;
- cache positions and cache length after every segment;
- every generated latent state;
- complete final input embeddings;
- supervised logits and scalar loss;
- retained gradient of every latent state;
- input-embedding gradient;
- every trainable full, LoRA, and DoRA parameter gradient;
- one optimizer update from identical initial optimizer state;
- hand-unrolled toy outputs and gradients at every segment boundary;
- proof that the final full decoder call used `use_cache=False`.

Use FP32 eager tolerances `atol=1e-6`, `rtol=1e-5`. A mismatch, missing
gradient, detached cache, wrong cursor, or cache use in the final pass kills the
candidate before CUDA work.

This structural witness has not been recorded as durable project evidence.
Informal review probes are not production authorization.

### 10.2 Required unit and regression matrix

Before the production-shape first nail, run exactly the pre-capacity subset:
the first nine bullets below plus the final-`use_cache=False` assertion from
Section 10.1. These tests exercise the engine directly and do not require
launcher routing, sidecars, resume, or ordinary model installation. After the
capacity first nail passes, run the entire list.

Add focused tests for:

- zero, one, and multiple latent blocks;
- consecutive latent slots and inter-block text gaps;
- multimodal three-axis position slices;
- attention-mask and cache-position shapes;
- all-one-mask enforcement and rejection of padding masks;
- cache identity and per-layer length after each delta;
- no cache allocation for zero latent spans;
- diagnostics remaining observability-only;
- `retain_latent_grads` behavior;
- candidate rejection for every nonzero active dropout;
- candidate rejection with gradient checkpointing;
- eval and inference routing to the oracle even when a training candidate is
  installed;
- default/legacy routing to the untouched oracle;
- no candidate cache state in outputs or checkpoints;
- recurrence-sidecar strict parsing, mismatch rejection, and resume identity;
- ordinary generation and Stage 1 cache regressions remaining unchanged.

The oracle and candidate comparison must use independently constructed call
traces. A test that computes its expected cursor or cache length by calling the
candidate helper is circular and prohibited.

### 10.3 BF16 and FlashAttention 2 gate

On the active H100, compare the same candidate and oracle with the pinned BF16
and FlashAttention 2 stack. Repeat all three full/LoRA/DoRA parameter-boundary
cases from Section 10.1 and repeat the one-step optimizer-update witness. Use:

- loss, every latent, and supervised logits: `atol=2e-2`, `rtol=2e-2`;
- each nonzero trainable gradient: cosine similarity at least `0.999` and
  relative L2 error at most `0.02`;
- zero-reference gradients: `atol=2e-2`, `rtol=0`;
- all values finite;
- identical non-`None` gradient set and frozen-parameter boundary.

For a nonzero oracle gradient `g_o`, relative L2 is
`norm(g_candidate - g_o) / max(norm(g_o), 1e-12)`. A zero-reference gradient is
one whose FP32 norm is exactly zero before tolerance comparison. Compute cosine
similarity in FP32 after flattening each individual parameter gradient; do not
concatenate parameters or omit a nonzero gradient.

After the one optimizer step, compare every trainable parameter and every
floating optimizer-state tensor with `atol=2e-2`, `rtol=2e-2`; compare integer
step counters and all non-floating state exactly. The updated non-`None` state
key set must also match. Frozen parameters must remain byte-identical to their
own initial state in both executions.

Exercise single-token deltas and multi-token gaps because FlashAttention uses
shape-dependent kernels. Verify the installed backend's bottom-right causal
alignment for query length smaller than key length through observed oracle
parity; do not infer it from a version string alone.

### 10.4 Production-shape capacity first nail

Immediately after the minimal engine passes Section 10.1, the explicitly named
pre-capacity subset of Section 10.2, and the direct-engine BF16 comparison in
Section 10.3, run fresh single-H100 candidate optimizer steps in BF16
FlashAttention 2 with DoRA rank 32, alpha 64, dropout zero, rank-local batch one,
and checkpointing disabled. Exercise the sealed p50 representative first, then
p95, then maximum work. Record completion, finiteness, loss, gradient norm, peak
allocated bytes, peak reserved bytes, and the exact recurrence-work identity
for every probe, including a failed probe.

Run each representative in its own supervised child process from the same
validated initial model bytes and fresh optimizer/scheduler state. After model,
adapter, optimizer, and scheduler placement, synchronize CUDA, reset peak
allocated/reserved statistics, then collate and execute exactly one zero-grad,
forward, backward, gradient-norm, optimizer, and scheduler step. Synchronize
again before reading peak statistics. The matched operational-oracle object must
have the identical raw index, scaffold digest, sequence length, latent positions,
and recurrence work. The controller allows 30 minutes per child, signals only
the child process group it created on timeout, preserves that child's log, and
exclusively creates the aggregate result after the child exits. Thus an OOM,
hard exit, or timeout still yields a fail-closed result without reusing a CUDA
context or overwriting evidence.

Write `stage2-kv-single-h100-capacity.json` with exactly `schema_version`
(integer `1`), `experiment_id`, `backend` (exactly `segmented_kv`),
`gradient_checkpointing` (exactly `false`), `manifest_sha256`, `device`,
`p50_capacity`, `p95_capacity`, and `maximum_capacity`. `device` uses the exact
shape from Section 8. Each capacity object uses the Section 11.1 capacity shape
plus `loss` and `gradient_norm`, which are finite numbers exactly when the probe
completed and otherwise null. Failed and skipped probes use the same
fail-closed false/null/zero convention as the prerequisite result.

Any OOM, hang, detached or non-finite gradient, or candidate peak allocation
above 105% of the matched schema-v2 operational-oracle representative stops the
candidate before launcher, sidecar, evidence, or DDP integration. Do not
substitute an extrapolation or reduce any model, adapter, sequence, latent, or
batch geometry. This is the load-bearing first nail for design commitment v3.

### 10.5 Distributed, save, and resume gate

Only after the single-device gates pass, run the production-shaped DDP geometry
with the same sealed logical batches and effective batch. Require:

- all ranks select the same recurrence identity;
- every rank observes the same optimizer step and expected sample occurrence;
- finite loss and trainable gradients;
- gradient synchronization only at the accumulation boundary;
- rank 0 writes a complete adapter/checkpoint and recurrence sidecar;
- a same-topology resume advances exactly one optimizer step;
- a backend mismatch or missing field in a present sidecar fails before training
  collectives;
- a missing-sidecar legacy checkpoint resumes only under the explicit no-cache
  compatibility branch in Section 7, while a legacy-to-KV request fails.

No cross-backend optimizer resume is tested or supported.

One logical backend qualification consists of four fresh DDP child jobs from
the same initial state: one optimizer step for p50, one for p95, one for maximum
work, then the separate three-warm-up/ten-measured-frame benchmark job. The
qualification controller launches only these owned jobs, preserves rank-local
logs, and assembles the four final rank-evidence files only after all four jobs
exit successfully. Each capacity job broadcasts the same sealed representative
to every rank, resets CUDA peaks after construction as in Section 10.4, and
records one capacity object per rank. A missing child result, timeout, OOM,
non-finite value, rank disagreement, or missing final rank file makes capacity
false; no partial file is treated as evidence.

The occurrence witness is the exact `occurrences` array and digest defined in
Section 11.1. The gradient-synchronization witness is the qualification-only DDP
communication-hook count per microstep. For resume, select the highest complete
numbered checkpoint written by the candidate save/resume job, require its
recurrence, sampler, Trainer, optimizer, scheduler, and RNG state, and read its
completed-frame cursor. The resumed one-step job must consume exactly the next
sealed frame and publish global step `checkpoint_step + 1`; any skipped,
repeated, or different occurrence fails the gate.

After each side of the resume probe, exclusively create an immutable result file
under `./results/`: `stage2-kv-resume-source.json` and
`stage2-kv-resume-result.json`. Each has exactly `schema_version` (integer `1`),
`experiment_id`, `role` (`source` or `result`), `checkpoint_reference`,
`checkpoint_tree_sha256`, `recurrence_sidecar_sha256`,
`sampler_sidecar_sha256`, `trainer_state_sha256`, `global_step`,
`completed_frame_cursor`, `expected_next_frame_sha256`, and nullable
`observed_frame_sha256`. `checkpoint_reference` is a repository-relative path
inside `checkpoints/`; recompute its Section 7 JSONL tree digest and the three
named file digests when reading the witness. The source has
`observed_frame_sha256=null`; the result records the actually gathered frame.
The result must have source global step plus one, source cursor plus one, and an
observed frame equal to the source's expected-next digest. Both witnesses and
both referenced checkpoint-tree digests are immutable validator inputs.

## 11. Capacity and performance adoption gate

Compare exactly two fresh dropout-zero operational configurations:

| Control | Recurrence backend | Gradient checkpointing |
|---|---|---|
| Operational oracle | `no_cache` | enabled |
| Candidate | `segmented_kv` | disabled |

Use the same commit, initial weights, manifest, sampler schedule, sample order,
effective batch, precision, FlashAttention backend, optimizer, seed, and number
of steps. Backend and gradient checkpointing are the only permitted differences.
This measures the end-to-end advantage of two feasible operational stacks; it
must never be reported as the isolated speedup of KV reuse. Seal the benchmark
cursor and frame IDs before timing. Use three warm-up optimizer steps and ten
measured optimizer steps; synchronize device work at timing boundaries and
report maximum-rank time for distributed runs.

Record raw per-step values for:

- end-to-end optimizer-step milliseconds;
- recurrence decoder, final decoder/loss, backward, optimizer, and data time;
- decoder call count and processed query/key token counts;
- peak allocated and peak reserved CUDA memory per rank;
- retained examples, non-padding examples, tokens, and latent blocks per second;
- loss and gradient norm;
- cache length and total cached tensor bytes at each latent boundary.

### 11.1 Raw evidence schema and independent validator

Neutral Trainer timing hooks, not the cache engine, write one immutable
schema-v2 `stage2-kv-rank-evidence.json` per backend and global rank. Each file
is strict canonical JSON with exactly:

- `schema_version`: integer `2`;
- `experiment_id`: nonempty single-path-component string;
- `backend`: `no_cache` or `segmented_kv`;
- `comparison_role`: `operational_oracle` or `candidate`;
- `gradient_checkpointing`: boolean;
- `global_rank`: integer in `[0, world_size)`;
- `world_size`: positive integer, fixed to 4 for adoption;
- `runtime_identity_sha256` and `schedule_sha256`: 64-lowercase-hex strings;
- `benchmark_cursor`: non-negative integer;
- `warmup_frame_sha256`: array of exactly three 64-lowercase-hex strings;
- `measured_frame_sha256`: array of exactly ten 64-lowercase-hex strings;
- `steps`: array of exactly ten step objects;
- `p50_capacity`, `p95_capacity`, and `maximum_capacity`: capacity objects.

Each step object has exactly:

- `index`: integer `0..9` in array order;
- `frame_sha256`: the matching measured-frame digest;
- finite positive `step_ms`;
- finite non-negative `data_ms`, `forward_loss_ms`, `backward_ms`, `sync_ms`,
  `optimizer_ms`, `overhead_ms`, `recurrence_decoder_ms`, and
  `final_decoder_ms`; `data_ms + forward_loss_ms + backward_ms + optimizer_ms +
  overhead_ms` equals `step_ms` within `1e-3` milliseconds,
  while `sync_ms` is an observed subset of `backward_ms` and is not added again;
- positive integers `peak_allocated_bytes`, `peak_reserved_bytes`,
  `query_token_count`, `key_token_count`, `decoder_call_count`,
  `latent_block_count`, `retained_example_count`, `token_count`, and
  `non_padding_example_count` not greater than `retained_example_count`;
- finite non-negative `retained_examples_per_second`,
  `non_padding_examples_per_second`, `tokens_per_second`, and
  `latent_blocks_per_second`;
- finite JSON numbers `loss` and `gradient_norm`;
- `occurrence_sha256`: 64-lowercase-hex digest of the ordered frame occurrence
  records;
- `occurrences`: an array of exactly `gradient_accumulation` records in local
  microstep order, each containing the six schedule-v2 occurrence fields plus
  non-negative integers `microstep` and `global_rank`;
- `sync_calls_by_microstep`: an array of exactly `gradient_accumulation`
  non-negative integers;
- `cache_boundaries`: an array with one object per generated latent, each
  containing exactly non-negative integers `microstep`,
  `local_example_ordinal`, `latent_ordinal`, `cache_length`, and `cache_bytes`.

`recurrence_decoder_ms` and `final_decoder_ms` are synchronized CUDA-event
subsets of `forward_loss_ms`; they are not added to the wall partition.
Top-level decoder pre/post hooks count only logical forward calls and exclude
decoder-layer recomputation during checkpoint backward. `query_token_count` is
the sum of query lengths of those calls, including the final full-sequence
call. `key_token_count` is the sum of effective attention key lengths: each
no-cache prefix contributes its full prefix length, each candidate delta
contributes its cache-after-update length, and the final call contributes the
full sequence length. Throughputs are their named frame totals divided by
`step_ms / 1000`. Candidate boundary bytes sum `numel * element_size` over all
key/value tensors in the active cache; oracle boundary entries have zero length
and bytes. These counters describe logical work, while `step_ms` and
`backward_ms` include checkpoint recomputation.

Order `cache_boundaries` by local microstep, local example, then latent
generation order. `local_example_ordinal` is always zero in version 1 because
rank-local batch is one, and `latent_ordinal` resets to zero for every example.

Canonicalize `occurrences` with the Section 7 JSONL rule to obtain
`occurrence_sha256`. Their gathered microstep-major, rank-major order must equal
the sealed eight-slot frame exactly. Install a counting DDP communication hook
for qualification: every non-final accumulation microstep must have zero sync
calls and the final microstep must have at least one. The hook delegates the
actual reduction unchanged and is removed outside qualification.

Each capacity object has exactly `raw_index` non-negative integer,
`sequence_length` positive integer, `latent_positions` ordered non-negative
integer array, `scaffold_sha256` 64-lowercase-hex string, `recurrence_work`
positive integer, `completed` boolean, `peak_allocated_bytes` and
`peak_reserved_bytes` non-negative integers, and `finite` boolean. Recompute
`recurrence_work` from the recorded length/positions and require exact identity
with the sealed representative. A failed or OOM probe records
`completed=false`; it does not omit the object.

Implement `src/stage2/kv_evidence.py` as a separate reader/validator. It must
not import the candidate engine or accept a candidate-supplied decision bit. It
strictly parses all eight rank files and requires within-backend runtime
identity agreement. Across backends it permits exactly
`(no_cache, operational_oracle, true)` versus
`(segmented_kv, candidate, false)`; every other decoded execution, model,
workload, and batch-geometry field must match. The validator consumes and
strictly parses each run's canonical `stage2-recurrence.json`, verifies that its
digest equals every corresponding rank's `runtime_identity_sha256`, and uses
those sidecars for the decoded comparison. It invokes the fixed correctness and
save/resume verification commands and derives the result. Store each invoked
argv array, exit code, and complete stdout/stderr SHA256; a nonzero or missing
command is a failed gate.

Run commands without a shell from the repository root. The fixed command map is:

| Gate | Environment | Argv |
|---|---|---|
| `fp32_correctness` | `PYTHONPATH=transformers/src:src`, `CUDA_VISIBLE_DEVICES=` | `./venv/ilvr-venv/bin/pytest -q tests/stage2/test_kv_correctness.py` |
| `bf16_correctness` | `PYTHONPATH=transformers/src:src`, `CUDA_VISIBLE_DEVICES=0` | `./venv/ilvr-venv/bin/pytest -q tests/stage2/integration/test_cuda_backends.py -k "segmented_kv and bf16"` |
| `distributed_correctness` | `PYTHONPATH=transformers/src:src` | `./venv/ilvr-venv/bin/python -m stage2.kv_verify --kind distributed --input-manifest INPUT` |
| `save_resume` | `PYTHONPATH=transformers/src:src` | `./venv/ilvr-venv/bin/python -m stage2.kv_verify --kind save_resume --input-manifest INPUT` |

`INPUT` is the repository-relative path to immutable
`stage2-kv-verification-input.json` in the comparison result directory. That
strict schema-v1 object has exactly `schema_version`, four ordered
`oracle_rank_paths`, four ordered `candidate_rank_paths`,
`oracle_recurrence_sidecar`, `candidate_recurrence_sidecar`,
`operational_oracle_prerequisite`, `candidate_single_h100_capacity`,
`candidate_resume_source`, and `candidate_resume_result`; every path is
repository-relative, resolves inside `results/` or `checkpoints/` as
appropriate, and is included in the comparison input digests. The two capacity
paths must name the exact files described in Sections 8 and 10.4. Implement
`stage2.kv_verify` independently from the candidate engine. The resume fields
name the two regular witness JSON files defined in Section 10.5, never checkpoint
directories directly.

For each command, exclusively create `<gate>-command.json` and sibling
`<gate>.stdout`/`<gate>.stderr` files under the comparison result directory.
The command record has exactly `schema_version` (integer `1`), `gate`, `argv`
(the exact string array after substituting `INPUT`), `environment` (only the
listed overrides), `exit_code`, `stdout_sha256`, and `stderr_sha256`. The
validator rejects a missing output, digest mismatch, unexpected argv or
environment, nonzero exit, or non-regular file. It never accepts a caller-set
gate boolean.

The validator writes one immutable schema-v2 `stage2-kv-comparison.json` with
exactly:

- `schema_version`: integer `2`;
- `comparison_id`, `oracle_run_id`, `candidate_run_id`: nonempty
  single-path-component strings;
- `comparison_kind`: exact string `operational_stack`;
- `input_sha256`: object mapping the eight rank-evidence relative paths, two
  recurrence-sidecar relative paths, the schema-v2 operational-oracle
  prerequisite, the single-H100 candidate capacity result, the verification
  input manifest, both resume-witness files, and every command
  record/stdout/stderr relative path to their 64-lowercase-hex digests; the
  validator also recomputes the checkpoint-tree digests embedded in the resume
  witnesses;
- `metrics`: exact object defined below;
- `gates`: exact object defined below;
- `accepted`: boolean derived as the conjunction of every gate.

`metrics` contains only finite numbers
`oracle_rank_max_median_step_ms`, `candidate_rank_max_median_step_ms`,
`speed_improvement_fraction`, `oracle_peak_allocated_bytes`,
`candidate_peak_allocated_bytes`, and `memory_regression_fraction`.
The byte metrics are encoded as non-negative integers. `gates` contains exactly
the booleans `runtime_identity`, `workload_identity`, `fp32_correctness`,
`bf16_correctness`, `distributed_correctness`, `capacity`, `save_resume`,
`speed`, and `memory`.

For backend `b` and measured step `i`, let
`T[b,i] = max_rank(step_ms[b,rank,i])`. Then:

```text
oracle_rank_max_median_step_ms    = median_i T[no_cache,i]
candidate_rank_max_median_step_ms = median_i T[segmented_kv,i]
speed_improvement_fraction         = 1 - candidate_median / oracle_median

peak[b] = max_rank,max_step peak_allocated_bytes[b,rank,step]
memory_regression_fraction = peak[segmented_kv] / peak[no_cache] - 1
```

The speed gate is `speed_improvement_fraction >= 0.15`. The memory gate is
`memory_regression_fraction <= 0.05`. Capacity is true only when p50, p95, and
maximum objects on every rank have `completed=true` and `finite=true`; the
separate single-H100 candidate first-nail result must also have completed finite
p50, p95, and maximum objects. The validator sets `accepted=true` only when
every named gate is true; no caller can override that value. The raw rank files
and final comparison use the canonicalization and duplicate-key rejection rules
from Section 7.

Run p50, p95, and known maximum-work admission examples. The candidate is
adopted only when:

1. every correctness, distributed, save, and resume gate passes;
2. median steady-state end-to-end step time improves by at least 15%;
3. peak allocated CUDA memory is no more than 5% above the operational oracle;
4. no p50, p95, or maximum-work probe OOMs, hangs, or produces non-finite values;
5. no comparison setting other than the declared backend/checkpointing pair
   differs or changes Stage 2 semantics or effective batch;
6. the immutable comparison result is complete and internally consistent.

Do not lower tolerances or thresholds after observing results. A miss is a valid
negative result: keep `no_cache` as the only production backend.

This experiment establishes runtime behavior only for dropout-zero training. It
does not establish model-quality equivalence to the canonical dropout `0.05`
run.

## 12. Fail-closed behavior

Reject `segmented_kv` before model preparation or before the first forward, as
appropriate, in both qualification and production when any intrinsic condition
below is true:

- active dropout or another stochastic prefix-changing module is found;
- gradient checkpointing is enabled;
- rank-local batch is not one;
- the attention mask contains padding;
- the selected attention backend is outside version 1's `eager` FP32 and
  `flash_attention_2` BF16 combinations;
- candidate routing is requested outside training mode;
- the checkpoint is legacy, no-cache, incomplete, or backend-incompatible;
- cache position, cache identity, layer count, or sequence length is wrong;
- any cached tensor or latent is detached;
- any required evidence digest or runtime identity differs across ranks;
- any value is non-finite.

Additionally reject only a `production` request when capacity or performance
adoption is absent or false, or when the supplied accepted comparison does not
match the requested runtime. A `qualification` request does not require prior
adoption or a previously accepted attention result because qualification creates
that evidence; it may execute only the gate it is designated to produce. The
workflow ordering in Sections 9 and 10 remains mandatory, and the final
validator fails if its fixed BF16 command does not pass. A failed immutable
qualification result cannot be reused or relabeled as a pass.

Never catch one of these errors and continue with `no_cache` under the same run
ID. Abort, preserve the log, and start an explicitly selected fresh oracle run.

## 13. Design commitment and critique record

### 13.1 Superseded commitment v2

- **Problem:** remove repeated deterministic prefix computation while preserving
  the continuous-latent recurrence, full final loss pass, and autograd graph.
- **Selected design:** a training-only sibling `DynamicCache` engine with a
  monotonic segment cursor; the default, final loss pass, and inference remain
  `no_cache`.
- **Change from proposal v1:** numerical-drift gates were insufficient for
  nonzero dropout. Version 2 excludes every stochastic prefix-changing module
  and gradient checkpointing, so existing DoRA dropout `0.05` runs are not
  eligible.
- **First nail:** deterministic tiny-Qwen FP32 forward/backward parity across two
  separated latent blocks. It is specified but not recorded as durable evidence.
- **Material divergence:** independent reviews agreed that cache reuse with
  nonzero dropout changes the stochastic computation rather than only rounding.
  They also required explicit cursor semantics, a sibling oracle-independent
  engine, and backend-specific mask/RoPE gates.
- **Prism result:** correctness passes only inside the deterministic subset;
  operational compatibility passes with fresh-run identity and fail-closed
  resume; performance remains unresolved because graph-attached dynamic
  concatenation may increase memory. The resolving question is whether the
  actual H100 p95 run passes parity while improving median step time by 15%
  within the 5% memory limit.
- **Unvalidated premises:** the compatible no-cache profile has at least 30%
  recurrence work, DynamicCache backward fits H100 without checkpointing,
  FlashAttention incremental shapes stay within declared tolerances, and DDP
  preserves enough compute gain for the adoption threshold.
- **Risk boundary:** approval does not waive method, correctness, capacity,
  resume, or performance gates. It does not authorize dropout `0.05`, automatic
  fallback, B200-derived approval, or a quality-equivalence claim.

Version 2 is retained as history. Its production-scale matched-no-checkpoint
premise was falsified by the p50 OOM recorded in Section 1.2 and is replaced by
version 3 below.

### 13.2 Final commitment v3

- **Problem and scope:** remove repeated deterministic prefix computation while
  preserving the continuous-latent recurrence, full final loss pass, autograd
  graph, model/data/batch geometry, and fail-closed resume behavior. Production
  training remains outside this authorization.
- **Final design:** keep the training-only sibling `DynamicCache` engine and
  checkpointing-disabled candidate. Split oracle roles: non-checkpointed
  `no_cache` is the small deterministic semantic oracle; checkpointed
  `no_cache` is the production operational baseline.
- **Changes from proposal v3:** none. The frozen proposal already moved direct
  candidate p50/p95/maximum capacity immediately after the minimal engine and
  semantic gates, and labeled the final result as an operational-stack
  comparison rather than an algorithm-only speed claim.
- **First nail:** after the minimal engine passes independent FP32 and small
  BF16 correctness, run full candidate forward/backward at the exact p50, p95,
  and maximum production geometry on one H100. Any OOM, non-finite value,
  detached gradient, or memory-gate failure ends the candidate before broader
  integration. It has not run because the candidate does not yet exist; all
  four target H100s were observed idle on 2026-09-03, so resource availability
  is not the remaining uncertainty.
- **Material divergence:** three stripped fresh reads found no incompatible
  alternative, but independently emphasized that full-geometry capacity must
  precede launcher/evidence integration and that the checkpointed oracle is the
  strongest feasible production comparator. Their convergence is reassurance,
  not evidence.
- **Prism result:** correctness passes only after no-cache
  checkpointing-equivalence and full candidate gradient/update witnesses;
  operational capacity is fail-closed at p50/p95/maximum; benchmark validity
  requires the validator to admit exactly the declared checkpointing asymmetry
  and forbid isolated-KV attribution. The three lenses had no unresolved
  disagreement.
- **Unvalidated premises:** graph-connected candidate backward fits the maximum
  H100 geometry without checkpointing, incremental BF16 FlashAttention shapes
  pass the declared correctness tolerances, and the operational candidate beats
  checkpointed `no_cache` by at least 15% while staying within the 5% peak-memory
  limit.
- **Risk boundary:** approval does not waive any semantic, correctness,
  capacity, resume, distributed, speed, or memory gate. It does not authorize
  dropout `0.05`, geometry reduction, automatic fallback, candidate
  checkpointing, ZeRO-3, a production run, or an algorithm-only speed claim.

### 13.3 Rejected alternatives

- Production-scale no-checkpoint oracle comparison: rejected because the
  reference OOMs at p50 and therefore is not an executable control.
- Making mutable cache replay checkpoint-safe in version 1: rejected because it
  expands the state/recomputation surface before basic candidate capacity is
  known.
- Switching to LoRA or reducing sequence, latent, adapter, or batch geometry:
  rejected because it changes the requested training configuration rather than
  testing the candidate at the required operating point.
- Making cache the default: rejected because eligibility is narrow and evidence
  is absent.
- Accepting DoRA dropout `0.05` as numerical drift: rejected because it changes
  stochastic prefix reuse.
- Keying new stateless dropout masks by token: rejected because it would change
  the current oracle's stochastic behavior.
- Recomputing dropout-affected history: rejected because it removes the
  load-bearing KV reuse benefit.
- Static/offloaded/quantized cache: rejected for version 1 because it expands
  the autograd and backend surface without evidence.
- Caching the final loss pass: rejected because the complete final decoder and
  supervision path are part of the Stage 2 oracle.

## 14. Explicit non-goals

- changing Stage 2 latent generation, objective, masks, or final pass;
- making KV caching available to the current DoRA dropout `0.05` run;
- claiming dropout-zero model quality equals dropout `0.05` quality;
- changing effective batch size, sampler, data membership, or learning rate;
- inference or generation acceleration;
- cache persistence or cross-sample reuse;
- gradient-checkpoint-compatible cache mutation;
- broad ZeRO/FSDP/tensor/sequence-parallel support;
- generalized retry, recovery, autoscaling, or cache infrastructure.

## 15. Authoritative local references

- `transformers/src/transformers/models/qwen2_5_vl/stage2_latent_rollout.py`
- `transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
- `transformers/src/transformers/cache_utils.py`
- `src/main.py`
- `src/stage2/efficiency_probe.py`
- `tests/stage2/test_rollout.py`
- `tests/stage2/test_qwen_forward.py`
- `tests/stage2/test_efficiency_probe.py`
- `docs/superpowers/specs/2026-09-02-stage2-efficiency-design.md`
- `documents/multi-node-stage2-training-20260920.md`

## 16. External references

- [Transformers cache explanation](https://huggingface.co/docs/transformers/cache_explanation)
- [Transformers KV-cache strategies](https://huggingface.co/docs/transformers/kv_cache)
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [FlashAttention repository and release notes](https://github.com/Dao-AILab/flash-attention)
