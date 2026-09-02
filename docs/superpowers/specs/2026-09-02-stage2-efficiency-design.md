# Stage 2 Training Efficiency Design

**Date:** 2026-09-02
**Status:** Approved design commitment v2; awaiting written-spec review
**Scope:** Make Stage 2 training faster and less memory-intensive without changing the existing Stage 2 mechanism or training objective.

## 1. Non-negotiable Stage 2 semantics

The current no-cache implementation is the semantic oracle. Every accepted optimization must preserve all of the following:

1. The input scaffold and ordered latent blocks are unchanged.
2. For every latent slot, the inserted continuous latent is the final decoder hidden state of the causal prefix immediately before that slot.
3. Each generated latent remains attached to the autograd graph and influences every later latent and the final loss. No latent or cached key/value tensor may be detached.
4. Discrete latent-pad embeddings never substitute for generated continuous latents.
5. After all latent slots are filled, Stage 2 still performs the existing full-sequence causal decoder pass and computes the same shifted assistant-token cross-entropy objective.
6. The existing label mask, optional latent-boundary supervision, image features, RoPE positions, attention mask, latent order, and parameter-training boundary are unchanged.
7. Existing checkpoints and all training configurations continue to use `no_cache` unless a separately versioned backend is explicitly selected for a new compatible run.

An optimization that fails any semantic witness is rejected even if it is faster. A numerically approximate backend may not be described as equivalent until it passes the predeclared tolerances below.

## 2. Approaches considered

### A. Replace the default recurrence with KV caching immediately

Rejected. The production LoRA/DoRA default has dropout `0.05`, and the current recurrence re-evaluates a prefix with new dropout masks at each latent slot. Reusing cached prefix states changes that stochastic objective. Training-time cache behavior is also incompatible with the current gradient-checkpointing path.

### B. Optimize only the current no-cache implementation

Safe but incomplete. It removes known exact waste while leaving the dominant repeated-prefix transformer work untouched if profiling confirms that it dominates.

### C. Exact no-cache optimizations followed by a gated KV experiment

Selected. Exact changes are implemented and measured first. A segmented KV backend is built only if the optimized no-cache profile proves that redundant prefix work is a large enough bottleneck. It remains default-off and cannot replace the oracle without correctness and performance evidence.

## 3. Measurement baseline and first nail

Before changing runtime behavior, add a focused Stage 2 profile using actual CoMT examples near the median and p95 of both sequence length and latent-block count. Record, under `results/` and `logs/`:

- data/collation time;
- image encoding time;
- latent recurrence decoder time;
- final full-sequence decoder and loss time;
- backward and optimizer-step time;
- CUDA peak allocated memory;
- number of decoder calls and processed prefix tokens.

Use CUDA synchronization around timed GPU regions, three warm-up steps, and ten measured steps. Compare configurations at the same model, example set, effective batch size, precision, attention backend, tuning method, and DeepSpeed setting.

The load-bearing objection is that a dropout-free, checkpointing-free KV path may still be slower or use more memory than the best no-cache path because the final full forward and distributed collectives remain mandatory. The first nail is therefore:

1. Optimize and re-profile the no-cache path.
2. Proceed to a KV implementation only if repeated-prefix decoder work is at least 30% of median end-to-end step time on the representative B200 profile.
3. Stop the KV work if a minimal segmented causal-composability probe fails the FP32 oracle checks.

This first nail has not been run. Running it before approval would have required new profiling/prototype code and B200 runtime. The implementation plan will authorize and order it before any KV backend work.

## 4. Exact no-cache optimizations

Each optimization is independently testable and must pass against the untouched no-cache oracle before the optimizations are combined.

### 4.1 Supervised-row loss head

Add an explicit Stage 2 loss-only forward option. When labels are present and the caller requests loss only:

1. Run the unchanged latent recurrence and final full decoder.
2. Form the shifted supervision mask from `labels[..., 1:] != -100`.
3. Select only the corresponding final hidden-state rows from positions before those labels.
4. Apply `lm_head`, FP32 conversion, and cross-entropy only to those rows.

`Stage2Trainer.compute_loss` uses this option only when `return_outputs=False`. Evaluation, debugging, generation, and every normal model call preserve the current full-logit output contract. The loss-only result must not construct a full `[batch, sequence, vocabulary]` tensor.

### 4.2 Opt-in rollout diagnostics

Add an explicit diagnostics microbatch interval. Its default is `0`, which disables latent norms and cosine calculations during normal training. A positive value samples diagnostics on that interval. This changes observability cost only; it does not change recurrence, loss, RNG, or checkpoint state.

### 4.3 Prefix-sized embedding construction

Replace repeated full-sequence `_replace_embedding` concatenations with two operations:

- for each recurrence call, construct only the prefix required by that latent slot from immutable base-embedding slices and the already generated graph-attached latent states;
- after the final latent, construct the complete embedding sequence once for the mandatory full forward.

The generated tensors, their order, and their autograd connections remain identical. No in-place update is allowed if it would sever or alias a gradient path.

### 4.4 Single length manifest and measured data workers

The launcher creates one immutable metadata-only length manifest before distributed workers start. It is stored at `results/<experiment-id>/stage2-length-manifest.json`; it contains raw example indices, exact untruncated lengths, latent-block counts, retained indices, and summary statistics. It contains no model artifacts.

The manifest identity binds:

- the ordered dataset file SHA-256;
- resolved image path, file size, modification time in nanoseconds, and decoded dimensions for each referenced image;
- processor, tokenizer, and chat-template identity;
- latent size, boundary-supervision setting, and maximum sequence length;
- the manifest schema version.

The launcher validates or creates this manifest once, then passes its path to every rank. Ranks prepare the same retained raw indices without re-running image-aware length measurement. A stale or mismatched manifest fails before distributed model preparation. Resume reuses the run's validated manifest.

Expose a fixed dataloader worker count through the launcher. Benchmark `0`, `2`, and `4` workers on the representative samples and select the fastest value that does not increase peak host memory beyond available capacity. Do not add runtime auto-tuning.

## 5. Conditional segmented KV backend

### 5.1 Selection and preflight

The backend choices are `no_cache` and `segmented_kv`. `no_cache` remains the default.

`segmented_kv` is permitted only for a fresh run when all of the following hold:

- gradient checkpointing is disabled;
- model configuration attention dropout is zero;
- every active `torch.nn.Dropout` module, including PEFT adapter dropout, has `p == 0`;
- model training is strictly prefix-causal for the selected attention backend;
- no stochastic routing or other training-time module can change a reused prefix;
- the segmented masks, position IDs, RoPE inputs, multimodal state, and cache positions pass the causal-composability probe.

The check walks the active configured model, not merely launcher arguments. Any violation fails before training. Existing PEFT checkpoints with dropout `0.05` cannot switch to `segmented_kv`.

### 5.2 Recurrence data flow

The experimental engine keeps the existing one-call-per-latent schedule plus the final full call:

1. Decode the initial prefix ending immediately before the first latent slot and retain graph-attached per-layer key/value state.
2. Insert the returned final hidden state as the first latent.
3. For consecutive latent positions, advance with the previous generated latent as the next one-token segment.
4. Across text or structural gaps, process the exact intervening embedding segment once, ending immediately before the next latent.
5. Carry graph-attached cache state without detaching it and without retaining obsolete cache concatenation versions.
6. Assemble all generated latents into the complete sequence and run the unchanged final full-sequence no-cache decoder and loss path.

The prototype must use an explicit Stage 2 cache adapter rather than silently depending on mutable inference-cache behavior. Cache state is runtime-only and is never stored as model or optimizer state.

### 5.3 Runtime identity and resume

Write a separate `stage2-recurrence.json` sidecar rather than changing the strict PEFT metadata schema. Schema version 1 records:

- backend (`no_cache` or `segmented_kv`);
- algorithm version (`no-cache-v1` or `segmented-kv-v1`);
- a digest of recurrence-semantic configuration;
- global step for checkpoint copies.

A legacy checkpoint with no recurrence sidecar deterministically means `no_cache` with algorithm `no-cache-v1`. It may resume only as `no_cache`. Publication and validation follow the existing collective runtime-identity pattern: every rank reaches the same validation decision before DeepSpeed/model preparation, and only the designated writer publishes the immutable file.

## 6. Correctness and leakage-resistant validation

The current no-cache code remains structurally independent of the candidate path and is the primary oracle. A tiny deterministic hand-unrolled causal model supplies a second oracle for segment boundaries and cache advancement. Candidate code may not be shared with either oracle.

Predeclare these gates:

### FP32 eager structural gate

For loss, every latent state, every supervised-position logit, and every trainable parameter gradient:

- `atol = 1e-6`;
- `rtol = 1e-5`.

The same limits apply to every trainable parameter after one identical optimizer step and after a save/resume continuation witness.

### BF16/FlashAttention 2 integration gate

For loss, latent states, and supervised-position logits:

- `atol = 2e-2`;
- `rtol = 2e-2`.

For every trainable gradient tensor with a nonzero reference norm:

- cosine similarity at least `0.999`;
- relative L2 error at most `0.02`.

Zero-reference gradient tensors must remain all-close at `atol = 2e-2`, `rtol = 0`.

Test full tuning, LoRA, and DoRA first on a tiny single-device model. Widen to FlashAttention 2, the selected DeepSpeed/ZeRO configuration, distributed sampler alignment, and resume only after the cheapest representative B200 falsification run passes.

## 7. Performance adoption gate

Compare against the fastest feasible optimized `no_cache` configuration, including gradient checkpointing when it improves feasible throughput. Use the same effective batch size and representative examples.

`segmented_kv` can be recommended for explicit dropout-zero runs only if:

- median steady-state end-to-end training-step time improves by at least 15%;
- CUDA peak allocated memory regresses by no more than 5%;
- the p95 length/block-count sample does not OOM;
- every correctness, optimizer-step, distributed, and resume witness passes.

Failure of any condition leaves `no_cache` as the only supported production backend. No tolerance or performance threshold may be adjusted after observing candidate results without declaring a new experiment version.

## 8. Error handling

- Loss-only mode without labels fails immediately.
- A manifest identity mismatch fails before launching distributed workers.
- Invalid or mixed recurrence identities are collected consistently across ranks and fail before training collectives.
- `segmented_kv` with nonzero stochasticity or gradient checkpointing fails with the exact incompatible setting named.
- Any non-finite loss, latent state, supervised logit, or gradient fails the equivalence probe.
- A performance-gate miss is a valid negative result, not an implementation failure; it stops backend adoption.

## 9. Critique synthesis and remaining uncertainty

The original proposal v1 was revised materially after independent review:

- “adapter dropout is zero” became “all active stochastic modules are proven deterministic”;
- supervised logits, optimizer-step, and resume witnesses were added to aggregate loss and gradient checks;
- the comparison baseline became the best feasible no-cache configuration, including gradient checkpointing;
- legacy missing-sidecar behavior was fixed as `no_cache`, and algorithm version/digest were added;
- representative profiling and a 30% prefix-time gate were moved before KV implementation;
- the manifest was bound to exact dataset, image, processor, template, and length semantics.

Reviewers agreed on the staged exact-first direction. Their material divergences were compatible rather than contradictory: one identified stochastic/causal equivalence as the strongest risk, while another identified lack of end-to-end performance headroom. Both are handled as independent gates.

Unvalidated premises remain:

- repeated-prefix work may be less than 30% after exact optimizations;
- a graph-attached segmented cache may retain too much activation memory without checkpointing;
- segment shape changes may create BF16 drift beyond the declared limits;
- DeepSpeed/ZeRO collectives may erase the token-compute speedup.

There is no waiver for changing Stage 2 semantics. Approval authorizes measurement and a default-off deterministic prototype only; it does not authorize adopting a backend that misses an invariant or gate.

## 10. Explicit non-goals

- changing the Stage 2 latent definition, scaffold, label objective, or supervision boundary;
- making KV caching the default;
- approximating or reusing dropout masks to force cache compatibility;
- enabling cache on existing dropout-bearing checkpoints;
- changing batch size, sampler policy, or dataset membership;
- patching upstream PEFT DoRA internals, including its `torch.eye` construction;
- adding generalized retry, recovery, or profiling infrastructure.
