# Packed admission for the Step 2B DoRA ablation

## Status

Design commitment v11, pending explicit user approval.

This commitment amends the capacity gates and training population in
`2026-09-04-step-preference-dora-ablation-readiness-design.md`. Everything not
changed here remains governed by the approved design commitment v5.

## Observed blocker

The mixed attention path now uses vision FlashAttention2, ordinary/reference
language SDPA, and packed-decoder FlexAttention. That change allowed source row
1078 to complete exact negative generation, producing 12 valid pairs. The
canonical DoRA 32/64/0.05 capacity probe passed row 2102 at packed padded length
1280 with 24,440,865,792 peak allocated bytes, but row 1078 OOMed at clean
length 8097 and packed padded length 8960 with 91,857,799,168 peak allocated
bytes and 98,962,505,728 reserved bytes.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` did not prevent the OOM.
Canonical LoRA also OOMed on the same packed shape, although its allocation was
about 4.14 GiB lower. The observed primary blocker is therefore the admitted
packed layout, not vision SDPA, allocator fragmentation, or DoRA alone.

The existing 8192 limit measures the clean sequence, not the actual packed
model input. It therefore does not bound packed attention memory. A complete
clean scan measured 3450 source rows: 3379 pass the clean 8192 limit, and 3374
have clean length at most 7168.

## Frozen experiment identity

The training source is `data/TRAIN.jsonl` (3450 rows, SHA-256
`5bc1313a594f5713975b67388c745c6632275c37fd260055b27230c9cfedbed8`) and
the image root is `data`. The evaluation source is `data/TEST.jsonl` (400 rows,
SHA-256
`6a1dccf32bb5196d09941adfd8ea79e29b58c34316f286673b557ed29dc69618`).
The full Stage 2A base is
`checkpoints/zero-stage-compatibility/stage2-eval-full-smoke-20260827`; its
`config.json` hash is
`cbec86a55c91603e7f91d506d94c975113de3af4ef6e9cce6e124e6825bc7b44`
and its `model.safetensors.index.json` hash is
`d28f59cc1060d05418f59a2320b4f53f83ba50374d7283d59ee559aefe35864b`.

All three arms use seed 42, BF16, local batch 1, gradient accumulation 2, two
epochs, two local processes, learning rate `5e-5`, `adamw_torch`, weight decay
0.01, a linear scheduler with 10 warmup steps, gradient clipping 1.0, and
`configs/deepspeed_zero2.json` (SHA-256
`20615765b665d38647ee0447ca984ea28d16a1b8738363b505b054871bfc3f85`).
DoRA targets are
`q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj`. Latent size is 8,
latent boundaries are supervised, dataloader workers are zero, and the
assistant rendering identity is `adjacent-text-space-lf-v1`.

The objective coefficient triplets `(CE,DPO,entropy)` are CE-only `(1,0,0)`,
DPO-only `(0,1,0)`, and entropy-only `(0,0,1)`. DPO beta and entropy margin are
both 0.1. Every arm still computes all three raw components through the same
packed forward graph; only scalar coefficients differ.

One shared negative artifact is generated greedily with
`scripts/generate_step_negatives.py`, `min_tokens=2`, `max_new_tokens=32`,
language SDPA, vision FlashAttention2, BF16, and `cuda:0`. Its existing strict
schema-3 manifest authenticates its source, frozen checkpoint files, tokenizer
configuration, rendering, generation settings, selected order, and JSONL hash.
The existing strict schema-2 Stage 2 length manifest authenticates the clean
source ordering and measured no-truncation lengths.

Training model state is written only to
`checkpoints/step-pref-dora-{ce,dpo,entropy}-<UTC>/`; metrics and samples go to
the matching `results/` directory and logs to the matching `logs/` directory.
The three arms launch sequentially through `run_stage2_training.sh`. Evaluation
uses `eval.py`, the same Stage 2A base plus each final adapter, the full TEST
set, language SDPA/vision FlashAttention2, `max_sequence_length=32768`,
`max_new_tokens=4096`, repetition penalty 1.05, temperature 0, and top-p 1.
The primary comparison is coverage accuracy (`correct / 400`); exact equality
is a tie and valid-gold accuracy is secondary only.

## Normative terms and entry points

- **Clean length** is the contiguous non-padding length of the single collated
  root `input_ids`, equal to `PackedTreePlan.root_length`, after the fixed
  space-LF rendering and before any preference sidecars are appended.
- **Natural packed plan** is the unmodified result of
  `step_preference.packing.build_packed_tree_plan` from the production
  `Stage2DataCollator` batch and its joined schema-3 negatives. Its unpadded
  length is the root plus all valid natural sidecar tokens. Its padded length is
  `ceil(unpadded_length / 128) * 128`, the fixed FlexAttention block rule.
- **Production collator** means `stage2.data.Stage2DataCollator` configured by
  the same strict Stage 2B preflight used by `src/main.py`, not a duplicate
  tokenizer or planner implementation.
- **Combined-positive objective** uses coefficients `(1,1,0.1)` with DPO beta
  0.1 and entropy margin 0.1. The canonical single-device probe is
  `scripts/probe_packed_stage2b_capacity.py --device cuda:0 --tuning_method
  dora --gradient_checkpointing` with the frozen identities above.
- The admission producer and verifier will be implemented as separate
  subcommands of `python -m step_preference.packed_admission`: `build` writes
  the manifest and retained JSONL; `verify` writes only a verifier result. The
  implementation plan must give their complete commands and tests before any
  runtime use.
- This document fixes design behavior and interfaces; it is not the executable
  implementation plan. The parent v5 commitment and this amendment form the
  complete design input. After approval, `superpowers:writing-plans` must pin
  exact CLI invocations, output filenames, per-field Python/JSON types, and
  causal RED/GREEN tests before code or runtime work begins.

## Decision and scientific scope

Exclude whole examples whose unchanged natural packed plan exceeds a new
packed length cap of 7168. Keep the clean tokenizer/model limit at 8192. Do not
truncate, split, or prune branches. A retained example keeps its exact natural
packed plan, targets, masks, selected positions, and losses.

This changes the estimand: the experiment compares CE-only, DPO-only, and
entropy-only on the **capacity-admitted subset**, not on the full original
training population. Sharing the exact retained subset across arms preserves
their internal comparison, but it does not establish performance on excluded
long or structurally complex examples. Every result must report that limitation
and the complete exclusion profile.

Branch splitting or microstep backward is deferred because it changes loss
normalization, optimizer-step semantics, and distributed scheduling. ZeRO-3 or
offload is deferred because it changes the fixed runtime architecture. LoRA and
shorter negatives are rejected because they change the requested method or
artifact, and LoRA already failed on the observed layout.

## Fixed runtime and admission interface

- Keep `MAX_SEQUENCE_LENGTH=8192` as the clean upper bound.
- Add `max_packed_sequence_length=7168`, exposed as
  `--step_max_packed_sequence_length` and
  `STEP_MAX_PACKED_SEQUENCE_LENGTH=7168`. It must be positive when packing is
  enabled.
- Keep DoRA 32/64/0.05, BF16, language SDPA, vision FlashAttention2, packed
  FlexAttention, non-reentrant checkpointing, two-rank ZeRO-2, seeds,
  optimization settings, negative protocol, and objective triplets unchanged.
- Evaluation remains the full 400-row TEST set because it does not construct
  packed preference trees.

## First nail: cheap tail pilot

Before implementing the manifest path or generating full negatives, generate
exact one-row schema-3 artifacts for these fixed source indices from the
authenticated clean scan:

| Source index | Clean length | Reasoning blocks | Purpose |
|---:|---:|---:|---|
| 1372 | 6065 | 5 | closest expected natural layout below the cap |
| 1714 | 5660 | 5 | fallback boundary if row 1372 packs over the cap |
| 1582 | 5256 | 6 | higher branch-count shape |

Build each unchanged natural packed plan. A pilot is ignored only if its
natural padded length exceeds 7168, because such a row would not be admitted.
At least two pilots must be admitted, including the admitted row with maximum
padded length and the six-block row. Run the canonical one-H100
combined-positive DoRA forward/backward probe on every admitted pilot. Each
must produce finite component losses, a finite nonzero trainable-gradient norm,
and peak reserved memory no greater than 85% of total device memory. Any failure
rejects the 7168 cap before full generation; the cap must not be lowered
silently.

Here and throughout this document, source indices are zero-based JSONL line
ordinals. “Maximum admitted pilot” means the greatest natural padded length
among these three candidates, with the packed-length audit tuple used to break
ties. The clean-device and CUDA measurement rules in the post-generation
section apply unchanged to pilot probes. The finite raw components are CE,
DPO, entropy, and total as emitted by the canonical probe; its gradient witness
is the FP32 L2 norm across all trainable DoRA parameters with missing gradients
ignored, and it must observe at least one gradient.

Concretely, if row 1582 is over-cap, the mandatory six-block condition fails
and the 7168 design is rejected; no replacement row is selected without a new
explicit design decision. If fewer than two rows are admitted, the design is
also rejected. Thus “ignored” means no capacity execution for an excluded row,
not that a required witness can be waived.

A missing, invalid, duplicate, or generation-failed pilot negative rejects the
pilot. It is never converted into an admission exclusion or silently retried.

The combined-positive probe evaluates every objective branch and backpropagates
their sum. Since each final arm uses the same forward graph and trainable
parameters while zeroing two scalar coefficients, this is a conservative
single-device graph-capacity probe, not an accuracy proxy.

## Clean-admissible source and shared negatives

Only after the pilot passes, create the schema-2 clean manifest and an ordered
`clean-admissible.jsonl` containing exactly the original rows whose clean length
is at most 7168. Record the original TRAIN SHA-256, clean-manifest SHA-256,
ordered original indices/sample IDs, and subset SHA-256. This prefilter cannot
remove a packed-admissible row because packed length is never less than clean
length.

Generate one schema-3 negative artifact against this source. Every arm must use
the exact same source and negative paths and hashes.

`clean-admissible.jsonl` is an exact byte-preserving ordered subset of the
original JSONL and is the input to shared negative generation. The later
packed-retained JSONL is an exact byte-preserving ordered subset of
`clean-admissible.jsonl` and is the authoritative training source for every arm;
the admission manifest is the authority for its original zero-based indices and
hash. Each emitted row ends in one LF. Any missing, duplicate, invalid, or
generation-failed negative makes shared generation fail as a whole; it is not a
filter reason.

## PackedAdmissionManifest schema 1

One producer creates an exclusive manifest that binds:

- original TRAIN and clean-admissible source paths and hashes, including the
  ordered original indices/sample IDs;
- negative artifact and negative-manifest paths and hashes;
- rendering and tokenizer/config identities, language and vision backends,
  latent size, clean limit 8192, and packed cap 7168;
- for every clean-admissible row: sample ID, clean length, packed unpadded and
  padded lengths, vision-token count, image count, reasoning-block count,
  branch count, scored-token count, and admission reason;
- ordered retained and filtered indices/sample IDs, counts, and boundary
  witness identities.

A separate read-only verifier entry point independently reads the bound source
and negative artifacts, rebuilds every natural packed plan through the
production collator, and compares the complete ordered records and partition
with the manifest. It writes one verification result and never modifies the
manifest. Every arm requires the same manifest path and expected SHA-256 plus
the same verifier-result path and expected SHA-256.

Stage 2B preflight validates all bound artifact and configuration identities and
selects only the recorded retained order. Normal training collation rebuilds
each selected plan. A padded length that differs from the manifest record or
exceeds 7168 raises before model forward. There is no dynamic skipping or
arm-specific filtering.

Both files use exact-field JSON schema 1, UTF-8, no duplicate keys, no NaN or
Infinity, and canonical serialization with sorted object keys, compact
separators, `ensure_ascii=False`, followed by one LF. A SHA-256 always covers
the complete on-disk bytes including that LF. Writers publish exclusively to a
new regular file and never follow or create a symlink.

The admission manifest has exact top-level fields
`schema_version,kind,identity,records,retained_source_indices,retained_sample_ids,filtered_source_indices,filtered_sample_ids,boundary_witnesses,counts`.
`schema_version` is integer 1 and `kind` is
`"stage2b-packed-admission"`. `identity` contains nonempty resolved paths and
64-lowercase-hex hashes for the original source, clean-admissible source,
clean-length manifest, negative JSONL, negative manifest, checkpoint config and
weight index, plus the rendering, tokenizer/config identity, language backend,
vision backend, latent size, clean limit, and packed cap. Integer fields reject
booleans.

Each ordered `records` entry has exact fields
`source_index,sample_id,clean_length,packed_unpadded_length,packed_padded_length,vision_token_count,image_count,reasoning_block_count,branch_count,scored_token_count,admission_reason`.
Counts and lengths are nonnegative integers, lengths are positive, sample IDs
are nonempty strings, and source indices/sample IDs are unique. The only reason
values are `"retained"` and `"packed_length_exceeds_limit"`, with the value
required to agree with `packed_padded_length <= 7168`. The four ordered
retained/filtered arrays must be the exact stable partition of `records` in
source order. `counts` contains exact integers
`original,clean_admissible,retained,filtered`; `boundary_witnesses` contains the
ordered Pareto source indices and the selected maximum-packed source index.

`records` contains exactly the 3374 clean-admissible rows, in original source
order. `filtered` means only packed-over-cap rows among those records. The
schema-2 clean manifest separately records the 76 original rows rejected by the
clean prefilter; the final capacity summary joins both sources to report all
original exclusions and reasons. The identity also binds the image root, the
packed-retained JSONL path/hash, the producing git commit, and a clean-worktree
flag. The existing negative-manifest checkpoint inventory binds every referenced
weight shard, not only the config and index files named in the frozen identity.
Producer and verifier both assert `packed_unpadded_length >= clean_length`.

The verifier result has exact fields
`schema_version,kind,status,manifest_path,manifest_sha256,clean_admissible_sha256,packed_retained_sha256,negative_sha256,negative_manifest_sha256,verified_record_count,retained_source_indices,filtered_source_indices,errors`.
Its kind is `"stage2b-packed-admission-verification"`; status is `"pass"` only
when every independently reconstructed record, order, partition, identity, and
hash equals the manifest. Otherwise it publishes `"fail"` with at least one
error and exits nonzero. Training accepts only a pass result whose manifest
path/hash, artifact hashes, count, and two ordered index arrays match the
separately supplied admission manifest.

## Post-generation capacity witnesses

From retained rows, select the unique Pareto boundary over four recorded adverse
features: packed padded length, vision-token count, branch count, and
scored-token count. Row A strictly dominates row B only when A is at least as
large on all four features and strictly larger on at least one. Exact equal
four-tuples form one equivalence class; retain the member with the lowest source
index as its deterministic Pareto representative and record every class member
and its representative in `boundary_witnesses`. Separately choose the global
maximum-packed witness by
`(packed_padded_length,packed_unpadded_length,clean_length,source_index)`,
matching the existing packed-length audit. The final probe set is the union of
the nondominated Pareto representatives and that global maximum witness,
deduplicated and sorted by source index. If this final set contains more than
eight rows, stop for explicit review rather than choosing a convenient subset.

Run the canonical one-H100 combined-positive DoRA forward/backward probe on
every row in the final probe set. Each must have finite component losses, a
finite nonzero gradient, and peak reserved memory at most 85% of total device
memory.

These probes run only when `nvidia-smi` reports no foreign process and zero MiB
used on the selected physical NVIDIA H100 NVL. Set `CUDA_VISIBLE_DEVICES=0`,
load the model and batch, synchronize `cuda:0`, record the allocator baseline,
then call `torch.cuda.reset_peak_memory_stats(cuda:0)` immediately before the
single `Stage2BTrainingStep`. Synchronize immediately after backward and read
`torch.cuda.max_memory_reserved(cuda:0)`. Read total bytes from
`torch.cuda.get_device_properties(cuda:0).total_memory`; pass only when
`peak_reserved_bytes / device_total_bytes <= 0.85`. Record the GPU name/UUID,
baseline allocated/reserved, peak allocated/reserved, total bytes, exact command,
git commit, and exit status. Unrelated allocations or a nonempty device stop the
probe rather than changing the denominator.

An empty retained set rejects the design. If the Pareto set exceeds eight, the
producer and verifier may publish their complete pass artifacts, but no
capacity probe or training begins; changing witness selection requires a new
explicit commitment. Resetting peak statistics does not clear the allocator,
so the recorded baseline remains included intentionally in peak reserved
memory. The implementation plan must specify one read-only GPU preflight that
rejects any pre-existing compute process, binds physical GPU 0 by UUID, and is
used identically for pilots and Pareto probes.

Then run two optimizer steps on the maximum-packed-length witness with the exact
two-rank ZeRO-2 stack and combined-positive objective. The first step witnesses
lazy optimizer-state allocation and the second steady execution. Require a
valid DoRA checkpoint and no OOM or non-finite value. Any failure stops before
ablation training.

The distributed gate launches `run_stage2_training.sh` with two local
processes, local batch 1, accumulation 1, `(1,1,0.1)`, the frozen DoRA and
backend settings, the shared authenticated artifacts, and `--max_steps 2
--epochs 1`. It passes only with exit zero, logged global steps 1 and 2, finite
raw and total losses at both steps, no skipped update/OOM, and a final artifact
inside its fresh `checkpoints/<run-id>/`. `stage2.checkpointing.inspect_stage2_artifact`
must classify that artifact as Stage 2 PEFT with tuning method `dora`; regular
nonempty `adapter_config.json`, `adapter_model.safetensors`, and
`stage2_peft_metadata.json` must exist, and the metadata must bind the frozen
Stage 2A identity, latent configuration, target modules, and DoRA parameters.
The capacity summary records hashes of those three files.

For this capacity-only singleton gate, the ordinary distributed sampler pads
the one retained occurrence onto both ranks; the Trainer restarts the iterator
to reach step 2. Both ranks must report the same sample ID at both steps. The
capacity gate's accumulation 1 intentionally maximizes optimizer-step frequency
and covers optimizer allocation, while final accumulation 2 has the same
per-microbatch activation shape and fewer optimizer boundaries. A mismatch in
rank sample IDs or step count fails the gate.

This does not mathematically prove that every retained row fits: allocator
behavior and feature interactions may be non-monotonic. Exhaustively probing
roughly 3000 rows would approach an additional training epoch and is not part of
this commitment. The 15% headroom, multidimensional Pareto boundary,
fail-closed runtime length check, and two-step distributed gate define the
accepted practical evidence boundary.

## Matched arms and stopping rules

CE-only, DPO-only, and entropy-only use the identical clean-admissible source,
shared negative artifact, packed-admission manifest and verifier result,
retained order, cap, and runtime. Normalized launch records may differ only in
run/output identities and the coefficient triplet. Each result records original,
clean-admissible, packed-retained, and packed-filtered counts, every excluded ID
and reason, all governing hashes, and the capacity-admitted-subset limitation.

If packed admission excludes more than 1% of the 3374 clean-admissible rows,
stop for a new explicit decision. The 1% threshold is only a review trigger; it
is not evidence that a smaller exclusion is unbiased.

The integer rule is `filtered >= 34`; it applies only to the second-stage packed
filter, not to the 76 rows already excluded by the clean 7168 prefilter. Arms
launch in the fixed order CE-only, DPO-only, entropy-only. A failed arm stops
later arms; no arm resumes or reuses partial Trainer state without a new
explicit decision.

## Critique synthesis and risk waiver

The load-bearing objection (`hate`) is that 7168 may still fail on a natural
high-branch layout, wasting later manifest and full-generation work. Its exact
result is `{root: "the proposed cap can still admit a natural high-branch layout
that OOMs, invalidating the launch premise", first_nail: "generate and probe
fixed tail rows 1372, 1714, and 1582 before full negatives; reject on an
admitted failure or peak reserved memory above 85%"}`. The first nail has not
run because executing it is implementation-like and this commitment is not yet
approved.

Fresh independent readings (`macrothink`) converged only as reassurance on an
arm-blind feasible cohort, a frozen shared identity/order, real boundary probes,
and the capacity-admitted estimand. Their material incompatible divergence was
that some readers required every Pareto witness under two-rank ZeRO-2, or each
loss arm separately, rather than one-GPU Pareto probes plus one two-rank maximum
witness. That stronger option is not adopted: all arms compute the same three
raw loss graphs over the same trainable parameters, combined-positive backward
activates all of them, optimizer-state size is coefficient-independent, and up
to eight separate distributed launches would add cost without isolating a known
different graph. The unresolved uncertainty is that distributed allocator or
communication behavior could nevertheless be non-monotonic across Pareto
shapes. A second divergent suggestion—deliberately find the closest excluded
row that OOMs—is also not adopted because the observed natural 8960 layout
already establishes a real failure beyond the cap and boundary hunting adds
intentional OOM work without validating retained rows.

All independent readings identified population shift as the largest scientific
limitation. This document therefore makes the capacity-admitted population
explicit rather than claiming full-population results.

Independent correctness, validity, and cost lenses (`prism`) all pass this v11
commitment. Correctness accepts the staged fail-closed escalation and capacity
witnesses after the global maximum was explicitly unioned with Pareto
representatives; validity accepts the separately hashed verifier and explicit
conditional estimand; cost accepts the three early pilot rows, the final maximum
of eight boundary probes, and two distributed optimizer steps. The v10
correctness failure and its v11 repair are preserved in the execution ledger.
Their agreement is review evidence, not runtime proof.

Approval of this commitment explicitly waives these remaining risks:

1. Filtering changes the scientific population.
2. The empirical 7168/85% envelope can miss a non-monotonic retained-row OOM.
3. Full negative generation can still reveal artifact or runtime failures after
   the pilot.
4. The separate verifier shares the authoritative production collator/planner,
   so a defect in that common primitive can affect both producer and verifier.
5. Only the maximum-packed witness receives the exact two-rank gate; other
   Pareto witnesses are single-device probes, so shape-specific distributed
   non-monotonicity remains possible.
