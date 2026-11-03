# Multi-node Stage 2 Training Guidance and Specification

## 1. Status and scope

- Filename version: `20260920`
- Environment evidence captured: 2026-09-02
- Target: two nodes, four NVIDIA H100 NVL GPUs, Stage 2 DoRA training
- Primary goal: minimize end-to-end training time without changing the Stage 2
  recurrence, objective, data membership, or tuning method
- Status: design proposal v2; implementation and production benchmark are not
  yet complete

This document defines the canonical multi-node candidate, its launch contract,
the evidence required to adopt it, and the fallback behavior. It does not claim
that four GPUs are faster until the comparison in Section 11 passes.

### 1.1 Design commitment v2 approval candidate

- **Problem and scope:** define the fastest credible two-node Stage 2 candidate
  without changing recurrence, loss, data membership, DoRA, or effective batch
  size.
- **Final design:** plain four-rank DDP is the candidate; `.41` is static rank 0,
  `.40` is machine rank 1, `bond0.11` is the exact communication interface,
  and the batch geometry is micro-batch `1` with accumulation `2` for effective
  batch `8`. ZeRO-1 is a capacity-only fallback, not an automatic alternative.
- **Change from proposal v1:** world-size-based bucket padding was disproved by
  a CPU occurrence-count witness. Sampler v2 instead creates world-size-
  independent logical frames of eight, so the two-GPU control and four-GPU
  candidate consume identical ordered occurrences and multiplicities.
- **First nail:** run one warm-up and two timed updates on the same sealed p95
  logical frame for two-GPU/accumulation-4 and four-GPU/accumulation-2. Reject
  the four-GPU candidate if its maximum-rank time is not lower. This has not
  been run because the multi-node launcher and sampler v2 are not implemented;
  therefore the performance premise remains unvalidated.
- **Critique synthesis:** fresh-read alternatives converged on DDP-first but
  required one sharding layer, recurrence-work-aware frames, a maximum-work
  capacity check, and collection of all rank RNG sidecars. The independent
  correctness, performance, and resume lenses agree that production launch is
  blocked until their witnesses pass. The performance lens remains unresolved
  until the first nail; this disagreement is carried into the adoption gate,
  not averaged away.
- **Validation independence:** performance frames and their cursor are sealed
  before timing; both controls replay the same frames, and adoption uses raw
  rank-max synchronized timings. The p95 frame is only a falsification probe,
  not the sole adoption workload.
- **Unvalidated premises:** the selected model fits one H100 under the declared
  capacity limit, all trainable DoRA parameters participate as required, four
  GPUs beat two across the 10 Gbps link, and four-rank save/resume reconstructs
  complete RNG and sampler state.
- **Risk boundary:** approving this specification accepts those premises only
  as explicitly gated uncertainties. It does not waive the capacity,
  correctness, sampler, save/resume, or 15% performance gates and does not
  authorize a production launch that fails them.

## 2. Non-negotiable Stage 2 method

Multi-node execution changes only data-parallel placement and gradient
reduction. It must preserve all existing Stage 2 semantics:

1. Use the optimized `no_cache` recurrence. Do not add or enable a KV-cache
   backend.
2. Generate every continuous latent from the final decoder hidden state of its
   causal prefix, in the existing order.
3. Keep every generated latent attached to autograd through later latents and
   the final loss.
4. Keep the mandatory final full-sequence decoder pass, shifted assistant-token
   cross-entropy, label masks, latent-boundary setting, image features, RoPE
   positions, and attention mask unchanged.
5. Keep DoRA with `r=32`, `alpha=64`, dropout `0.05`, and the existing decoder
   target modules. Do not replace it with LoRA or full tuning for the canonical
   run.
6. Keep rank-local micro-batch size at `1` and effective global batch size at
   `8`.
7. Keep the sealed length manifest and retained raw-index membership. New
   multi-node runs use the world-size-independent logical-batch schedule in
   Section 7.2 so two- and four-GPU controls consume identical sample
   multiplicities. Existing sampler-v1 checkpoints keep their original sampler
   and resume contract.

The automatic exact optimizations remain enabled: supervised-row loss-only
projection, prefix-sized embedding assembly, one final full embedding assembly,
opt-in diagnostics with the production default disabled, and immutable length
manifest consumption.

## 3. Observed target environment

| Role | Host | DDP machine rank | Address | GPUs | Artifact storage |
|---|---|---:|---|---|---|
| Master/writer | `korea-21` | 0 | `172.16.11.41` | 2× H100 NVL 95,830 MiB | `/root/workspace/ILVR/checkpoints` → `/raid/workspace/ILVR/checkpoints` |
| Worker | `korea-20` | 1 | `172.16.11.40` | 2× H100 NVL 95,830 MiB | local canonical `./checkpoints` symlink |

Both hosts were observed with driver `580.173.02`. Both repository environments
reported Python packages with PyTorch `2.7.1+cu128`, CUDA `12.8`, Accelerate
`1.10.1`, DeepSpeed `0.18.0`, and local Transformers `4.50.0.dev0`.

The inter-node path is `bond0.11`, an active-backup bond with an active 10 Gbps
link on each host. It is not a 20 Gbps aggregate. Therefore the design avoids
parameter all-gathers inside Stage 2's repeated decoder forwards.

The observations above are not permanent configuration truth. Every launch
must rerun the preflight in Section 8.

## 4. Selected parallelism

### 4.1 Canonical candidate: plain DDP

Run one process per GPU with four global processes. Each process holds the BF16
base model and its local DoRA adapter replica, including the permitted sparse
latent-token rows when the selected Stage 1 continuation contract contains
them. DDP reduces all declared trainable gradients after backward; it does not
shard or gather the frozen decoder parameters for each latent-prefix forward.

This is the best first candidate when the selected Stage 1 base plus DoRA model
passes the one-GPU p95 capacity probe in Section 5.1. DoRA trains a small subset
of parameters, while the 10 Gbps inter-node link is the scarce resource. PyTorch
recommends DDP when a model fits on one GPU and `torchrun` or an equivalent
launcher for multi-node execution. Accelerate is already part of this repository
and is a passthrough launcher for this model and Trainer stack.

### 4.2 Capacity fallback: ZeRO-1

Consider `configs/deepspeed_zero1.json` only if the DDP candidate narrowly
misses the declared VRAM limit and a model-state estimate shows that partitioning
optimizer state alone is sufficient. ZeRO-1 keeps parameters replicated and
does not solve an activation- or parameter-capacity failure. It requires its own
bounded smoke and is not benchmarked automatically.

### 4.3 Rejected defaults

- ZeRO-2 is not the default because DoRA optimizer and gradient state are small
  relative to replicated frozen parameters, so its extra sharding does not
  address the expected bottleneck.
- ZeRO-3 is not the default because Stage 2 invokes the decoder once per latent
  prefix and once for the final full pass. Parameter sharding can introduce
  collectives on each invocation across the 10 Gbps link. Rank-varying latent
  counts also make forward-time collective alignment more fragile.
- FSDP, tensor parallelism, pipeline parallelism, KV caching, CPU offload, NVMe
  offload, and elastic membership are out of scope.

## 5. Canonical training configuration

| Setting | Value |
|---|---|
| Tuning | DoRA |
| Recurrence | optimized `no_cache` |
| Precision | BF16 |
| Attention | FlashAttention 2 |
| Nodes / GPUs | 2 / 4 |
| Processes | 4 total, 2 per node, 1 per GPU |
| Per-device batch | 1 |
| Gradient accumulation | 2 |
| Effective global batch | `1 × 4 × 2 = 8` |
| Learning rate | `1e-5`, without automatic batch scaling |
| DoRA rank / alpha / dropout | 32 / 64 / 0.05 |
| DoRA targets | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` |
| Latent size | 8 |
| Max sequence length | 8192 |
| Latent-boundary supervision | enabled |
| Diagnostics interval | 0 |
| Seed | 42 |
| Rendezvous | static, no elastic restart |
| Multi-node sampler | logical-batch schedule v2, batch width 8 |
| DDP unused-parameter search | disabled only after all-trainable coverage witness |

The single-node control uses the same values except world size `2` and gradient
accumulation `4`, so its effective global batch is also `8`.

### 5.1 Gradient checkpointing selection

The preferred candidate disables gradient checkpointing because recomputation
adds work to an already repeated-forward method. Select p95 by recurrence work,
defined as the sum of every latent-prefix decoder input length plus the final
full-sequence length, rather than final sequence length alone. Run a real
forward, backward, and first optimizer step so activation graphs and Adam state
are both allocated:

- keep checkpointing disabled when every rank completes and peak allocated VRAM
  is at most 85% of local GPU capacity;
- enable the existing non-reentrant checkpointing path if any rank exceeds 85%
  or raises CUDA OOM;
- do not change sequence length, latent size, batch size, or Stage 2 semantics as
  an automatic fallback.

Also run the known maximum-recurrence-work retained example as an admission
check. A p95 pass alone does not prove that the upper 5% cannot OOM.

This is a one-time selection recorded in the run configuration, not runtime
auto-tuning.

### 5.2 DataLoader worker selection

Measure worker counts `0`, `2`, and `4` independently on both nodes using the
sealed dataset and representative batches. After two warm-up batches, measure
twenty collated batches or one retained epoch when shorter. Select one worker
count for all ranks that minimizes the slower node's median collation time
without exceeding host-memory capacity. Do not tune workers during training.

## 6. Network and rendezvous contract

Set these values on both nodes:

```bash
export MASTER_ADDR=172.16.11.41
export MASTER_PORT=29520
export NCCL_SOCKET_IFNAME='=bond0.11'
export GLOO_SOCKET_IFNAME=bond0.11
export NCCL_SOCKET_FAMILY=AF_INET
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
```

The exact-interface form for `NCCL_SOCKET_IFNAME` prevents NCCL from selecting
Docker, Tailscale, VXLAN, or another UP interface. NVIDIA documents that manual
interface selection bypasses automatic selection. Do not retain diagnostic
variables such as `NCCL_DEBUG=INFO` in the production command; enable them only
for a failing smoke run.

Use Accelerate's static rendezvous with:

- `--num_machines 2`;
- `--num_processes 4` total;
- `--machine_rank 0` on `.41` and `1` on `.40`;
- `--main_process_ip 172.16.11.41`;
- `--main_process_port 29520`;
- `--same_network`;
- `--rdzv_backend static`;
- no restart or changing membership.

Both node commands must be started for the job to form. The rank-1 command may
start first and wait at rendezvous. The port must be unused and reachable from
`.40` before launch.

## 7. Code, data, manifest, and artifact identity

### 7.1 Code and environment

Before launch, both nodes must report the same full Git commit and a clean
tracked tree for runtime files. The local worktree and the remote canonical
repository were not at the same Stage 2 feature commit when this document was
drafted; this is a current launch blocker, not a tolerated difference.

Both nodes must use `/root/workspace/ILVR/venv/ilvr-venv`. Record hashes for:

- the Git commit;
- `run_stage2_training.sh` or its multi-node successor;
- `src/main.py`;
- `src/stage2/` runtime modules;
- the local Qwen Stage 2 rollout and modeling files;
- the selected launcher configuration.

### 7.2 Dataset, manifest, and logical-batch schedule

The same `TRAIN.jsonl`, image set, processor source, chat template, latent
settings, and maximum sequence length must exist locally on both nodes. Rank 0
creates one immutable Stage 2 length manifest. Copy the exact bytes to rank 1
and verify the SHA256 there before starting any distributed process.

The existing sampler pads every latent-block bucket to `world_size`. A focused
CPU witness showed that the same five-example bucket geometry produced one
padded sample at world size 2 and three at world size 4, changing per-sample
multiplicity. That behavior cannot serve as a controlled two-versus-four-GPU
comparison.

Fresh multi-node runs therefore require sampler schedule schema v2:

1. Give every retained example a stable raw-index identity, latent-block count,
   measured length, and recurrence-work value. Seal their ordered digest.
2. For each epoch and latent-block-count bucket, deterministically shuffle with
   a seed derived only from the dataset digest, base seed, epoch, and bucket.
   Sortish-pack the shuffled entries by recurrence work so one slow sample does
   not unnecessarily stall ranks with much cheaper work.
3. Place every retained raw index into the schedule exactly once before padding.
   Pad each latent-block bucket to a multiple of eight by cycling from an
   epoch-rotating offset in that bucket's shuffled IDs. Record a distinct
   occurrence ID, source raw index, and padding flag for every slot.
4. Chunk each bucket into homogeneous logical optimizer frames of eight, then
   deterministically shuffle the completed frames. Materialize the schedule
   once and seal its digest. Both controls consume the exact same ordered frames
   and padding occurrences.
5. For the two-GPU control, shard each logical frame as two ranks over four
   microsteps. For the four-GPU candidate, shard it as four ranks over two
   microsteps. In both cases, global frame slot `s` is assigned by
   `s = microstep * world_size + global_rank`; concatenating slots in that
   microstep-major, rank-major order must reconstruct the sealed frame exactly.
6. Publish a sampler sidecar containing schema version, manifest digest, seed,
   epoch, logical batch size `8`, schedule digest, ordered occurrence and raw
   indices, padding map, multiplicity histogram, completed-frame cursor, runtime
   world size, and gradient accumulation.

The schedule construction is byte-reproducible, not implementation-defined:

- Canonical records are UTF-8 JSON with sorted keys, compact separators, integer
  values, and a terminal newline. The schedule digest is SHA256 over the
  concatenated canonical occurrence records in output order.
- Derive each random seed as the unsigned little-endian integer from the first
  eight bytes of SHA256 over the NUL-separated UTF-8 fields `schema-v2`, dataset
  digest, decimal base seed, decimal epoch, bucket ID, and operation label. Mask
  it to 63 bits and use it only with a CPU `torch.Generator` plus
  `torch.randperm` under the pinned PyTorch version.
- The bucket shuffle label is `bucket-shuffle`. After shuffling, divide the
  bucket into consecutive windows of 64 entries and sort each window by
  descending recurrence work, then ascending raw index as the tie-breaker.
- The padding start is `seed("padding-offset") % bucket_size`; append cyclic
  entries from that position until the bucket size is divisible by eight.
  Occurrence ID is the tuple `(epoch, bucket_id, post-shuffle_slot,
  padding_ordinal)`, where non-padding entries use padding ordinal zero and each
  replay uses its one-based ordinal.
- Chunk in current order into eight-slot frames. Shuffle the completed frame
  indices with label `frame-shuffle`; never shuffle slots inside a completed
  frame. Empty buckets emit no frames and are never used in modulo arithmetic.

The logical schedule makes sample membership and multiplicity independent of
runtime world size while keeping rank-local micro-batch one and effective batch
eight. Its padding is included in reported processed throughput; non-padding
throughput is reported separately.

There must be exactly one sharding layer. Flatten the sealed frames and expose a
global batch sampler that yields one singleton list `[occurrence_index]` per
slot. For world size `W` and global rank `r`, Accelerate's `BatchSamplerShard`
must use `split_batches=False` and `even_batches=False`, so that the rank-local
iterator receives flat slots `r, r + W, r + 2W, ...`, each as a micro-batch of
one. Set `dispatch_batches=False`; do not pre-shard the sampler or let Trainer,
the DataLoader, or another framework pad, split, or dispatch it again. Accumulate
exactly `8 / W` local singleton batches before each optimizer step and assert
the gathered occurrence trace reconstructs every sealed frame.

Sampler v2 is fresh-run-only. A sampler-v1 checkpoint resumes only with v1 and
its original world size. No checkpoint silently migrates between schemas.

### 7.3 Checkpoints, results, and logs

All model-bearing output remains under each repository's canonical
`./checkpoints/` symlink. Rank 0 on `.41` is the authoritative writer. The run
uses `./checkpoints/zero-stage-compatibility/$RUN_ID/`,
`./results/zero-stage-compatibility/$RUN_ID/`, and
`./logs/zero-stage-compatibility/$RUN_ID/`.

Plain DDP saves only from global process zero. Rank 1 must not publish a competing
checkpoint. Before resume, replicate the complete authoritative checkpoint into
the identical `./checkpoints/zero-stage-compatibility/$RUN_ID/checkpoint-$STEP`
relative path on `.40` and
verify a deterministic file inventory plus SHA256 digests. Do not resume from a
partial adapter-only directory that lacks the expected optimizer, scheduler,
RNG, processor, recurrence, and sampler sidecars.

At an optimizer boundary, rank 0 first writes the common checkpoint payload to a
fresh numbered directory without a completion marker. Every rank serializes its
Python, NumPy, CPU Torch, and local CUDA RNG state to bytes and participates in
one `torch.distributed.gather_object(..., dst=0)`. Rank 0 writes the gathered
bytes as `rng_state_<global-rank>.pth`, including its local rank 1 and both
remote ranks, then writes a SHA256 inventory manifest. It atomically creates the
completion marker last and broadcasts save success before the final barrier.
Resume and replication reject any directory without that marker or any of the
four RNG entries. Node-local Trainer RNG files are non-authoritative; the worker
copy is never authoritative.

Each process needs a rank-qualified log file or Accelerate rank-qualified
stdout/stderr capture. The current shared `logging.FileHandler(args.log_file)`
is not a sufficient multi-node logging contract because multiple local ranks
can append to the same file. Multi-node implementation must make this
rank-safe before production launch.

## 8. Fail-fast preflight

Preflight runs before rendezvous and must stop the launch on any mismatch:

1. Confirm host identity, `bond0.11` IPv4 address, both H100 devices, free
   rendezvous port, and `.40 → .41` TCP reachability.
2. Confirm identical driver, PyTorch, CUDA, Accelerate, Transformers,
   FlashAttention, PEFT, TRL, and DeepSpeed versions.
3. Confirm identical Git commit and runtime-file hashes.
4. Confirm `TRAIN.jsonl`, image inventory, Stage 1 base/adapter, processor, and
   manifest digests.
5. Confirm canonical checkpoint/result/log paths use one fresh run ID and do not
   overwrite an existing run.
6. Confirm DoRA parameter targets and frozen/trainable boundary.
7. Confirm effective batch `8`, world size `4`, micro-batch `1`, accumulation
   `2`, sampler schema v2, logical batch width `8`, and the sealed schedule
   digest.
8. Confirm `use_cache=False`, diagnostics interval `0`, BF16, FlashAttention 2,
   and the selected checkpointing value.
9. Confirm one global sampler, one Accelerate sharding layer, and no framework
   re-padding or batch dispatch from rank 0.

Do not use the first address from `hostname -I`; use the fixed
`172.16.11.41` rendezvous address and `bond0.11` interface.

## 9. Required launcher behavior

The current `run_stage2_training.sh` only launches one node: it has no machine
rank or machine-count arguments, and without a DeepSpeed config it invokes one
Python process. Therefore the canonical DDP command below is a target interface,
not a claim that the current wrapper is already multi-node ready.

The smallest acceptable launcher change must:

1. preserve the existing manifest, path, resume, tuning, and argument
   validation;
2. add explicit distributed mode `ddp`, total process count `4`, machine count
   `2`, machine rank, master IP, master port, and static rendezvous;
3. invoke Accelerate even when no DeepSpeed config is selected;
4. require the pre-created, byte-identical manifest on both nodes;
5. select and shard the same sealed sampler-v2 schedule on both world sizes;
6. make logs rank-safe while keeping only global rank 0 as the artifact writer;
7. print a dry-run command without loading models or data;
8. reject DeepSpeed, FSDP, KV-cache, or topology changes not declared by this
   specification.

The target wrapper commands are identical except for machine rank. The operator
first exports the common Stage 1, dataset, output, DoRA, manifest, and training
values required by Sections 5 and 7.

```bash
# korea-21 / 172.16.11.41
STAGE2_DISTRIBUTED_MODE=ddp \
STAGE2_NUM_MACHINES=2 STAGE2_NUM_PROCESSES=4 \
STAGE2_MACHINE_RANK=0 STAGE2_MAIN_PROCESS_IP=172.16.11.41 \
STAGE2_MAIN_PROCESS_PORT=29520 \
./run_stage2_training.sh
```

```bash
# korea-20 / 172.16.11.40
STAGE2_DISTRIBUTED_MODE=ddp \
STAGE2_NUM_MACHINES=2 STAGE2_NUM_PROCESSES=4 \
STAGE2_MACHINE_RANK=1 STAGE2_MAIN_PROCESS_IP=172.16.11.41 \
STAGE2_MAIN_PROCESS_PORT=29520 \
./run_stage2_training.sh
```

The wrapper must translate those values to Accelerate's `--multi_gpu`, static
rendezvous, machine-rank, machine-count, total-process-count, and main-process
arguments. The common training arguments must explicitly encode Section 5 and
must omit `--deepspeed_config` for the DDP candidate.

## 10. Validation sequence

Run only the following staged checks; do not launch the full training job before
they pass.

### 10.1 Static and unit verification

- Run the complete Stage 2 test suite and artifact-path tests in the repository
  venv.
- Add launcher tests for both machine ranks, total/local process counts,
  interface environment, rank-safe logs, and the no-DeepSpeed DDP branch.
- Add sampler-v2 tests proving identical ordered sample IDs and multiplicities
  for two-GPU/accumulation-4 and four-GPU/accumulation-2 sharding, including
  final-tail padding and resume cursor behavior.
- Add a double-sharding regression that records every occurrence ID observed by
  every rank and reconstructs the sealed microstep-major, rank-major frame.
- Add preflight tests that reject code, data, manifest, checkpoint, world-size,
  batch, and recurrence mismatches before rendezvous.

### 10.2 Four-rank communication smoke

Run one small NCCL all-reduce per GPU. Verify global ranks `0..3`, local ranks
`0..1` on each node, expected sums, and collective completion. Use
`NCCL_DEBUG=INFO` only if this smoke fails.

### 10.3 Deterministic method witness

With dropout disabled on the tiny model, compare one single-process reference
against four-rank DDP for the same effective batch. Check loss, every latent,
supervised logits, trainable gradients, and one optimizer step using the existing
Stage 2 tolerances:

- FP32 eager: loss/latent/logit/gradient `atol=1e-6`, `rtol=1e-5`;
- BF16 FlashAttention 2: loss/latent/logit `atol=2e-2`, `rtol=2e-2`;
- nonzero gradients: cosine similarity at least `0.999` and relative L2 error at
  most `0.02`;
- zero-reference gradients: `atol=2e-2`, `rtol=0`.

### 10.4 Production-shaped DoRA smoke

Run one optimizer step with DoRA dropout `0.05`, the real manifest, and p50 then
p95 representatives. Require:

- finite loss, latents, supervised logits, gradients, and optimizer state;
- DoRA parameters change after the step;
- frozen visual and base parameters remain unchanged;
- every declared trainable parameter receives a gradient on the union of the
  representative frames; set DDP unused-parameter search to false only after
  this coverage witness passes;
- all ranks report the same optimizer step and sampler identity;
- the eight logical-batch sample IDs and padding flags match the sealed schedule;
- gradient synchronization is suppressed on non-boundary accumulation
  microsteps and occurs at the optimizer boundary;
- rank 0 writes a loadable adapter checkpoint and complete sidecars;
- a four-rank resume advances exactly one additional optimizer step.

Dropout makes rank-count numerical identity inappropriate for this production
smoke. It verifies the same mechanism and invariants, not bitwise equality with
a two-rank run. Exact sample multiplicity does not imply identical dropout masks
or floating-point reduction order across world sizes; topology-independent RNG
is not required or claimed.

## 11. Performance experiment and adoption gate

The load-bearing uncertainty is whether two additional GPUs overcome 10 Gbps
gradient synchronization and input overhead. Test exactly two configurations:

| Control | World size | Accumulation | Effective batch |
|---|---:|---:|---:|
| Single node DDP | 2 | 4 | 8 |
| Multi-node DDP | 4 | 2 | 8 |

Use the same commit, manifest, precision, attention, checkpointing selection,
DoRA settings, seed, optimizer settings, and measured optimizer-step count.
Before either timing run, seal one benchmark cursor in the sampler-v2 schedule.
Run the same three preceding logical frames as warm-up and the same next ten
consecutive logical frames as measured steps in both controls. Do not select or
replace measured frames after seeing timing results. Synchronize device work at
each timing boundary and use the maximum elapsed time reported by any rank for
each effective batch.
Record:

- median and p95 optimizer-step milliseconds;
- retained examples and non-padding examples per second;
- tokens and latent blocks processed per second;
- peak allocated VRAM per rank;
- data/collation, forward/loss, backward, gradient synchronization, and optimizer
  time when measurable without changing execution;
- network bytes and link utilization when available;
- four-GPU speedup over two GPUs;
- scaling efficiency: `throughput_4gpu / (2 × throughput_2gpu)`.

Before that full comparison, run the cheapest end-to-end first nail on one
sealed eight-occurrence p95-work frame: one warm-up update and two timed updates
for each control, with saving and evaluation disabled. Compare maximum-rank
seconds per effective batch. Stop the multi-node candidate immediately if four
GPUs are not faster. A synthetic all-reduce alone is insufficient because it
cannot expose recurrence stragglers or real gradient scheduling.

Adopt multi-node DDP as canonical only when:

1. median non-padding examples per second improves by at least 15%;
2. the p50, p95, and maximum-work capacity/smoke probes complete without OOM,
  hang, non-finite values, sampler mismatch, or sample-multiplicity divergence;
3. every correctness, save, and resume witness passes;
4. no measurement changes effective batch or Stage 2 semantics.

If the gate misses, report the result as a valid negative finding and keep the
two-GPU control as the most efficient supported run. Do not lower the 15%
threshold after observing results without a new experiment version.

## 12. Failure and resume behavior

- Use static rendezvous and zero automatic restarts. A rank failure terminates
  the whole job.
- Preserve logs from the failed run. Identify the root cause, make the smallest
  targeted fix, add one causal regression test, and retry with a fresh run ID.
- Do not switch DDP to ZeRO, change checkpointing, reduce sequence length, change
  batch geometry, or alter the dataset during a run.
- Signal only the recorded launcher process group. Never use broad `pkill` or
  pattern-based termination that could affect another job.
- Resume only with the same commit, four-rank topology, sampler schema and
  schedule digest, manifest digest, DoRA configuration, checkpointing setting,
  and complete replicated checkpoint digest.
- A checkpoint created with world size 2 is not a world-size-4 optimizer resume
  source. It may be an explicitly validated model initialization source for a
  fresh run, not a continuation.

## 13. Operator guidance

1. Integrate the Stage 2 feature commit, then place that exact commit on both
   nodes.
2. Synchronize data, Stage 1 artifacts, processor files, and the rank-0-created
   manifest; verify hashes.
3. Run preflight on both nodes.
4. Select DataLoader workers and checkpointing with the bounded probes.
5. Run the four-rank communication smoke, deterministic witness, one-step DoRA
   smoke, save, and resume checks.
6. Run the two-configuration performance comparison.
7. Publish the evidence under one run ID beneath `results/` and `logs/`.
8. Launch full multi-node Stage 2 training only if the adoption gate passes.

## 14. Required result record

The immutable result JSON for the comparison must include:

- run and candidate IDs;
- full Git commit and runtime-file digests;
- hostnames, IPs, GPU names/counts, driver and package versions;
- interface and rendezvous settings;
- dataset, image inventory, processor, Stage 1 artifact, and manifest digests;
- tuning, recurrence, precision, attention, checkpointing, worker count, sampler,
  and batch geometry;
- logical-batch schedule digest, ordered sample IDs, padding IDs, and per-sample
  multiplicities;
- warm-up and measured counts;
- the predeclared benchmark cursor and exact warm-up/measured frame IDs;
- raw per-step samples and aggregate metrics from Section 11;
- capacity, correctness, save, and resume outcomes;
- adoption decision and exact reason.

The record must distinguish directly observed values from configuration intent.
Test names, comments, or declared flags alone are not runtime evidence.

## 15. Explicit non-goals

- changing the Stage 2 algorithm or objective;
- enabling any KV-cache backend;
- changing DoRA to LoRA or full tuning;
- changing effective global batch size from 8;
- adding generalized retry, recovery, autoscaling, archival, or scheduler
  infrastructure;
- benchmarking a broad DDP/ZeRO/FSDP matrix;
- treating historical B200 evidence as H100 multi-node evidence;
- claiming multi-node efficiency before the Section 11 gate passes.

## 16. Authoritative references

- [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)
- [PyTorch Multi-node Training](https://docs.pytorch.org/tutorials/intermediate/ddp_series_multinode)
- [Hugging Face Accelerate CLI](https://huggingface.co/docs/accelerate/package_reference/cli)
- [Hugging Face TRL Distributed Training](https://huggingface.co/docs/trl/distributing_training)
- [NVIDIA NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NVIDIA NCCL Networking Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html)
