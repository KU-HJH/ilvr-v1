# Stage 2 native DDP coordinator — design proposal v1

Status: frozen proposal v1 for design-commitment critique; NOT implementation-authorizing approval.

The user approved the direction and preliminary architecture with “DDP 추가하자”.
That approval covers extending the existing Stage 2 coordinator, not deploying
over the canonical dirty checkout or launching training. The final written
commitment requires explicit approval before an implementation plan or code.

## 1. Problem and scope

The approved three-term Stage 2 objective currently rejects DeepSpeed because
its auxiliary execution is unvalidated there. The existing two-node coordinator
only launches DeepSpeed. Add an explicit native DDP route through that SAME
coordinator so CE-only, CE+entropy, and CE+entropy+Vision can use the same
Stage 1 source/data/seed and native Trainer semantics.

This is an execution-path extension, not a new objective or a claim that ZeRO
cannot support the objective. Keep the objective design in
`2026-09-14-stage2-entropy-vision-design.md` authoritative. Do not reuse V-off,
change F/M/reference behavior, add Stage 1 DDP, enable auxiliary ZeRO, redesign
the historical lifecycle/capacity matrix, or build a generic backend framework.

Alternatives considered: a separate DDP coordinator duplicates existing
preflight/ownership/transfer contracts; a generic backend refactor changes more
interfaces than this task requires. Extend the existing coordinator instead.

## 2. Evidence and boundaries

Inspected at objective branch `abf7462` in the existing isolated worktree:

- `src/multinode.py`: `LaunchPlan`, CLI, node-info, checkpoint identity and
  training completion all assume ZeRO 1/2/3 plus a configuration digest.
- `run_stage2_training.sh`: its multi-node branch unconditionally selects
  `--use_deepspeed`. Installed Accelerate's `multi_gpu_launcher` calls
  `torch.distributed.run`; explicit `--multi_gpu` selects that branch.
- `src/multinode_artifacts.py`: checkpoint observation expects `latest`,
  `global_stepN` and DeepSpeed model/optimizer shards. Existing transport collects
  local-owned files onto remote canonical storage and redistributes before resume.
- Fork Trainer: global rank 0 saves model, optimizer, scheduler and Trainer
  state with default `save_on_each_node=False`; every rank saves its own RNG.
  Stage2 runtime/sampler/objective/PEFT sidecars are also rank-0 publications.
- `deepspeed-runtime.json` already represents disabled DeepSpeed with null
  stage/config/digest and records world size. Reuse it without renaming/migration.
- Stage2 producer evidence presently requires an integer ZeRO stage. Model
  artifact classification already runs on rank 0 and broadcasts its result.

These are traced source paths, not four-rank filesystem runtime evidence.
Objective CPU Gloo tests observed two-process Trainer gradients/updates, not
this coordinator's native checkpoint transfer. No native DDP GPU run exists.

Canonical main has user edits overlapping launchers/coordinator/docs. Work stays
in `.worktrees/stage2-entropy-vision`; preserve those edits and resolve integration
with the user later. No automatic checkout replacement, sync, merge, or push.

## 3. Launch interface and compatibility

Stage2 `train-stage2` and standalone preflight accept
`--backend {deepspeed,ddp}`, default `deepspeed`.

| Selection | ZeRO/config fields | Worker launch |
| --- | --- | --- |
| Existing/default DeepSpeed | Existing required ZeRO 1/2/3 and selected config/digest | Existing flags unchanged |
| Explicit Stage2 DDP | `zero_stage`, config path and digest are null | Explicit `accelerate launch --multi_gpu` |

DDP rejects an explicit `--zero-stage`, a nonempty `DEEPSPEED_CONFIG`, and
conflicting enabled Accelerate DeepSpeed/FSDP/plugin environment settings.
The worker repeats the relevant checks on its own node: coordinator-local
validation alone does not control inherited remote environments. No `zero_stage=0`,
silent fallback, or backend inference from an empty path. Stage1 rejects DDP.

Add a defaulted backend field to LaunchPlan, with an exact two-way invariant.
Pass a coordinator-owned `ILVR_STAGE2_BACKEND` to workers; do not let forwarded
training arguments override it. Preserve the current Stage1 and ZeRO argv,
selected configurations and worker behavior. DDP requires complete multi-node
topology; it cannot enter the worker's legacy single-node fallback.

Native launch retains 2 machines, 4 processes, remote machine rank0/local rank1,
master `10.1.102.41`, interface `bond1.102`, same explicit port and repository
venv. BF16 is explicit for the comparison examples. No new optimizer, gradient
accumulation implementation, or DDP wrapping is added to Trainer.

## 4. Preflight and process ownership

Reuse existing node-info checks, same committed code, compatible installed
environments, two GPUs/node, input readability, storage and fresh run/attempt
paths. Skip only DeepSpeed configuration selection/digest validation for DDP;
do not weaken shared environment or ownership checks. Keeping the existing
installed DeepSpeed version check does not mean DDP executes DeepSpeed.

Include configured reference source and its local original DoRA base among
required inputs when entropy is enabled; validate local-only sources through
existing reference metadata resolution. Reference is not a student resume
checkpoint and is not copied into each checkpoint. The same readable original
source/base paths must exist on both nodes. Resume still validates their saved
reference identity in the actual Trainer before optimizer work.

Reuse owned remote-rank0 readiness, local-rank1 start, tmux supervisors,
timeouts, exact-owned cleanup, and command digests. Backend selection is bound
into commands/digests and checkpoint identity. No process-name signaling,
new retries, ownership subsystem or provisioning automation.

`--dry-run` prints the actual backend, topology, environment/argv and destinations
without starting processes or creating model artifacts. It is not a successful
remote preflight or training witness.

## 5. Native checkpoint identity and storage

Keep legacy artifact manifest/identity V1 emitters and accepted field sets
unchanged for Stage1/Stage2 ZeRO. Add one strict DDP-only V2 variant to the
existing manifest reader/writer, not a generic registry or new transport.

V2 keeps the existing run/method/world/rank/step/commit/state-required fields and
adds `execution_backend="ddp"` and `objective_enabled: bool` to runtime identity.
It requires `stage="stage2"`, `zero_stage=null`,
`deepspeed_config_digest=null`, `model_state_layout="trainer"`.
Only this exact combination is valid; reject mixed V1/V2 identities, invalid
types and backend conversion on resume. Old V1 files remain readable and old
commands continue emitting V1. Old code need not read new V2 files; both nodes
must run the same new committed code before a DDP launch.

Use the existing disabled DeepSpeed sidecar as the positive observed profile:
`enabled=false`, stage/path/digest null, world_size=4. Verify checkpoint-N
against `trainer_state.json.global_step`; native checkpoints do not require
`latest` or a DeepSpeed `global_stepN` directory and must reject those markers.

Expected ORIGINAL node-local files before collection:

| Remote node (global ranks 0,1) | Local node (global ranks 2,3) |
| --- | --- |
| Rank0 model/adapter and required tokenizer/config files | `rng_state_2.pth`, `rng_state_3.pth` |
| `optimizer.pt`, `scheduler.pt`, `trainer_state.json`, `training_args.bin` | No assumed model/optimizer/Trainer-state copy |
| Runtime + sampler sidecars; PEFT metadata/sparse rows when required by loader | No assumed common sidecars |
| `stage2_objective.json` iff objective enabled | |
| `rng_state_0.pth`, `rng_state_1.pth` | |

Require the actual method-specific loader's full or PEFT artifact contract;
support existing Full/LoRA/DoRA save layouts without copying model files into
results. Reuse exact path/symlink/duplicate validation. Native optimizer and
scheduler must be rank0-owned top-level files, not four sharded copies. Require
exactly all four rank RNG files in the combined/canonical checkpoint.

The original local RNG-only manifest cannot derive Trainer identity from absent
common files. Handle that case explicitly: enumerate and validate assigned RNG
ownership against the trusted requested identity, then bind it to the independently
observed remote common identity during collection. Do not pretend local sidecars
were observed or fabricate them. Canonical/resume manifests require full common
identity observation and all four RNG files, regardless of filesystem location.

## 6. Collection and resume

Retain `collect_checkpoint_artifacts` and `redistribute_resume_checkpoint`
control flow. Collect only after both owned node exits are zero. Merge remote
rank0 common state/RNG0,1 and local RNG2,3; push only the local-owned entries;
re-observe the complete remote canonical checkpoint. Require the objective
sidecar when identity says enabled, including enabled CE-only.

Before a resumed attempt, distribute the complete canonical checkpoint back
to the local node so every rank can load model/optimizer/scheduler and its RNG.
Keep missing-only/no-overwrite transfers, rank ownership and immutable identity
checks. A destination conflict is an error, not a repair request. Cross-backend
resume is unsupported and rejected before launch. Keep source checkpoint,
objective/reference/vocabulary configuration, optimizer/scheduler settings and
sampler contract unchanged; use a fresh attempt ID/port under the same run.

All model-bearing files remain under checkpoints. Results contain evidence only;
logs remain under logs. Do not put reference weights into Stage2 outputs.

## 7. Producer and completion evidence

Keep current ZeRO producer/witness V1 unchanged. Introduce native Stage2 producer
V2 with `execution_backend="ddp"`, null `zero_stage`, and
`objective_identity_sha256` (null iff objective is off). Retain actual four-rank
records for loss, gradient, optimizer/global step and errors. Confirm prepared
runtime is native distributed with world4 and DeepSpeed disabled; do not label
the requested CLI backend as observed evidence.

Only rank0 observes model/common checkpoint files and hashes the actual objective
sidecar; broadcast the validated artifact result/digest. Every rank contributes
its own actual numerical/step record. Rank0 observation errors must be propagated
collectively before peers wait on a success broadcast; reuse existing small
failure-reduction/publication patterns. Do not require absent common files on
the local RNG-only node before redistribution.

Coordinator compares producer backend/run/method/step against the plan, checks
the objective digest and enabled flag against the collected canonical sidecar,
and compares its config with the requested resolved objective config. Final
training witness has a DDP-specific version/identity and references validated
producer evidence. Missing ranks, nonfinite values, missing/different sidecar,
profile mismatch or an incomplete checkpoint cannot become success.

Do not extend historical ZeRO matrix/capacity result formats or reinterpret
their fields as DDP. New DDP training witnesses are not cells in that matrix.
Inference/evaluation migration is out of scope; resume includes the objective
sidecar, while existing inference file filtering continues excluding it.

## 8. Files and isolation

Expected production edits: `src/multinode.py`, `src/multinode_runtime.py`,
`src/multinode_artifacts.py`, `run_stage2_training.sh`, `src/stage2/evidence.py`,
and the Stage2 evidence call in `src/main.py`. Reuse `src/deepspeed_runtime.py`
disabled profile without schema changes. No Stage1 model/trainer or objective
math/attention/rollout edits are part of this extension.

Focused tests belong in `tests/multinode/` and Stage2 launcher/evidence tests.
Add native-DDP operator instructions to `docs/multinode-training.md` without
rewriting historical examples. The original objective plan's Task5 owns the
three comparison examples and objective documentation; connect it to this
backend after implementation/review, avoiding duplicate wrappers.

## 9. Validation and first-nail candidate

Write causal failing tests before production changes, then one covering set:

1. Actual shell dry-run and coordinator command construction: default ZeRO
   unchanged; explicit DDP has --multi_gpu, fixed topology and no DS flags;
   invalid backend/ZeRO/env/single-node combinations fail before launch.
2. Real tiny native Trainer save to two logical node-local roots in a CPU
   Gloo subprocess test; observe original rank ownership rather than manually
   assuming it. Exercise original manifests, collection/canonical re-observation,
   redistribution and native resume through the production functions. Full and
   PEFT layouts need coverage. CPU test transport may map remote operations to
   the second local test directory; it must not replace checkpoint logic with
   preconstructed passing manifests. Model fixtures live under checkpoints.
3. Resume reaches the next optimizer step with state/RNG and frozen reference
   identity preserved. Negative tests remove optimizer/RNG/objective files,
   change backend/config/reference, or add DS markers. No changed source data,
   disabled assertions or relaxed tolerances to make the tests pass.
4. Exercise native producer and final witness with observed four-rank records,
   rank0-only common files and collectively propagated observation failures.
   Legacy V1 fixtures still round-trip through their unchanged contracts.
5. Retain the existing objective Gloo accumulation/unequal-M-count/gradient
   oracle; this extension must not replace its Trainer arithmetic.

Cheapest decisive first-nail candidate: the tiny CPU native save -> actual
collection -> redistribution -> resume witness above, with physically separate
logical node roots. It can expose the central rank0-only-common-state assumption
before target-model training. It is not yet run and requires implementation-like
test work after approval. No fabricated four-node or GPU claim follows from it.

GPU acceptance is separate: existing coordinator only, two real nodes/four
GPUs, first a short save/resume run with all rank witnesses, then target-model
CE/entropy/combined memory/finite-loss/latent-gradient checks at supported
eager/SDPA/BF16/non-reentrant GC settings. No launch is authorized by this
design approval; obtain a separate run decision and pass operator preflight.
Do not add a new GPU lifecycle orchestrator merely to run this check; use
train-stage2 initial/resume attempts. Exact smoke budgets are explicit test
protocols, not new reproduction defaults.

## 10. Baseline examples and unvalidated premises

Three examples keep the reproduced Stage1 checkpoint-431, retained data IDs,
seed42, four ranks, microbatch1, accumulation2/global8, LR5e-5, warmup10 and
DoRA r32/alpha64/dropout0.05. Use distinct run IDs, change only objective weights
and their necessary skipped branches. Derive actual update budget from retained
samples and Trainer remainder behavior; 431 is not a Stage2 constant. Default
examples are dry-run until an explicit launch request and preflight.

Native DDP retains unsharded student/optimizer state and an independent frozen
reference per rank. Target-model memory fit, NCCL behavior, PEFT save timing
across real nodes and transfer time are unvalidated. DoRA can have a smaller
optimizer than full tuning; that is not proof the combined objective fits.
Do not alter the objective, batch baseline or sequence cap silently after OOM.

Last observed local GPUs were occupied by foreign compute work; no process was
signaled. Original Stage1 checkpoint was complete remotely but incomplete
locally. Canonical code is not deployed. Those are launch blockers to resolve,
not reasons to claim the DDP design or CPU contract tests failed.

Known objective pre-remedy broad regression had three independently reproduced
baseline Python-version tests plus two subsequently focused-fixed failures.
Current-head full regression is still outstanding. Report exact fresh test
evidence; do not claim a green full suite, GPU support, successful training or
task-performance improvement from design/code review or from increasing D/G.

## 11. Approval gate

This proposal will be critiqued unchanged under hate, conditional macrothink
and prism, then synthesized into an exact design commitment. Material changes
require a new frozen proposal and corresponding critique passes. The final
commitment must state first-nail status, dissent, unresolved premises and the
explicit risk acceptance requested. Only human approval of that commitment
routes to writing-plans and implementation.

## 12. Design commitment v1 — awaiting final human approval

Sections 1–11 above remain the frozen proposal that all critique passes used.
This section is the Superpowers synthesis and the implementation-authorizing
commitment offered for review. It adopts the complete proposal without changes
to scope, interfaces, checkpoint semantics or objective behavior.

**Agreed problem/scope.** Add Stage2-only native DDP to the existing fixed
two-node/four-rank coordinator to enable the approved three comparison examples.
Keep Stage1, legacy ZeRO, objective math/F/M/reference and training baseline
unchanged. No new coordinator, generic backend framework, historical matrix
expansion, evaluation migration or deployment automation.

**Final design.** Adopt sections 3–8: explicit `--backend ddp`, null DS-only
fields, worker `--multi_gpu`, inherited-environment conflict checks on both
nodes, existing preflight/owned process lifecycle, strict DDP-only V2 checkpoint
and producer identities while legacy V1 emission stays unchanged, rank0 common
state plus per-rank RNG, explicit RNG-only original local manifests, existing
collection/redistribution transport, exact same-backend resume, and objective
sidecar presence/config/digest validation. Actual rank records and prepared
runtime establish evidence; requested configuration alone does not.

**Changes from proposal.** No behavioral or interface change. Clarify the
validation order and terminology: section9's cheap CPU save/transfer/resume
candidate is the first implementation contract test, NOT evidence for the
load-bearing GPU capacity premise. Retain it and the separate real-node
save/resume witness. Only the GPU experiment below can falsify capacity for the
selected real workload. Unsupported historical matrix/evaluation commands stay
outside DDP scope even if they share a parser/helper with train-stage2.

**hate result.** `{root: the requested real comparisons remain infeasible if
the unchanged combined objective cannot fit a GPU with unsharded student state,
reference and retained latent graphs; first_nail: one complete combined-objective
optimizer update with accumulation2 through the real two-node/four-rank
coordinator on representative high-memory retained input, recording actual rank
participation, update completion and per-rank peak memory.}` This is cheaper
than three training runs. It is NOT RUN. A passing selected example is not a
guarantee of memory fit across the entire retained dataset.

**Not-run rationale and uncertainty.** The DDP implementation does not yet
exist, canonical code is undeployed, the original source is incomplete locally,
and foreign GPU load was last observed. No separate GPU experiment authorization
has been given. Neither a proposed probe, CPU Gloo tests, static source tracing
nor code-review agreement establishes target-model capacity or NCCL execution.

**Material divergence.** Both independent macrothink reads challenged the
premise that a DDP launcher alone delivers runnable full comparisons: per-rank
memory remains untested. Their compatible reframing was to deliver a narrow
execution path and test a real complete optimizer update before spending on
three runs. Agreement is reassurance only, not proof or a majority decision.

**prism disagreement.** Checkpoint correctness and legacy/data/process safety
accept the explicit design contracts at design level; resource feasibility
remains unclear. This is not averaged away. The resolving human decision is
whether to authorize the bounded implementation with GPU feasibility unvalidated,
while withholding training-readiness claims until runtime evidence exists.

**Unvalidated premises and requested risk acceptance.** Approval accepts the
possibility of implementing a correct DDP route that later fails target-model
capacity, along with still-unrun four-rank save/resume/PEFT timing and transfer
checks. It does not waive training topology, input/resource checks, fixed loss,
batch/sequence settings, reference identity, or any acceptance test. On a real
capacity failure, report the failing protocol and request a new execution
decision; do not silently change the objective or baseline.

**Execution after approval.** Use writing-plans, subagent-driven-development,
TDD, task review, and evidence-matched final verification. Run CPU tests locally;
do not launch GPU jobs, merge, push, or overwrite/synchronize canonical user
changes without the separate required user decision. Rejoin original objective
Task5 for the three explicit same-source/data/seed dry-run examples. Document
implemented paths, actually passed tests, baseline failures and every unrun
GPU/training check separately. No performance-improvement claim is authorized.

**Cold-read clarifications (no behavioral change).** A context-free shower
reader understood the scope/approval boundary but had to infer the enabled
flag and RNG-only identity source. Make those existing meanings explicit:

| Stage2 mode | `objective_enabled` | Objective sidecar/digest |
| --- | --- | --- |
| Feature off / legacy objective | false | Absent / null |
| New CE-only: feature flag true, both weights zero | true | Present / SHA-256 |
| New CE+entropy: feature flag true, dyn positive, vis zero | true | Present / SHA-256 |
| New combined: feature flag true, both weights positive | true | Present / SHA-256 |

`objective_enabled` is exactly the resolved `enable_stage2_objective` boolean,
not `config.active` and not inferred from reference presence. Enabled CE-only
still records its configuration; its reference/vocabulary identity fields are
absent as prescribed by the approved objective. DDP is also usable feature-off;
adding the backend does not implicitly enable the objective. A feature-off
request with a checkpoint bearing an enabled objective identity is rejected.

The trusted DDP runtime identity is built coordinator-side from validated
LaunchPlan fields plus the resolved Stage2 objective enable flag, never from
an untrusted local shard or environment read at collection time. Its exact
fields are `stage`, `tuning_method`, `zero_stage`,
`deepspeed_config_digest`, `world_size`, `rank_mapping`, `global_step`,
`model_state_layout`, `optimizer`, `scheduler`, `rng_ranks`, `stage_sidecar`,
`commit`, `run_id`, `execution_backend`, `objective_enabled`. Fixed values
come from sections3/5; run/method/step come from validated launch/resume arguments;
commit comes from the matching node preflight. The method-specific primary
sidecar remains `stage2_peft_metadata.json` for LoRA/DoRA or `stage2-sampler.json`
for full tuning. The objective sidecar is an additional conditional requirement.

Both original node manifests carry that same expected identity. Remote common
files must independently agree on all observable fields (runtime profile,
step, full/PEFT method and enabled objective). The local RNG-only exception
is permitted only for original machine-rank1 manifests with assigned ranks2/3,
correct run/checkpoint ancestry, and precisely their top-level RNG files.
Every carried identity field must equal the trusted identity and its peer's
identity exactly; any mismatch rejects collection before transfer. That equality
does not falsely claim the local node observed absent model metadata. After
redistribution, the local canonical view takes the full-observation path.

This is an architecture/acceptance contract, not a standalone implementation
plan. The approved objective spec and existing loader/ownership source are
required implementation references. Writing-plans will bind task-specific
function signatures and method file requirements from those inspected sources;
it may not weaken these contracts or invent a new loader/ownership API.

## Approval record

2026-09-14: user replied “ㅇㅋ” to the final commitment v1 approval question.
Section12 is approved for implementation with GPU feasibility explicitly
unvalidated. Frozen proposal and commitment text above are retained as reviewed.
This approval does not authorize deployment, GPU training, merge, or push.
