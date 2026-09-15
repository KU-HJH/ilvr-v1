# Stage 2 entropy transition and counterfactual Vision objective

This is the operator contract for the opt-in Stage 2 objective. It preserves
the existing self-generated-latent supervised CE and can add two blockwise
terms. It does not change Stage 1, the input trajectory, image order, marker
supervision, or the existing trainable-parameter policy.

## Fixed objective

For every block in the final valid set `K`, entropy is averaged inside each
selected window before a ratio or nonlinear loss is formed:

```text
r_pre[k]  = (H_theta_pre[k]    + eps) / (H_ref_pre[k]    + eps)
r_post[k] = (H_theta_post_F[k] + eps) / (H_ref_post_F[k] + eps)
D[k]      = r_pre[k] - r_post[k]
loss_dyn[k] = -sigmoid(D[k] - gamma_H)

S_F[k] = mean_j log p_F(y_kj)
S_M[k] = mean_j log p_M(y_kj)
G[k] = S_F[k] - stop_gradient(S_M[k])
loss_vis[k] = softplus(m_V - G[k])

loss_total = loss_ce_F
           + lambda_dyn * mean_k(loss_dyn[k])
           + lambda_vis * mean_k(loss_vis[k])
```

`F` is the normal student forward. `M_k` is an independent forward of the
current student in training mode and under `no_grad`, with only block `k`'s
latent-slot access ablated. The frozen reference is a separate Stage 1 model in
`eval`/`no_grad`; it generates its own latent trajectory and never shares
student parameters, optimizer ownership, or student latent states. `S_M` and
reference values are detached; CE, student entropy, `S_F`, and the normal
self-generated latent path retain gradients.

`S_F` and `S_M` use full-output-vocabulary, teacher-forced log probabilities.
Entropy instead renormalizes with natural-log `log_softmax` over a tokenizer-
resolved text vocabulary in FP32. The recorded exclusion set covers padding,
special/modality controls, image/video/latent placeholders, structural
tool/FIM tokens, and unmapped model output rows. Ordinary added text tokens
remain eligible. Token IDs are resolved from the loaded tokenizer/config; they
are not constants. Nonfinite values are errors, not clipped or replaced.

## Windows and the local intervention

For latent markers at positions `s=<|latent_start|>` and
`e=<|latent_end|>`, the default one-token windows are:

```text
index:       0   1       2          3   4       5        6   7
token:      p0  p1  latent_start   z0  z1  latent_end   y0  y1
logit used:     pre                                      post/y0
position:        1                                         5
```

The pre-window is the last `pre_window` text-state positions in the immediately
preceding assistant-text segment, including `s-1`. It is not filtered according
to whether its next label is text; marker CE stays supervised. The post-window
is the first `post_window` valid supervised text targets in the immediately
following assistant-text segment and uses logits at target position minus one.
Those same targets define `S_F` and `S_M`. Windows never cross a turn, latent
block, padding, or truncation boundary.

After collecting all candidate blocks, any logit position used by both a pre-
and post-window is removed from both sides simultaneously, without refill or
iteration. A block empty on either side is excluded. The resulting identical
positions are used for the student, reference, and Vision branches. Malformed,
nested, or zero-length marker spans fail with sample/block context.

`M_k` preserves the original IDs, image tensors, markers, slot count, sequence
length, teacher-forced text, padding mask for position calculation, and RoPE.
At every decoder layer/head and every relevant prefix/final forward, it adds
this restriction to the normal causal/padding attention mask:

```text
block attention(q, j) when q >= e_k and s_k < j < e_k
```

The restriction starts at the latent-end query so information cannot leak
through the marker. `M_k` starts from the original processed input, reruns later
latent generation, and does not reuse F latents, downstream states, or a suffix
cache. It does not delete slots, zero embeddings, change positions, or use a
mutable global ablation flag. F RNG is captured and replayed around each M while
the student remains in training mode; the outer CPU/initialized-CUDA RNG state
is restored.

## Modes and distributed reduction

All wrapper modes explicitly enable the new feature and native DDP:

| Wrapper mode | Weights | Work intentionally skipped |
| --- | --- | --- |
| `ce` | `lambda_dyn=0`, `lambda_vis=0` | reference, vocabulary/windows, denominator collective, and M |
| `entropy` | positive operator-supplied `lambda_dyn`, `lambda_vis=0` | M |
| `combined` | positive operator-supplied values for both weights | neither auxiliary branch |

Enabled zero-weight CE is distinct from feature-off legacy execution. It keeps
the normal F CE and finite graph-connected zero for the all-ignore-label edge
case, and records an enabled objective identity with absent reference/vocabulary
identity. No weight in this guide is presented as optimal.

For active auxiliaries, the valid block count `N` is collected across every
rank and microbatch in the actual optimizer accumulation window, including
zero-block ranks. With DDP world size `W` and Trainer loss divisor `A`, each
local differentiable block sum is scaled by `W*A/N` before backward. DDP's
`/W` and Trainer's `/A` recover the global block mean. The pinned Trainer
continues to divide every microbatch by the configured accumulation factor `A`,
including a final partial window; `N` uses the actual globally valid block count
for that window. Do not replace `A` with the partial-window microbatch count.
Existing CE reduction is unchanged. Logged means are separately count-weighted
global reductions; they are not the backward-scaled scalar.

## Supported execution boundary

The active auxiliary path is Stage 2 native DDP, two nodes/four ranks,
microbatch one, eager or SDPA, FP32/BF16, and the validated native
non-reentrant gradient-checkpoint function. The comparison wrapper selects
SDPA, BF16 through the Stage 2 TrainingArguments, gradient checkpointing on,
and the reproduced DoRA settings (`r=32`, `alpha=64`, `dropout=0.05`, decoder
targets, latent size 8).

Active auxiliaries reject DeepSpeed/ZeRO, FlashAttention 2, effective sliding-
window attention, nonzero native attention dropout, FSDP/compiled or custom
accumulation plugins, and unsupported checkpoint functions. Active auxiliaries
also reject FP16 requested through settings or inherited mixed precision, and
FP16 model parameters.
Feature-off and enabled zero-weight CE do not acquire restrictions required
only by auxiliary computation.

Native DDP checkpoints use Trainer-format model/adapter, optimizer, scheduler,
Trainer state, per-rank RNG, sampler, runtime, and conditional objective
sidecars under `checkpoints/<run-id>/checkpoint-N`. Full, LoRA, and DoRA layouts
are recognized by the native artifact path. Resume must preserve backend,
method, run, saved objective/reference/vocabulary identity, optimizer/scheduler,
sampler, and per-rank RNG state. This comparison wrapper creates fresh DoRA runs;
resume remains an operator action through the coordinator.

## Three reproducible comparisons

Run from `/root/workspace/ILVR`. The default Stage 1 source for fresh direct
Stage 2 DoRA comparisons is the reproduced checkpoint below, not
`comt_ckpt` or the older local checkpoint-215. Set `DATA` and `IMAGES` to the
same verified retained Stage 2 inputs under this repository's `data/` tree.
Set `STEPS` from retained sample count and the actual Trainer sampler/remainder
behavior; for example, 3,379 retained samples at global batch 8 require 423
updates for one full-coverage epoch. `431` is specific to the Stage 1
reproduction and is not a Stage 2 default.

Choose three currently free ports. `LAMBDA_DYN` and `LAMBDA_VIS` are required
experiment choices, not recommendations. The commands default to coordinator
dry-run and therefore print the actual remote/local commands without starting
ranks or creating run artifacts.

```bash
cd /root/workspace/ILVR
STAGE1=/root/workspace/ILVR/checkpoints/stage1-reproduce-mn-egb8-s431-lr5e5-20260914-055210/checkpoint-431
: "${DATA:?set the verified retained Stage 2 JSONL under /root/workspace/ILVR/data}"
: "${IMAGES:?set its verified image root under /root/workspace/ILVR/data}"
: "${STEPS:?derive and verify the Stage 2 update budget}"
: "${LAMBDA_DYN:?choose a positive entropy weight}"
: "${LAMBDA_VIS:?choose a positive Vision weight}"
: "${CE_PORT:?choose a free port in 1024..65535}"
: "${ENTROPY_PORT:?choose a different free port}"
: "${COMBINED_PORT:?choose a third free port}"

stamp="$(date -u +%Y%m%d-%H%M%S)-$$"
CE_RUN="stage2-objective-ce-$stamp"
ENTROPY_RUN="stage2-objective-entropy-$stamp"
COMBINED_RUN="stage2-objective-combined-$stamp"

./run_stage2_objective_examples.sh ce \
  --stage1 "$STAGE1" --data "$DATA" --image-root "$IMAGES" \
  --run-id "$CE_RUN" --max-steps "$STEPS" \
  --main-process-port "$CE_PORT"

./run_stage2_objective_examples.sh entropy \
  --stage1 "$STAGE1" --data "$DATA" --image-root "$IMAGES" \
  --run-id "$ENTROPY_RUN" --max-steps "$STEPS" \
  --main-process-port "$ENTROPY_PORT" --lambda-dyn "$LAMBDA_DYN"

./run_stage2_objective_examples.sh combined \
  --stage1 "$STAGE1" --data "$DATA" --image-root "$IMAGES" \
  --run-id "$COMBINED_RUN" --max-steps "$STEPS" \
  --main-process-port "$COMBINED_PORT" \
  --lambda-dyn "$LAMBDA_DYN" --lambda-vis "$LAMBDA_VIS"
```

Each command fixes four ranks, microbatch 1, accumulation 2/global batch 8,
learning rate `5e-5`, seed 42, warmup 10 optimizer updates, BF16, AdamW/linear
schedule, weight decay 0.01, max gradient norm 1.0, DoRA r32/alpha64/dropout0.05,
latent size 8, supervised boundary markers, SDPA, native non-reentrant gradient
checkpointing, and terminal `save_steps=max_steps`. It changes only run/port and
the objective branches/weights required by the mode. These three wrapper examples
explicitly forward `--stage2_m_execution packed_suffix`; each remains a dry-run
until `--launch` is supplied.

### Packed M suffix execution

`stage2_m_execution` accepts `full_replay` (the default) and `packed_suffix`.
The default preserves existing and older saved objective configurations: a saved
configuration with no mode field is interpreted as `full_replay`. A resumed run
with an explicit mode different from its saved objective configuration fails;
choose the desired mode before creating or resuming the run. The three wrapper
examples above are the supported packed opt-in path. Direct coordinator users
may pass `--stage2_m_execution packed_suffix`; pass `full_replay` to opt out.

Packed mode leaves normal Stage 2 latent recurrence and the F/reference call
schedules unchanged. When Vision has valid blocks, it captures the unchanged
final F cache and runs one graph-free decoder call for all M suffixes. For a
block with start `s` and END `e`, each packed query can see active root keys
before `e` except latent slots strictly between `s` and `e`, plus only its own
causal suffix keys. It cannot read clean F END/post keys or another branch's
suffix. The cache uses physical packed positions while the original three-axis
mRoPE positions are passed separately.

This CPU-supported path is limited to B=1, eager/SDPA, FP32/BF16, zero native
attention dropout, native non-reentrant checkpointing, native DDP, and one
unmerged supported Linear LoRA/DoRA adapter layout (ordinary adapter dropout is
replayed at its recorded F RNG state). It does not establish support for
ZeRO, FlashAttention, FP16, FSDP, compilation, CUDA dropout behavior, real-image
equivalence, two-host execution, or a GPU training run.

Excluding the unchanged latent recurrence, expected decoder prediction calls
are CE=1, CE+entropy=2, CE+Vision=2, and combined=3. In packed mode one M
decoder call is deliberately **not** a top-level model call:
`model_forwards_M=0`, `decoder_forwards_M=1`, `m_branches=K`, and
`m_query_rows=Q` for a nonempty packed microbatch. In full replay,
`m_query_rows` still reports the same conceptual selected contiguous suffix-row
total `Q`; it is not the total rows recomputed across separate legacy M
rollouts. These counters are runtime observations, not a memory, throughput, or
constant-work guarantee as `K` grows.

After inspecting the dry-run, do not merely append `--launch`. First run the
live two-node preflight described in [multinode-training.md](multinode-training.md),
confirm identical committed code and complete readable inputs on both hosts,
verify all four GPUs/resources and each free port, and confirm every destination
is fresh. A separately authorized execution uses the same command plus
`--launch`; `--launch` and `--dry-run` are mutually exclusive.

## Evidence and limits

Evidence must be read at its actual scope:

### Current CPU verification — 2026-09-15

User-requested repair `cc9ba9f` changes only
`tests/multinode/test_capacity.py`,
`tests/stage1/test_evaluation_diagnostics.py`, and
`tests/stage2/test_launcher.py`. Production source, training behavior and the
objective are unchanged from `165342c`; the prior failures are preserved below
as historical evidence, not erased or excluded.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q tests -m 'not gpu'
```

Exit 0: **2632 passed, 7 skipped, 3 deselected, 10 warnings in 421.86 seconds**.
Full output: `logs/stage2-ddp-dev/final-project-cpu-cc9ba9f.log`.
This is a passing project CPU suite, not GPU/real-host training verification.
The ten warnings are the existing PEFT missing-config fallback in Stage2
evaluation/inference tests; they were not suppressed. The branch-finishing
CPU gate passed; merge, push, deployment and training still require the user's
separate decision. Only documentation/evidence changed after this run.

Repair evidence: the exact original four tests first failed in 4.56 seconds
(`logs/stage2-ddp-dev/resumed-4red.log`). The amended three-file covering command
passed **112 tests in 14.55 seconds** (`resumed-covering.log`):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/python -m pytest -q \
  tests/multinode/test_capacity.py \
  tests/stage1/test_evaluation_diagnostics.py tests/stage2/test_launcher.py
```

The scoped review approved the repair and inspected actual regression failures
for five in-memory mutants: omitted/all-marker normalization, omitted
diagnostics/main restoration, and an incorrect inferred final step. Their
full traces are in `logs/stage2-ddp-dev/resumed-actual-regression-mutations.log`;
the executed command/script body is retained in `resumed-mutation-command.log`
in the same directory. No production mutation persisted. Earlier checks that
duplicated marker assertions are supplemental only, not the final mutation
witness. Test portability now comes from explicit marked/unmarked input and
real import-boundary observations; the capacity fixture uses a real validated
LaunchPlan without changing its topology/config validation.

### Historical and task-level evidence

Historical verification at production/test commit `092a95e` (before the
user-requested test repair `cc9ba9f`):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/transformers/src:$PWD/src" \
  ./venv/ilvr-venv/bin/pytest -q tests -m 'not gpu'
```

Exit 1: **2626 passed, 4 failed, 7 skipped, 3 deselected, 10 warnings in
429.00 seconds**. Full output is retained in
`logs/stage2-ddp-dev/final-project-cpu-092a95e.log`. No passing-full-suite or
ready-to-merge claim was made at that revision. The finishing gate stopped
before integration options; the worktree was preserved without deployment.

The four failures at that revision were:

1. `tests/multinode/test_capacity.py::test_stage1_coordinator_uses_training_evidence_step_without_max_steps`:
   an unchanged Stage 1 test replaces `build_launch_plan` with a
   `SimpleNamespace` lacking `backend`; the new witness keyword selection at
   `src/multinode.py:2129` accesses that field. This was a newly exposed
   integration-test incompatibility, **not** one of the three known
   platform baseline failures. The real builder returns `LaunchPlan` with a
   backend field, so this traceback alone does not demonstrate a production
   Stage 1 execution failure. It blocked the full regression pass then.
2. `tests/stage1/test_evaluation_diagnostics.py::test_diagnostic_entrypoint_leaves_an_extra_build_marker_invalid`.
3. `tests/stage1/test_evaluation_diagnostics.py::test_diagnostic_entrypoint_restores_platform_after_heavy_import_failure`.
4. `tests/stage2/test_launcher.py::test_production_entrypoint_imports_with_build_marker_then_restores_platform`.

Failures 2–4 reproduce the earlier canonical-baseline symptoms recorded below.
The final ten warnings are located in Stage 2 evaluation/inference tests and
report PEFT's missing-config fallback; this identifies their observed location,
not causal attribution to the new objective. Syntax checks passed for all three
Stage 2 wrappers/workers. An actual import probe resolved Transformers
`4.50.0.dev0` and Qwen to this worktree's `transformers/src`, not the installed
unmodified package. Only documentation/checklist evidence changed before the
subsequent user-requested test repair.

- Combined whole-branch review and the single scoped re-review are complete
  through `092a95e`. All seven findings were addressed: loadable reference
  symlink/indexed-shard hashing, active provenance presence and runtime binding,
  disabled-reference preflight, FP16 rejection, collator typing, asymmetric
  blockwise-nonlinearity regression, and contextual numerical errors.
  The amended-file covering run passed 169 tests in 74.05 seconds; complete
  output: `logs/stage2-ddp-dev/final-fix-covering.log`. Actual four-CPU producer
  records in `results/stage2-ddp-dev/native-provenance-rdgde5bm/` reject missing
  and different reference/vocabulary identities on all ranks before restored
  valid publication. This is not a GPU or full-project pass.
- Three final actual-coordinator dry-runs are captured in
  `logs/stage2-ddp-dev/final-dryrun-092a95e-{ce,entropy,combined}.json`.
  They use the same local reproduced checkpoint-431 path, `data/TRAIN.jsonl`,
  image root and seed 42, with weights `(0,0)`, `(0.4,0)`, `(0.4,0.6)`.
  The supplied 423-update budget and weights are command-construction inputs,
  not validated retained-data counts or recommended research hyperparameters.
  All three exited zero and created no experiment artifacts in either local
  repository root. No live port/input/resource check occurred, and their
  canonical worker commit is still the undeployed `31f745e`; do not launch
  these captured commands as evidence of deployment or readiness.

- The executable wrapper boundary in `tests/stage2/test_objective_examples.py`
  has 26 passing CPU shell tests for all three argv/environment variants,
  validation, explicit-launch selection, collision rejection, and dry-run
  no-artifact behavior. A separate actual-coordinator combined dry-run is
  retained verbatim in `logs/stage2-ddp-dev/task4-actual-dry-run.log`; it emitted
  the two owned commands and the resolved DDP/objective plan, but started no
  ranks and performed no live preflight. That task-time JSON targets the
  canonical `/root/workspace/ILVR` worker roots and records canonical commit
  `31f745e02a81619d6ee12183b8f2e9c699c80902`, where this feature delivery was
  not deployed. It therefore verifies command construction only, not matching
  execution code or readiness; both canonical nodes must first receive the same
  reviewed commit through a separately authorized integration step.
- Objective unit/integration evidence is owned by
  `tests/stage2/test_objective_math.py`, `test_objective_windows.py`,
  `test_objective_config.py`, `test_objective_ablation.py`,
  `test_objective_reference.py`, `test_objective_trainer.py`, and
  `test_objective_distributed.py`. The reviewed objective record reports 32
  focused passes plus three separate original-mask/eager/SDPA controls after
  the historical implementation was removed, observed RED, and reimplemented.
  That remedy does not make the original work uniformly test-first.
- The final native-DDP Task 3 covering command over
  `tests/multinode/test_native_ddp_evidence.py`,
  `test_native_ddp_checkpoint.py`, `test_coordinator.py`, and
  `tests/stage2/integration/test_distributed_contract.py` produced
  `182 passed, 4 skipped`; its complete output is in
  `logs/stage2-ddp-dev/task3-covering-final.log`. The executed native fixtures
  used four CPU Gloo processes and two logical filesystem roots. They exercised
  actual Trainer-format full/DoRA save, collection, redistribution/resume,
  objective identity, four-rank publication, and native completion witnesses.
  The four skips were one CUDA-only case and three opt-in tiny-train cases.
- The known baseline log `logs/stage2-objective-dev/platform-baseline.log`
  records three failures: two Stage 1 evaluation-diagnostic platform-marker
  tests and the Stage 2 launcher platform-marker test. Host Python `3.11.15`
  lacks the expected `+`, and extra-marker acceptance differs. These failures
  are not waived and must stay separate from supported-path passes.

No GPU/NCCL or real-host save/resume, SSH/rsync/container transfer, target-model
capacity, full retained-data training, all-three-model comparison, ZeRO active-
objective execution, or task evaluation has been run. CPU Gloo and dry-run do
not prove any of those. A dry-run is not preflight, rank participation, or
training readiness. Increasing `D` does not ensure both entropy directions;
increasing `G` can result from lowering `S_M`; neither value nor a lower loss
establishes semantic visual grounding, training stability, novelty, or task
performance improvement.

### Acceptance evidence map

| Criterion | Executed witness and scope |
| --- | --- |
| A: math and gradients | `test_objective_math.py`: literal values/signs, detached reference/M, finite small-entropy/BF16 cases; asymmetric scalar and broadcast mean-first mutations rejected. |
| B: alignment and masks | `test_objective_windows.py`, `test_data.py`, `test_objective_ablation.py`: causal offsets, vocabulary, marker/latent masks, padding, overlap and invalid/empty windows; local tokenizer/processor probe recorded in the objective plan. |
| C: intervention | `test_objective_ablation.py`: actual tiny Qwen F depends on selected latent; M post/downstream perturbation difference is zero, prefix/input/RoPE preserved, no F suffix reuse. |
| D: integration/regression | `test_objective_trainer.py`, `test_objective_distributed.py`, native checkpoint/evidence tests: actual Trainer gradients, checkpointing, fixed reference, full/DoRA CPU resume and global-count oracle; legacy controls are bounded by the final suite result. |
| E: execution | Local CPU tests and actual multi-process Gloo integration only. Three coordinator dry-runs construct commands. Real two-host GPU/NCCL, production sparse-row resume, capacity, full training and evaluation remain unrun. |

The resumed Stage1 capacity-test mutation demonstrates rejection of an
incorrect inferred final step by the real evidence decoder. Its final
directory-existence assertion alone does not prove post-validation checkpoint
selection, because that test creates `checkpoint-2` before calling `_main`.
This pre-existing evidence limit is distinct from the native-DDP actual
save/collection/resume fixtures described above.

### Implementation rulings

These choices did not change the approved research objective:

1. Zero-weight CE bypasses auxiliary metadata/vocabulary/count collectives;
   otherwise unused window parsing could reject valid CE inputs. If wrong,
   unused diagnostic metadata is unavailable; CE arithmetic is unchanged.
2. One fixed-run/checkpoint-step canonical objective reader supplies in-memory
   bytes to the existing witness path, because original node1 is RNG-only.
   If wrong, one narrow internal read API is unnecessary maintenance.
3. An explicit internal `resume_destination` mode recognizes a fully validated
   previously redistributed native checkpoint as well as original RNG2/3-only
   state; partial/mixed state still fails. If wrong, one internal mode is
   unnecessary maintenance, without weaker validation or new recovery.
4. Native `zero_stage` is required-but-nullable, not an omitted keyword, to
   preserve legacy call shape. If wrong, callers supply an unnecessary `None`;
   serialized/runtime behavior is unchanged.
5. Load-relevant local symlink/indexed-shard bytes are hashed rather than
   banning sources the loader already accepts. If wrong, linked artifacts
   remain an availability dependency instead of being rejected; no remote
   loading or generic traversal infrastructure was added.
6. The resumed repair corrects host-dependent marker tests and the partial
   LaunchPlan fixture without changing training code. Explicit inputs and real
   import-boundary observations protect one-marker normalization/restoration;
   a real builder supplies the complete plan. If wrong, downstream dependency
   invalid-version rejection is no longer asserted by that boundary test;
   production validation was not relaxed. Actual repaired tests reject all
   five normalization/restoration/inferred-step mutations.
