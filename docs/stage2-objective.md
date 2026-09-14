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
accumulation plugins, and unsupported checkpoint functions. FP16 is outside the
declared objective support boundary even though a lower-level parser may accept
it. Feature-off and enabled zero-weight CE do not acquire restrictions required
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
the objective branches/weights required by the mode.

After inspecting the dry-run, do not merely append `--launch`. First run the
live two-node preflight described in [multinode-training.md](multinode-training.md),
confirm identical committed code and complete readable inputs on both hosts,
verify all four GPUs/resources and each free port, and confirm every destination
is fresh. A separately authorized execution uses the same command plus
`--launch`; `--launch` and `--dry-run` are mutually exclusive.

## Evidence and limits

Evidence must be read at its actual scope:

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
