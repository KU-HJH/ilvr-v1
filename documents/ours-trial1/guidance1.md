# ours-trial1 Stage-2 Evidence–Window Continuation Guide

**Purpose:** self-contained handoff for a new session continuing the approved
ILVR Stage-2 evidence-window credit implementation.

**Recorded:** 2026-08-23

**Feature branch:** `ours-trial1`

**Verified implementation HEAD:** `e5c1012`

**Task-13 base:** `3a7294e2632b983a7750b269be6d007c22a6e640`

**Task-13 implementation commit:** `02503265067b524f52e9266894079cca0c6ff822`

**Task-13 witness fix commit:** `dddaa441d7c97494f9b8af54254917153b09c558`

**Task-13 review fix commit:** `9f97f6790f4614cca5dd45632f690e24452ce1b8`

**Task-13 distributed-publication fix commit:**
`33f6fd0834760d959b7f659bedbadc3183e244dc`

**Task-13 terminal-completion fix commit:**
`f0e85215c819d2d5e352b982d822a8c18aeceefe`

**Task-13 early-coordination fix commit:**
`08dc0b6116c162a4a2cd1dd64a62fcd6f8743b48`

**Current state:** Tasks 1–14 are implemented and final runtime verification is
complete. The full CPU suite passed (`442 passed`), the real 7B six-case matrix
completed, and two-rank online frozen-reference smokes completed under ZeRO-1
and ZeRO-2. The independent whole-branch review found no Critical, Important,
or Minor issues. Reuse the sealed witnesses listed in Section 15; do not rerun
them merely to produce a new manifest.

## 1. Canonical locations

Main repository:

```text
/root/workspace/ILVR
```

Feature worktree:

```text
/root/workspace/ILVR/.worktrees/ours-trial1
```

Canonical guidance file (outside the worktree Git checkout):

```text
/root/workspace/ILVR/documents/ours-trial1/guidance1.md
```

The worktree's `documents` entry is an intentionally untracked symlink to the
canonical directory. Never stage, replace, or clean it.

Required Python environment:

```text
/root/workspace/ILVR/venv/ilvr-venv
```

Use this environment for every Python, test, training, or evaluation command.
Do not use system Python or the legacy `./ilvr-venv` directory.

## 2. New-session protocol

1. Read the repository-root `AGENTS.md` and this file completely.
2. Change into the `ours-trial1` feature worktree.
3. Confirm branch, HEAD, and status without cleaning anything.
4. Read the approved spec, plan, Task-13 report, and Task-14 state.
5. Treat the untracked `documents` symlink as expected user-owned workspace
   state.
6. Reuse recorded expensive-validation witnesses. Before reading more than
   10 GB, hashing a model/checkpoint, or starting a run expected to exceed five
   minutes, check for equivalent evidence and obtain explicit approval if a
   second pass is genuinely needed.

Read-only preflight:

```bash
cd /root/workspace/ILVR/.worktrees/ours-trial1
git branch --show-current
git rev-parse HEAD
git status --short --untracked-files=all
```

## 3. Architectural contract

The implementation is a stage harmonization, not a generic token-weighting
patch.

### Stage 1: node grounding, unchanged

Stage 1 retains `CustomTrainerStage1`, `collate_fn_stage1`, its EMA teacher,
helper-image feature supervision, latent selection, CE/similarity objective,
masks, and BF16 behavior:

```text
L_stage1 = lambda_CE * L_CE + lambda_sim * L_align
```

No version-2 Stage-2 objective attaches to Stage 1. The launcher accepts no EW
config or a disabled historical version-1 config and rejects an enabled one.
Historical v1 modules/configs/tests remain for reproducibility but are not the
new method.

### Stage 2: edge grounding

Stage 2 is explicit: `CustomTrainerStage2` plus `collate_fn_stage2`. It requires
an existing Stage-1 checkpoint through `--load_model_path`, sets
`config.stage = "stage2"`, uses self-generated latent states, and trains
assistant CE. It never creates the Stage-1 EMA teacher, decodes helper images,
teacher-forces helper features, or adds Stage-1 similarity loss.

With every version-2 coefficient zero, Stage 2 returns ordinary CE after one
normal forward and does no parser, SDPA replay, reference lookup, diagnostic
matrix, or hidden-state work.

## 4. Exact `Z_j`, `R_k`, and edge semantics

### Latent evidence `Z_j`

`Z_j` is marker-inclusive:

```text
<|latent_start|> <|latent_pad|> ... <|latent_pad|> <|latent_end|>
```

The configured/tokenizer-resolved IDs are used; no Qwen token ID is hard-coded.
A delimiter pair without at least one latent-pad interior is structurally
parsed but is not valid evidence. Sequence length and positions never change.

### Reasoning window `R_k`

`R_k` is a contiguous supervised assistant-text run between latent spans. Its
mask contains only positions with active attention and `labels != -100` and
excludes padding, user/system text, latent markers/pads, tokenizer special
tokens, assistant termination, and empty regions between adjacent latent spans.

An initial assistant window before any latent remains part of CE but has no
evidence predecessor. Preserved `sequence_plan` types are validated against
the token parser, including legal leading-latent plans.

### Causal predictor-query shift

The target token at row `t` is scored from the decoder representation/logit at
row `t-1`. Therefore the attention intervention uses predictor query rows
`{t-1 : t in R_k}`, while token log-probability aggregation remains aligned to
target rows `t`. This shift is tested explicitly; reverting to target query row
`t` makes one-token windows incorrect.

All causally earlier valid evidence spans are candidate edges. A positive
`max_evidence_edges_per_window` keeps the nearest predecessors deterministically;
zero means exhaustive valid edges.

## 5. Exact window-local intervention

The only version-2 intervention is `window_attention_block`. For one edge
`(Z_j, R_k)`, a boolean `[B,T,T]` mask has `True` exactly at:

```text
query in predictor_queries(R_k) AND key/value in marker-inclusive Z_j
```

All other entries are false. It preserves previous reasoning computation,
other visual evidence, token IDs, embeddings, positions, causal masking,
padding, and multimodal RoPE assumptions. It is not latent zeroing, deletion,
global masking, or all-visual removal.

Normal CE retains FlashAttention2. Auxiliary full and blocked passes both use a
same-weight SDPA override in the vendored Qwen2.5-VL attention path. The 2D
padding mask remains separate. The block mask is sliced on both query and key
axes during the Stage-2 latent-prefix loop. An intervention without explicit
SDPA is rejected. Fully blocked rows are handled without leaking attention.

The result is a fixed-history conditional/direct dependency estimator, not the
total causal effect of removing `Z_j` and regenerating downstream reasoning.

## 6. Scores and implemented objective

Default contribution reduction is mean; sum is opt-in:

```text
ell_M(R_k | H) = mean_{t in R_k} log p_M(y_t | H, y_<t)
C^M_jk = ell_M(R_k | full) - ell_M(R_k | block R_k -> Z_j)
```

Mean avoids window-length bias. The engine performs one matched full auxiliary
forward and one exact blocked forward per selected edge. Both current-policy
branches remain attached to autograd.

The Stage-2 objective is:

```text
L_stage2 = L_CE
         + lambda_ew(r)  * L_EW
         + lambda_rel(r) * L_relation
         + lambda_res(r) * L_residual
```

Reference-free EW:

```text
L_EW = mean_(j,k in E) -log sigmoid(beta * C_policy_jk)
```

Fixed-reference EW-DPO:

```text
M_jk = C_policy_jk - stop_gradient(C_ref_jk)
L_EW-DPO = mean_(j,k in E) -log sigmoid(beta * M_jk)
```

Relation KD normalizes only valid candidates for each window:

```text
q_M(j|k) = softmax_{j in E_k}(C^M_jk / temperature)
L_relation = mean_k JSD(q_ref(.|k), q_policy(.|k))
```

A one-candidate window contributes a graph-connected zero JSD. Relation KD
does not maximize entropy or demand uniform evidence use.

Hidden residual KD pools final normalized hidden effects on predictor-query
rows:

```text
delta_h^M_jk = mean_q h^M_full[q] - mean_q h^M_LOO[j,k][q]
L_residual = mean_(j,k,h) Huber_fp32(
    delta_h_policy_jkh,
    stop_gradient(delta_h_ref_jkh)
)
```

Policy/reference hidden widths must match. There is no projection, hidden
residual subtraction, residualized prediction branch, or version-2 logit-
residual objective.

Each coefficient has configurable warmup and ramp ratios. Effective weight is
zero before warmup, linear over the ramp, and full afterward; zero ramp switches
at the boundary. Reference lookup and hidden collection follow effective—not
merely configured—weights.

## 7. Current version-2 variants

All files are under `configs/evidence_window/stage2/`:

| File | Enabled objective | Reference | Required outputs |
| --- | --- | --- | --- |
| `baseline.json` | none | none | none |
| `ew_reference_free.json` | EW | none | token log-probs |
| `ew_precomputed_dpo.json` | EW-DPO | precomputed | token log-probs |
| `relation_kd.json` | relation JSD | precomputed | token log-probs |
| `hidden_residual_kd.json` | residual Huber | precomputed | token log-probs + hidden states |
| `online_frozen_zero1.json` | EW-DPO | online frozen | token log-probs |
| `online_frozen_zero2.json` | EW-DPO | online frozen | token log-probs |

Every non-baseline file enables only the named objective. Absolute artifact or
checkpoint values are explicit placeholders to replace in a copied run config.
No source edit is needed to switch experiments.

Online JSON does not select DeepSpeed. Pair:

```text
online_frozen_zero1.json + configs/config_stage1.json  # ZeRO-1
online_frozen_zero2.json + configs/config_stage2.json  # ZeRO-2
```

`configs/config_stage2.json` is the production 8-rank sequence/data-parallel
mesh (`2 x 4`). The validated two-GPU smoke instead uses
`.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/zero2_smoke.json`,
which keeps ZeRO-2 and the same scientific objective but omits sequence
parallelism.

The similarly named DeepSpeed files never select ILVR Stage 1/2. Online frozen
is rejected under ZeRO-0 and ZeRO-3 as well as any unsupported stage. The
large-model harness additionally enforces the exact named-case pairing, so a
ZeRO-1 case cannot run with the ZeRO-2 file or vice versa. Named cases validate
their resolved objective/reference/output/intervention signature; a custom
`--ew-config` must preserve that signature and is disallowed for the mixed
`all-single-gpu` matrix.

Recommended progression:

1. explicit Stage-2 CE baseline;
2. reference-free EW to test edge-local credit without reference semantics;
3. precomputed EW-DPO;
4. relation-only KD;
5. hidden-residual-only KD;
6. online-frozen EW under ZeRO-1 and ZeRO-2 only as runtime/reference
   comparisons, not distinct scientific losses.

Keep data, Stage-1 checkpoint, seed, edge cap, reduction, beta, LoRA/dropout,
and optimization schedule fixed unless that factor is the experiment.

## 8. Reference modes and artifacts

### `none`

Valid for reference-free EW and baseline. Relation/residual KD are rejected.

### `precomputed` (recommended for 7B training)

`src/precompute_stage2_reference.py` evaluates the fixed Stage-1 initialization
using Stage-2 self-generated latents, matched SDPA, exact local blocks,
`eval()`, and `torch.no_grad()`. Scalar contributions are stored for DPO and
relation. `--include-hidden-residuals` adds pooled residual vectors.

Artifacts are immutable shards plus a small manifest. Identity includes one
dataset SHA256 witness, explicit checkpoint ID/witness, tokenizer and special
tokens, parser version, reduction, intervention/backend, edge cap, and model
identity. The producer hashes the dataset at most once unless an authenticated
witness is supplied and never hashes checkpoint weights. Training reconstructs
its own expected identity from CLI arguments and rejects any mismatch.

### `online_frozen`

The exact Stage-1 policy initialization checkpoint is loaded independently,
not copied from a DeepSpeed-managed policy. It runs with `config.stage =
"stage2"`, self-generated latents, eval/no-grad, and no optimizer state. It is
not the Stage-1 EMA teacher.

Before Accelerator preparation, all floating reference parameters and buffers
are placed on the execution device and normalized to BF16 on CUDA or FP32 on
CPU. This is required because the validated Stage-1 checkpoint contains mixed
physical dtypes and online reference scoring does not rely on autocast.

ZeRO-1 and ZeRO-2 do not parameter-shard the frozen reference, so every rank
owns a full replica plus transient reference activations. The current model
estimates should reserve roughly one extra 7B BF16 parameter copy per rank.
ZeRO-3 is rejected before construction rather than silently changing objective
semantics. Precomputed targets are the lower-memory recommendation.

## 9. Complexity and distributed behavior

For `E_local` selected edges, one policy step has:

```text
1 ordinary CE forward
+ 1 matched full auxiliary forward
+ E_local pair-blocked auxiliary forwards
```

Thus local auxiliary complexity is `O(1 + E_local)`. This exact first
implementation is per edge, not the earlier per-evidence-span optimization.
Distributed ranks coordinate edge counts and execute one full plus
`E_global_max` edge slots. Ranks with fewer edges use all-false same-policy
dummy slots with graph-zero participation. Online references use the same
padded no-grad schedule. Hidden states are requested only for active residual
loss.

## 10. Logging and scientific interpretation

The trainer aggregates distributed sufficient statistics for CE/EW/relation/
residual losses, scheduled weights, mean contribution, positive-edge ratio,
edge counts, per-window means, global diagnostic margin and outer coefficient,
top-1 evidence mass, normalized entropy, threshold counts, and residual effect
norms. Interval-gated matrices contain only numeric values and opaque IDs.

Do not claim that LOO is total causal effect, attention weights prove use,
sparse dependency is bad, every window needs every evidence, or high entropy is
inherently better. Implementing diagnostics does not verify the saturation
hypothesis. Single-evidence LOO cannot identify all synergy/redundancy; group,
Shapley, and supervised required-edge methods are out of scope.

## 11. Commands

```bash
cd /root/workspace/ILVR/.worktrees/ours-trial1
source /root/workspace/ILVR/venv/ilvr-venv/bin/activate
```

Reference-free Stage 2:

```bash
accelerate launch src/main.py \
  --stage stage2 \
  --model /absolute/path/to/Qwen2.5-VL-7B-Instruct \
  --load_model_path /absolute/path/to/stage1-checkpoint \
  --task zebra-cot \
  --data_path /absolute/path/to/TRAIN.jsonl \
  --save_model_path /absolute/path/to/stage2-output \
  --log_file /absolute/path/to/stage2.log \
  --cache_dir /absolute/path/to/cache \
  --ew_config configs/evidence_window/stage2/ew_reference_free.json \
  --deepspeed_config configs/config_stage3.json
```

Precompute reference contributions (add `--include-hidden-residuals` only for
residual KD):

```bash
/root/workspace/ILVR/venv/ilvr-venv/bin/python \
  src/precompute_stage2_reference.py \
  --data-path /absolute/path/to/TRAIN.jsonl \
  --reference-checkpoint /absolute/path/to/stage1-checkpoint \
  --reference-checkpoint-id stage1-approved \
  --reference-checkpoint-witness /absolute/or-authenticated-checkpoint-witness \
  --artifact-dir /absolute/path/to/stage2-reference-artifact \
  --tokenizer-id /absolute/path/to/stage1-checkpoint \
  --model-config-id qwen2.5-vl-stage2-self-latent-v1 \
  --contribution-reduction mean
```

Focused CPU smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:transformers/src \
  /root/workspace/ILVR/venv/ilvr-venv/bin/python \
  -m pytest tests/evidence_window/test_stage2_experiment_configs.py \
  tests/evidence_window/test_stage2_smoke.py -q
```

Callable large harness (check prior expensive evidence first):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:transformers/src \
  /root/workspace/ILVR/venv/ilvr-venv/bin/python \
  .superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/stage2_large_model_smoke.py \
  --case all-single-gpu \
  --model /absolute/path/to/Qwen2.5-VL-7B-Instruct \
  --stage1-checkpoint /absolute/path/to/stage1-checkpoint \
  --data /absolute/path/to/TRAIN.jsonl \
  --image-root /absolute/path/to/image-root \
  --sample-index 920 \
  --attempt-id ew-v2-smoke-001 \
  --deepspeed-config configs/config_stage3.json \
  --reference-artifact-dir /absolute/path/to/stage2-reference-artifact-with-hidden-residuals \
  --reference-dataset-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --reference-checkpoint-id stage1-approved \
  --reference-checkpoint-witness /absolute/or-authenticated-checkpoint-witness \
  --reference-tokenizer-id /absolute/path/to/stage1-checkpoint \
  --reference-parser-version stage2-marker-v1 \
  --reference-model-config-id qwen2.5-vl-stage2-self-latent-v1 \
  --output-dir /absolute/path/to/stage2-smoke-results
```

The output path must not exist. A single-process invocation may omit launcher
state. If any distributed launcher variable is present, the harness requires a
complete valid `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, and
`MASTER_PORT`; missing values are never defaulted. Optional `LOCAL_WORLD_SIZE`,
the port range, and the local rank versus visible CUDA devices are validated
before selection or rendezvous. The harness creates the launcher-provided
`env://` process group with NCCL on the CUDA target (Gloo on CPU). A group that
is already initialized is reused only after its rank/world/backend and local
device match this derived contract. Any mismatch fails before processor/model
load. Rank 0 then atomically creates the output path as an exclusive ownership
claim, all ranks collectively agree on claim success, and only afterward does
each rank validate the claim and run preflight. A concurrent launcher or retry
therefore loses on every rank before any model load. `--attempt-id` is path-safe
and is bound into `attempt.json`, every rank filename and JSON, and the final
manifest. Retries use both a new attempt ID and a new output path.

For precomputed cases the harness independently reconstructs
`ArtifactIdentity` from these consumer-side values plus the current token IDs,
parser/reduction/intervention/backend, and edge cap. It never treats the
artifact manifest as its own expected identity and never hashes checkpoint
weights. Before model work, all selected cases are resolved together and their
config signature, exact online ZeRO pairing, obvious paths, reference manifest,
and independent identity fields are checked. Exact tokenizer special-token
identity remains the processor-derived check.

After training, every rank validates its case-specific loss/edge/window,
finite total, exact policy/reference/teacher forward counts, nonzero policy
learning, and frozen reference/EMA gradient witnesses. Those errors are
collectively aggregated before any result publication. Rank JSON files use
same-directory temporaries that are serialized, flushed, and file-`fsync`ed
before an atomic no-overwrite hard link; pre-link/link publication errors are
collectively aggregated. Every rank then checks the complete expected file set
and attempt identity in the final precommit collective. After that agreement,
non-rank0 processes may exit and rank 0 links `completion.json` last. A
successful hard link is the logical terminal commit. Temporary cleanup and
directory `fsync` occur afterward as best-effort, non-throwing work: failure
cannot retract visible completion, although crash durability was not
additionally confirmed when directory `fsync` fails. Link failure leaves no
destination. No collective or required success operation follows the terminal
link. Consumers treat only `completion.json` as success. A failure before or at
the completion link leaves no completion; once linked, it remains logically
committed even if cleanup/sync or a later process exit fails. Partial attempts
remain immutable diagnostic state and are never reused.

Online ZeRO-1/2 cases require separate two-rank `accelerate launch` commands
with their matching DeepSpeed files. The EW JSON never controls ZeRO.

## 12. TDD and implementation commits

Commits through the Task-13 base:

```text
26e753e feat: define stage2 evidence-window config
b2e70cf feat: add helper-free stage2 data path
7d67174 feat: separate explicit stage2 baseline
d0bd413 feat: parse stage2 evidence-window edges
04e4f96 fix: accept leading latent stage2 plans
a696d6d feat: build exact window attention blocks
244dd67 feat: add exact stage2 sdpa intervention
553a120 fix: preserve auxiliary attention reach
cc1f656 feat: compute exact stage2 evidence effects
dd44ad5 fix: align stage2 effects with causal queries
9cd4f31 fix: prevent fully blocked attention leakage
4cc5244 feat: train stage2 with local evidence preference
ce851f4 fix: coordinate stage2 auxiliary replays
9931db1 feat: add precomputed stage2 reference targets
0b5a282 fix: harden stage2 reference artifacts
582cce5 feat: support stage2 frozen reference on zero1 and zero2
b503a3a fix: harden stage2 online reference semantics
ae4443f feat: distill stage2 evidence relations
1f94e93 fix: stabilize stage2 relation distillation
ea6def7 fix: preserve relation graph zero across dtypes
2e65d0d feat: distill stage2 hidden evidence effects
f5f41f3 fix: coordinate stage2 residual target preflight
3a7294e fix: allow edge-free precomputed residual rows
0250326 docs: add stage2 evidence-window variants
dddaa44 fix: record stage1 smoke forward witnesses
9f97f67 fix: harden stage2 smoke witnesses
33f6fd0 fix: publish distributed smoke completion
f0e8521 fix: make smoke completion terminal
08dc0b6 fix: coordinate distributed smoke startup
```

Task reports under
`.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit-implementation/`
record RED/GREEN evidence and review rounds. Task 13 observed the new config/
smoke suite fail because all seven v2 files and the large harness were missing,
then pass after implementation. Important-review fix round 1 observed `15
failed, 14 passed`, then `32 passed` after named-case signatures, exact ZeRO
pairing, independently reconstructed artifact identity, validated atomic result
publication, real CE/auxiliary call accounting, and absolute-venv documentation
were added. Fix round 2 observed `6 failed, 23 passed` (plus one focused
malformed-manifest RED), then `40 passed` after all-case preflight and the
completion-manifest distributed publication protocol were implemented. Task 14
owns the full suite and final review. Fix round 3 observed `14 failed, 21
passed`, then `44 passed` after exclusive attempt ownership, attempt-bound
no-overwrite files, final collective precommit, and terminal completion with no
post-publication collective were added.
Fix round 4 observed `5 failed, 36 passed`, then `50 passed` after the harness
initialized a real launcher process group before ownership/preflight, enforced
rank-0 claim followed by collective agreement and all-rank preflight, and made
post-link cleanup/directory sync non-throwing. The focused tests include actual
spawned two-rank Gloo success, asymmetric-preflight, and concurrent-launcher
cases plus injected post-link and hard-link failures.
Fix round 5 observed the fail-closed launcher, preinitialized-group/device, and
simultaneous rendezvous-port reservation tests fail against round 4, then pass
as part of `74 passed`. The harness now rejects incomplete or malformed
distributed state before rendezvous, validates and selects the local device
before reusing a matching group, rejects rank/world/backend mismatches, and the
test launcher holds all candidate ports open together until a controlled
release immediately before child start.

## 13. Known risks and limitations

- Exact per-edge replay is expensive; batching/FlexAttention optimizations are
  deferred until equivalence with the SDPA oracle is established.
- Online frozen reference replicates full weights per ZeRO-1/2 rank and may be
  memory-infeasible. No ZeRO-3 fallback exists.
- Hidden artifacts can be large and are eagerly loaded to CPU.
- Production preprocessing resolves relative image paths through
  `task_deepseed.TRAIN_IMAGE_ROOT`; `src/main.py` has no image-root CLI. The
  smoke harness sets this from `--image-root`. Production data should use
  absolute paths or an environment-specific repository setting.
- The validated online-reference CUDA dtype is BF16. FP16/FP32 CUDA online
  reference operation has not been established.
- The exact intervention is implemented only for customized Qwen2.5-VL.
- Long-range eligibility is causal, not supervised as task-required.
- Relation-only initialization can have legitimate zero JSD at identical
  initialization; gradient/graph connectivity remains the smoke witness.

## 14. Source map

```text
README.md
docs/superpowers/specs/2026-08-22-ilvr-evidence-window-credit-design.md
docs/superpowers/plans/2026-08-22-ilvr-stage2-evidence-window-credit-implementation.md
.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit-implementation/progress.md
.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit-implementation/task-13-report.md
src/main.py
src/stage2_data.py
src/trainer_stage2.py
src/precompute_stage2_reference.py
src/evidence_window/stage2_config.py
src/evidence_window/stage2_layout.py
src/evidence_window/attention.py
src/evidence_window/stage2_effects.py
src/evidence_window/stage2_engine.py
src/evidence_window/stage2_reference.py
src/evidence_window/artifacts.py
src/evidence_window/losses.py
src/evidence_window/metrics.py
transformers/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
configs/evidence_window/stage2/
tests/evidence_window/test_stage2_*.py
```

## 15. Final verification witnesses and how to continue

Detailed commands, results, failure diagnoses, and limitations are recorded at:

```text
/root/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/verification.md
```

Successful immutable completion manifests:

```text
/root/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/large-results/single-009/completion.json
/root/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/large-results/zero1-011/completion.json
/root/workspace/ILVR/.worktrees/ours-trial1/.superpowers/sdd/2026-08-22-ilvr-stage2-evidence-window-credit/large-results/zero2-013/completion.json
```

The final full suite result is `442 passed in 262.99s`; compileall and
`git diff --check` passed. The two-rank ZeRO-3 latent ledger also passed with
unequal rank-local latent counts (1 and 3), exactly eight decoder calls on both
ranks, finite backward, and a LoRA update.

Relevant Task-14 completion commits after the earlier list include:

```text
bedcc2e fix: render stage2 latent placeholders safely
cff0422 fix: normalize precompute reference dtype
00dde2c fix: preserve smoke lora checkpoint gradients
79ad791 fix: make stage2 prefix checkpoints zero3-safe
67a55ba test: authenticate smoke global grad norm
0d97866 fix: normalize online reference dtype
e5c1012 test: add two-rank zero2 smoke config
```

For a new session, inspect the branch and the three small completion manifests,
then continue only with a new explicitly requested experiment or integration
action. Do not rerun the successful 7B/B200 validators, rehash the checkpoint,
restart the architecture, substitute zeroing for attention blocking, or
silently change reference modes to save memory.
